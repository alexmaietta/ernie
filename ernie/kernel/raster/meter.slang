// SPDX-License-Identifier: BSD-3-Clause
// Exposure metering: a histogram of the base-2 logarithm of luminance (cd/m^2) over one layer of
// HdrColor, for automatic exposure and exposure meters. Each workgroup counts its pixels into
// shared bins, then adds them to the result with one atomic per bin; pixels without light
// (luminance 0, such as the background) count in none.
//
// Bindings: 0 HdrColor (linear sRGB in cd/m^2), 1 the result: kBins counts, cleared to 0.
//
// Automatic exposure stays on the GPU: `adaptMain` turns the histogram into the stops that bring
// the scene to middle grey, adapting a key's state towards them, and the next frame's
// `applyMain` scales its exposure by them where its uniforms hold it, so no frame waits for the
// host to read the meter.
//
// Reference: K. Narkowicz, "Automatic Exposure", 2016,
// https://knarkowicz.wordpress.com/2016/01/09/automatic-exposure/ (a log-luminance histogram
// averaged without its darkest and brightest tails).

static const uint kBins = 64;

// The linear value an average scene is keyed to.
static const float kMiddleGrey = 0.18;

struct MeterConstants {
    // xy: image size, z: the layer.
    uint4 size;
    // x: log2 luminance at the first bin's low edge, y: bins per stop; adapting, z the share of
    // the way to the target taken (1 at once) and w the camera's linear exposure scale.
    float4 range;
    // Adapting: the histogram (kBins counts).
    uint* histogram;
    // Each automatic exposure's state: x its stops, y the luminance last metered (cd/m^2), z 1
    // once metered.
    float4* states;
    // Applying: the sensor's photographic exposure scale (`SensorUniforms::lens[2]`), the first
    // view's exposure (`Uniforms::eye[3]`), and the gain the presented SpectralRadiance takes.
    float* lens;
    float* eyes;
    float* gain;
    // x: the state's index, y: views, z: floats between views' exposures.
    uint4 layout;
}

[[vk::binding(0, 0)]] Texture2DArray<float4> hdrColor;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> bins;
[[vk::push_constant]] ConstantBuffer<MeterConstants> constants;

groupshared uint counts[kBins];

[shader("compute")]
[numthreads(16, 16, 1)]
void meterMain(uint3 id : SV_DispatchThreadID, uint index : SV_GroupIndex) {
    if (index < kBins) {
        counts[index] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (all(id.xy < constants.size.xy)) {
        let rgb = hdrColor.Load(int4(int2(id.xy), int(constants.size.z), 0)).rgb;
        let luminance = dot(rgb, float3(0.2126, 0.7152, 0.0722));
        if (luminance > 0.0) {
            let position = (log2(luminance) - constants.range.x) * constants.range.y;
            let bin = uint(clamp(position, 0.0, float(kBins - 1)));
            InterlockedAdd(counts[bin], 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (index < kBins && counts[index] > 0) {
        InterlockedAdd(bins[index], counts[index]);
    }
}

// One thread: the mean log luminance of the histogram's pixels between the darkest tenth and the
// brightest twentieth, so a few very dark or bright ones do not swing it, and the stops that key
// it to middle grey, taken `range.z` of the way from the state's (at once on its first). A frame
// without lit pixels leaves the state as it is.
[shader("compute")]
[numthreads(1, 1, 1)]
void adaptMain() {
    var total = 0u;
    for (uint b = 0; b < kBins; ++b) {
        total += constants.histogram[b];
    }
    if (total == 0) {
        return;
    }
    let low = 0.10 * float(total);
    let high = 0.95 * float(total);
    var seen = 0.0;
    var weighted = 0.0;
    var counted = 0.0;
    for (uint b = 0; b < kBins; ++b) {
        let count = float(constants.histogram[b]);
        let from = max(seen, low);
        let to = min(seen + count, high);
        if (to > from) {
            weighted += (constants.range.x + (float(b) + 0.5) / constants.range.y) * (to - from);
            counted += to - from;
        }
        seen += count;
    }
    if (counted <= 0.0) {
        return;
    }
    let luminance = exp2(weighted / counted);
    let target = log2(kMiddleGrey / (luminance * constants.range.w));
    let index = constants.layout.x;
    var state = constants.states[index];
    state.x = state.z > 0.0 ? state.x + (target - state.x) * constants.range.z : target;
    state.y = luminance;
    state.z = 1.0;
    constants.states[index] = state;
}

// One thread per view: scales the frame's exposure, where its uniforms hold it, by its automatic
// exposure's stops (none before its first metering).
[shader("compute")]
[numthreads(64, 1, 1)]
void applyMain(uint3 id : SV_DispatchThreadID) {
    let state = constants.states[constants.layout.x];
    let scale = state.z > 0.0 ? exp2(state.x) : 1.0;
    if (id.x == 0) {
        constants.lens[0] *= scale;
        constants.gain[0] = scale;
    }
    if (id.x < constants.layout.y) {
        constants.eyes[id.x * constants.layout.z] *= scale;
    }
}
