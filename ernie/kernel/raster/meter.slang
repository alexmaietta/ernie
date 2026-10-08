// Exposure metering: a histogram of the base-2 logarithm of luminance (cd/m^2) over one layer of
// HdrColor, for automatic exposure and exposure meters. Each workgroup counts its pixels into
// shared bins, then adds them to the result with one atomic per bin; pixels without light
// (luminance 0, such as the background) count in none.
//
// Bindings: 0 HdrColor (linear sRGB in cd/m^2), 1 the result: kBins counts, cleared to 0.
//
// Reference: K. Narkowicz, "Automatic Exposure", 2016,
// https://knarkowicz.wordpress.com/2016/01/09/automatic-exposure/ (a log-luminance histogram
// averaged without its darkest and brightest tails).

static const uint kBins = 64;

struct MeterConstants {
    // xy: image size, z: the layer.
    uint4 size;
    // x: log2 luminance at the first bin's low edge, y: bins per stop.
    float4 range;
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
