// SPDX-License-Identifier: BSD-3-Clause
// Motion blur and rolling shutter: sums the sub-frames' raster radiance, rendered at evenly spaced
// times t_k across the shutter and readout, into what each row integrates over its exposure. The
// radiance between sub-frames is taken as their linear interpolation, so sub-frame k weighs, for a
// row exposed over [lo, hi], the mean of its hat function (1 at t_k, 0 at its neighbours) there;
// an instantaneous row (lo = hi) interpolates the two sub-frames around it. With a rolling shutter
// the rows' intervals shift: the top row's comes `rolling` frames before the bottom row's.
//
// Pixels no prim covers hold radiance only with a sky; otherwise they count as none. The share of
// each pixel's exposure that prims covered accumulates too, which the sensor blends the clear
// colour by.
//
// A camera rendered on cube faces times each face texel by the sensor row its lens maps it to,
// from a map of them (sensor.slang's `faceRowsMain`), not by its own row.
//
// Bindings: 0 the sub-frame's raster radiance (the sum after the last), 1 the sum so far, 2 the
// sub-frame's ids (G-buffer 2), 3 the coverage (z of an RGBA32F image), 4 a cube camera's faces'
// sensor rows (a layer per face; a placeholder for other cameras).
//
// References:
// - P. Haeberli and K. Akeley, "The Accumulation Buffer: Hardware Support for High-Quality
//   Rendering", SIGGRAPH '90, 309-318, 1990. https://doi.org/10.1145/97880.97913
// - R. L. Cook, T. Porter and L. Carpenter, "Distributed Ray Tracing", SIGGRAPH '84, 137-145,
//   1984. https://doi.org/10.1145/964965.808590
// - C.-K. Liang, L.-W. Chang and H. H. Chen, "Analysis and Compensation of Rolling Shutter
//   Effect", IEEE Transactions on Image Processing 17(8), 1323-1330, 2008.
//   https://doi.org/10.1109/TIP.2008.925384

struct MotionConstants {
    // x: this sub-frame, y: sub-frames, z: bands, w: 1 when the background holds the sky.
    uint4 frame;
    // x: the first sub-frame's time, y: the spacing, z: the shutter's opening, w: its closing,
    // all in frames.
    float4 times;
    // x: the rolling readout, in frames; y: the cube faces whose rows `faceRows` holds, 0 for
    // none.
    float4 rolling;
}

static const uint kBackground = 0xFFFFFFFF;

[[vk::binding(0, 0)]] [[vk::image_format("r32f")]] RWTexture2DArray<float> radiance;
[[vk::binding(1, 0)]] [[vk::image_format("r32f")]] RWTexture2DArray<float> sum;
[[vk::binding(2, 0)]] Texture2DArray<uint4> ids;
// The coverage is z of `RenderTargets::Raster`, whose xy are left as they are.
[[vk::binding(3, 0)]] [[vk::image_format("rgba32f")]] RWTexture2DArray<float4> coverage;
// Float bits of the share of the sensor's height down to the row; 0xFFFFFFFF for none.
[[vk::binding(4, 0)]] [[vk::image_format("r32ui")]] RWTexture2DArray<uint> faceRows;
[[vk::push_constant]] ConstantBuffer<MotionConstants> constants;

// The integral of the hat max(0, 1 - |x|) from minus infinity to u.
float hatIntegral(float u) {
    if (u <= -1.0) {
        return 0.0;
    }
    if (u <= 0.0) {
        return 0.5 * (u + 1.0) * (u + 1.0);
    }
    if (u <= 1.0) {
        return 1.0 - 0.5 * (1.0 - u) * (1.0 - u);
    }
    return 1.0;
}

// Sub-frame k's weight in the exposure [lo, hi].
float subframeWeight(uint k, float lo, float hi) {
    let step = constants.times.y;
    let t = constants.times.x + float(k) * step;
    if (hi - lo <= 1e-6 * step) {
        return max(0.0, 1.0 - abs(lo - t) / step);
    }
    return step * (hatIntegral((hi - t) / step) - hatIntegral((lo - t) / step)) / (hi - lo);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void motionMain(uint3 id : SV_DispatchThreadID) {
    uint width = 0;
    uint height = 0;
    uint layers = 0;
    radiance.GetDimensions(width, height, layers);
    if (id.x >= width || id.y >= height) {
        return;
    }
    let k = constants.frame.x;
    let count = constants.frame.y;
    let bands = constants.frame.z;
    let view = id.z / bands;
    // The row's exposure, the top row's `rolling` frames before the bottom row's.
    var row = (float(id.y) + 0.5) / float(height);
    let faces = uint(constants.rolling.y);
    if (faces > 1) {
        let bits = faceRows[uint3(id.xy, view % faces)];
        row = bits == 0xFFFFFFFFu ? 0.5 : asfloat(bits);
    }
    let shift = -constants.rolling.x * (1.0 - row);
    let w = subframeWeight(k, constants.times.z + shift, constants.times.w + shift);
    let covered = ids.Load(int4(int2(id.xy), int(view), 0)).x != kBackground;
    let value = covered || constants.frame.w != 0 ? radiance[id] : 0.0;
    let total = (k == 0 ? 0.0 : sum[id]) + w * value;
    if (k + 1 < count) {
        sum[id] = total;
    } else {
        radiance[id] = total;
    }
    if (id.z % bands == 0) {
        let at = uint3(id.xy, view);
        var texel = coverage[at];
        texel.z = (k == 0 ? 0.0 : texel.z) + (covered ? w : 0.0);
        coverage[at] = texel;
    }
}
