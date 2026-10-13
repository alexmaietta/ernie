// SPDX-License-Identifier: BSD-3-Clause
// Long-range shadows (raster/long_range_shadows.h): the share of a distant light's disc that the
// casters in its long-range view (shadow_cascades.h's `FitLongRangeView`) leave the receivers, such
// as a moon eclipsing the sun, for any caster shape at any distance.
//
// `coverageMain` marks the view's texels a caster covers, those whose depth falls short of the far
// plane, onto which the receivers' own casters clamp: x 1 covered, y its depth there. Then
// `coverageMipMain` averages them down a mip chain, x the covered fraction and y the covered
// depths' sum over the texels, so y / x is their mean depth.
//
// `visibilityMain` integrates the light's limb-darkened disc over each texel of the visibility map,
// which spans the receivers' part of the view's (`region`), not the margin round it that holds the
// casters off it: the casters there, at their mean depth, see the light's disc as a circle
// on the map of radius their distance from the receivers times its angular radius's tangent,
// about the receivers' point; across rings of it, the fraction the casters cover, read at the mip
// as fine as the ring's width, weighted by the brightness the power law of limb darkening gives
// each ring, as atmosphere.slang's `eclipseVisibility` weights a body's. The receivers lie at one
// depth, their box's middle (`LongRangeConstants::params.y`): for casters far beyond them, such as
// a moon thousands of times farther than their box is deep, every receiver sees the same penumbra.
// It is written at four wavelengths (`params.w` the visibility's layer, `wavelengths`), between
// which shading interpolates by the limb darkening exponent.
//
// References:
// - S. Schneegans, J. Gilg, V. Ahlers and A. Gerndt, "Real-Time Rendering of Eclipses without
//   Incorporation of Atmospheric Effects", Computer Graphics Forum 41(7):279-289, 2022 (the
//   occluded fraction of a limb-darkened disc, ring by ring).
// - D. Hestroffer and C. Magnan, "Wavelength dependency of the Solar limb darkening", Astronomy
//   and Astrophysics 333:338-342, 1998 (the power law).

struct LongRangeConstants {
    // x: texels across the view's map (in its layer's corner of the shadow maps), y: its layer,
    // z: texels across the visibility map, w: the coverage's mips.
    uint4 size;
    // x: the tangent of the light's angular radius in map widths per unit of map depth, y: the
    // receivers' depth (past the far plane, so above 1), z: the visibility's layer.
    float4 params;
    // The wavelengths (nm) the visibility is written at.
    float4 wavelengths;
    // The square of the view's map the visibility map spans, the receivers' box: xy its corner
    // nearest the origin and z its side, in map coordinates (0 to 1).
    float4 region;
}

[[vk::push_constant]] ConstantBuffer<LongRangeConstants> constants;

// Depths at or past this lie on the far plane: no long-range caster.
static const float kFarPlane = 0.999999;
// Rings across the light's disc (atmosphere.slang's `kEclipseRings`).
static const uint kRings = 16;
static const float kPi = 3.14159265358979;

// `coverageMain`: the shadow maps (an array) and the coverage's first mip.
[[vk::binding(0, 0)]] Texture2DArray<float> shadowMaps;
[[vk::binding(1, 0)]] RWTexture2D<float2> coverageOut;

[shader("compute")]
[numthreads(8, 8, 1)]
void coverageMain(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= constants.size.xx)) {
        return;
    }
    let depth = shadowMaps.Load(int4(int2(id.xy), int(constants.size.y), 0));
    let covered = depth < kFarPlane ? 1.0 : 0.0;
    coverageOut[id.xy] = float2(covered, covered * depth);
}

// `coverageMipMain`: a finer mip and the next, `size.x` the finer's texels across.
[[vk::binding(2, 0)]] Texture2D<float2> finer;
[[vk::binding(3, 0)]] RWTexture2D<float2> coarser;

[shader("compute")]
[numthreads(8, 8, 1)]
void coverageMipMain(uint3 id : SV_DispatchThreadID) {
    let texels = max(constants.size.x / 2, 1u);
    if (any(id.xy >= uint2(texels))) {
        return;
    }
    let last = int(constants.size.x) - 1;
    let at = int2(id.xy) * 2;
    var sum = float2(0.0);
    for (uint i = 0; i < 4; ++i) {
        let t = min(at + int2(i & 1, i >> 1), int2(last));
        sum += finer.Load(int3(t, 0));
    }
    coarser[id.xy] = 0.25 * sum;
}

// `visibilityMain`: the coverage with all its mips, and the visibility maps (an array).
[[vk::binding(4, 0)]] Texture2D<float2> coverage;
[[vk::binding(5, 0)]] RWTexture2DArray<float4> visibilityOut;

// The coverage at `uv` (the view's map) and mip `mip` (rounded down), bilinear between texels;
// nothing off the map.
float2 coverageAt(float2 uv, float mip) {
    if (any(uv < float2(0.0)) || any(uv > float2(1.0))) {
        return float2(0.0);
    }
    let at = min(uint(max(mip, 0.0)), constants.size.w - 1);
    let texels = max(constants.size.x >> at, 1u);
    let p = uv * float(texels) - 0.5;
    let base = int2(floor(p));
    let f = p - floor(p);
    let edge = int2(int(texels) - 1);
    let a = coverage.Load(int3(clamp(base, int2(0), edge), int(at)));
    let b = coverage.Load(int3(clamp(base + int2(1, 0), int2(0), edge), int(at)));
    let c = coverage.Load(int3(clamp(base + int2(0, 1), int2(0), edge), int(at)));
    let d = coverage.Load(int3(clamp(base + int2(1, 1), int2(0), edge), int(at)));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

// Hestroffer and Magnan's limb darkening exponent at `wavelength` nm (atmosphere.slang's
// `limbDarkening`).
float limbDarkening(float wavelength) {
    return max(-0.023 + 292.0 / wavelength, 0.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void visibilityMain(uint3 id : SV_DispatchThreadID) {
    let texels = constants.size.z;
    if (any(id.xy >= uint2(texels))) {
        return;
    }
    let layer = uint(constants.params.z);
    let uv = constants.region.xy + (float2(id.xy) + 0.5) / float(texels) * constants.region.z;
    let mapTexels = float(constants.size.x);
    let tanUv = constants.params.x;
    let receivers = constants.params.y;
    // The casters near enough to shade the receivers here: those within the widest penumbra, a
    // caster's at the near plane, read at the mip that wide.
    let widest = receivers * tanUv;
    let near = coverageAt(uv, log2(max(2.0 * widest * mapTexels, 1.0)));
    if (near.x <= 1e-6) {
        visibilityOut[uint3(id.xy, layer)] = float4(1.0);
        return;
    }
    // Their mean depth, and the light's disc's radius on the map as they see it.
    let casters = near.y / near.x;
    let radius = max(receivers - casters, 0.0) * tanUv;
    // A disc within a texel: the coverage under the receivers' point.
    if (radius * mapTexels < 0.5) {
        visibilityOut[uint3(id.xy, layer)] = float4(saturate(1.0 - coverageAt(uv, 0.0).x));
        return;
    }
    var exponent = float4(0.0);
    for (uint k = 0; k < 4; ++k) {
        exponent[k] = 0.5 * limbDarkening(constants.wavelengths[k]) + 1.0;
    }
    // Each ring read at the mip as fine as its width, at points about as far apart.
    let width = radius / float(kRings);
    let mip = log2(max(width * mapTexels, 1.0));
    var covered = float4(0.0);
    var inside = float4(0.0);
    for (uint ring = 0; ring < kRings; ++ring) {
        let outer = float(ring + 1) / float(kRings);
        let rho = (float(ring) + 0.5) * width;
        let points = clamp(uint(ceil(2.0 * kPi * rho / width)), 4u, 32u);
        var fraction = 0.0;
        for (uint j = 0; j < points; ++j) {
            let a = 2.0 * kPi * (float(j) + 0.5 * float(ring & 1)) / float(points);
            fraction += coverageAt(uv + rho * float2(cos(a), sin(a)), mip).x;
        }
        fraction /= float(points);
        let within = 1.0 - pow(float4(max(1.0 - outer * outer, 0.0)), exponent);
        covered += fraction * (within - inside);
        inside = within;
    }
    visibilityOut[uint3(id.xy, layer)] = saturate(1.0 - covered);
}
