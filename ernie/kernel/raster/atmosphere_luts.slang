// SPDX-License-Identifier: BSD-3-Clause
// The atmosphere's tables (atmosphere.slang), rebuilt when the atmosphere or the spectral bands
// change: the transmittance to the top of the atmosphere, then Hillaire's multiple scattering
// Psi_ms from it. Dispatch z is the band group, each thread four bands.
//
// Multiple scattering, per height and sun zenith cosine: over 64 directions spread evenly on the
// sphere (a Fibonacci lattice), a march to the top or the ground gathers the light scattered
// once from the sun with an isotropic phase function (L_2nd) and the fraction of light arriving
// from all around that scatters there (f_ms), plus the sun's light the ground reflects
// (Lambertian). Light scattered n times is then L_2nd f_ms^(n-2), whose sum is
// L_2nd / (1 - f_ms); f_ms < 1 as every path loses light to the ground, absorption or space,
// though it nears 1 in thick, bright atmospheres such as dusty Mars, where the closed form keeps
// orders a truncated sum would lose. Each step integrates its in-scattering analytically for
// constant coefficients (Hillaire 2015).
//
// Bindings: 0 the medium, 1 the constituents' densities, 2 the transmittance table (storage),
// 3 the transmittance table (sampled), 4 a linear sampler, 5 the multiple scattering table
// (storage). Every image is in the general layout.
//
// References:
// - S. Hillaire, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique",
//   Computer Graphics Forum 39(4):13-22, 2020, section 5.5. https://doi.org/10.1111/cgf.14050
// - S. Hillaire, "Physically Based and Unified Volumetric Rendering in Frostbite", SIGGRAPH 2015
//   Advances in Real-Time Rendering course (energy-conserving integration of in-scattering).
// - E. Bruneton, "Precomputed Atmospheric Scattering: a New Implementation", 2017 (the
//   transmittance table). https://ebruneton.github.io/precomputed_atmospheric_scattering/
// - R. Marques, C. Bouville, M. Ribardiere, L. P. Santos and K. Bouatouch, "Spherical Fibonacci
//   Point Sets for Illumination Integrals", Computer Graphics Forum 32(8):134-143, 2013.
//   https://doi.org/10.1111/cgf.12190
import atmosphere;

static const float PI = 3.14159265358979;
// Steps along each transmittance ray: a horizontal ray crosses some 1100 km of Earth's atmosphere,
// and its haze's scale height is 1.2 km.
static const uint kTransmittanceSteps = 500;
static const uint kDirections = 64;
// Steps along each direction: rays near the horizon cross hundreds of kilometres, and fewer than
// about a hundred steps miss the haze near the ground (Hillaire's 20 suit a camera's sky, not a
// reference).
static const uint kScatteringSteps = 128;

[[vk::binding(0, 0)]] StructuredBuffer<float4> medium;
[[vk::binding(1, 0)]] StructuredBuffer<float> densities;
[[vk::image_format("rgba32f")]] [[vk::binding(2, 0)]] RWTexture2DArray<float4> transmittanceOut;
[[vk::binding(3, 0)]] Texture2DArray<float4> transmittance;
[[vk::binding(4, 0)]] SamplerState linear;
[[vk::image_format("rgba32f")]] [[vk::binding(5, 0)]] RWTexture2DArray<float4> multipleOut;

[shader("compute")]
[numthreads(8, 8, 1)]
void transmittanceMain(uint3 id: SV_DispatchThreadID) {
    if (id.x >= kTransmittanceWidth || id.y >= kTransmittanceHeight) {
        return;
    }
    let bottom = atmosphereBottom(medium);
    let top = atmosphereTop(medium);
    let uv = (float2(id.xy) + 0.5) / float2(kTransmittanceWidth, kTransmittanceHeight);
    float r;
    float mu;
    transmittanceRMu(uv, bottom, top, r, mu);
    let span = distanceToTop(r, mu, top);
    let step = span / float(kTransmittanceSteps);
    var depth = float4(0.0);
    for (uint i = 0; i < kTransmittanceSteps; ++i) {
        let t = (float(i) + 0.5) * step;
        let radius = sqrt(r * r + t * t + 2.0 * r * mu * t);
        float4 scattering;
        float4 extinction;
        mediumAt(medium, densities, radius - bottom, id.z, scattering, extinction);
        depth += extinction * step;
    }
    transmittanceOut[id] = exp(-depth);
}

// Direction `i` of `count` on the unit sphere, by the spherical Fibonacci lattice.
float3 fibonacciDirection(uint i, uint count) {
    let z = 1.0 - (2.0 * float(i) + 1.0) / float(count);
    let radius = sqrt(max(1.0 - z * z, 0.0));
    let phi = float(i) * 2.39996322972865;  // the golden angle, pi (3 - sqrt(5))
    return float3(radius * cos(phi), radius * sin(phi), z);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void multipleScatteringMain(uint3 id: SV_DispatchThreadID) {
    let size = kMultipleScatteringSize;
    if (id.x >= size || id.y >= size) {
        return;
    }
    let bottom = atmosphereBottom(medium);
    let top = atmosphereTop(medium);
    let group = id.z;
    // The texel centres span [0, 1] end to end (`multipleScatteringUv`).
    let muSun = 2.0 * float(id.x) / float(size - 1) - 1.0;
    let height = (top - bottom) * float(id.y) / float(size - 1);
    // A little inside the shell, so rays start in the medium.
    let r = clamp(bottom + height, bottom + 1e-3, top - 1e-3);
    let origin = float3(0.0, 0.0, r);
    let sun = float3(sqrt(max(1.0 - muSun * muSun, 0.0)), 0.0, muSun);
    let albedo = groundAlbedo(medium, group);
    let isotropic = 1.0 / (4.0 * PI);

    var secondOrder = float4(0.0);
    var transfer = float4(0.0);
    for (uint i = 0; i < kDirections; ++i) {
        let direction = fibonacciDirection(i, kDirections);
        let mu = direction.z;
        let ground = hitsGround(r, mu, bottom);
        let span = ground ? distanceToBottom(r, mu, bottom) : distanceToTop(r, mu, top);
        let step = span / float(kScatteringSteps);
        var throughput = float4(1.0);
        var light = float4(0.0);
        var gathered = float4(0.0);
        for (uint s = 0; s < kScatteringSteps; ++s) {
            let t = (float(s) + 0.5) * step;
            let point = origin + t * direction;
            let radius = length(point);
            float4 scattering;
            float4 extinction;
            mediumAt(medium, densities, radius - bottom, group, scattering, extinction);
            let sampleTransmittance = exp(-extinction * step);
            let muPoint = dot(point, sun) / radius;
            let toSun = hitsGround(radius, muPoint, bottom)
                            ? float4(0.0)
                            : transmittanceToTop(transmittance, linear, radius, muPoint, group,
                                                 bottom, top);
            // In-scattering over the step for constant coefficients:
            // integral of S exp(-sigma_t x) dx = S (1 - exp(-sigma_t step)) / sigma_t.
            let fraction = select(extinction > 0.0,
                                  (1.0 - sampleTransmittance) / max(extinction, 1e-20),
                                  float4(step));
            light += throughput * scattering * toSun * isotropic * fraction;
            gathered += throughput * scattering * fraction;
            throughput *= sampleTransmittance;
        }
        if (ground) {
            let hit = origin + span * direction;
            let normal = normalize(hit);
            let muGround = dot(normal, sun);
            let toSun = transmittanceToTop(transmittance, linear, bottom, muGround, group, bottom,
                                           top);
            light += throughput * toSun * saturate(muGround) * albedo / PI;
        }
        // Uniform directions sample the isotropic phase function's own density, so the integrals
        // over the sphere weighted by it are the means.
        secondOrder += light / float(kDirections);
        transfer += gathered / float(kDirections);
    }
    multipleOut[id] = secondOrder / (1.0 - min(transfer, float4(0.999)));
}
