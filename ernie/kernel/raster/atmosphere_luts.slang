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
// Sky irradiance, third, per height and sun zenith cosine: the sky's radiance (single scattering by
// each constituent's phase function, multiple scattering from Psi_ms, as the sky pass marches it)
// integrated over the upper hemisphere of a level surface, by 64 directions spread evenly over its
// projected disc (a cosine-weighted Fibonacci spiral), so the irradiance is pi times their mean.
//
// Bindings: 0 the medium, 1 the constituents' densities, 2 the transmittance table (storage),
// 3 the transmittance table (sampled), 4 a linear sampler, 5 the multiple scattering table
// (storage), 6 the constituents' phase functions, 7 the sky irradiance table (storage), 8 the
// multiple scattering table (sampled), 9 the refraction table (storage). Every image is in the
// general layout.
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
[[vk::binding(6, 0)]] StructuredBuffer<float> phases;
[[vk::image_format("rgba32f")]] [[vk::binding(7, 0)]] RWTexture2DArray<float4> irradianceOut;
[[vk::binding(8, 0)]] Texture2DArray<float4> multiple;
[[vk::image_format("rgba32f")]] [[vk::binding(9, 0)]] RWTexture2DArray<float4> refractionOut;
[[vk::binding(10, 0)]] Texture2DArray<float4> refraction;
// x: the sine of the sun's angular radius, y: the cloud tops' height (km), below which clouds
// block rays grazing the planet; z: the first band's lower wavelength (nm), w: the band width;
// then x: the fraction of the limb the clouds cover.
[[vk::binding(12, 0)]] StructuredBuffer<float4> shadowSettings;

// Refraction, per apparent zenith angle and height: a ray traced out of the atmosphere by the ray
// equation d(n t)/ds = grad n through its spherically layered refractive index, the bands' bending
// summed along one path (theirs part by microradians), each step's |dn/dr| sin(angle to the
// vertical) / n. Steps grow from 0.2 km at the ground, where n changes fastest.
static const uint kRefractionSteps = 20000;

[shader("compute")]
[numthreads(8, 8, 1)]
void refractionMain(uint3 id: SV_DispatchThreadID) {
    if (id.x >= kRefractionWidth || id.y >= kRefractionHeight) {
        return;
    }
    let bottom = atmosphereBottom(medium);
    let top = atmosphereTop(medium);
    let uv = (float2(id.xy) + 0.5) / float2(kRefractionWidth, kRefractionHeight);
    float r;
    float mu;
    refractionRMu(uv, bottom, top, r, mu);
    r = clamp(r, bottom + 1e-4, top - 1e-4);
    // In the plane of the ray and the planet's centre: x across, y up through the start.
    var p = float2(0.0, r);
    var t = float2(sqrt(max(1.0 - mu * mu, 0.0)), mu);
    var bent = float4(0.0);
    for (uint i = 0; i < kRefractionSteps; ++i) {
        var radius = length(p);
        if (radius > top && dot(p, t) > 0.0) {
            break;
        }
        if (radius < bottom) {
            // A ray that meets the ground goes on straight through the planet and out again, as
            // CosmoScout VR traces it: its bending then grows smoothly into that of the rays that
            // just clear the ground, so the table interpolates without a step at the apparent
            // horizon (which `escapesAtmosphere` decides apart).
            let along = dot(p, t);
            p += (-along + sqrt(max(along * along - (radius * radius - bottom * bottom), 0.0)) +
                  1e-3) * t;
            radius = length(p);
        }
        let altitude = radius - bottom;
        let step = 0.2 + 0.05 * altitude;
        let h = 0.05;
        let below = refractivityAt(medium, densities, max(altitude - h, 0.0), id.z);
        let above = refractivityAt(medium, densities, altitude + h, id.z);
        let gradient = (above - below) / (min(altitude, h) + h);
        let n = 1.0 + refractivityAt(medium, densities, altitude, id.z);
        let up = p / radius;
        let sine = abs(up.x * t.y - up.y * t.x);
        bent += -gradient * sine / n * step;
        // The shared path bends with the bands' mean.
        let g = dot(gradient, float4(0.25)) * up;
        t = normalize(t + step * (g - dot(g, t) * t) / dot(n, float4(0.25)));
        p += step * t;
    }
    refractionOut[id] = bent;
}

uint group(uint3 id) {
    return id.z;
}

static const uint kIrradianceSteps = 32;
// The sine of the Sun's angular radius, 0.266 degrees, over which it sets behind the planet.
static const float kSunSine = 0.00465;

[shader("compute")]
[numthreads(8, 8, 1)]
void skyIrradianceMain(uint3 id: SV_DispatchThreadID) {
    if (id.x >= kIrradianceWidth || id.y >= kIrradianceHeight) {
        return;
    }
    let bottom = atmosphereBottom(medium);
    let top = atmosphereTop(medium);
    let uv = (float2(id.xy) + 0.5) / float2(kIrradianceWidth, kIrradianceHeight);
    float r;
    float muSun;
    irradianceRMu(uv, bottom, top, r, muSun);
    // A little inside the shell, so rays start in the medium.
    r = clamp(r, bottom + 1e-3, top - 1e-3);
    let origin = float3(0.0, 0.0, r);
    let sun = float3(sqrt(max(1.0 - muSun * muSun, 0.0)), 0.0, muSun);
    var irradiance = float4(0.0);
    for (uint i = 0; i < kDirections; ++i) {
        let disc = sqrt((float(i) + 0.5) / float(kDirections));
        let phi = float(i) * 2.39996322972865;  // the golden angle
        let direction =
            float3(disc * cos(phi), disc * sin(phi), sqrt(max(1.0 - disc * disc, 0.0)));
        float start;
        float end;
        bool ground;
        if (!clipToAtmosphere(origin, direction, bottom, top, start, end, ground)) {
            continue;
        }
        float4 light;
        float4 through;
        scatterAlong(medium, densities, phases, transmittance, multiple, linear, origin, direction,
                     start, end, sun, kSunSine, noEclipse(), id.z, kIrradianceSteps, light,
                     through);
        irradiance += light;
    }
    irradianceOut[id] = PI * irradiance / float(kDirections);
}

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

// The planet's shadow (atmosphere.slang's `planetShadow`), in the transmittance table's layers
// after its own, per distance from the planet and
// separation of the planet's centre from the sun's: the sun's light that reaches the point past
// the atmosphere's disc, plus the light that reaches it through the atmosphere, over the sun's
// whole light. Through the atmosphere, each direction the point looks in, at angle alpha from the
// planet's centre, is a ray of impact parameter b = D sin alpha that grazes the planet at the
// radius r_t where n(r_t) r_t = b (Bouguer's invariant), bent towards the planet by twice the
// refraction table's bending from r_t level (in and out alike) and dimmed by the square of the
// transmittance from r_t level to the top. It carries the sun's limb-darkened radiance from the
// direction it came from, alpha - bending from the planet's centre (across it when negative); its
// radiance is kept along the ray, so the integral over the point's sky, of solid angle
// sin alpha dalpha dphi, holds the focusing of the light the atmosphere bends into the umbra as
// well as its reddening. Rays grazing the ground are blocked, and below the cloud tops the share
// the clouds cover. Light the atmosphere scatters towards the shadow is left out: next to the
// refracted sunlight it matters only deep in the umbra after volcanic eruptions (Garcia Munoz
// and Palle 2011).
//
// Across each grazing height, the directions whose rays come from the sun lie within an angle
// psi_max of the line through the sun's centre, from the half-angle formula
// sin^2(gamma / 2) = sin^2((a - sigma) / 2) + sin a sin sigma sin^2(psi / 2) for a source at a
// from the planet's centre and the sun's at sigma, which keeps small angles accurate.
//
// References:
// - F. Link, "Eclipse Phenomena in Astronomy", Springer, 1969, chapter 2 (the theory of the
//   umbra's light, refracted and extinguished in the atmosphere).
//   https://doi.org/10.1007/978-3-642-86808-8
// - A. Garcia Munoz and E. Palle, "Lunar eclipse theory revisited: Scattered sunlight in both the
//   quiescent and the volcanically perturbed atmosphere", Journal of Quantitative Spectroscopy
//   and Radiative Transfer 112(10):1609-1621, 2011. https://doi.org/10.1016/j.jqsrt.2011.03.017
// - S. Schneegans et al., "Physically Based Real-Time Rendering of Eclipses", Computer Graphics
//   Forum 44, 2025 (shadow maps of the light through and past an atmosphere, by the receiver's
//   view of the planet and the sun). https://diglib.eg.org/handle/10.1111/cgf70017
static const uint kShadowHeights = 128;
static const uint kShadowAzimuths = 32;

[shader("compute")]
[numthreads(8, 8, 1)]
void planetShadowMain(uint3 id: SV_DispatchThreadID) {
    if (id.x >= kPlanetShadowWidth || id.y >= kPlanetShadowHeight) {
        return;
    }
    let bottom = atmosphereBottom(medium);
    let top = atmosphereTop(medium);
    let sunSine = shadowSettings[0].x;
    let cloudTop = shadowSettings[0].y;
    let clear = 1.0 - shadowSettings[1].x;
    let sunRadius = asin(sunSine);
    let group = id.z;
    float4 wavelengths;
    float4 exponent;
    for (uint k = 0; k < 4; ++k) {
        wavelengths[k] = shadowSettings[0].z + (float(group * 4 + k) + 0.5) * shadowSettings[0].w;
        exponent[k] = limbDarkening(wavelengths[k]);
    }
    let uv = (float2(id.xy) + 0.5) / float2(kPlanetShadowWidth, kPlanetShadowHeight);
    float distance;
    float sigma;
    planetShadowDistance(uv, top, sunRadius, distance, sigma);

    // Past the atmosphere's disc.
    var eclipse = noEclipse();
    eclipse.bodies[0] = float4(0.0, 0.0, distance, top);
    eclipse.wavelengths = wavelengths;
    var light =
        eclipseVisibility(eclipse, float3(0.0), float3(sin(sigma), 0.0, cos(sigma)), sunSine);

    // Through it: grazing heights crowded towards the ground, b = bottom + (top - bottom) t^2.
    let halfSun = sin(0.5 * sunRadius);
    let solidAngle = 4.0 * PI * halfSun * halfSun;
    let sinSigma = sin(sigma);
    for (uint i = 0; i < kShadowHeights; ++i) {
        let t = (float(i) + 0.5) / float(kShadowHeights);
        let b = bottom + (top - bottom) * t * t;
        let db = (top - bottom) * 2.0 * t / float(kShadowHeights);
        let alpha = asin(b / distance);
        let dalpha = db / sqrt(max(distance * distance - b * b, 1e-6));
        let refractivity = refractivityAt(medium, densities, b - bottom, group);
        for (uint k = 0; k < 4; ++k) {
            // The grazing radius, n(r_t) r_t = b, in two steps.
            var grazing = b / (1.0 + refractivity[k]);
            grazing = b / (1.0 + refractivityAt(medium, densities, grazing - bottom, group)[k]);
            if (grazing <= bottom) {
                continue;
            }
            let clouds = grazing <= bottom + cloudTop ? clear : 1.0;
            let bending =
                2.0 * refractionDeviation(refraction, linear, grazing, 0.0, group, bottom, top)[k];
            let through = transmittanceToTop(transmittance, linear, grazing, 0.0, group, bottom,
                                             top)[k];
            let source = abs(alpha - bending);
            let halfOff = sin(0.5 * (source - sigma));
            let across = sin(source) * sinSigma;
            let room = halfSun * halfSun - halfOff * halfOff;
            if (room <= 0.0) {
                continue;
            }
            // The directions, either side of the sun's line, whose source lies on the sun.
            let psiMax = across <= room ? PI : 2.0 * asin(sqrt(room / across));
            var radiance = 0.0;
            for (uint j = 0; j < kShadowAzimuths; ++j) {
                let psi = (float(j) + 0.5) / float(kShadowAzimuths) * psiMax;
                let s = sin(0.5 * psi);
                let halfGamma = sqrt(max(halfOff * halfOff + across * s * s, 0.0));
                let ratio = min(halfGamma / halfSun, 1.0);
                let mu = sqrt(max(1.0 - ratio * ratio, 0.0));
                radiance += pow(mu, exponent[k]);
            }
            radiance *= (exponent[k] + 2.0) / (2.0 * solidAngle) * 2.0 * psiMax /
                        float(kShadowAzimuths);
            light[k] += clouds * through * through * radiance * sin(alpha) * dalpha;
        }
    }
    let groups = uint(medium[1].x);
    transmittanceOut[uint3(id.xy, groups + group)] = light;
}
