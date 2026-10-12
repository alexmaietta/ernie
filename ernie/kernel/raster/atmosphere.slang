// SPDX-License-Identifier: BSD-3-Clause
// A planet's atmosphere for the atmosphere's table passes and shading: a spherical shell of gases
// and aerosols around the planet's centre, lengths in kilometres and coefficients in km^-1, the
// spectral bands four to a float4 ("band groups").
//
// The medium (`AtmosphereLuts`): `medium[0]` the bottom and top radii, the constituent count and
// the density samples per constituent; `medium[1]` the band groups, the bands and the phase table's
// angles; then per constituent its scattering coefficients for each group and its absorption
// coefficients for each group, where its density is 1; then the ground albedo for each group.
// `densities` holds each constituent's relative density at `samples` altitudes evenly from the
// ground to the top, and `phases` each constituent's phase function (sr^-1) per band, band-major,
// at scattering angles theta_i = pi (i / (angles - 1))^2 (`PhaseCosines`).
//
// The transmittance table holds the transmittance from radius r along cosine mu (from the zenith)
// to the top of the atmosphere for rays that miss the ground, in Bruneton's parameterisation,
// which spends texels where the transmittance changes fastest, near the horizon. The multiple
// scattering table holds Hillaire's Psi_ms: the radiance scattered more than once towards a point,
// per unit sun illuminance and scattering coefficient there, by height and sun zenith cosine.
//
// References:
// - E. Bruneton and F. Neyret, "Precomputed Atmospheric Scattering", Computer Graphics Forum
//   27(4):1079-1086, 2008. https://doi.org/10.1111/j.1467-8659.2008.01245.x
// - E. Bruneton, "Precomputed Atmospheric Scattering: a New Implementation", 2017, sections
//   "Transmittance" and "Precomputation" (texture parameterisations).
//   https://ebruneton.github.io/precomputed_atmospheric_scattering/
// - S. Hillaire, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique",
//   Computer Graphics Forum 39(4):13-22, 2020, sections 5.5 (multiple scattering) and 5.6 (the
//   sky's ray march, with multiple scattering from Psi_ms). https://doi.org/10.1111/cgf.14050
// - S. Hillaire, "Physically Based and Unified Volumetric Rendering in Frostbite", SIGGRAPH 2015
//   Advances in Real-Time Rendering course (energy-conserving integration of in-scattering).
// - S. Schneegans, T. Meyran, I. Ginkel, G. Zachmann and A. Gerndt, "Physically Based Real-Time
//   Rendering of Atmospheres using Mie Theory", Computer Graphics Forum 43(2), 2024 (tabulated
//   phase functions per constituent). https://doi.org/10.1111/cgf.15010
// - L. H. Auer and E. M. Standish, "Astronomical Refraction: Computational Method for All Zenith
//   Angles", The Astronomical Journal 119(5):2472-2474, 2000 (refraction by integrating the ray
//   through a spherically layered atmosphere). https://doi.org/10.1086/301325
// - A. T. Young, "Sunset Science. IV. Low-Altitude Refraction", The Astronomical Journal
//   127(6):3622-3637, 2004 (bending near the horizon, and its temperature gradient).
//   https://doi.org/10.1086/420806
// - S. Schneegans et al., "Physically Based Real-Time Rendering of Eclipses", Computer Graphics
//   Forum 44, 2025, section 4 (a table of ray deviations by height and angle, traced on through
//   the planet for continuity). https://diglib.eg.org/handle/10.1111/cgf70017
// - D. Hestroffer and C. Magnan, "Wavelength dependency of the Solar limb darkening", Astronomy
//   and Astrophysics 333:338-342, 1998 (the power law fitted to Neckel and Labs's and Pierce and
//   Slaughter's measurements). https://ui.adsabs.harvard.edu/abs/1998A&A...333..338H
module atmosphere;

static const float kPi = 3.14159265358979;

public static const uint kTransmittanceWidth = 256;
public static const uint kTransmittanceHeight = 64;
public static const uint kMultipleScatteringSize = 32;
// The sky irradiance table: by the sun's zenith cosine from -0.2 (twilight, below which the sky's
// light is negligible) to 1 across, and the height up.
public static const uint kIrradianceWidth = 64;
public static const uint kIrradianceHeight = 16;
public static const float kIrradianceMinMuSun = -0.2;

public float atmosphereBottom(StructuredBuffer<float4> medium) {
    return medium[0].x;
}

public float atmosphereTop(StructuredBuffer<float4> medium) {
    return medium[0].y;
}

// The relative density of constituent `c` at `altitude` (km), linear between its samples.
float densityAt(StructuredBuffer<float4> medium, StructuredBuffer<float> densities, uint c,
                float altitude) {
    let samples = uint(medium[0].w);
    let height = medium[0].y - medium[0].x;
    let x = clamp(altitude / height, 0.0, 1.0) * float(samples - 1);
    let i = min(uint(x), samples - 2);
    let t = x - float(i);
    let base = c * samples;
    return lerp(densities[base + i], densities[base + i + 1], t);
}

// The scattering and extinction coefficients (km^-1) of band group `group` at `altitude` (km).
public void mediumAt(StructuredBuffer<float4> medium, StructuredBuffer<float> densities,
                     float altitude, uint group, out float4 scattering, out float4 extinction) {
    let constituents = uint(medium[0].z);
    let groups = uint(medium[1].x);
    scattering = float4(0.0);
    extinction = float4(0.0);
    for (uint c = 0; c < constituents; ++c) {
        let density = densityAt(medium, densities, c, altitude);
        let s = medium[2 + c * 2 * groups + group];
        let a = medium[2 + c * 2 * groups + groups + group];
        scattering += density * s;
        extinction += density * (s + a);
    }
}

public float4 groundAlbedo(StructuredBuffer<float4> medium, uint group) {
    let constituents = uint(medium[0].z);
    let groups = uint(medium[1].x);
    return medium[2 + constituents * 2 * groups + group];
}

// The refractive index less 1 of band group `group` at `altitude` (km): each constituent's
// refractivity where its density is 1 (after the ground albedo in the medium) times its density.
public float4 refractivityAt(StructuredBuffer<float4> medium, StructuredBuffer<float> densities,
                             float altitude, uint group) {
    let constituents = uint(medium[0].z);
    let groups = uint(medium[1].x);
    let first = 2 + constituents * 2 * groups + groups;
    var refractivity = float4(0.0);
    for (uint c = 0; c < constituents; ++c) {
        refractivity += densityAt(medium, densities, c, altitude) * medium[first + c * groups + group];
    }
    return refractivity;
}

// Whether a ray from radius r (km) at the apparent zenith cosine mu escapes to space for each
// band of group `group` rather than meeting the ground, by Bouguer's invariant n r sin(zenith),
// constant along a ray in a spherically layered atmosphere: it meets the ground only heading down
// with n0 r sin z below the ground's n times its radius (with n r growing with height, as it does
// without ducting).
public bool4 escapesAtmosphere(StructuredBuffer<float4> medium, StructuredBuffer<float> densities,
                               float r, float mu, uint group) {
    let bottom = atmosphereBottom(medium);
    if (mu >= 0.0) {
        return bool4(true);
    }
    let n0 = 1.0 + refractivityAt(medium, densities, r - bottom, group);
    let nb = 1.0 + refractivityAt(medium, densities, 0.0, group);
    return n0 * r * sqrt(max(1.0 - mu * mu, 0.0)) >= nb * bottom;
}

// The refraction table's coordinates: across, the apparent zenith angle from 0 to 2 degrees below
// the geometric horizon, crowded towards that end, where the bending changes fastest
// (x = 1 - sqrt(1 - zenith / largest)); up, the square root of the height's fraction of the
// atmosphere, crowded towards the ground; and back.
static const float kRefractionMargin = 0.0349;  // 2 degrees
public static const uint kRefractionWidth = 256;
public static const uint kRefractionHeight = 32;

float largestZenith(float r, float bottom) {
    return acos(horizonCosine(r, bottom)) + kRefractionMargin;
}

public float2 refractionUv(float r, float mu, float bottom, float top) {
    let zenith = acos(clamp(mu, -1.0, 1.0));
    let x = 1.0 - sqrt(max(1.0 - zenith / largestZenith(r, bottom), 0.0));
    let y = sqrt(clamp((r - bottom) / (top - bottom), 0.0, 1.0));
    return float2(textureCoord(clamp(x, 0.0, 1.0), float(kRefractionWidth)),
                  textureCoord(y, float(kRefractionHeight)));
}

public void refractionRMu(float2 uv, float bottom, float top, out float r, out float mu) {
    let y = unitRange(uv.y, float(kRefractionHeight));
    r = bottom + y * y * (top - bottom);
    let x = unitRange(uv.x, float(kRefractionWidth));
    let s = 1.0 - x;
    mu = cos(largestZenith(r, bottom) * (1.0 - s * s));
}

// The bending (radians) of rays of band group `group` leaving radius r at apparent zenith cosine
// mu, as the refraction table holds it: towards the ground, so the true direction of what such a
// ray sees lies that much further from the zenith than the ray.
public float4 refractionDeviation(Texture2DArray<float4> table, SamplerState linear, float r,
                                  float mu, uint group, float bottom, float top) {
    let uv = refractionUv(clamp(r, bottom, top), mu, bottom, top);
    return table.SampleLevel(linear, float3(uv, float(group)), 0.0);
}

// Direction `d` turned in its vertical plane (through `up`) by `angle` radians towards the zenith;
// a negative angle turns it towards the nadir.
public float3 turnTowardsZenith(float3 d, float3 up, float angle) {
    let across = up - dot(up, d) * d;
    let size = length(across);
    if (size < 1e-6) {
        return d;
    }
    return d * cos(angle) + across / size * sin(angle);
}

// Distance from radius r along cosine mu to the sphere of radius `radius` outside it (the top).
public float distanceToTop(float r, float mu, float top) {
    let discriminant = r * r * (mu * mu - 1.0) + top * top;
    return max(0.0, -r * mu + sqrt(max(discriminant, 0.0)));
}

// Distance from radius r along cosine mu to the ground, for rays that hit it.
public float distanceToBottom(float r, float mu, float bottom) {
    let discriminant = r * r * (mu * mu - 1.0) + bottom * bottom;
    return max(0.0, -r * mu - sqrt(max(discriminant, 0.0)));
}

public bool hitsGround(float r, float mu, float bottom) {
    return mu < 0.0 && r * r * (mu * mu - 1.0) + bottom * bottom >= 0.0;
}

// The cosine of the horizon's direction from the zenith at radius r.
public float horizonCosine(float r, float bottom) {
    let ratio = min(bottom / r, 1.0);
    return -sqrt(max(1.0 - ratio * ratio, 0.0));
}

// Texture coordinate of x in [0, 1] whose ends land on the first and last texel centres, and back.
float textureCoord(float x, float size) {
    return 0.5 / size + x * (1.0 - 1.0 / size);
}

float unitRange(float u, float size) {
    return (u - 0.5 / size) / (1.0 - 1.0 / size);
}

// Bruneton's transmittance coordinates: x_r = rho / H, the distance to the horizon over the
// largest; x_mu = (d - d_min) / (d_max - d_min) of the distance to the top.
public float2 transmittanceUv(float r, float mu, float bottom, float top) {
    let H = sqrt(top * top - bottom * bottom);
    let rho = sqrt(max(r * r - bottom * bottom, 0.0));
    let d = distanceToTop(r, mu, top);
    let dMin = top - r;
    let dMax = rho + H;
    let xMu = dMax > dMin ? (d - dMin) / (dMax - dMin) : 0.0;
    let xR = rho / H;
    return float2(textureCoord(xMu, float(kTransmittanceWidth)),
                  textureCoord(xR, float(kTransmittanceHeight)));
}

public void transmittanceRMu(float2 uv, float bottom, float top, out float r, out float mu) {
    let xMu = unitRange(uv.x, float(kTransmittanceWidth));
    let xR = unitRange(uv.y, float(kTransmittanceHeight));
    let H = sqrt(top * top - bottom * bottom);
    let rho = H * xR;
    r = sqrt(rho * rho + bottom * bottom);
    let dMin = top - r;
    let dMax = rho + H;
    let d = dMin + xMu * (dMax - dMin);
    mu = d == 0.0 ? 1.0 : clamp((H * H - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
}

// The transmittance of band group `group` from radius r along mu to the top, for a ray that misses
// the ground.
public float4 transmittanceToTop(Texture2DArray<float4> table, SamplerState linear, float r,
                                 float mu, uint group, float bottom, float top) {
    let uv = transmittanceUv(clamp(r, bottom, top), mu, bottom, top);
    return table.SampleLevel(linear, float3(uv, float(group)), 0.0);
}

// The transmittance of band group `group` towards the sun from a point `p` (km from the planet's
// centre), with the planet's own shadow: the part of the sun's disc of angular radius sine
// `sunSine` above the horizon. 1 from outside the atmosphere where the ray misses it.
public float4 sunTransmittance(Texture2DArray<float4> table, SamplerState linear, float3 p,
                               float3 sun, float sunSine, uint group, float bottom, float top) {
    var r = length(p);
    var mu = r > 0.0 ? dot(p, sun) / r : 1.0;
    if (r > top) {
        // Above the atmosphere: to where the ray towards the sun enters it, if it does.
        let discriminant = r * r * (mu * mu - 1.0) + top * top;
        if (mu >= 0.0 || discriminant < 0.0) {
            return float4(1.0);
        }
        let entry = -r * mu - sqrt(discriminant);
        mu = (r * mu + entry) / top;
        r = top;
    }
    r = max(r, bottom);
    let horizon = horizonCosine(r, bottom);
    let visible = smoothstep(-sunSine, sunSine, mu - horizon);
    if (visible <= 0.0) {
        return float4(0.0);
    }
    return visible * transmittanceToTop(table, linear, r, max(mu, horizon), group, bottom, top);
}

// The multiple scattering table's coordinates: the sun's zenith cosine and the height's fraction of
// the atmosphere, each over [0, 1] between the first and last texel centres.
public float2 multipleScatteringUv(float r, float muSun, float bottom, float top) {
    let size = float(kMultipleScatteringSize);
    return float2(textureCoord(0.5 + 0.5 * muSun, size),
                  textureCoord(clamp((r - bottom) / (top - bottom), 0.0, 1.0), size));
}

public float4 multipleScattering(Texture2DArray<float4> table, SamplerState linear, float r,
                                 float muSun, uint group, float bottom, float top) {
    let uv = multipleScatteringUv(r, muSun, bottom, top);
    return table.SampleLevel(linear, float3(uv, float(group)), 0.0);
}

// The sky irradiance table's coordinates, as Bruneton's: the sun's zenith cosine over
// [kIrradianceMinMuSun, 1] and the height's fraction of the atmosphere, each between the first
// and last texel centres; and back.
public float2 irradianceUv(float r, float muSun, float bottom, float top) {
    let x = (muSun - kIrradianceMinMuSun) / (1.0 - kIrradianceMinMuSun);
    return float2(textureCoord(clamp(x, 0.0, 1.0), float(kIrradianceWidth)),
                  textureCoord(clamp((r - bottom) / (top - bottom), 0.0, 1.0),
                               float(kIrradianceHeight)));
}

public void irradianceRMu(float2 uv, float bottom, float top, out float r, out float muSun) {
    let x = unitRange(uv.x, float(kIrradianceWidth));
    muSun = kIrradianceMinMuSun + x * (1.0 - kIrradianceMinMuSun);
    r = bottom + unitRange(uv.y, float(kIrradianceHeight)) * (top - bottom);
}

// The sky's irradiance on a level surface at radius `r` with the sun at zenith cosine `muSun`, per
// unit sun irradiance above the atmosphere, for band group `group`; 0 below the table's twilight.
public float4 skyIrradiance(Texture2DArray<float4> table, SamplerState linear, float r,
                            float muSun, uint group, float bottom, float top) {
    if (muSun < kIrradianceMinMuSun) {
        return float4(0.0);
    }
    let uv = irradianceUv(r, muSun, bottom, top);
    return table.SampleLevel(linear, float3(uv, float(group)), 0.0);
}

// The phase function (sr^-1) of constituent `c` in band `band` at scattering cosine `nu`, linear
// between its table's angles.
float phaseAt(StructuredBuffer<float4> medium, StructuredBuffer<float> phases, uint c, uint band,
              float nu) {
    let bands = uint(medium[1].y);
    let angles = uint(medium[1].z);
    let u = sqrt(acos(clamp(nu, -1.0, 1.0)) / kPi);
    let x = u * float(angles - 1);
    let i = min(uint(x), angles - 2);
    let base = (c * bands + min(band, bands - 1)) * angles;
    return lerp(phases[base + i], phases[base + i + 1], x - float(i));
}

// Where a ray from `p` (km from the planet's centre) along unit `d` crosses the atmosphere: from
// `start` to `end`, the ground or the top, and whether it ends on the ground; false if it misses
// the atmosphere. A point below the ground starts on it.
public bool clipToAtmosphere(float3 p, float3 d, float bottom, float top, out float start,
                             out float end, out bool ground) {
    start = 0.0;
    end = 0.0;
    ground = false;
    let r = max(length(p), bottom);
    let mu = dot(p, d) / max(length(p), 1e-6);
    let toTop = r * r * (mu * mu - 1.0) + top * top;
    if (toTop < 0.0) {
        return false;
    }
    if (r > top) {
        if (mu >= 0.0) {
            return false;
        }
        start = -r * mu - sqrt(toTop);
    }
    end = -r * mu + sqrt(toTop);
    let toGround = r * r * (mu * mu - 1.0) + bottom * bottom;
    if (mu < 0.0 && toGround >= 0.0) {
        end = max(-r * mu - sqrt(toGround), start);
        ground = true;
    }
    return end > start;
}

// The part of the segment from `p` (km from the planet's centre) along unit `d` for `span` km
// that lies inside the top of the atmosphere, from `start` to `end`; false if none does. The
// ground does not clip it: the segment ends on a surface, which may lie below the planet's mean
// sphere or, on a flat terrain, beyond its horizon.
public bool clipSegment(float3 p, float3 d, float span, float top, out float start,
                        out float end) {
    start = 0.0;
    end = 0.0;
    let r = length(p);
    let mu = dot(p, d) / max(r, 1e-6);
    let toTop = r * r * (mu * mu - 1.0) + top * top;
    if (toTop < 0.0) {
        return false;
    }
    if (r > top) {
        start = -r * mu - sqrt(toTop);
    }
    end = min(span, -r * mu + sqrt(toTop));
    return end > max(start, 0.0);
}

// The light the atmosphere scatters towards `p` (km from the planet's centre) along the segment
// from `p + start d` to `p + end d`, per unit sun irradiance (sr^-1), for band group `group`; and
// the transmittance across it. Single scattering takes each constituent's own phase function and
// the sun's transmittance with the planet's shadow; multiple scattering takes Psi_ms. Each step's
// in-scattering is integrated exactly for constant coefficients. Steps crowd towards the start
// when it is inside the atmosphere, where the air is densest along rays from the ground, and are
// even across rays from space.
//
// `eclipse`'s bodies (planet-centred) dim the sun at each step (`eclipseVisibility`), both its
// single and its multiple scattering: Psi_ms assumes the sun lights all the air around a point
// alike, so inside a moon's umbra, about a hundred kilometres across, the light scattered in
// from the sunlit air beyond is left out, and the sky there is darker than it should be.
public void scatterAlong(StructuredBuffer<float4> medium, StructuredBuffer<float> densities,
                         StructuredBuffer<float> phases, Texture2DArray<float4> transmittanceTable,
                         Texture2DArray<float4> multipleTable, SamplerState linear, float3 p,
                         float3 d, float start, float end, float3 sun, float sunSine,
                         Eclipse eclipse, uint group, uint steps, out float4 radiance,
                         out float4 transmittance) {
    SkyLights lights;
    lights.count = 1;
    lights.towards[0] = float4(sun, sunSine);
    lights.irradiance[0] = float4(1.0);
    scatterAlongLights(medium, densities, phases, transmittanceTable, multipleTable, linear, p, d,
                       start, end, lights, eclipse, group, steps, radiance, transmittance);
}

// Distant lights the air scatters, such as the sun and earthshine: towards each (xyz) and the
// sine of its angular radius (w), and its irradiance above the atmosphere in a band group.
public static const uint kSkyLights = 4;
// Constituents whose phase functions a march keeps per step: as many as the presets have (air,
// ozone and haze; carbon dioxide and dust) and one spare.
static const uint kSkyConstituents = 4;
public struct SkyLights {
    public float4 towards[kSkyLights];
    public float4 irradiance[kSkyLights];
    public uint count;
}

// `scatterAlong` for `lights`, the light scattered being theirs (W m^-2 sr^-1 nm^-1) rather than
// per unit irradiance; the air's coefficients are found once per step for them all.
public void scatterAlongLights(StructuredBuffer<float4> medium, StructuredBuffer<float> densities,
                               StructuredBuffer<float> phases,
                               Texture2DArray<float4> transmittanceTable,
                               Texture2DArray<float4> multipleTable, SamplerState linear, float3 p,
                               float3 d, float start, float end, SkyLights lights, Eclipse eclipse,
                               uint group, uint steps, out float4 radiance,
                               out float4 transmittance) {
    let bottom = atmosphereBottom(medium);
    let top = atmosphereTop(medium);
    let constituents = uint(medium[0].z);
    let groups = uint(medium[1].x);
    let crowd = start == 0.0;
    radiance = float4(0.0);
    transmittance = float4(1.0);
    var previous = start;
    for (uint s = 0; s < steps; ++s) {
        let x = float(s + 1) / float(steps);
        let next = start + (end - start) * (crowd ? x * x : x);
        let step = next - previous;
        let t = 0.5 * (previous + next);
        previous = next;
        let point = p + t * d;
        let r = length(point);
        let altitude = r - bottom;
        float4 extinction = float4(0.0);
        float4 scattering = float4(0.0);
        float4 coefficient[kSkyConstituents];
        for (uint c = 0; c < constituents; ++c) {
            let density = densityAt(medium, densities, c, altitude);
            let sc = density * medium[2 + c * 2 * groups + group];
            let ab = density * medium[2 + c * 2 * groups + groups + group];
            extinction += sc + ab;
            scattering += sc;
            coefficient[min(c, kSkyConstituents - 1)] = sc;
        }
        let up = point / r;
        var source = float4(0.0);
        for (uint i = 0; i < lights.count; ++i) {
            let sun = lights.towards[i].xyz;
            let sunSine = lights.towards[i].w;
            let nu = dot(d, sun);
            float4 phased = float4(0.0);
            for (uint c = 0; c < min(constituents, kSkyConstituents); ++c) {
                float4 phase;
                for (uint k = 0; k < 4; ++k) {
                    phase[k] = phaseAt(medium, phases, c, group * 4 + k, nu);
                }
                phased += coefficient[c] * phase;
            }
            let toSun = sunTransmittance(transmittanceTable, linear, point, sun, sunSine, group,
                                         bottom, top);
            let multiple =
                multipleScattering(multipleTable, linear, r, dot(up, sun), group, bottom, top);
            source += lights.irradiance[i] * (phased * toSun + scattering * multiple) *
                      eclipseVisibility(eclipse, point, sun, sunSine);
        }
        let sample = exp(-extinction * step);
        let fraction = select(extinction > 0.0, (1.0 - sample) / max(extinction, 1e-20),
                              float4(step));
        radiance += transmittance * source * fraction;
        transmittance *= sample;
    }
}

// The radiance of the sun's disc towards `d`, per unit irradiance (sr^-1), in band `band` of
// `wavelength` nm: Hestroffer and Magnan's limb darkening, I(mu) / I(0) = mu^alpha with
// alpha = -0.023 + 0.292 / lambda (um), over a disc of angular radius sine `sunSine`, normalised so
// the disc integrates to 1 (its mean is 2 / (alpha + 2) of its centre). A disc smaller than the
// pixel, of solid angle `pixelSolidAngle`, fills the pixel whose centre lies within the pixel's
// own angular radius of the sun's centre, keeping its irradiance. 0 off the disc.
public float sunDisc(float3 d, float3 sun, float sunSine, float wavelength, float pixelSolidAngle) {
    let cosine = dot(d, sun);
    let sine = sqrt(max(1.0 - cosine * cosine, 0.0));
    let solidAngle = 2.0 * kPi * (1.0 - sqrt(max(1.0 - sunSine * sunSine, 0.0)));
    if (pixelSolidAngle > solidAngle) {
        let pixelRadius = sqrt(pixelSolidAngle / kPi);
        return cosine > 0.0 && sine < pixelRadius ? 1.0 / pixelSolidAngle : 0.0;
    }
    if (cosine <= 0.0 || sine >= sunSine) {
        return 0.0;
    }
    let alpha = limbDarkening(wavelength);
    let ratio = sine / sunSine;
    let mu = sqrt(max(1.0 - ratio * ratio, 0.0));
    return pow(mu, alpha) * (alpha + 2.0) / (2.0 * solidAngle);
}

// An eye at `p` (km from the planet's centre) lifted onto the ground, a metre up, when it is below
// it, as Hillaire's sky lifts cameras: the atmosphere is seen from the ground at least, not from
// inside the planet, where its rays would start below the ground they are clipped to.
//
// References:
// - S. Hillaire, UnrealEngineSkyAtmosphere, RenderSkyRayMarching.hlsl (the camera kept
//   PLANET_RADIUS_OFFSET above the ground). https://github.com/sebh/UnrealEngineSkyAtmosphere
public float3 aboveGround(float3 p, float bottom) {
    let r = length(p);
    let lowest = bottom + 1e-3;
    return r < lowest ? (r > 0.0 ? p * (lowest / r) : float3(0.0, 0.0, lowest)) : p;
}

// Hestroffer and Magnan's limb darkening exponent at `wavelength` nm (`sunDisc`).
public float limbDarkening(float wavelength) {
    return max(-0.023 + 292.0 / wavelength, 0.0);
}

// Bodies that may eclipse the sun, such as moons: xyz each centre and w its radius, in km in the
// frame of the points it is asked about (0 radius for none); and the wavelengths (nm) of the band
// group's bands, whose limb darkening differs.
public struct Eclipse {
    public float4 bodies[4];
    public float4 wavelengths;
}

// No bodies.
public Eclipse noEclipse() {
    Eclipse eclipse;
    for (uint i = 0; i < 4; ++i) {
        eclipse.bodies[i] = float4(0.0);
    }
    eclipse.wavelengths = float4(550.0);
    return eclipse;
}

// Rings across the sun's disc over which an eclipse is integrated.
static const uint kEclipseRings = 16;

// The fraction of the sun's light that reaches `p` (km, in the bodies' frame) past `eclipse`'s
// bodies, per band: the part of the sun's disc, of angular radius sine `sunSine` towards `sun`,
// that no body covers, weighted by its limb-darkened brightness (`sunDisc`). Each body covers a
// disc of angular radius asin(radius / distance). Across rings of the sun's disc, a body covers
// the fraction acos((s^2 + rho^2 - R^2) / (2 s rho)) / pi of the ring at angular radius rho, for a
// body of angular radius R whose centre is s from the sun's (the discs taken as flat, as both are
// small); the power law's brightness inside radius rho is 1 - (1 - (rho / rho_sun)^2)^(alpha/2 + 1)
// of the whole, exactly, which weights the rings. Several bodies dim the light in turn. 0 inside a
// body.
//
// References:
// - S. Schneegans, J. Gilg, V. Ahlers and A. Gerndt, "Real-Time Rendering of Eclipses without
//   Incorporation of Atmospheric Effects", Computer Graphics Forum 41(7):279-289, 2022, sections
//   3 and 4 (the eclipse geometry, and the occluded fraction with limb darkening).
//   https://doi.org/10.1111/cgf.14676
// - Hestroffer and Magnan 1998 (the limb darkening, `sunDisc`).
public float4 eclipseVisibility(Eclipse eclipse, float3 p, float3 sun, float sunSine) {
    var visible = float4(1.0);
    let sunRadius = asin(sunSine);
    var exponent = float4(0.0);
    for (uint k = 0; k < 4; ++k) {
        exponent[k] = 0.5 * limbDarkening(eclipse.wavelengths[k]) + 1.0;
    }
    for (uint i = 0; i < 4; ++i) {
        let body = eclipse.bodies[i];
        if (body.w <= 0.0) {
            continue;
        }
        let toBody = body.xyz - p;
        let distance = length(toBody);
        if (distance <= body.w) {
            return float4(0.0);
        }
        let direction = toBody / distance;
        if (dot(direction, sun) <= 0.0) {
            continue;
        }
        // Angles: the body's radius, and its centre's separation from the sun's, which the
        // chord's arc sine keeps accurate for small angles.
        let bodyRadius = asin(body.w / distance);
        let separation = 2.0 * asin(min(0.5 * length(direction - sun), 1.0));
        if (separation >= sunRadius + bodyRadius) {
            continue;
        }
        if (separation + sunRadius <= bodyRadius) {
            return float4(0.0);
        }
        var covered = float4(0.0);
        var inside = float4(0.0);
        for (uint ring = 0; ring < kEclipseRings; ++ring) {
            let outer = float(ring + 1) / float(kEclipseRings);
            let rho = (float(ring) + 0.5) / float(kEclipseRings) * sunRadius;
            let cosine = (separation * separation + rho * rho - bodyRadius * bodyRadius) /
                         max(2.0 * separation * rho, 1e-20);
            let arc = acos(clamp(cosine, -1.0, 1.0)) / kPi;
            let within = 1.0 - pow(float4(max(1.0 - outer * outer, 0.0)), exponent);
            covered += arc * (within - inside);
            inside = within;
        }
        visible *= saturate(1.0 - covered);
    }
    return visible;
}

// The planet's shadow table (`planetShadowMain`): the fraction of the sun's light reaching a point
// behind the planet, through and past its atmosphere, per band. Across, the point's distance D from
// the planet's centre, from the top of the atmosphere to `kPlanetShadowReach` times it, evenly in
// log D; up, the angle sigma between the planet's centre and the sun's as the point sees them, from
// 0 to where the atmosphere's disc no longer covers any of the sun (its angular radius plus the
// sun's). Beyond, the sun is whole. It is the transmittance table's size, and shares its image, in
// the layers after the transmittance's, which keeps shading within its descriptors.
public static const uint kPlanetShadowWidth = kTransmittanceWidth;
public static const uint kPlanetShadowHeight = kTransmittanceHeight;
public static const float kPlanetShadowReach = 256.0;

// The largest separation the shadow table holds at distance `distance`.
public float planetShadowSeparation(float distance, float top, float sunRadius) {
    return asin(min(top / distance, 1.0)) + sunRadius;
}

public float2 planetShadowUv(float distance, float separation, float top, float sunRadius) {
    let x = log(max(distance / top, 1.0)) / log(kPlanetShadowReach);
    let y = separation / planetShadowSeparation(distance, top, sunRadius);
    return float2(textureCoord(saturate(x), float(kPlanetShadowWidth)),
                  textureCoord(saturate(y), float(kPlanetShadowHeight)));
}

public void planetShadowDistance(float2 uv, float top, float sunRadius, out float distance,
                                 out float separation) {
    distance = top * exp(unitRange(uv.x, float(kPlanetShadowWidth)) * log(kPlanetShadowReach));
    separation = unitRange(uv.y, float(kPlanetShadowHeight)) *
                 planetShadowSeparation(distance, top, sunRadius);
}

// The fraction of the sun's light, per band of the group in table layer `layer`, that reaches a
// point `toPlanet` (km) from the planet's centre, with the sun of angular radius sine `sunSine` towards `sun`: from the
// shadow table, the light refracted through its atmosphere into its umbra included. 1 inside the
// atmosphere (the sky's own tables light it), beyond the table's reach the plain geometric eclipse
// by the atmosphere's disc.
public float4 planetShadow(Texture2DArray<float4> table, SamplerState linear, float3 toPlanet,
                           float3 sun, float sunSine, float4 wavelengths, uint layer, float top) {
    let distance = length(toPlanet);
    if (distance <= top) {
        return float4(1.0);
    }
    let direction = toPlanet / distance;
    let separation = 2.0 * asin(min(0.5 * length(direction - sun), 1.0));
    let sunRadius = asin(sunSine);
    if (separation >= planetShadowSeparation(distance, top, sunRadius)) {
        return float4(1.0);
    }
    if (distance > top * kPlanetShadowReach) {
        var eclipse = noEclipse();
        eclipse.bodies[0] = float4(toPlanet, top);
        eclipse.wavelengths = wavelengths;
        return eclipseVisibility(eclipse, float3(0.0), sun, sunSine);
    }
    let uv = planetShadowUv(distance, separation, top, sunRadius);
    return table.SampleLevel(linear, float3(uv, float(layer)), 0.0);
}
