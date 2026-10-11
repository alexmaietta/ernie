// SPDX-License-Identifier: BSD-3-Clause
// A planet's atmosphere for the atmosphere's table passes and shading: a spherical shell of gases
// and aerosols around the planet's centre, lengths in kilometres and coefficients in km^-1, the
// spectral bands four to a float4 ("band groups").
//
// The medium (`AtmosphereLuts`): `medium[0]` the bottom and top radii, the constituent count and
// the density samples per constituent; `medium[1].x` the band groups; then per constituent its
// scattering coefficients for each group and its absorption coefficients for each group, where
// its density is 1; then the ground albedo for each group. `densities` holds each constituent's
// relative density at `samples` altitudes evenly from the ground to the top.
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
//   Computer Graphics Forum 39(4):13-22, 2020, section 5.5 (multiple scattering).
//   https://doi.org/10.1111/cgf.14050
module atmosphere;

public static const uint kTransmittanceWidth = 256;
public static const uint kTransmittanceHeight = 64;
public static const uint kMultipleScatteringSize = 32;

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
