// SPDX-License-Identifier: BSD-3-Clause
// Shared inputs and helpers for reflectance models. A model is a struct implementing IBrdf whose
// radianceFactor returns the radiance factor I/F (pi times the bidirectional reflectance) for unit
// incident irradiance.
//
// Every backend evaluates models through the same helpers: brdfInputOf turns world or local
// directions into a BrdfInput, evaluateBrdf gives the cosine-weighted BRDF a light sample or path
// vertex multiplies incident radiance by, and sampleBrdf and brdfPdf importance sample incident
// directions for ray and path tracers.

public static const float PI = 3.14159265358979;

// Geometry of one evaluation, following Hapke's notation: i incidence angle, e emission angle,
// g phase angle, psi azimuth between the planes of incidence and emission (0 when the light and the
// viewer are on the same side of the normal). Hosts only evaluate models with mu0 and mu > 0.
public struct BrdfInput {
    public float mu0;
    public float mu;
    public float cosG;
    public float cosPsi;
    // Albedo input; each model documents how it uses it (e.g. Hapke's single-scattering albedo w).
    public float3 albedo;
    // Variance per axis of the surface slopes about the normal below what the host resolves (a
    // filtered normal map's, `MipFilter::NormalSlopes`), 0 when there are none. Models with a
    // roughness widen it by this, as the same slopes would at a finer scale; others ignore it.
    public float slopeVariance;
    // The wavelengths (nm) of the three bands in the RGB slots, for models whose reflectance
    // depends on them beyond their colours (a thin film's interference); 550 where the host has
    // none.
    public float3 wavelengths;
    // The model's colour parameters (`BrdfParam::colour`) in its order, each as the host's
    // spectral uplift of its RGB at the three bands, as the albedo is: white where the host has
    // none, and for colours past `kBrdfColours`.
    public float3 colours[kBrdfColours];
    // The directions towards the light and the viewer in the surface's frame: x its tangent (the
    // host's, along which its texture coordinates' u grows, where it has one), y its bitangent,
    // z its normal, so z is mu0 and mu. For anisotropic models; isotropic ones need only the
    // angles above.
    public float3 light;
    public float3 view;
}

// The most colour parameters a host uplifts into `BrdfInput::colours`.
public static const uint kBrdfColours = 6;
// Parameters packed with the material (`BrdfParams::values`); the rest, up to 48, behind
// `BrdfParams::extended`.
public static const uint kBrdfCoreParams = 16;

// Hapke's mean slope angle theta-bar (radians) widened by unresolved slopes of per-axis variance
// `slopeVariance`. Hapke's roughness distribution has Gaussian slopes of variance
// pi tan^2(theta-bar) / 2 per axis, and independent slopes' variances add.
public float widenedMeanSlope(float thetaBar, float slopeVariance) {
    let tanT = tan(thetaBar);
    return atan(sqrt(tanT * tanT + 2.0 * slopeVariance / PI));
}

// Up to 48 model parameters in the order the model declares them: the first 16 packed with the
// material, read with it; the rest behind `extended`, read only where the model asks for them (a
// layer a material uses), after the material's feature bits (`BrdfModel::features`). `extended`
// is null for models of 16 or fewer.
public struct BrdfParams {
    public float4 values[4];
    public float* extended;

    public float get(uint i) {
        return i < kBrdfCoreParams ? values[i / 4][i % 4] : extended[1 + i - kBrdfCoreParams];
    }

    // The material's features (`BrdfModel::features`): the layers it uses. Only for models with
    // extended parameters, whose materials always have them (a pointer's comparison with null
    // would need 64-bit integers).
    public uint features() {
        return asuint(extended[0]);
    }
}

// How a surface reflects light arriving from all around it, as from a dome or ambient light: the
// weight on the irradiance over pi (Lambert's is its albedo), and up to two specular lobes, each a
// weight on the radiance from around the mirror direction and the lobe's perceptual roughness
// (the square root of its GGX alpha), over which the host prefilters that radiance.
public struct EnvironmentResponse {
    public float3 diffuse;
    public float3 specular0;
    public float roughness0;
    public float3 specular1;
    public float roughness1;
    // The normal the first lobe mirrors the viewer about, in the surface's frame
    // (`BrdfInput::light` and `view`'s): bent for an anisotropic lobe, whose reflections stretch
    // across its tangent; the normal, (0, 0, 1), otherwise.
    public float3 normal0;
    // The part of `diffuse` that enters the surface to leave it elsewhere (`IBrdf::subsurface`).
    public float3 subsurface;
}

// A purely diffuse response of weight `albedo`.
public EnvironmentResponse diffuseEnvironment(float3 albedo) {
    EnvironmentResponse r;
    r.diffuse = albedo;
    r.specular0 = float3(0.0);
    r.roughness0 = 1.0;
    r.specular1 = float3(0.0);
    r.roughness1 = 1.0;
    r.normal0 = float3(0.0, 0.0, 1.0);
    r.subsurface = float3(0.0);
    return r;
}

// The share of a radiance factor that facets reflect at the surface (a specular lobe), which
// polarizes it by Fresnel's laws about the facets' plane of incidence, and the facets' index of
// refraction; the rest is scattered beneath the surface and leaves depolarized.
public struct SpecularReflection {
    public float3 factor;
    public float ior;
}

// How much of the light from behind a surface comes through it towards the viewer, per band
// (`factor`; what it reflects is its radiance factor's): through a solid body of index of
// refraction `ior`, bent at its surface, or a thin-walled one (`thinWalled`, such as a pane),
// which lets it through unbent.
public struct Transmission {
    public float3 factor;
    public float ior;
    public bool thinWalled;
}

public interface IBrdf {
    float3 radianceFactor(BrdfInput s, BrdfParams p);

    // The part of `radianceFactor` that enters the surface to leave it around where it entered,
    // per band: light scattered beneath it, which the host spreads by the model's subsurface
    // radius (`BrdfModel::subsurfaceRadius`). Models that do not override it scatter none so.
    float3 subsurface(BrdfInput s, BrdfParams p) {
        return float3(0.0);
    }

    // The light from behind the surface it lets through towards the viewer at cos(e) `s.mu`.
    // Models that do not override it are opaque.
    Transmission transmission(BrdfInput s, BrdfParams p) {
        Transmission t;
        t.factor = float3(0.0);
        t.ior = 1.0;
        t.thinWalled = true;
        return t;
    }

    // The specular part of `radianceFactor` for polarization. Models that do not override it
    // reflect nothing specularly: all their light diffuse.
    SpecularReflection specularReflection(BrdfInput s, BrdfParams p) {
        SpecularReflection r;
        r.factor = float3(0.0);
        r.ior = 1.5;
        return r;
    }

    // The response to surrounding light for a viewer at cos(e) `mu` (`s.mu`; the other angles of
    // `s` are unused), for the dome and ambient terms of a rasterizer. Models that do not
    // override it reflect it diffusely with their albedo input as the weight.
    EnvironmentResponse environment(BrdfInput s, BrdfParams p) {
        return diffuseEnvironment(s.albedo);
    }
}

// The geometry of unit directions towards the light l and the viewer v about the unit normal n,
// any frame, with the given albedo input. mu0 <= 0 means the light is below the horizon, where
// models are not evaluated; mu is clamped to at least 1e-3 so views grazing or just below the
// horizon (as with interpolated shading normals) stay finite. The surface's frame is
// `orthonormalBasis`'s about n (`brdfInputIn` for a tangent's).
public BrdfInput brdfInputOf(float3 n, float3 l, float3 v, float3 albedo) {
    float3 t;
    float3 b;
    orthonormalBasis(n, t, b);
    return brdfInputIn(t, b, n, l, v, albedo);
}

// `brdfInputOf` in the surface frame of unit tangent t, bitangent b and normal n.
public BrdfInput brdfInputIn(float3 t, float3 b, float3 n, float3 l, float3 v, float3 albedo) {
    let mu0 = dot(n, l);
    let mu = dot(n, v);
    let lPlane = l - n * mu0;
    let vPlane = v - n * mu;
    var cosPsi = 1.0;
    if (length(lPlane) > 1e-5 && length(vPlane) > 1e-5) {
        cosPsi = dot(normalize(lPlane), normalize(vPlane));
    }
    BrdfInput input;
    input.mu0 = mu0;
    input.mu = max(mu, 1e-3);
    input.cosG = dot(l, v);
    input.cosPsi = cosPsi;
    input.albedo = albedo;
    input.slopeVariance = 0.0;
    input.wavelengths = float3(550.0);
    for (uint c = 0; c < kBrdfColours; ++c) {
        input.colours[c] = float3(1.0);
    }
    input.light = float3(dot(l, t), dot(l, b), mu0);
    input.view = float3(dot(v, t), dot(v, b), input.mu);
    return input;
}

// The BRDF times cos(i), f_r mu0 = radianceFactor / pi: the radiance reflected towards v per unit
// radiance arriving from l per steradian. Zero with the light below the horizon; never negative.
public float3 evaluateBrdf<B : IBrdf>(B model, float3 n, float3 l, float3 v, float3 albedo,
                                       BrdfParams p) {
    let input = brdfInputOf(n, l, v, albedo);
    if (input.mu0 <= 0.0) {
        return float3(0.0);
    }
    return max(model.radianceFactor(input, p), float3(0.0)) / PI;
}

// A sampled incident direction: `direction` towards the light (unit, above the horizon), its
// probability density per steradian, and `weight`, evaluateBrdf over the density, by which the
// radiance arriving along the direction scales into the radiance reflected towards the viewer.
public struct BrdfSample {
    public float3 direction;
    public float pdf;
    public float3 weight;
}

// A right-handed orthonormal basis (t, b, n) around the unit vector n, without branches on the
// normal's direction (Duff et al., "Building an Orthonormal Basis, Revisited", JCGT 6(1), 2017).
public void orthonormalBasis(float3 n, out float3 t, out float3 b) {
    let s = n.z >= 0.0 ? 1.0 : -1.0;
    let a = -1.0 / (s + n.z);
    let c = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

// A cosine-distributed direction about the unit normal n for uniform u in [0, 1)^2, by Malley's
// method: a concentric disk sample (Shirley and Chiu 1997) lifted onto the hemisphere.
public float3 cosineHemisphere(float3 n, float2 u) {
    let d = 2.0 * u - 1.0;
    var r = 0.0;
    var phi = 0.0;
    if (d.x != 0.0 || d.y != 0.0) {
        if (abs(d.x) > abs(d.y)) {
            r = d.x;
            phi = (PI / 4.0) * (d.y / d.x);
        } else {
            r = d.y;
            phi = (PI / 2.0) - (PI / 4.0) * (d.x / d.y);
        }
    }
    let x = r * cos(phi);
    let y = r * sin(phi);
    let z = sqrt(max(0.0, 1.0 - x * x - y * y));
    float3 t;
    float3 b;
    orthonormalBasis(n, t, b);
    return normalize(x * t + y * b + z * n);
}

// The density per steradian with which sampleBrdf picks the incident direction l: cosine
// weighted, cos(i) / pi above the horizon and zero below. Models share it, since every built-in one
// is dominated by the cos(i) factor of its radiance factor; MIS weights use it with light sampling.
public float brdfPdf(float3 n, float3 l) {
    return max(dot(n, l), 0.0) / PI;
}

// Samples an incident direction for the viewer direction v with brdfPdf, from uniform u in
// [0, 1)^2. The weight is radianceFactor / mu0, exactly the albedo for Lambert.
public BrdfSample sampleBrdf<B : IBrdf>(B model, float3 n, float3 v, float3 albedo, BrdfParams p,
                                        float2 u) {
    BrdfSample sample;
    sample.direction = cosineHemisphere(n, u);
    sample.pdf = brdfPdf(n, sample.direction);
    sample.weight = float3(0.0);
    if (sample.pdf > 0.0) {
        sample.weight = evaluateBrdf(model, n, sample.direction, v, albedo, p) / sample.pdf;
    }
    return sample;
}

public float phaseAngle(BrdfInput s) {
    return acos(clamp(s.cosG, -1.0, 1.0));
}

// Single-lobe Henyey-Greenstein phase function in Hapke's sign convention: xi < 0 backscatters.
// Normalised so its average over the sphere is 1.
public float henyeyGreenstein(float xi, float cosG) {
    let xi2 = xi * xi;
    return (1.0 - xi2) / pow(max(1.0 + 2.0 * xi * cosG + xi2, 1e-6), 1.5);
}

// Two-parameter double Henyey-Greenstein phase function (Hapke 2012): lobe width b in [0, 1), and c
// in [-1, 1] weighting the backscattering lobe ((1 + c) / 2) against the forward lobe.
public float doubleHenyeyGreenstein(float b, float c, float cosG) {
    let b2 = b * b;
    let back = (1.0 - b2) / pow(max(1.0 - 2.0 * b * cosG + b2, 1e-6), 1.5);
    let forward = (1.0 - b2) / pow(max(1.0 + 2.0 * b * cosG + b2, 1e-6), 1.5);
    return 0.5 * (1.0 + c) * back + 0.5 * (1.0 - c) * forward;
}

// GGX microfacet helpers shared by the models with specular lobes.
//
// References:
// - E. Heitz, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs", JCGT
//   3(2), 2014.
// - E. Turquin, "Practical multiple scattering compensation for microfacet models", ILM, 2019.
// - B. Karis, "Physically Based Shading on Mobile", 2014 (analytic environment BRDF).

// GGX normal distribution D(h) for alpha^2 `a2` (Trowbridge and Reitz 1975; Walter et al. 2007).
public float ggx(float nDotH, float a2) {
    let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Height-correlated Smith visibility G / (4 mu0 mu) for GGX (Heitz 2014).
public float smithVisibility(float mu0, float mu, float a2) {
    let lambdaV = mu0 * sqrt(mu * mu * (1.0 - a2) + a2);
    let lambdaL = mu * sqrt(mu0 * mu0 * (1.0 - a2) + a2);
    return 0.5 / max(lambdaV + lambdaL, 1e-8);
}

public float3 schlick(float3 f0, float vDotH) {
    let m = pow(saturate(1.0 - vDotH), 5.0);
    return f0 + (1.0 - f0) * m;
}

// The split-sum environment BRDF's scale and bias on f0 for perceptual roughness `roughness` at
// cos(e) `mu`, from Lazarov's analytic fit (Karis 2014), so no lookup table is needed.
public float2 environmentBrdf(float roughness, float mu) {
    let c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    let c1 = float4(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * mu)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}

// The factor restoring the energy a GGX lobe of perceptual roughness `roughness` loses to its
// single-scattering approximation, for reflectance `f0` at cos(e) `mu`: 1 + f0 (1 / E - 1), E the
// lobe's directional albedo for f0 = 1 (the environment BRDF's scale plus bias), as Filament does.
public float3 energyCompensation(float3 f0, float roughness, float mu) {
    let ab = environmentBrdf(roughness, saturate(mu));
    return 1.0 + f0 * (1.0 / max(ab.x + ab.y, 1e-3) - 1.0);
}

// The cosine at which Schlick's Fresnel times the split sum's scale plus bias, A + B, equals its
// f0 A + B for every f0: (1 - mu)^5 = B / (A + B). A model's own Fresnel there (an exact
// dielectric's, a tinted metal's, a thin film's) stands in for Schlick's in the split sum.
public float environmentCosine(float2 ab) {
    return 1.0 - pow(saturate(ab.y / max(ab.x + ab.y, 1e-6)), 0.2);
}

// `energyCompensation` for direct light: the geometric mean of the factors at the incidence and
// emission angles, so the BRDF stays reciprocal (it equals the emission one when they match).
public float3 reciprocalCompensation(float3 f0, float roughness, float mu0, float mu) {
    return sqrt(energyCompensation(f0, roughness, mu0) * energyCompensation(f0, roughness, mu));
}

// The perceptual roughness of a lobe of `roughness` widened by unresolved slopes of `variance`.
public float widenedRoughness(float roughness, float variance) {
    let alpha = roughness * roughness;
    return sqrt(sqrt(min(alpha * alpha + 2.0 * max(variance, 0.0), 1.0)));
}

// Fresnel's reflectance of unpolarized light at cos(i) `cosI` onto a dielectric of relative index
// of refraction `eta` (from the side of index 1; eta < 1 from inside), 1 past the critical angle.
//
// Reference: M. Born and E. Wolf, "Principles of Optics", 7th ed., Cambridge, 1999, sec. 1.5.
public float fresnelDielectric(float cosI, float eta) {
    let c = saturate(cosI);
    let sinT2 = (1.0 - c * c) / (eta * eta);
    if (sinT2 >= 1.0) {
        return 1.0;
    }
    let cosT = sqrt(1.0 - sinT2);
    let rs = (c - eta * cosT) / (c + eta * cosT);
    let rp = (eta * c - cosT) / (eta * c + cosT);
    return 0.5 * (rs * rs + rp * rp);
}

// The hemispherical mean of `fresnelDielectric` (cosine weighted) onto a dielectric of eta >= 1,
// as fitted by Kulla and Conty: the light a smooth dielectric reflects from all around.
//
// Reference: C. Kulla and A. Conty, "Revisiting Physically Based Shading at Imageworks",
// SIGGRAPH 2017 course: Physically Based Shading in Theory and Practice.
public float meanFresnelDielectric(float eta) {
    let n = max(eta, 1.0);
    return (n - 1.0) / (4.08567 + 1.00071 * n);
}
