// SPDX-License-Identifier: BSD-3-Clause
import brdf;

// OpenPBR Surface's reflectance: a base of a dielectric (an energy-preserving Oren-Nayar diffuse
// under a GGX specular lobe of exact Fresnel) mixed with a metal (F82-tint Fresnel), each
// specular lobe optionally through a thin film's interference; under an optional coat (a second
// GGX lobe of its own index, its colour absorbing what crosses it, darkening the base by the
// light it reflects back down and roughening its lobe) and an optional fuzz (Zeltner's sheen).
// Layers combine by albedo scaling: each one passes on what its directional albedo leaves,
// reciprocally (the geometric mean of the albedos at cos(i) and cos(e)).
//
// The albedo input is base_color. Parameters, in order: 0 base_weight, 1 base_metalness,
// 2 base_diffuse_roughness, 3 specular_weight, 4 specular_roughness, 5 specular_ior,
// 6-8 specular_color, 9 coat_weight, 10 fuzz_weight, 11 thin_film_weight, 12 coat_roughness,
// 13 coat_ior, 14 coat_darkening, 15 fuzz_roughness; past the first 16, read only for a layer
// the material has: 16-18 coat_color, 19-21 fuzz_color, 22 thin_film_thickness (um),
// 23 thin_film_ior, 24 specular_roughness_anisotropy (along the surface's tangent,
// `BrdfInput::light` and `view`'s frame), 25 transmission_weight, 26-28 transmission_color,
// 29 geometry_thin_walled (0 or 1), 30 subsurface_weight, 31-33 subsurface_color,
// 34 subsurface_radius (scene units), 35-37 subsurface_radius_scale. Subsurface scattering
// replaces that share of the dielectric base's diffuse with a Lambertian lobe of
// subsurface_color, which `subsurface` reports for the host to spread over the radius, per band
// (`BrdfModel::subsurfaceRadius`). A transmitting material's dielectric base lets through, in
// place of its diffuse, what its specular lobe, coat and fuzz leave (`transmission`), unblurred
// by its roughness. Colours arrive uplifted at the bands (`BrdfInput::colours`: specular_color,
// coat_color, fuzz_color), and the film's interference is at the bands' wavelengths.
//
// Layers cost nothing where no material has them: the renderer defines BRDF_FEATURES as the
// union of its materials' (`kOpenPbrCoat`, `kOpenPbrFuzz`, `kOpenPbrThinFilm`), so code under the
// others falls away, and a material without one skips it per pixel.
//
// References:
// - J. Andersson, A. Portsmouth, et al., "OpenPBR Surface Specification", v1.1, Academy Software
//   Foundation, 2024. https://academysoftwarefoundation.github.io/OpenPBR/
// - J. Portsmouth, P. Kutz and S. Hill, "EON: A practical energy-preserving rough diffuse BRDF",
//   Journal of Computer Graphics Techniques 14(1), 2025 (its listing's E_FON fit).
// - N. Hoffman, "Generalization of Adobe's Fresnel Model", 2023 (F82 tint); P. Kutz, M. Hasan
//   and P. Edmondson, "Novel aspects of the Adobe Standard Material", 2021.
// - T. Zeltner, B. Burley and M. J.-Y. Chiang, "Practical Multiple-Scattering Sheen Using Linearly
//   Transformed Cosines", SIGGRAPH 2022 Talks, as MaterialX's mx_zeltner_sheen fits.
// - L. Belcour and P. Barla, "A Practical Extension to Microfacet Theory for the Modeling of
//   Varying Iridescence", ACM Transactions on Graphics 36(4), 2017 (the film's Airy
//   reflectance, here per band rather than in its Fourier form, the bands being wavelengths).
// - C. Kulla and A. Conty, "Revisiting Physically Based Shading at Imageworks", SIGGRAPH 2017
//   course (albedo scaling, mean Fresnel).
// - R. Guy and M. Agopian, "Physically Based Rendering in Filament", 2018, section 4.6.1 (the
//   bent reflection vector of an anisotropic lobe's image-based lighting).
//   https://google.github.io/filament/Filament.html
// - B. Burley, "Physically-Based Shading at Disney", SIGGRAPH 2012 course, and E. Heitz,
//   "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs", JCGT 3(2), 2014
//   (anisotropic GGX and its height-correlated Smith visibility).

#ifndef BRDF_FEATURES
#define BRDF_FEATURES 0xFFFFFFFFu
#endif

// Feature bits (scene/brdf.cpp's OpenPbr features).
public static const uint kOpenPbrCoat = 1u;
public static const uint kOpenPbrFuzz = 2u;
public static const uint kOpenPbrThinFilm = 4u;
public static const uint kOpenPbrTransmission = 8u;
public static const uint kOpenPbrSubsurface = 16u;
// Anisotropy reads the surface's tangent: `BrdfRegistry::kTangentsFeature`.
public static const uint kOpenPbrAnisotropy = 0x80000000u;

// Anisotropic GGX's distribution D(h) for the half vector `h` in the surface's frame (x along the
// tangent) and roughnesses `at` and `ab` along the tangent and bitangent.
float ggxAnisotropic(float3 h, float at, float ab) {
    let d = h.x * h.x / (at * at) + h.y * h.y / (ab * ab) + h.z * h.z;
    return 1.0 / (PI * at * ab * d * d);
}

// Its height-correlated Smith visibility G / (4 mu0 mu) for light `l` and view `v` in that frame.
float smithVisibilityAnisotropic(float3 l, float3 v, float at, float ab) {
    let lambdaV = l.z * length(float3(at * v.x, ab * v.y, v.z));
    let lambdaL = v.z * length(float3(at * l.x, ab * l.y, l.z));
    return 0.5 / max(lambdaV + lambdaL, 1e-8);
}

// Whether this material has layer `bit`: never where no material of the scene does.
bool hasLayer(BrdfParams p, uint bit) {
    return (BRDF_FEATURES & bit) != 0u && (p.features() & bit) != 0u;
}

// EON's Fujii Oren-Nayar constants and the directional albedo of its single scattering for
// rho = 1, at cos `mu` and roughness `r`.
static const float kFon1 = 0.5 - 2.0 / (3.0 * PI);
static const float kFon2 = 2.0 / 3.0 - 28.0 / (15.0 * PI);

float fonAlbedo(float mu, float r) {
    let m = 1.0 - saturate(mu);
    let gOverPi = m * (0.0571085289 + m * (0.491881867 + m * (-0.332181442 + m * 0.0714429953)));
    return (1.0 + r * gOverPi) / (1.0 + kFon1 * r);
}

// EON's f_r for albedo `rho` and roughness `r`: Fujii's Oren-Nayar single scattering and the
// multiple scattering that makes it lose no energy.
float3 eon(float3 rho, float r, float mu0, float mu, float cosG) {
    let s = cosG - mu0 * mu;
    let sOverT = s > 0.0 ? s / max(max(mu0, mu), 1e-6) : s;
    let a = 1.0 / (1.0 + kFon1 * r);
    let single = rho / PI * a * (1.0 + r * sOverT);
    let mean = a * (1.0 + kFon2 * r);
    let rhoMs = rho * rho * mean / (1.0 - rho * (1.0 - mean));
    let multiple = rhoMs / PI * max(1.0 - fonAlbedo(mu0, r), 1e-7) *
                   max(1.0 - fonAlbedo(mu, r), 1e-7) / max(1.0 - mean, 1e-7);
    return single + multiple;
}

// EON's directional albedo at cos `mu`.
float3 eonAlbedo(float3 rho, float r, float mu) {
    let e = fonAlbedo(mu, r);
    let mean = (1.0 + kFon2 * r) / (1.0 + kFon1 * r);
    let rhoMs = rho * rho * mean / (1.0 - rho * (1.0 - mean));
    return rho * e + rhoMs * (1.0 - e);
}

// The F82-tint metal's Fresnel: Schlick's of `f0` less a term making it `tint` times Schlick's at
// cos 1/7 (about 82 degrees).
float3 f82(float3 f0, float3 tint, float mu) {
    let muBar = 1.0 / 7.0;
    let mBar = 1.0 - muBar;
    let schlickBar = f0 + (1.0 - f0) * pow(mBar, 5.0);
    let a = schlickBar * (1.0 - tint) / (muBar * pow(mBar, 6.0));
    let m = 1.0 - saturate(mu);
    return saturate(f0 + (1.0 - f0) * pow(m, 5.0) - a * saturate(mu) * pow(m, 6.0));
}

// The reflectance of a film of index `nf` and thickness `nm` between a medium of index `n1` and
// a substrate of real index `n3`, at cos `cos1` in the medium and wavelength `lambda` (nm): the
// Airy sum of its two interfaces' amplitudes, s and p averaged.
float airy(float cos1, float n1, float nf, float n3, float nm, float lambda) {
    let sin1 = 1.0 - cos1 * cos1;
    let sinF = (n1 / nf) * (n1 / nf) * sin1;
    if (sinF >= 1.0) {
        return 1.0;
    }
    let cosF = sqrt(1.0 - sinF);
    let cos3 = sqrt(max(1.0 - (n1 / n3) * (n1 / n3) * sin1, 0.0));
    let r12s = (n1 * cos1 - nf * cosF) / (n1 * cos1 + nf * cosF);
    let r12p = (nf * cos1 - n1 * cosF) / (nf * cos1 + n1 * cosF);
    let r23s = (nf * cosF - n3 * cos3) / max(nf * cosF + n3 * cos3, 1e-6);
    let r23p = (n3 * cosF - nf * cos3) / max(n3 * cosF + nf * cos3, 1e-6);
    let c = cos(4.0 * PI * nf * nm * cosF / max(lambda, 1.0));
    let s = (r12s * r12s + r23s * r23s + 2.0 * r12s * r23s * c) /
            (1.0 + r12s * r12s * r23s * r23s + 2.0 * r12s * r23s * c);
    let p = (r12p * r12p + r23p * r23p + 2.0 * r12p * r23p * c) /
            (1.0 + r12p * r12p * r23p * r23p + 2.0 * r12p * r23p * c);
    return saturate(0.5 * (s + p));
}

// The real index of refraction whose reflectance at normal incidence is `f0`.
float iorOf(float f0) {
    let root = sqrt(clamp(f0, 0.0, 0.98));
    return (1.0 + root) / (1.0 - root);
}

// Zeltner's sheen: its directional albedo at cos `mu` for roughness `r`, and the inverse LTC
// matrix's coefficients (a over the tangent axes, b mixing the normal into the view's tangent).
float sheenAlbedo(float mu, float r) {
    let s = r * (0.0206607 + 1.58491 * r) / (0.0379424 + r * (1.32227 + r));
    let m = r * (-0.193854 + r * (-1.14885 + r * (1.7932 - 0.95943 * r * r))) / (0.046391 + r);
    let o = r * (0.000654023 + (-0.0207818 + 0.119681 * r) * r) / (1.26264 + r * (-1.92021 + r));
    let z = (mu - m) / s;
    return saturate(exp(-0.5 * z * z) / (s * sqrt(2.0 * PI)) + o);
}

float sheenAInv(float mu, float r) {
    return (2.58126 * mu + 0.813703 * r) * r / (1.0 + 0.310327 * mu * mu + 2.60994 * mu * r);
}

float sheenBInv(float mu, float r) {
    return sqrt(1.0 - mu) * (r - 1.0) * r * r * r /
           (0.0000254053 + 1.71228 * mu - 1.71506 * mu * r + 1.34174 * r * r);
}

// The sheen's distribution of light towards the viewer over directions to the light (f_r cos(i)
// over its albedo, integrating to 1): a clamped cosine through the inverse LTC matrix.
float sheenLtc(BrdfInput s, float r) {
    let sinI = sqrt(max(1.0 - s.mu0 * s.mu0, 0.0));
    let sinPsi = sqrt(max(1.0 - s.cosPsi * s.cosPsi, 0.0));
    let aInv = sheenAInv(s.mu, r);
    let bInv = sheenBInv(s.mu, r);
    let w = float3(aInv * sinI * s.cosPsi + bInv * s.mu0, aInv * sinI * sinPsi, s.mu0);
    let lengthSquared = dot(w, w);
    return max(s.mu0, 0.0) / PI * (aInv / lengthSquared) * (aInv / lengthSquared);
}

public struct OpenPbr : IBrdf {
    // One material's layers, as read from its parameters and the evaluation's colours.
    struct Material {
        float3 base;  // base_weight times base_color, darkened under the coat
        float metalness;
        float diffuseRoughness;
        float specularWeight;
        float specularRoughness;  // roughened under the coat, before unresolved slopes
        float anisotropy;  // specular_roughness_anisotropy, 0 without the feature
        float specularIor;
        float3 tint;
        float ambient;  // the base's surrounding index: the coat's as far as it covers it
        float coatWeight;
        float coatRoughness;
        float coatIor;
        float3 coatTransmission;
        float fuzzWeight;
        float fuzzRoughness;
        float3 fuzzColour;
        float filmWeight;
        float filmNm;
        float filmIor;
        float transmission;
        float3 transmissionColour;
        bool thinWalled;
        float subsurface;
        float3 subsurfaceColour;
    }

    static Material materialOf(BrdfInput s, BrdfParams p) {
        Material m;
        m.base = max(p.get(0), 0.0) * s.albedo;
        m.metalness = saturate(p.get(1));
        m.diffuseRoughness = saturate(p.get(2));
        m.specularWeight = max(p.get(3), 0.0);
        m.specularRoughness = saturate(p.get(4));
        m.specularIor = max(p.get(5), 1e-3);
        m.tint = max(s.colours[0], float3(0.0));
        m.anisotropy = hasLayer(p, kOpenPbrAnisotropy) ? saturate(p.get(24)) : 0.0;
        m.ambient = 1.0;
        m.coatWeight = 0.0;
        m.coatRoughness = 0.0;
        m.coatIor = 1.5;
        m.coatTransmission = float3(1.0);
        if (hasLayer(p, kOpenPbrCoat)) {
            m.coatWeight = saturate(p.get(9));
            m.coatRoughness = saturate(p.get(12));
            m.coatIor = max(p.get(13), 1.0);
            m.coatTransmission = lerp(float3(1.0), saturate(s.colours[1]), m.coatWeight);
            m.ambient = lerp(1.0, m.coatIor, m.coatWeight);
            // Light the coat reflects back down (its internal reflectance K) darkens the base
            // as a rough base sends it back up: rho (1 - K) / (1 - rho K).
            let k = 1.0 - (1.0 - meanFresnelDielectric(m.coatIor)) / (m.coatIor * m.coatIor);
            let darkening = m.coatWeight * saturate(p.get(14));
            m.base = lerp(m.base, m.base * (1.0 - k) / max(1.0 - saturate(m.base) * k, 1e-4),
                          darkening);
            // The base's lobe seen through a rough coat, wider.
            let r2 = m.specularRoughness * m.specularRoughness;
            let c2 = m.coatRoughness * m.coatRoughness;
            m.specularRoughness =
                min(sqrt(sqrt(r2 * r2 + 2.0 * m.coatWeight * c2 * c2)), 1.0);
        }
        m.fuzzWeight = 0.0;
        m.fuzzRoughness = 0.5;
        m.fuzzColour = float3(1.0);
        if (hasLayer(p, kOpenPbrFuzz)) {
            m.fuzzWeight = saturate(p.get(10));
            m.fuzzRoughness = clamp(p.get(15), 0.01, 1.0);
            m.fuzzColour = saturate(s.colours[2]);
        }
        m.filmWeight = 0.0;
        m.filmNm = 0.0;
        m.filmIor = 1.5;
        if (hasLayer(p, kOpenPbrThinFilm)) {
            m.filmWeight = saturate(p.get(11));
            m.filmNm = max(p.get(22), 0.0) * 1000.0;
            m.filmIor = max(p.get(23), 1.0);
        }
        m.subsurface = 0.0;
        m.subsurfaceColour = float3(1.0);
        if (hasLayer(p, kOpenPbrSubsurface)) {
            m.subsurface = saturate(p.get(30));
            m.subsurfaceColour = saturate(s.colours[4]);
        }
        m.transmission = 0.0;
        m.transmissionColour = float3(1.0);
        m.thinWalled = false;
        if (hasLayer(p, kOpenPbrTransmission)) {
            m.transmission = saturate(p.get(25));
            m.transmissionColour = saturate(s.colours[3]);
            m.thinWalled = p.get(29) > 0.5;
        }
        return m;
    }

    // The dielectric's and the metal's Fresnel at cos `c` (onto the facets), weighted by the
    // specular weight and tinted, through the film where the material has one.
    static void fresnels(Material m, float c, float3 wavelengths, out float3 dielectric,
                         out float3 metal) {
        let eta = m.specularIor / m.ambient;
        dielectric = m.specularWeight * m.tint * fresnelDielectric(c, eta);
        metal = m.specularWeight * f82(m.base, m.tint, c);
        if (m.filmWeight > 0.0) {
            float3 filmDielectric;
            float3 filmMetal;
            for (uint k = 0; k < 3; ++k) {
                filmDielectric[k] =
                    airy(c, m.ambient, m.filmIor, m.specularIor, m.filmNm, wavelengths[k]);
                filmMetal[k] = airy(c, m.ambient, m.filmIor, iorOf(m.base[k]), m.filmNm,
                                    wavelengths[k]);
            }
            dielectric = lerp(dielectric, m.specularWeight * m.tint * filmDielectric, m.filmWeight);
            metal = lerp(metal, m.specularWeight * filmMetal, m.filmWeight);
        }
    }

    // The dielectric specular lobe's normal reflectance, for its energy compensation and its
    // directional albedo.
    static float dielectricF0(Material m) {
        let eta = m.specularIor / m.ambient;
        let r0 = (eta - 1.0) / (eta + 1.0);
        return r0 * r0;
    }

    // The directional albedo at cos `mu` of the dielectric specular lobe (perceptual roughness
    // `roughness`), which the diffuse under it does not get.
    static float3 dielectricAlbedo(Material m, float roughness, float mu) {
        let ab = environmentBrdf(roughness, saturate(mu));
        let f0 = dielectricF0(m);
        return m.specularWeight * m.tint * (f0 * ab.x + ab.y) *
               energyCompensation(float3(f0), roughness, mu);
    }

    // The coat's directional albedo at cos `mu`.
    static float coatAlbedo(Material m, float roughness, float mu) {
        let ab = environmentBrdf(roughness, saturate(mu));
        let r0 = (m.coatIor - 1.0) / (m.coatIor + 1.0);
        return m.coatWeight * (r0 * r0 * ab.x + ab.y);
    }

    struct Lobes {
        float3 diffuse;
        float3 specular;
        float ior;
        float3 subsurface;  // the part of `diffuse` scattered beneath the surface
    }

    // f_r's diffuse part (the base's diffuse and the fuzz, unpolarized) and specular part (the
    // base's and the coat's lobes), and the outer specular facets' index of refraction.
    static Lobes lobesOf(BrdfInput s, BrdfParams p) {
        let m = materialOf(s, p);
        let mu0 = s.mu0;
        let mu = s.mu;
        // The half vector from cos(g): |l + v| = sqrt(2 + 2 cos g).
        let sum = sqrt(max(2.0 + 2.0 * s.cosG, 1e-8));
        let nDotH = saturate((mu0 + mu) / sum);
        let vDotH = saturate(0.5 * sum);
        let unresolved = 2.0 * max(s.slopeVariance, 0.0);

        let alpha = m.specularRoughness * m.specularRoughness;
        var a2 = clamp(alpha * alpha + unresolved, 1e-6, 1.0);
        var microfacet = 0.0;
        if (m.anisotropy > 0.0) {
            // OpenPBR's split of the roughness along the tangent and across it, each widened by
            // the unresolved slopes; compensated as their geometric mean.
            let along = alpha * sqrt(2.0 / (1.0 + (1.0 - m.anisotropy) * (1.0 - m.anisotropy)));
            let at = sqrt(clamp(along * along + unresolved, 1e-6, 1.0));
            let across = (1.0 - m.anisotropy) * along;
            let ab = sqrt(clamp(across * across + unresolved, 1e-6, 1.0));
            let h = normalize(s.light + s.view);
            microfacet = ggxAnisotropic(h, at, ab) *
                         smithVisibilityAnisotropic(s.light, s.view, at, ab);
            a2 = at * ab;
        } else {
            microfacet = ggx(nDotH, a2) * smithVisibility(mu0, mu, a2);
        }
        let roughness = sqrt(sqrt(a2));
        float3 dielectric;
        float3 metal;
        fresnels(m, vDotH, s.wavelengths, dielectric, metal);
        let specular = lerp(dielectric * reciprocalCompensation(float3(dielectricF0(m)),
                                                                roughness, mu0, mu),
                            metal * reciprocalCompensation(m.base, roughness, mu0, mu),
                            m.metalness) *
                       microfacet;
        let under = sqrt(max((1.0 - dielectricAlbedo(m, roughness, mu0)) *
                                 (1.0 - dielectricAlbedo(m, roughness, mu)),
                             float3(0.0)));
        Lobes lobes;
        let base = (1.0 - m.metalness) * (1.0 - m.transmission) * under;
        lobes.subsurface = base * m.subsurface * m.subsurfaceColour / PI;
        lobes.diffuse = base * (1.0 - m.subsurface) *
                            eon(saturate(m.base), m.diffuseRoughness, mu0, mu, s.cosG) +
                        lobes.subsurface;
        lobes.specular = specular;
        let meanBase = (m.base.x + m.base.y + m.base.z) / 3.0;
        lobes.ior = lerp(m.specularIor, iorOf(meanBase), m.metalness);

        if (m.coatWeight > 0.0) {
            let coatAlpha = m.coatRoughness * m.coatRoughness;
            let coatA2 = clamp(coatAlpha * coatAlpha + unresolved, 1e-6, 1.0);
            let coatRoughness = sqrt(sqrt(coatA2));
            let r0 = (m.coatIor - 1.0) / (m.coatIor + 1.0);
            let coat = m.coatWeight * fresnelDielectric(vDotH, m.coatIor) *
                       ggx(nDotH, coatA2) * smithVisibility(mu0, mu, coatA2) *
                       reciprocalCompensation(float3(r0 * r0), coatRoughness, mu0, mu).x;
            let through = m.coatTransmission *
                          sqrt(max((1.0 - coatAlbedo(m, coatRoughness, mu0)) *
                                       (1.0 - coatAlbedo(m, coatRoughness, mu)),
                                   0.0));
            lobes.diffuse *= through;
            lobes.subsurface *= through;
            lobes.specular = lobes.specular * through + coat;
            lobes.ior = lerp(lobes.ior, m.coatIor, m.coatWeight);
        }
        if (m.fuzzWeight > 0.0) {
            let albedo0 = m.fuzzWeight * sheenAlbedo(mu0, m.fuzzRoughness);
            let albedo = m.fuzzWeight * sheenAlbedo(mu, m.fuzzRoughness);
            let below = sqrt(max((1.0 - albedo0) * (1.0 - albedo), 0.0));
            // Its f_r cos(i) is its albedo towards the viewer times its distribution.
            let fuzz = m.fuzzColour * albedo * sheenLtc(s, m.fuzzRoughness) / max(mu0, 1e-4);
            lobes.diffuse = lobes.diffuse * below + fuzz;
            lobes.subsurface *= below;
            lobes.specular *= below;
        }
        return lobes;
    }

    public float3 radianceFactor(BrdfInput s, BrdfParams p) {
        let lobes = lobesOf(s, p);
        // The radiance factor is pi times f_r times cos(i).
        return PI * s.mu0 * (lobes.diffuse + lobes.specular);
    }

    public override float3 subsurface(BrdfInput s, BrdfParams p) {
        // None where no material of the scene scatters beneath its surface.
        if ((BRDF_FEATURES & kOpenPbrSubsurface) == 0u) {
            return float3(0.0);
        }
        return PI * s.mu0 * lobesOf(s, p).subsurface;
    }

    public override SpecularReflection specularReflection(BrdfInput s, BrdfParams p) {
        let lobes = lobesOf(s, p);
        SpecularReflection r;
        r.factor = PI * s.mu0 * lobes.specular;
        r.ior = lobes.ior;
        return r;
    }

    // The dielectric base's transmission where it does not reflect: what its specular lobe at the
    // viewer's angle leaves, tinted, then what the coat and the fuzz let through; refracted at
    // the base's index unless thin-walled.
    public override Transmission transmission(BrdfInput s, BrdfParams p) {
        let m = materialOf(s, p);
        Transmission t;
        t.ior = m.specularIor;
        t.thinWalled = m.thinWalled;
        t.factor = float3(0.0);
        if (m.transmission <= 0.0) {
            return t;
        }
        let mu = saturate(s.mu);
        let roughness = widenedRoughness(m.specularRoughness, s.slopeVariance);
        t.factor = (1.0 - m.metalness) * m.transmission * m.transmissionColour *
                   max(1.0 - dielectricAlbedo(m, roughness, mu), float3(0.0));
        if (m.coatWeight > 0.0) {
            let coatRoughness = widenedRoughness(m.coatRoughness, s.slopeVariance);
            t.factor *= m.coatTransmission * (1.0 - coatAlbedo(m, coatRoughness, mu));
        }
        if (m.fuzzWeight > 0.0) {
            t.factor *= 1.0 - m.fuzzWeight * sheenAlbedo(mu, m.fuzzRoughness);
        }
        return t;
    }

    // The split-sum approximation: the base's specular lobe (the metal's F82 tint on the
    // environment BRDF's grazing bias) and the coat's, the diffuse what they leave; the fuzz,
    // broad, with the diffuse.
    public override EnvironmentResponse environment(BrdfInput s, BrdfParams p) {
        let m = materialOf(s, p);
        let mu = saturate(s.mu);
        // An anisotropic lobe prefiltered as an isotropic one of its alphas' geometric mean,
        // alpha k sqrt(1 - anisotropy), k the stretch along the tangent.
        let across = 1.0 - m.anisotropy;
        let k = sqrt(2.0 / (1.0 + across * across));
        let roughness = widenedRoughness(m.specularRoughness * sqrt(k * sqrt(across)),
                                         s.slopeVariance);
        let ab = environmentBrdf(roughness, mu);
        let f0 = dielectricF0(m);
        // The dielectric's exact Fresnel, the metal's F82 tint and the film's interference at
        // the split sum's cosine, rather than Schlick's.
        float3 dielectricFresnel;
        float3 metalFresnel;
        fresnels(m, environmentCosine(ab), s.wavelengths, dielectricFresnel, metalFresnel);
        let dielectric = dielectricFresnel * (ab.x + ab.y) *
                         energyCompensation(float3(f0), roughness, mu);
        let metal = metalFresnel * (ab.x + ab.y) * energyCompensation(m.base, roughness, mu);
        var r = diffuseEnvironment(float3(0.0));
        r.roughness0 = roughness;
        if (m.anisotropy > 0.0) {
            // Mirrored about a normal bent towards the plane across the rough tangent, so the
            // reflections stretch along it (Filament's bent reflection vector).
            let b = float3(0.0, 1.0, 0.0);
            let bentNormal = cross(cross(b, s.view), b);
            let bend = m.anisotropy * saturate(5.0 * m.specularRoughness);
            if (dot(bentNormal, bentNormal) > 1e-8) {
                r.normal0 = normalize(lerp(float3(0.0, 0.0, 1.0), normalize(bentNormal), bend));
            }
        }
        r.specular0 = lerp(dielectric, metal, m.metalness);
        let base = (1.0 - m.metalness) * (1.0 - m.transmission) *
                   max(1.0 - dielectric, float3(0.0));
        r.subsurface = base * m.subsurface * m.subsurfaceColour;
        r.diffuse = base * (1.0 - m.subsurface) *
                        eonAlbedo(saturate(m.base), m.diffuseRoughness, mu) +
                    r.subsurface;
        if (m.coatWeight > 0.0) {
            let coatRoughness = widenedRoughness(m.coatRoughness, s.slopeVariance);
            let coat = coatAlbedo(m, coatRoughness, mu);
            let through = m.coatTransmission * (1.0 - coat);
            r.diffuse *= through;
            r.subsurface *= through;
            r.specular0 *= through;
            r.specular1 = float3(coat);
            r.roughness1 = coatRoughness;
        }
        if (m.fuzzWeight > 0.0) {
            let albedo = m.fuzzWeight * sheenAlbedo(mu, m.fuzzRoughness);
            r.diffuse = r.diffuse * (1.0 - albedo) + m.fuzzColour * albedo;
            r.subsurface *= 1.0 - albedo;
            r.specular0 *= 1.0 - albedo;
            r.specular1 *= 1.0 - albedo;
        }
        return r;
    }
}
