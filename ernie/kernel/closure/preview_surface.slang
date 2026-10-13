// SPDX-License-Identifier: BSD-3-Clause
import brdf;

// UsdPreviewSurface's reflectance, as glTF 2.0's metallic-roughness BRDF defines it (which the
// preview surface's metallic workflow matches): a Lambert diffuse lobe and a GGX specular lobe with
// height-correlated Smith visibility and Schlick's Fresnel,
//
//   dielectric = (1 - F(f0)) diffuse / pi + F(f0) D V,    f0 = ((ior - 1) / (ior + 1))^2
//   metal      = F(albedo) D V
//   f_r        = lerp(dielectric, metal, metallic)
//
// and in the specular workflow (1 - F(s)) diffuse / pi + F(s) D V with s the specularColor. A
// clearcoat adds a second GGX lobe of f0 = 0.04 weighted by clearcoat, over the base attenuated by
// its Fresnel. D and V take alpha = roughness^2, widened by the unresolved slopes of
// `BrdfInput::slopeVariance` as alpha^2 + 2 variance (Toksvig; Olano and Baker's LEAN mapping).
// The albedo input is the diffuseColor (the base colour). The bands are wavelengths, not RGB, so
// specularColor counts as its mean.
//
// Parameters: 0 roughness, 1 metallic, 2 ior, 3 useSpecularWorkflow (0 or 1), 4-6 specularColor,
// 7 clearcoat, 8 clearcoatRoughness.
//
// Light from all around (`environment`) follows the split-sum approximation. Both scale the base
// specular lobe by 1 + f0 (1 / E - 1), E its directional albedo, for the energy single scattering
// loses on rough surfaces (direct light by the geometric mean of the factors at cos(i) and
// cos(e), which keeps it reciprocal); the clearcoat, smooth and weak, goes without.
//
// References:
// - E. Turquin, "Practical multiple scattering compensation for microfacet models", Industrial
//   Light & Magic technical report, 2019.
// - R. Guy and M. Agopian, "Physically Based Rendering in Filament", 2018, section 4.7.2.
//   https://google.github.io/filament/Filament.html
// - B. Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013 course notes (split sum).
// - B. Karis, "Physically Based Shading on Mobile", 2014, after D. Lazarov, "Getting More
//   Physical in Call of Duty: Black Ops II", SIGGRAPH 2013 course (analytic environment BRDF).
//   https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile

public struct PreviewSurface : IBrdf {
    // The split-sum approximation (Karis 2013): each lobe's weight is its f0 times the
    // environment BRDF's scale plus its bias, and the diffuse lobe keeps what the dielectric's
    // specular does not take.
    public override EnvironmentResponse environment(BrdfInput s, BrdfParams p) {
        let mu = saturate(s.mu);
        let roughness = widenedRoughness(saturate(p.get(0)), s.slopeVariance);
        let ab = environmentBrdf(roughness, mu);
        var r = diffuseEnvironment(float3(0.0));
        r.roughness0 = roughness;
        if (p.get(3) > 0.5) {
            let f0 = (p.get(4) + p.get(5) + p.get(6)) / 3.0;
            let specular = f0 * ab.x + ab.y;
            r.specular0 = specular * energyCompensation(float3(f0), roughness, mu);
            r.diffuse = s.albedo * (1.0 - specular);
        } else {
            let metallic = saturate(p.get(1));
            let ior = max(p.get(2), 1.0);
            let r0 = (ior - 1.0) / (ior + 1.0);
            let dielectric = r0 * r0 * ab.x + ab.y;
            let metal = (s.albedo * ab.x + ab.y) * energyCompensation(s.albedo, roughness, mu);
            r.specular0 = lerp(dielectric * energyCompensation(float3(r0 * r0), roughness, mu),
                               metal, metallic);
            r.diffuse = s.albedo * (1.0 - metallic) * (1.0 - dielectric);
        }
        let clearcoat = saturate(p.get(7));
        if (clearcoat > 0.0) {
            let coatRoughness = widenedRoughness(saturate(p.get(8)), s.slopeVariance);
            let coatAb = environmentBrdf(coatRoughness, mu);
            let coat = (0.04 * coatAb.x + coatAb.y) * clearcoat;
            r.diffuse *= 1.0 - coat;
            r.specular0 *= 1.0 - coat;
            r.specular1 = float3(coat);
            r.roughness1 = coatRoughness;
        }
        return r;
    }

    public float3 radianceFactor(BrdfInput s, BrdfParams p) {
        let lobes = lobesOf(s, p);
        // The radiance factor is pi times f_r times cos(i).
        return PI * s.mu0 * (lobes.diffuse + lobes.specular);
    }

    // The specular lobes (the base's and the clearcoat's), which Fresnel's laws polarize, with
    // the index of refraction of the base's facets: its ior; a metal's, approximately, the real
    // index whose normal reflectance is its albedo's mean (the complex index's phases ignored);
    // in the specular workflow, the one whose normal reflectance is the specularColor's mean.
    public override SpecularReflection specularReflection(BrdfInput s, BrdfParams p) {
        let lobes = lobesOf(s, p);
        SpecularReflection r;
        r.factor = PI * s.mu0 * lobes.specular;
        r.ior = lobes.ior;
        return r;
    }

    // f_r's diffuse and specular parts, and the specular facets' index of refraction.
    struct Lobes {
        float3 diffuse;
        float3 specular;
        float ior;
    }

    static Lobes lobesOf(BrdfInput s, BrdfParams p) {
        let mu0 = s.mu0;
        let mu = s.mu;
        // The half vector from cos(g): |l + v| = sqrt(2 + 2 cos g).
        let sum = sqrt(max(2.0 + 2.0 * s.cosG, 1e-8));
        let nDotH = saturate((mu0 + mu) / sum);
        let vDotH = saturate(0.5 * sum);
        let unresolved = 2.0 * max(s.slopeVariance, 0.0);

        let roughness = saturate(p.get(0));
        let alpha = roughness * roughness;
        let a2 = clamp(alpha * alpha + unresolved, 1e-6, 1.0);
        let specular = ggx(nDotH, a2) * smithVisibility(mu0, mu, a2);
        // The energy multiple scattering adds back, per f0 (Turquin 2019).
        let lobeRoughness = sqrt(sqrt(a2));

        let metallic = saturate(p.get(1));
        let ior = max(p.get(2), 1.0);
        let r0 = (ior - 1.0) / (ior + 1.0);
        let diffuse = s.albedo / PI;
        Lobes lobes;
        if (p.get(3) > 0.5) {
            let f0 = (p.get(4) + p.get(5) + p.get(6)) / 3.0;
            let fresnel = schlick(float3(f0), vDotH);
            lobes.diffuse = (1.0 - fresnel) * diffuse;
            lobes.specular = fresnel * specular *
                             reciprocalCompensation(float3(f0), lobeRoughness, mu0, mu);
            lobes.ior = iorOfReflectance(f0);
        } else {
            let fDielectric = schlick(float3(r0 * r0), vDotH);
            let metal = schlick(s.albedo, vDotH) * specular *
                        reciprocalCompensation(s.albedo, lobeRoughness, mu0, mu);
            lobes.diffuse = (1.0 - metallic) * (1.0 - fDielectric) * diffuse;
            lobes.specular =
                lerp(fDielectric * specular *
                         reciprocalCompensation(float3(r0 * r0), lobeRoughness, mu0, mu),
                     metal, metallic);
            let albedoMean = (s.albedo.x + s.albedo.y + s.albedo.z) / 3.0;
            lobes.ior = lerp(ior, iorOfReflectance(albedoMean), metallic);
        }

        let clearcoat = saturate(p.get(7));
        if (clearcoat > 0.0) {
            let coatRoughness = saturate(p.get(8));
            let coatAlpha = coatRoughness * coatRoughness;
            let coatA2 = clamp(coatAlpha * coatAlpha + unresolved, 1e-6, 1.0);
            let coatFresnel = schlick(float3(0.04), vDotH).x * clearcoat;
            let coat = ggx(nDotH, coatA2) * smithVisibility(mu0, mu, coatA2);
            lobes.diffuse *= 1.0 - coatFresnel;
            lobes.specular = lobes.specular * (1.0 - coatFresnel) + coatFresnel * coat;
        }
        return lobes;
    }

    // The real index of refraction whose reflectance at normal incidence is `f0`:
    // f0 = ((n - 1) / (n + 1))^2.
    static float iorOfReflectance(float f0) {
        let root = sqrt(clamp(f0, 0.0, 0.98));
        return (1.0 + root) / (1.0 - root);
    }
}
