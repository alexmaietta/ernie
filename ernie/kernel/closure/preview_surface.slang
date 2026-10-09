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
// Light from all around (`environment`) follows the split-sum approximation.
//
// References:
// - B. Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013 course notes (split sum).
// - B. Karis, "Physically Based Shading on Mobile", 2014, after D. Lazarov, "Getting More
//   Physical in Call of Duty: Black Ops II", SIGGRAPH 2013 course (analytic environment BRDF).
//   https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile

// GGX normal distribution D(h) for alpha^2 `a2` (Trowbridge and Reitz 1975; Walter et al. 2007).
float ggx(float nDotH, float a2) {
    let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Height-correlated Smith visibility G / (4 mu0 mu) for GGX (Heitz 2014).
float smithVisibility(float mu0, float mu, float a2) {
    let lambdaV = mu0 * sqrt(mu * mu * (1.0 - a2) + a2);
    let lambdaL = mu * sqrt(mu0 * mu0 * (1.0 - a2) + a2);
    return 0.5 / max(lambdaV + lambdaL, 1e-8);
}

float3 schlick(float3 f0, float vDotH) {
    let m = pow(saturate(1.0 - vDotH), 5.0);
    return f0 + (1.0 - f0) * m;
}

// The split-sum environment BRDF's scale and bias on f0 for perceptual roughness `roughness` at
// cos(e) `mu`, from Lazarov's analytic fit (Karis 2014), so no lookup table is needed.
float2 environmentBrdf(float roughness, float mu) {
    let c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    let c1 = float4(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * mu)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}

// The perceptual roughness of a lobe of `roughness` widened by unresolved slopes of `variance`.
float widenedRoughness(float roughness, float variance) {
    let alpha = roughness * roughness;
    return sqrt(sqrt(min(alpha * alpha + 2.0 * max(variance, 0.0), 1.0)));
}

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
            r.specular0 = float3(specular);
            r.diffuse = s.albedo * (1.0 - specular);
        } else {
            let metallic = saturate(p.get(1));
            let ior = max(p.get(2), 1.0);
            let r0 = (ior - 1.0) / (ior + 1.0);
            let dielectric = r0 * r0 * ab.x + ab.y;
            r.specular0 = lerp(float3(dielectric), s.albedo * ab.x + ab.y, metallic);
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

        let metallic = saturate(p.get(1));
        let ior = max(p.get(2), 1.0);
        let r0 = (ior - 1.0) / (ior + 1.0);
        let diffuse = s.albedo / PI;
        float3 base;
        if (p.get(3) > 0.5) {
            let f0 = (p.get(4) + p.get(5) + p.get(6)) / 3.0;
            let fresnel = schlick(float3(f0), vDotH);
            base = (1.0 - fresnel) * diffuse + fresnel * specular;
        } else {
            let fDielectric = schlick(float3(r0 * r0), vDotH);
            let dielectric = (1.0 - fDielectric) * diffuse + fDielectric * specular;
            let metal = schlick(s.albedo, vDotH) * specular;
            base = lerp(dielectric, metal, metallic);
        }

        let clearcoat = saturate(p.get(7));
        if (clearcoat > 0.0) {
            let coatRoughness = saturate(p.get(8));
            let coatAlpha = coatRoughness * coatRoughness;
            let coatA2 = clamp(coatAlpha * coatAlpha + unresolved, 1e-6, 1.0);
            let coatFresnel = schlick(float3(0.04), vDotH).x * clearcoat;
            let coat = ggx(nDotH, coatA2) * smithVisibility(mu0, mu, coatA2);
            base = base * (1.0 - coatFresnel) + coatFresnel * coat;
        }
        // The radiance factor is pi times f_r times cos(i).
        return PI * mu0 * base;
    }
}
