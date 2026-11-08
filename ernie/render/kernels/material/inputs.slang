// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.material.spectral;

struct SurfaceInputs : IDifferentiable {
    SpectralValue baseColor;
    float metallic;
    float roughness;
    float specularWeight;
    SpectralValue specularColor;
    float ior;
    float coatWeight;
    float coatRoughness;
    SpectralValue emission;
    float opacity;
    float retroreflection;
    float radarBackscatter;

    [Differentiable]
    __init() {
        baseColor = SpectralValue.Rgb(float3(0.8f));
        metallic = 0;
        roughness = 0.3f;
        specularWeight = 1;
        specularColor = SpectralValue.Constant(1);
        ior = 1.5f;
        coatWeight = 0;
        coatRoughness = 0;
        emission = SpectralValue.Constant(0);
        opacity = 1;
        retroreflection = 0;
        radarBackscatter = 1;
    }
}
