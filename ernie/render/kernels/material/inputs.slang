// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.material.spectral;
__exported import ernie.render.kernels.material.radio;

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
    SpectralValue emissivity;
    bool emissivityAuthored;
    ElectricalValue electrical;
    bool electricalAuthored;
    float radarRoughness;
    float radarCorrelation;

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
        emissivity = SpectralValue.Constant(1);
        emissivityAuthored = false;
        electrical = ElectricalValue.Constant(5, 0.01f);
        electricalAuthored = false;
        radarRoughness = 0.001f;
        radarCorrelation = 0.01f;
    }
}
