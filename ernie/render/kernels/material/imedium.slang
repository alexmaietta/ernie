// SPDX-License-Identifier: BSD-3-Clause

struct MediumProperties : IDifferentiable {
    float3 absorption;
    float3 scattering;
    float3 emission;
    float anisotropy;

    [Differentiable]
    __init(float3 absorption, float3 scattering, float3 emission, float anisotropy) {
        this.absorption = absorption;
        this.scattering = scattering;
        this.emission = emission;
        this.anisotropy = anisotropy;
    }
}

interface IMedium : IDifferentiable {
    [Differentiable]
    MediumProperties Evaluate(float3 position);
}

struct Vacuum : IMedium {
    [Differentiable]
    MediumProperties Evaluate(float3 position) {
        return MediumProperties(float3(0), float3(0), float3(0), 0);
    }
}
