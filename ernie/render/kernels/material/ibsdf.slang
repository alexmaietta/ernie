// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.spectrum.spectrum;

struct BsdfValue<Spectrum : ISpectrum> : IDifferentiable {
    Spectrum diffuse;
    Spectrum specular;

    [Differentiable]
    __init(Spectrum diffuse, Spectrum specular) {
        this.diffuse = diffuse;
        this.specular = specular;
    }

    [Differentiable]
    static BsdfValue<Spectrum> Zero() {
        return BsdfValue<Spectrum>(Spectrum.Constant(0), Spectrum.Constant(0));
    }

    [Differentiable]
    Spectrum Sum() {
        return diffuse + specular;
    }

    [Differentiable]
    Spectrum Scaled(float diffuseScale, float specularScale) {
        return diffuse * diffuseScale + specular * specularScale;
    }
}

struct BsdfSample<Spectrum : ISpectrum> : IDifferentiable {
    no_diff float3 direction;
    BsdfValue<Spectrum> weight;
    no_diff float pdf;
    no_diff bool delta;
}

interface IBsdf<Spectrum : ISpectrum> : IDifferentiable {
    [Differentiable]
    BsdfValue<Spectrum> Evaluate(float3 outgoing, float3 incoming);

    [Differentiable]
    BsdfSample<Spectrum> Sample(float3 outgoing, no_diff float3 u);

    float Pdf(float3 outgoing, float3 incoming);
}
