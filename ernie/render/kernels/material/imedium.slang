// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.spectrum.spectrum;

struct MediumProperties<Spectrum : ISpectrum> : IDifferentiable {
    Spectrum absorption;
    Spectrum scattering;
    Spectrum emission;
    float anisotropy;

    [Differentiable]
    __init(Spectrum absorption, Spectrum scattering, Spectrum emission, float anisotropy) {
        this.absorption = absorption;
        this.scattering = scattering;
        this.emission = emission;
        this.anisotropy = anisotropy;
    }
}

interface IMedium<Spectrum : ISpectrum> : IDifferentiable {
    [Differentiable]
    MediumProperties<Spectrum> Evaluate(float3 position);
}

struct Vacuum<Spectrum : ISpectrum> : IMedium<Spectrum> {
    [Differentiable]
    MediumProperties<Spectrum> Evaluate(float3 position) {
        return MediumProperties<Spectrum>(Spectrum.Constant(0), Spectrum.Constant(0),
                                          Spectrum.Constant(0), 0);
    }
}
