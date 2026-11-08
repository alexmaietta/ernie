// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.spectrum.spectrum;

static const uint kSpectralConstant = 0;
static const uint kSpectralRgb = 1;
static const uint kSpectralLibrary = 2;

struct SpectralValue : IDifferentiable {
    no_diff uint kind;
    float scale;
    float3 data;

    [Differentiable]
    static SpectralValue Constant(float value) {
        SpectralValue spectral;
        spectral.kind = kSpectralConstant;
        spectral.scale = value;
        spectral.data = float3(0);
        return spectral;
    }

    [Differentiable]
    static SpectralValue Rgb(float3 rgb) {
        SpectralValue spectral;
        spectral.kind = kSpectralRgb;
        spectral.scale = 1;
        spectral.data = rgb;
        return spectral;
    }

    static SpectralValue Library(uint spectrum) {
        SpectralValue spectral;
        spectral.kind = kSpectralLibrary;
        spectral.scale = 1;
        spectral.data = float3(asfloat(spectrum), 0, 0);
        return spectral;
    }

    [Differentiable]
    SpectralValue Scaled(float factor) {
        SpectralValue spectral = this;
        spectral.scale = scale * factor;
        return spectral;
    }

    [Differentiable]
    Spectrum Evaluate<Spectrum : ISpectrum>() {
        if (kind == kSpectralRgb) {
            return Spectrum.FromRgb(data).Scale(scale);
        }
        if (kind == kSpectralLibrary) {
            return Spectrum.FromLibrary(no_diff asuint(detach(data.x))).Scale(scale);
        }
        return Spectrum.Constant(scale);
    }
}
