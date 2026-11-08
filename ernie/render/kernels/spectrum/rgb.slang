// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.spectrum.band;

struct RgbSpectrum : ISpectrum {
    float3 values;

    [Differentiable]
    __init(float3 values) {
        this.values = values;
    }

    [Differentiable]
    static RgbSpectrum Constant(float value) {
        return RgbSpectrum(float3(value));
    }

    [Differentiable]
    static RgbSpectrum FromRgb(float3 rgb) {
        return RgbSpectrum(rgb);
    }

    [Differentiable]
    static RgbSpectrum FromLibrary(no_diff uint spectrum) {
        if (!KnownSpectrum(spectrum)) {
            return RgbSpectrum(float3(0));
        }
        return RgbSpectrum(gSpectra->spectra.rgb[spectrum].Vector());
    }

    [Differentiable]
    static RgbSpectrum Blackbody(float kelvin) {
        return RgbSpectrum(
            float3(Planck(610, kelvin), Planck(550, kelvin), Planck(465, kelvin)));
    }

    [Differentiable]
    float3 ToRgb() {
        return values;
    }

    [Differentiable]
    BandValues Bands() {
        return BandSpectrum.FromRgb(values).Bands();
    }

    [Differentiable]
    RgbSpectrum Add(RgbSpectrum other) {
        return RgbSpectrum(values + other.values);
    }

    [Differentiable]
    RgbSpectrum Subtract(RgbSpectrum other) {
        return RgbSpectrum(values - other.values);
    }

    [Differentiable]
    RgbSpectrum Multiply(RgbSpectrum other) {
        return RgbSpectrum(values * other.values);
    }

    [Differentiable]
    RgbSpectrum Scale(float factor) {
        return RgbSpectrum(values * factor);
    }

    [Differentiable]
    float Average() {
        return (values.x + values.y + values.z) / 3;
    }

    [Differentiable]
    float Luminance() {
        return dot(values, kLuminanceWeights);
    }

    [Differentiable]
    float MaxValue() {
        return max(values.x, max(values.y, values.z));
    }
}
