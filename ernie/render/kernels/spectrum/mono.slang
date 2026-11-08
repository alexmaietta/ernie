// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.spectrum.band;
import ernie.render.kernels.spectrum.sigmoid;
import ernie.scene.records.render_tables;

using namespace ernie;

static float gMonoNanometers;

void UseWavelength(float nanometers) {
    gMonoNanometers = nanometers;
}

float LibraryValue(uint spectrum, float nanometers) {
    if (!KnownSpectrum(spectrum)) {
        return 0;
    }
    const Uint2 range = gSpectra->spectra.ranges[spectrum];
    if (range.y == 0) {
        return 0;
    }
    const Ptr<Float2> samples = gSpectra->spectra.samples + range.x;
    if (nanometers <= samples[0].x) {
        return samples[0].y;
    }
    for (uint i = 1; i < range.y; ++i) {
        if (nanometers <= samples[i].x) {
            const float t =
                (nanometers - samples[i - 1].x) / max(samples[i].x - samples[i - 1].x, 1e-6f);
            return lerp(samples[i - 1].y, samples[i].y, t);
        }
    }
    return samples[range.y - 1].y;
}

struct MonoSpectrum : ISpectrum {
    float value;

    [Differentiable] __init(float value) { this.value = value; }

        [Differentiable] static MonoSpectrum Constant(float value) {
        return MonoSpectrum(value);
    }

    [Differentiable] static MonoSpectrum FromRgb(float3 rgb) {
        return MonoSpectrum(
            no_diff FitSigmoid(gSpectra->spectra, detach(rgb)).Value(gMonoNanometers));
    }

        [Differentiable] static MonoSpectrum FromLibrary(no_diff uint spectrum) {
        return MonoSpectrum(no_diff LibraryValue(spectrum, gMonoNanometers));
    }

    [Differentiable] static MonoSpectrum Blackbody(float kelvin) {
        return MonoSpectrum(Planck(gMonoNanometers, kelvin));
    }

        [Differentiable] float3 ToRgb() {
        return float3(value);
    }

    [Differentiable] BandValues Bands() {
        BandValues values;
        values.low = float4(value);
        values.high = float4(value);
        return values;
    }

        [Differentiable] MonoSpectrum Add(MonoSpectrum other) {
        return MonoSpectrum(value + other.value);
    }

    [Differentiable] MonoSpectrum Subtract(MonoSpectrum other) {
        return MonoSpectrum(value - other.value);
    }

        [Differentiable] MonoSpectrum Multiply(MonoSpectrum other) {
        return MonoSpectrum(value * other.value);
    }

    [Differentiable] MonoSpectrum Scale(float factor) { return MonoSpectrum(value * factor); }

        [Differentiable] float Average() {
        return value;
    }

    [Differentiable] float Luminance() { return value; }

        [Differentiable] float MaxValue() {
        return value;
    }
}
