// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.spectrum.ispectrum;

static const uint kSpectrumBands = 8;
static const float kSpectrumFirstNm = 380;
static const float kSpectrumBandNm = 45;

static const float4x3 kLowBandRgb = {
    0.05f, 0.00f, 0.60f, 0.00f, 0.05f, 1.00f, 0.00f, 0.45f, 0.70f, 0.05f, 1.00f, 0.15f,
};
static const float4x3 kHighBandRgb = {
    0.45f, 0.85f, 0.00f, 1.00f, 0.35f, 0.00f, 0.80f, 0.05f, 0.00f, 0.25f, 0.00f, 0.00f,
};

struct BandSpectrum : ISpectrum {
    float4 low;
    float4 high;

    [Differentiable]
    __init(float4 low, float4 high) {
        this.low = low;
        this.high = high;
    }

    [Differentiable]
    static BandSpectrum Constant(float value) {
        return BandSpectrum(float4(value), float4(value));
    }

    [Differentiable]
    static BandSpectrum FromRgb(float3 rgb) {
        return BandSpectrum(mul(kLowBandRgb, rgb) / mul(kLowBandRgb, float3(1)),
                            mul(kHighBandRgb, rgb) / mul(kHighBandRgb, float3(1)));
    }

    [Differentiable]
    static BandSpectrum Blackbody(float kelvin) {
        float4 low;
        float4 high;
        [ForceUnroll]
        for (uint band = 0; band < 4; ++band) {
            low[band] = Planck(kSpectrumFirstNm + (band + 0.5f) * kSpectrumBandNm, kelvin);
            high[band] = Planck(kSpectrumFirstNm + (band + 4.5f) * kSpectrumBandNm, kelvin);
        }
        return BandSpectrum(low, high);
    }

    [Differentiable]
    float3 ToRgb() {
        return (mul(low, kLowBandRgb) + mul(high, kHighBandRgb)) /
               (mul(float4(1), kLowBandRgb) + mul(float4(1), kHighBandRgb));
    }

    [Differentiable]
    BandSpectrum Add(BandSpectrum other) {
        return BandSpectrum(low + other.low, high + other.high);
    }

    [Differentiable]
    BandSpectrum Subtract(BandSpectrum other) {
        return BandSpectrum(low - other.low, high - other.high);
    }

    [Differentiable]
    BandSpectrum Multiply(BandSpectrum other) {
        return BandSpectrum(low * other.low, high * other.high);
    }

    [Differentiable]
    BandSpectrum Scale(float factor) {
        return BandSpectrum(low * factor, high * factor);
    }

    [Differentiable]
    float Average() {
        return (dot(low, float4(1)) + dot(high, float4(1))) / kSpectrumBands;
    }

    [Differentiable]
    float Luminance() {
        return dot(ToRgb(), kLuminanceWeights);
    }

    [Differentiable]
    float MaxValue() {
        const float4 most = max(low, high);
        return max(max(most.x, most.y), max(most.z, most.w));
    }
}
