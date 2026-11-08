// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.spectrum.ispectrum;
import ernie.scene.records.render_tables;

using namespace ernie;

static const uint kSpectrumBands = 8;

static Ptr<DetectorRecord> gBandDetector;

void UseBands(Ptr<DetectorRecord> detector) {
    gBandDetector = detector;
}

float4 FromRgbBands(uint row) {
    return gBandDetector->bands.fromRgb[row].Vector();
}

float4 ToRgbBands(uint row) {
    return gBandDetector->bands.toRgb[row].Vector();
}

float BandCenter(uint band) {
    return gBandDetector->bands.first + (float(band) + 0.5f) * gBandDetector->bands.width;
}

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
        return BandSpectrum(rgb.x * no_diff FromRgbBands(0) + rgb.y * no_diff FromRgbBands(2) +
                                rgb.z * no_diff FromRgbBands(4),
                            rgb.x * no_diff FromRgbBands(1) + rgb.y * no_diff FromRgbBands(3) +
                                rgb.z * no_diff FromRgbBands(5));
    }

    [Differentiable]
    static BandSpectrum Blackbody(float kelvin) {
        float4 low;
        float4 high;
        [ForceUnroll]
        for (uint band = 0; band < 4; ++band) {
            low[band] = Planck(no_diff BandCenter(band), kelvin);
            high[band] = Planck(no_diff BandCenter(band + 4), kelvin);
        }
        return BandSpectrum(low, high);
    }

    [Differentiable]
    float3 ToRgb() {
        return float3(dot(low, no_diff ToRgbBands(0)) + dot(high, no_diff ToRgbBands(1)),
                      dot(low, no_diff ToRgbBands(2)) + dot(high, no_diff ToRgbBands(3)),
                      dot(low, no_diff ToRgbBands(4)) + dot(high, no_diff ToRgbBands(5)));
    }

    [Differentiable]
    BandValues Bands() {
        BandValues values;
        values.low = low;
        values.high = high;
        return values;
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
