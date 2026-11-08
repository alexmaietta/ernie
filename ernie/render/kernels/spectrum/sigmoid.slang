// SPDX-License-Identifier: BSD-3-Clause

import ernie.scene.records.render_tables;

using namespace ernie;

static const float kSigmoidFirstNm = 360;
static const float kSigmoidLastNm = 830;
static const float kVisibleFirstNm = 380;
static const float kVisibleLastNm = 780;

struct Sigmoid {
    float3 coefficients;
    float scale;

    float Value(float nanometers) {
        const float t = (nanometers - kSigmoidFirstNm) / (kSigmoidLastNm - kSigmoidFirstNm);
        const float x = (coefficients.x * t + coefficients.y) * t + coefficients.z;
        return scale * (0.5f + x / (2 * sqrt(1 + x * x)));
    }
}

float3 SigmoidEntry(SpectraRecord spectra, uint l, uint z, uint y, uint x) {
    const uint n = spectra.sigmoidResolution;
    const Ptr<float> at = spectra.sigmoids + (((l * n + z) * n + y) * n + x) * 3;
    return float3(at[0], at[1], at[2]);
}

Sigmoid FitSigmoid(SpectraRecord spectra, float3 rgb) {
    Sigmoid sigmoid;
    sigmoid.coefficients = float3(0, 0, -1e4f);
    sigmoid.scale = 1;
    rgb = max(rgb, float3(0));
    const float largest = max(rgb.x, max(rgb.y, rgb.z));
    const uint n = spectra.sigmoidResolution;
    if (largest <= 0 || n < 2) {
        return sigmoid;
    }
    sigmoid.scale = max(largest, 1.0f);
    rgb /= sigmoid.scale;
    const uint l = rgb.x >= rgb.y && rgb.x >= rgb.z ? 0 : rgb.y >= rgb.z ? 1 : 2;
    const float z = rgb[l];
    const float x = rgb[(l + 1) % 3] / z * float(n - 1);
    const float y = rgb[(l + 2) % 3] / z * float(n - 1);
    uint zi = 0;
    for (uint k = 1; k + 1 < n; ++k) {
        zi = spectra.sigmoidScales[k] <= z ? k : zi;
    }
    const float low = spectra.sigmoidScales[zi];
    const float dz = clamp((z - low) / max(spectra.sigmoidScales[zi + 1] - low, 1e-6f), 0.0f, 1.0f);
    const uint xi = min(uint(x), n - 2);
    const uint yi = min(uint(y), n - 2);
    const float dx = x - float(xi);
    const float dy = y - float(yi);
    float3 c = float3(0);
    for (uint corner = 0; corner < 8; ++corner) {
        const uint ox = corner & 1;
        const uint oy = (corner >> 1) & 1;
        const uint oz = corner >> 2;
        const float w = (ox != 0 ? dx : 1 - dx) * (oy != 0 ? dy : 1 - dy) * (oz != 0 ? dz : 1 - dz);
        c += w * SigmoidEntry(spectra, l, zi + oz, yi + oy, xi + ox);
    }
    sigmoid.coefficients = c;
    return sigmoid;
}

bool Visible(float first, float last) {
    return first >= kVisibleFirstNm - 1e-3f && last <= kVisibleLastNm + 1e-3f;
}
