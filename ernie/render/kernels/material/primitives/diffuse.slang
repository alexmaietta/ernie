// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
__exported import ernie.render.kernels.spectrum.spectrum;

float3 CosineHemisphere(float2 u) {
    const float r = sqrt(u.x);
    const float phi = 2 * kPi * u.y;
    return float3(r * cos(phi), r * sin(phi), sqrt(max(0.0f, 1 - u.x)));
}

struct DiffuseReflectionBsdf<Spectrum : ISpectrum> : IDifferentiable {
    Spectrum tint;

    [Differentiable]
    __init(Spectrum tint) {
        this.tint = tint;
    }

    [Differentiable]
    Spectrum Estimate() {
        return tint;
    }

    [Differentiable]
    Spectrum Evaluate() {
        return tint.Scale(1 / kPi);
    }

    float Pdf(float3 incoming) {
        return incoming.z / kPi;
    }

    float3 Direction(float2 u) {
        return CosineHemisphere(u);
    }
}
