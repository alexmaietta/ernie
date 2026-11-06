// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
__exported import ernie.render.kernels.material.ibsdf;

float3 CosineHemisphere(float2 u) {
    const float r = sqrt(u.x);
    const float phi = 2 * kPi * u.y;
    return float3(r * cos(phi), r * sin(phi), sqrt(max(0.0f, 1 - u.x)));
}

struct Lambert<Spectrum : ISpectrum> : IBsdf<Spectrum> {
    Spectrum albedo;

    [Differentiable]
    __init(Spectrum albedo) {
        this.albedo = albedo;
    }

    [Differentiable]
    BsdfValue<Spectrum> Evaluate(float3 outgoing, float3 incoming) {
        if (outgoing.z <= 0 || incoming.z <= 0) {
            return BsdfValue<Spectrum>.Zero();
        }
        return BsdfValue<Spectrum>(albedo * (1 / kPi), Spectrum.Constant(0));
    }

    [Differentiable]
    BsdfSample<Spectrum> Sample(float3 outgoing, no_diff float3 u) {
        BsdfSample<Spectrum> sample;
        sample.direction = CosineHemisphere(u.xy);
        sample.pdf = outgoing.z > 0 ? sample.direction.z / kPi : 0;
        sample.weight = sample.pdf > 0 ? BsdfValue<Spectrum>(albedo, Spectrum.Constant(0))
                                       : BsdfValue<Spectrum>.Zero();
        sample.delta = false;
        return sample;
    }

    float Pdf(float3 outgoing, float3 incoming) {
        return outgoing.z > 0 && incoming.z > 0 ? incoming.z / kPi : 0;
    }
}
