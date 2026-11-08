// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.material.iprimitive;

static const uint kFirstPrimitiveLobe = 3;
static const bool kPrimitives = false;

struct LobeParams {
    __init(float4 values) {}
}

bool PrimitiveGlossy(uint kind) {
    return false;
}

[Differentiable]
Spectrum PrimitiveEstimate<Spectrum : ISpectrum>(uint kind, Spectrum tint, float alpha,
                                                 no_diff LobeParams params,
                                                 no_diff float3 outgoing) {
    return Spectrum.Constant(0);
}

[Differentiable]
Spectrum PrimitiveEvaluate<Spectrum : ISpectrum>(uint kind, Spectrum tint, float alpha,
                                                 no_diff LobeParams params, no_diff float3 outgoing,
                                                 no_diff float3 incoming) {
    return Spectrum.Constant(0);
}

float PrimitivePdf(uint kind, float alpha, LobeParams params, float3 outgoing, float3 incoming) {
    return 0;
}

float3 PrimitiveDirection(uint kind, float alpha, LobeParams params, float3 outgoing, float2 u) {
    return float3(0, 0, 1);
}
