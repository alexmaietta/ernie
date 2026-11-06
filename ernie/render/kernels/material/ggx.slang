// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
import ernie.render.kernels.spectrum.spectrum;

[Differentiable]
float GgxD(float3 h, float alpha) {
    const float a2 = alpha * alpha;
    const float d = h.z * h.z * (a2 - 1) + 1;
    return a2 / (kPi * d * d);
}

[Differentiable]
float GgxLambda(float3 w, float alpha) {
    const float c2 = w.z * w.z;
    const float tan2 = max(0.0f, 1 - c2) / max(c2, 1e-12f);
    return (sqrt(1 + alpha * alpha * tan2) - 1) / 2;
}

[Differentiable]
float GgxG1(float3 w, float alpha) {
    return 1 / (1 + GgxLambda(w, alpha));
}

[Differentiable]
float GgxG2(float3 outgoing, float3 incoming, float alpha) {
    return 1 / (1 + GgxLambda(outgoing, alpha) + GgxLambda(incoming, alpha));
}

[Differentiable]
float SchlickWeight(float cosine) {
    const float m = clamp(1 - cosine, 0.0f, 1.0f);
    const float m2 = m * m;
    return m2 * m2 * m;
}

[Differentiable]
float Schlick(float f0, float cosine) {
    return lerp(f0, 1, SchlickWeight(cosine));
}

[Differentiable]
Spectrum Schlick<Spectrum : ISpectrum>(Spectrum f0, float cosine) {
    const float weight = SchlickWeight(cosine);
    return f0 * (1 - weight) + Spectrum.Constant(weight);
}

[Differentiable]
float GgxReflection(float3 outgoing, float3 incoming, float3 h, float alpha) {
    return GgxD(h, alpha) * GgxG2(outgoing, incoming, alpha) / (4 * outgoing.z * incoming.z);
}

float GgxReflectionPdf(float3 outgoing, float3 h, float alpha) {
    return GgxG1(outgoing, alpha) * GgxD(h, alpha) / (4 * outgoing.z);
}

float3 SampleGgxVisibleNormal(float3 outgoing, float alpha, float2 u) {
    const float3 v = normalize(float3(alpha * outgoing.x, alpha * outgoing.y, outgoing.z));
    const float length2 = v.x * v.x + v.y * v.y;
    const float3 t1 = length2 > 0 ? float3(-v.y, v.x, 0) * rsqrt(length2) : float3(1, 0, 0);
    const float3 t2 = cross(v, t1);
    const float r = sqrt(u.x);
    const float phi = 2 * kPi * u.y;
    const float p1 = r * cos(phi);
    const float s = 0.5f * (1 + v.z);
    const float p2 = (1 - s) * sqrt(max(0.0f, 1 - p1 * p1)) + s * r * sin(phi);
    const float3 n = p1 * t1 + p2 * t2 + sqrt(max(0.0f, 1 - p1 * p1 - p2 * p2)) * v;
    return normalize(float3(alpha * n.x, alpha * n.y, max(1e-6f, n.z)));
}
