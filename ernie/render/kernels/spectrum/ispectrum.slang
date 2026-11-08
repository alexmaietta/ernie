// SPDX-License-Identifier: BSD-3-Clause

static const float kPlanckFirst = 1.191042972e20f;
static const float kPlanckSecond = 1.438776877e7f;
static const float3 kLuminanceWeights = float3(0.2126f, 0.7152f, 0.0722f);

[Differentiable]
float Planck(no_diff float nanometers, float kelvin) {
    const float n2 = nanometers * nanometers;
    return kPlanckFirst / (n2 * n2 * nanometers) /
           (exp(kPlanckSecond / (nanometers * max(kelvin, 1e-6f))) - 1);
}

struct BandValues : IDifferentiable {
    float4 low;
    float4 high;
}

interface ISpectrum : IDifferentiable {
    [Differentiable]
    static This Constant(float value);

    [Differentiable]
    static This FromRgb(float3 rgb);

    [Differentiable]
    static This Blackbody(float kelvin);

    [Differentiable]
    float3 ToRgb();

    [Differentiable]
    BandValues Bands();

    [Differentiable]
    This Add(This other);

    [Differentiable]
    This Subtract(This other);

    [Differentiable]
    This Multiply(This other);

    [Differentiable]
    This Scale(float factor);

    [Differentiable]
    float Average();

    [Differentiable]
    float Luminance();

    [Differentiable]
    float MaxValue();
}

__generic<Spectrum : ISpectrum>
[Differentiable]
Spectrum operator +(Spectrum a, Spectrum b) {
    return a.Add(b);
}

__generic<Spectrum : ISpectrum>
[Differentiable]
Spectrum operator -(Spectrum a, Spectrum b) {
    return a.Subtract(b);
}

__generic<Spectrum : ISpectrum>
[Differentiable]
Spectrum operator *(Spectrum a, Spectrum b) {
    return a.Multiply(b);
}

__generic<Spectrum : ISpectrum>
[Differentiable]
Spectrum operator *(Spectrum a, float factor) {
    return a.Scale(factor);
}

__generic<Spectrum : ISpectrum>
[Differentiable]
Spectrum operator *(float factor, Spectrum a) {
    return a.Scale(factor);
}
