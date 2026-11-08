// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.material.ibsdf;
import ernie.render.kernels.material.primitives.diffuse;
import ernie.render.kernels.material.primitives.ggx;
__exported import ernie.render.kernels.material.registered;

static const uint kClosureLobes = 4;

static const uint kLobeNone = 0;
static const uint kLobeDiffuseReflection = 1;
static const uint kLobeMicrofacetGgx = 2;

struct Lobe<Spectrum : ISpectrum> : IDifferentiable {
    no_diff uint kind;
    float weight;
    Spectrum tint;
    float alpha;
    no_diff LobeParams params;

    [Differentiable]
    static Lobe<Spectrum> None() {
        Lobe<Spectrum> lobe;
        lobe.kind = kLobeNone;
        lobe.weight = 0;
        lobe.tint = Spectrum.Constant(0);
        lobe.alpha = 1;
        lobe.params = LobeParams(float4(0));
        return lobe;
    }

    [Differentiable]
    static Lobe<Spectrum> DiffuseReflection(Spectrum tint, float weight) {
        Lobe<Spectrum> lobe;
        lobe.kind = kLobeDiffuseReflection;
        lobe.weight = weight;
        lobe.tint = tint;
        lobe.alpha = 1;
        lobe.params = LobeParams(float4(0));
        return lobe;
    }

    [Differentiable]
    static Lobe<Spectrum> MicrofacetGgx(Spectrum f0, float alpha, float weight) {
        Lobe<Spectrum> lobe;
        lobe.kind = kLobeMicrofacetGgx;
        lobe.weight = weight;
        lobe.tint = f0;
        lobe.alpha = alpha;
        lobe.params = LobeParams(float4(0));
        return lobe;
    }

    [Differentiable]
    static Lobe<Spectrum> Primitive(no_diff uint kind, Spectrum tint, float alpha,
                                    no_diff float4 params, float weight) {
        Lobe<Spectrum> lobe;
        lobe.kind = kind;
        lobe.weight = weight;
        lobe.tint = tint;
        lobe.alpha = alpha;
        lobe.params = LobeParams(params);
        return lobe;
    }

    bool Registered() {
        return kPrimitives && kind >= kFirstPrimitiveLobe;
    }

    bool Active() {
        return kind != kLobeNone && detach(weight) > 0;
    }

    bool Specular() {
        return kind == kLobeMicrofacetGgx || (Registered() && PrimitiveGlossy(kind));
    }

    [Differentiable]
    Spectrum Estimate(float3 outgoing) {
        if (Registered()) {
            return PrimitiveEstimate<Spectrum>(kind, tint, alpha, params, outgoing).Scale(weight);
        }
        if (kind == kLobeMicrofacetGgx) {
            return MicrofacetGgxBsdf<Spectrum>(tint, alpha).Estimate(outgoing).Scale(weight);
        }
        return DiffuseReflectionBsdf<Spectrum>(tint).Estimate().Scale(weight);
    }

    [Differentiable]
    Spectrum Evaluate(float3 outgoing, float3 incoming, float3 h) {
        if (Registered()) {
            return PrimitiveEvaluate<Spectrum>(kind, tint, alpha, params, outgoing, incoming)
                .Scale(weight);
        }
        if (kind == kLobeMicrofacetGgx) {
            return MicrofacetGgxBsdf<Spectrum>(tint, alpha)
                .Evaluate(outgoing, incoming, h)
                .Scale(weight);
        }
        return DiffuseReflectionBsdf<Spectrum>(tint).Evaluate().Scale(weight);
    }

    float Pdf(float3 outgoing, float3 incoming, float3 h) {
        if (Registered()) {
            return PrimitivePdf(kind, alpha, params, outgoing, incoming);
        }
        if (kind == kLobeMicrofacetGgx) {
            return MicrofacetGgxBsdf<Spectrum>(tint, alpha).Pdf(outgoing, h);
        }
        return DiffuseReflectionBsdf<Spectrum>(tint).Pdf(incoming);
    }

    float3 Direction(float3 outgoing, float2 u) {
        if (Registered()) {
            return PrimitiveDirection(kind, alpha, params, outgoing, u);
        }
        if (kind == kLobeMicrofacetGgx) {
            return MicrofacetGgxBsdf<Spectrum>(tint, alpha).Direction(outgoing, u);
        }
        return DiffuseReflectionBsdf<Spectrum>(tint).Direction(u);
    }
}

struct OpticalClosure<Spectrum : ISpectrum> : IBsdf<Spectrum> {
    Lobe<Spectrum> lobes[kClosureLobes];
    no_diff uint count;
    Spectrum albedo;
    Spectrum emission;
    float opacity;
    no_diff float retroreflection;
    no_diff float radarBackscatter;
    Spectrum emissivity;
    no_diff bool emissivityAuthored;

    [Differentiable]
    __init() {
        [ForceUnroll]
        for (uint i = 0; i < kClosureLobes; ++i) {
            lobes[i] = Lobe<Spectrum>.None();
        }
        count = 0;
        albedo = Spectrum.Constant(0);
        emission = Spectrum.Constant(0);
        opacity = 1;
        retroreflection = 0;
        radarBackscatter = 1;
        emissivity = Spectrum.Constant(1);
        emissivityAuthored = false;
    }

    [Differentiable]
    static OpticalClosure<Spectrum> Diffuse(Spectrum albedo) {
        OpticalClosure<Spectrum> closure = OpticalClosure<Spectrum>();
        closure.AddLobe(Lobe<Spectrum>.DiffuseReflection(albedo, 1));
        closure.albedo = albedo;
        return closure;
    }

    [Differentiable]
    [mutating]
    void AddLobe(Lobe<Spectrum> lobe) {
        [ForceUnroll]
        for (uint i = 0; i < kClosureLobes; ++i) {
            if (i == count) {
                lobes[i] = lobe;
            }
        }
        count = min(count + 1, kClosureLobes);
    }

    float4 Chances(float3 outgoing) {
        uint active = 0;
        uint last = 0;
        [ForceUnroll]
        for (uint i = 0; i < kClosureLobes; ++i) {
            if (lobes[i].Active()) {
                ++active;
                last = i;
            }
        }
        if (active <= 1) {
            return float4(last == 0 ? 1 : 0, last == 1 ? 1 : 0, last == 2 ? 1 : 0,
                          last == 3 ? 1 : 0);
        }
        float4 chance = float4(0);
        [ForceUnroll]
        for (uint i = 0; i < kClosureLobes; ++i) {
            if (lobes[i].Active()) {
                chance[i] = detach(lobes[i].Estimate(outgoing)).Luminance();
            }
        }
        const float total = chance.x + chance.y + chance.z + chance.w;
        return total > 0 ? chance / total : float4(1, 0, 0, 0);
    }

    [Differentiable]
    BsdfValue<Spectrum> Evaluate(float3 outgoing, float3 incoming) {
        if (outgoing.z <= 0 || incoming.z <= 0) {
            return BsdfValue<Spectrum>.Zero();
        }
        const float3 h = normalize(outgoing + incoming);
        BsdfValue<Spectrum> value = BsdfValue<Spectrum>.Zero();
        [ForceUnroll]
        for (uint i = 0; i < kClosureLobes; ++i) {
            if (lobes[i].Active()) {
                const Spectrum lobe = lobes[i].Evaluate(outgoing, incoming, h);
                if (lobes[i].Specular()) {
                    value.specular = value.specular + lobe;
                } else {
                    value.diffuse = value.diffuse + lobe;
                }
            }
        }
        return value;
    }

    float Pdf(float3 outgoing, float3 incoming) {
        if (outgoing.z <= 0 || incoming.z <= 0) {
            return 0;
        }
        const float4 chance = Chances(outgoing);
        const float3 h = normalize(outgoing + incoming);
        float pdf = 0;
        [ForceUnroll]
        for (uint i = 0; i < kClosureLobes; ++i) {
            if (lobes[i].Active()) {
                pdf += chance[i] * lobes[i].Pdf(outgoing, incoming, h);
            }
        }
        return pdf;
    }

    [Differentiable]
    BsdfSample<Spectrum> Sample(float3 outgoing, no_diff float3 u) {
        BsdfSample<Spectrum> sample;
        sample.direction = float3(0, 0, 1);
        sample.weight = BsdfValue<Spectrum>.Zero();
        sample.pdf = 0;
        sample.delta = false;
        if (outgoing.z <= 0) {
            return sample;
        }
        const float3 view = detach(outgoing);
        const float4 chance = no_diff Chances(view);
        uint selected = 0;
        bool chosen = false;
        float cumulative = 0;
        [ForceUnroll]
        for (uint i = 0; i < kClosureLobes; ++i) {
            if (lobes[i].Active() && !chosen) {
                cumulative += chance[i];
                selected = i;
                chosen = u.z < cumulative;
            }
        }
        float3 incoming = float3(0, 0, 1);
        [ForceUnroll]
        for (uint i = 0; i < kClosureLobes; ++i) {
            if (i == selected) {
                incoming = no_diff lobes[i].Direction(view, u.xy);
            }
        }
        const float pdf = no_diff Pdf(view, incoming);
        if (pdf <= 0) {
            return sample;
        }
        const BsdfValue<Spectrum> value = Evaluate(outgoing, incoming);
        sample.direction = incoming;
        sample.pdf = pdf;
        sample.weight = BsdfValue<Spectrum>(value.diffuse.Scale(incoming.z / pdf),
                                            value.specular.Scale(incoming.z / pdf));
        return sample;
    }
}
