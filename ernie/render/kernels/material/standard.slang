// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.material.idefinition;
__exported import ernie.render.kernels.material.inputs;
import ernie.render.kernels.material.primitives.ggx;
import ernie.render.kernels.basis;

static const float kCoatReflectance = 0.04f;
static const float kMinimumAlpha = 1e-3f;

[Differentiable]
float DielectricReflectance(float ior) {
    const float r = (1 - ior) / (1 + ior);
    return r * r;
}

[Differentiable]
void LowerStandardSurface<Spectrum : ISpectrum>(SurfaceInputs inputs, float3 outgoing,
                                                inout OpticalClosure<Spectrum> closure) {
    const Spectrum base = inputs.baseColor.Evaluate<Spectrum>();
    const bool specular = inputs.specularWeight > 0 || inputs.metallic > 0;
    const float coat =
        inputs.coatWeight > 0 ? inputs.coatWeight * Schlick(kCoatReflectance, outgoing.z) : 0;
    const float under = 1 - coat;
    const Spectrum dielectric = inputs.specularColor.Evaluate<Spectrum>().Scale(
        inputs.specularWeight * DielectricReflectance(inputs.ior));
    const Spectrum f0 = dielectric.Scale(1 - inputs.metallic) + base.Scale(inputs.metallic);
    Spectrum diffuse = base.Scale((1 - inputs.metallic) * under);
    if (specular) {
        diffuse = diffuse * (Spectrum.Constant(1) - Schlick(f0, outgoing.z));
    }
    closure.AddLobe(Lobe<Spectrum>.DiffuseReflection(diffuse, 1));
    closure.AddLobe(Lobe<Spectrum>.MicrofacetGgx(
        f0, max(inputs.roughness * inputs.roughness, kMinimumAlpha), specular ? under : 0));
    closure.AddLobe(Lobe<Spectrum>.MicrofacetGgx(
        Spectrum.Constant(kCoatReflectance),
        max(inputs.coatRoughness * inputs.coatRoughness, kMinimumAlpha), inputs.coatWeight));
    closure.albedo = base;
    closure.emission = inputs.emission.Evaluate<Spectrum>();
    closure.opacity = inputs.opacity;
    closure.retroreflection = no_diff saturate(detach(inputs.retroreflection));
    closure.radarBackscatter = no_diff max(detach(inputs.radarBackscatter), 0.0f);
    closure.emissivityAuthored = inputs.emissivityAuthored;
    closure.radio.electrical = inputs.electricalAuthored ? inputs.electrical
                               : detach(inputs.metallic) > 0.5f ? ElectricalValue.Conductor()
                                                                : inputs.electrical;
    closure.radio.rmsHeight = detach(inputs.radarRoughness);
    closure.radio.correlationLength = detach(inputs.radarCorrelation);
    closure.emissivity = inputs.emissivity.Evaluate<Spectrum>();
}

interface IStandardMaterial {
    static void Material(inout ShadeContext context, inout SurfaceInputs inputs);

    static SpectralValue Albedo(inout ShadeContext context);
}

struct StandardDefinition<M : IStandardMaterial, Spectrum : ISpectrum>
    : IMaterialDefinition<Spectrum> {
    static void Shade(inout ShadeContext context, inout OpticalClosure<Spectrum> closure) {
        SurfaceInputs inputs = SurfaceInputs();
        M.Material(context, inputs);
        const float3 outgoing = Basis(context.point.shadingNormal).ToLocal(context.outgoing);
        LowerStandardSurface<Spectrum>(inputs, outgoing, closure);
    }

    static Spectrum Albedo(inout ShadeContext context) {
        return M.Albedo(context).Evaluate<Spectrum>();
    }
}

SpectralValue StandardAlbedo<M : IStandardMaterial>(inout ShadeContext context) {
    SurfaceInputs inputs = SurfaceInputs();
    M.Material(context, inputs);
    return inputs.baseColor;
}
