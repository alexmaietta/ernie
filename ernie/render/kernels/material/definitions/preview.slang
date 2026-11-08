// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
__exported import ernie.render.kernels.material.standard;
import ernie.render.kernels.material.mapping;

using namespace ernie;

struct PreviewMaterial : IStandardMaterial {
    static PreviewParams* Params(ShadeContext context) {
        return context.materials.Params<PreviewParams>(context.words);
    }

    static bool Textured(ShadeContext context) {
        return context.definition == kDefinitionPreviewTextured;
    }

    [noinline]
    static float3 Mapped<let Inputs : uint, let Normals : bool>(MaterialQuery query,
                                                                PreviewParams* d,
                                                                bool textured,
                                                                inout SurfacePoint point,
                                                                out SurfaceTexels<Inputs> base) {
        const bool detailed = d.detail.enableDetail != 0;
        const float2 tilt = d.detail.normal.Vector().xy;
        const bool bent = tilt.x != 0 || tilt.y != 0;
        base = SurfaceTexels<Inputs>(query, point.shadingNormal);
        if (textured) {
            base.inputs = &d.inputs[0];
        }
        float3 normal = point.shadingNormal;
        float3 diffuse = d.diffuseColor.Vector();
        if (textured || detailed || bent) {
            if (Normals) {
                base.SetNormal(textured ? &d.inputs[uint(PreviewInput::Normal)]
                                        : Ptr<TextureInput>(0),
                               d.detail.normal.Vector(), d.detail.normalStrength);
            }
            PlacementSettings settings;
            settings.uv.angle = radians(d.mapping.uvRotation);
            settings.uv.scale = d.mapping.uvScale.Vector();
            settings.uv.offset = d.mapping.uvOffset.Vector();
            settings.fine = false;
            settings.single = query.minimal;
            settings.hex = d.mapping.hexTiling != 0;
            settings.hexContrast = d.mapping.hexTilingContrast;
            settings.hexRotation = d.mapping.hexTilingRotation / 180;
            settings.frameNormal = point.shadingNormal;
            ForEachPlacement(query.hit, point, d.mapping, settings, base);
            normal = base.MappedNormal();
            diffuse = TexturedColor(d.diffuseColor, base.texels[0], base.Handle(0),
                                    &d.inputs[uint(PreviewInput::DiffuseColor)]);
            if (detailed) {
                SurfaceTexels<1> fine = SurfaceTexels<1>(query, normal);
                if (textured) {
                    fine.inputs = &d.inputs[uint(PreviewInput::DetailAlbedo)];
                }
                if (Normals) {
                    fine.SetNormal(textured ? &d.inputs[uint(PreviewInput::DetailNormal)]
                                            : Ptr<TextureInput>(0),
                                   d.detail.detailNormal.Vector(), d.detail.detailNormalStrength);
                }
                settings.fine = true;
                settings.detail.angle = radians(d.detail.detailRotation);
                settings.detail.scale = d.detail.detailScale.Vector();
                settings.detail.offset = d.detail.detailOffset.Vector();
                settings.hex = d.detail.detailHexTiling != 0;
                settings.frameNormal = normal;
                ForEachPlacement(query.hit, point, d.mapping, settings, fine);
                normal = fine.MappedNormal();
                const float3 overlay =
                    TexturedColor(d.detail.detailAlbedo, fine.texels[0], fine.Handle(0),
                                  &d.inputs[uint(PreviewInput::DetailAlbedo)]);
                diffuse *= lerp(float3(1), 2 * overlay, d.detail.detailAlbedoStrength);
            }
        }
        if (dot(normal, point.shadingNormal) > 0) {
            point.shadingNormal = normal;
        }
        return diffuse;
    }

    static SpectralValue Albedo(inout ShadeContext context) {
        SurfaceTexels<1> base;
        SurfacePoint point = context.point;
        const float3 albedo =
            Mapped<1, false>(context.materials, Params(context), Textured(context), point, base);
        context.point = point;
        return SpectralValue.Rgb(albedo);
    }

    static void Material(inout ShadeContext context, inout SurfaceInputs inputs) {
        PreviewParams* d = Params(context);
        SurfaceTexels<kTextureInputs> base;
        SurfacePoint point = context.point;
        inputs.baseColor = SpectralValue.Rgb(
            Mapped<kTextureInputs, true>(context.materials, d, Textured(context), point, base));
        context.point = point;
        inputs.emission = SpectralValue.Rgb(
            TexturedColor(d.emissiveColor, base.texels[1], base.Handle(1),
                          &d.inputs[uint(PreviewInput::EmissiveColor)]));
        inputs.roughness = TexturedFloat(d.roughness, base.texels[4], base.Handle(4),
                                         &d.inputs[uint(PreviewInput::Roughness)]);
        inputs.coatWeight = TexturedFloat(d.clearcoat, base.texels[5], base.Handle(5),
                                          &d.inputs[uint(PreviewInput::Clearcoat)]);
        inputs.coatRoughness =
            TexturedFloat(d.clearcoatRoughness, base.texels[6], base.Handle(6),
                          &d.inputs[uint(PreviewInput::ClearcoatRoughness)]);
        inputs.opacity = TexturedFloat(d.opacity, base.texels[7], base.Handle(7),
                                       &d.inputs[uint(PreviewInput::Opacity)]);
        if (d.useSpecularWorkflow != 0) {
            const float3 specular =
                TexturedColor(d.specularColor, base.texels[2], base.Handle(2),
                              &d.inputs[uint(PreviewInput::SpecularColor)]);
            inputs.metallic = 0;
            inputs.ior = 1.5f;
            inputs.specularColor = SpectralValue.Rgb(specular / DielectricReflectance(1.5f));
        } else {
            inputs.metallic = TexturedFloat(d.metallic, base.texels[3], base.Handle(3),
                                            &d.inputs[uint(PreviewInput::Metallic)]);
            inputs.ior = d.ior;
            inputs.specularColor = SpectralValue.Rgb(float3(1));
        }
    }
}
