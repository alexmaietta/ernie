// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
__exported import ernie.render.kernels.material.standard;
import ernie.render.kernels.material.mapping;

using namespace ernie;

struct PreviewMaterial : IStandardMaterial {
    [noinline]
    static float3 Mapped<let Inputs : uint, let Normals : bool>(MaterialQuery query,
                                                                uint material,
                                                                inout SurfacePoint point,
                                                                out SurfaceTexels<Inputs> base) {
        RenderTables* tables = query.hit.tables;
        PreviewRecord* d = &tables->previews[PrimRow(tables, material, kSlotUsdPreviewSurface)];
        const uint textures = query.Textures(material);
        const bool detailed = d.detail.enableDetail != 0;
        const float2 tilt = d.detail.normal.Vector().xy;
        const bool bent = tilt.x != 0 || tilt.y != 0;
        base = SurfaceTexels<Inputs>(query, point.shadingNormal);
        base.bindings = query.Bindings(textures, 0);
        float3 normal = point.shadingNormal;
        float3 diffuse = d.diffuseColor.Vector();
        if (textures != kNoTextures || detailed || bent) {
            const bool directX =
                NormalConvention(d.detail.normalConvention) == NormalConvention::DirectX;
            if (Normals) {
                base.normalHandle = query.Texture(textures, uint(PreviewTexture::Normal));
                base.normalValue = d.detail.normal.Vector();
                base.normalInfluence = d.detail.normalTextureInfluence;
                base.normalStrength = d.detail.normalStrength;
                base.directX = directX;
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
            diffuse = MixedColor(d.diffuseColor, base.texels[0], base.Handle(0),
                                 d.diffuseColorTextureChannel,
                                 d.diffuseColorTextureInfluence);
            if (detailed) {
                SurfaceTexels<1> fine = SurfaceTexels<1>(query, normal);
                fine.bindings = query.Bindings(textures, uint(PreviewTexture::DetailAlbedo));
                if (Normals) {
                    fine.normalHandle =
                        query.Texture(textures, uint(PreviewTexture::DetailNormal));
                    fine.normalValue = d.detail.detailNormal.Vector();
                    fine.normalInfluence = d.detail.detailNormalTextureInfluence;
                    fine.normalStrength = d.detail.detailNormalStrength;
                    fine.directX = directX;
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
                    MixedColor(d.detail.detailAlbedo, fine.texels[0], fine.Handle(0),
                               d.detail.detailAlbedoTextureChannel,
                               d.detail.detailAlbedoTextureInfluence);
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
        const float3 albedo = Mapped<1, false>(context.materials, context.material, point, base);
        context.point = point;
        return SpectralValue.Rgb(albedo);
    }

    static void Material(inout ShadeContext context, inout SurfaceInputs inputs) {
        RenderTables* tables = context.materials.hit.tables;
        PreviewRecord* d =
            &tables->previews[PrimRow(tables, context.material, kSlotUsdPreviewSurface)];
        SurfaceTexels<kTextureInputs> base;
        SurfacePoint point = context.point;
        inputs.baseColor = SpectralValue.Rgb(
            Mapped<kTextureInputs, true>(context.materials, context.material, point, base));
        context.point = point;
        inputs.emission = SpectralValue.Rgb(
            MixedColor(d.emissiveColor, base.texels[1], base.Handle(1),
                       d.emissiveColorTextureChannel, d.emissiveColorTextureInfluence));
        inputs.roughness = MixedFloat(d.roughness, base.texels[4], base.Handle(4),
                                      d.roughnessTextureChannel, d.roughnessTextureInfluence);
        inputs.coatWeight = MixedFloat(d.clearcoat, base.texels[5], base.Handle(5),
                                       d.clearcoatTextureChannel, d.clearcoatTextureInfluence);
        inputs.coatRoughness =
            MixedFloat(d.clearcoatRoughness, base.texels[6], base.Handle(6),
                       d.clearcoatRoughnessTextureChannel, d.clearcoatRoughnessTextureInfluence);
        inputs.opacity = MixedFloat(d.opacity, base.texels[7], base.Handle(7),
                                    d.opacityTextureChannel, d.opacityTextureInfluence);
        if (d.useSpecularWorkflow != 0) {
            const float3 specular =
                MixedColor(d.specularColor, base.texels[2], base.Handle(2),
                           d.specularColorTextureChannel, d.specularColorTextureInfluence);
            inputs.metallic = 0;
            inputs.ior = 1.5f;
            inputs.specularColor = SpectralValue.Rgb(specular / DielectricReflectance(1.5f));
        } else {
            inputs.metallic = MixedFloat(d.metallic, base.texels[3], base.Handle(3),
                                         d.metallicTextureChannel, d.metallicTextureInfluence);
            inputs.ior = d.ior;
            inputs.specularColor = SpectralValue.Rgb(float3(1));
        }
    }
}
