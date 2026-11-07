// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
__exported import ernie.render.kernels.material.imaterial;
__exported import ernie.render.kernels.material.lambert;
import ernie.render.kernels.material.ggx;
import ernie.render.kernels.material.mapping;

using namespace ernie;

static const float kClearcoatReflectance = 0.04f;
static const float kMinimumAlpha = 1e-3f;

struct PreviewSurfaceBsdf<Spectrum : ISpectrum> : IBsdf<Spectrum> {
    Spectrum diffuse;
    Spectrum specular;
    float alpha;
    float clearcoat;
    float clearcoatAlpha;

    [Differentiable]
    __init(Spectrum diffuse, Spectrum specular, float alpha, float clearcoat,
           float clearcoatAlpha) {
        this.diffuse = diffuse;
        this.specular = specular;
        this.alpha = alpha;
        this.clearcoat = clearcoat;
        this.clearcoatAlpha = clearcoatAlpha;
    }

    [Differentiable]
    float Coat(float cosine) {
        return clearcoat * Schlick(kClearcoatReflectance, cosine);
    }

    float3 LobeChances(float3 outgoing) {
        const float coat = Coat(outgoing.z);
        const Spectrum fresnel = Schlick(specular, outgoing.z);
        const float diffuseChance =
            (diffuse * (Spectrum.Constant(1) - fresnel)).Luminance() * (1 - coat);
        const float specularChance = fresnel.Luminance() * (1 - coat);
        const float total = diffuseChance + specularChance + coat;
        return total > 0 ? float3(diffuseChance, specularChance, coat) / total : float3(1, 0, 0);
    }

    [Differentiable]
    BsdfValue<Spectrum> Evaluate(float3 outgoing, float3 incoming) {
        if (outgoing.z <= 0 || incoming.z <= 0) {
            return BsdfValue<Spectrum>.Zero();
        }
        const float3 h = normalize(outgoing + incoming);
        const float cosine = dot(incoming, h);
        const float under = 1 - Coat(outgoing.z);
        const Spectrum diffuseValue =
            diffuse * (Spectrum.Constant(1) - Schlick(specular, outgoing.z)) * (under / kPi);
        const Spectrum specularValue =
            Schlick(specular, cosine) * (GgxReflection(outgoing, incoming, h, alpha) * under) +
            Spectrum.Constant(clearcoat * Schlick(kClearcoatReflectance, cosine) *
                       GgxReflection(outgoing, incoming, h, clearcoatAlpha));
        return BsdfValue<Spectrum>(diffuseValue, specularValue);
    }

    float Pdf(float3 outgoing, float3 incoming) {
        if (outgoing.z <= 0 || incoming.z <= 0) {
            return 0;
        }
        const float3 chance = LobeChances(outgoing);
        const float3 h = normalize(outgoing + incoming);
        return chance.x * incoming.z / kPi + chance.y * GgxReflectionPdf(outgoing, h, alpha) +
               chance.z * GgxReflectionPdf(outgoing, h, clearcoatAlpha);
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
        const float3 chance = no_diff LobeChances(view);
        float3 incoming = no_diff CosineHemisphere(u.xy);
        if (u.z >= chance.x) {
            const float a = detach(u.z < chance.x + chance.y ? alpha : clearcoatAlpha);
            const float3 h = no_diff SampleGgxVisibleNormal(view, a, u.xy);
            incoming = 2 * dot(view, h) * h - view;
        }
        const float pdf = no_diff Pdf(view, incoming);
        if (pdf <= 0) {
            return sample;
        }
        const BsdfValue<Spectrum> value = Evaluate(outgoing, incoming);
        sample.direction = incoming;
        sample.pdf = pdf;
        sample.weight = BsdfValue<Spectrum>(value.diffuse * (incoming.z / pdf),
                                     value.specular * (incoming.z / pdf));
        return sample;
    }
}

struct PreviewSurface<Spectrum : ISpectrum> : IMaterial<Spectrum> {
    typealias Bsdf = PreviewSurfaceBsdf<Spectrum>;
    typealias Medium = Vacuum<Spectrum>;

    float3 diffuseColor;
    float3 emissiveColor;
    no_diff bool useSpecularWorkflow;
    float3 specularColor;
    float metallic;
    float roughness;
    float clearcoat;
    float clearcoatRoughness;
    float opacity;
    float ior;

    [Differentiable]
    __init(float3 diffuseColor, float3 emissiveColor, no_diff bool useSpecularWorkflow,
           float3 specularColor, float metallic, float roughness, float clearcoat,
           float clearcoatRoughness, float opacity, float ior) {
        this.diffuseColor = diffuseColor;
        this.emissiveColor = emissiveColor;
        this.useSpecularWorkflow = useSpecularWorkflow;
        this.specularColor = specularColor;
        this.metallic = metallic;
        this.roughness = roughness;
        this.clearcoat = clearcoat;
        this.clearcoatRoughness = clearcoatRoughness;
        this.opacity = opacity;
        this.ior = ior;
    }

    [noinline]
    static float3 Mapped<let Inputs : uint, let Normals : bool>(MaterialQuery query,
                                                                uint material,
                                                                inout SurfacePoint point,
                                                                out SurfaceTexels<Inputs> base) {
        SceneData* scene = query.hit.scene;
        const uint row = PrimRow(scene, material, kSlotUsdPreviewSurface);
        UsdPreviewSurfaceData* d = &scene->schemas.usdPreviewSurface;
        SurfaceDetailAPIData* s = &scene->schemas.surfaceDetailAPI;
        const uint detail = ApiRow(scene, material, kApiSurfaceDetailAPI);
        const uint textures = query.Textures(material);
        const bool detailed = s.enableDetail[detail] != 0;
        const float2 tilt = s.normal[detail].Vector().xy;
        const bool bent = tilt.x != 0 || tilt.y != 0;
        base = SurfaceTexels<Inputs>(query, point.shadingNormal);
        base.bindings = query.Bindings(textures, 0);
        float3 normal = point.shadingNormal;
        float3 diffuse = d.diffuseColor[row].Vector();
        if (textures != kNoTextures || detailed || bent) {
            const bool directX =
                NormalConvention(s.normalConvention[detail]) == NormalConvention::DirectX;
            if (Normals) {
                base.normalHandle = query.Texture(textures, uint(PreviewTexture::Normal));
                base.normalValue = s.normal[detail].Vector();
                base.normalInfluence = s.normalTextureInfluence[detail];
                base.normalStrength = s.normalStrength[detail];
                base.directX = directX;
            }
            const uint mapping = ApiRow(scene, material, kApiTextureMappingAPI);
            TextureMappingAPIData* m = &scene->schemas.textureMappingAPI;
            PlacementSettings settings;
            settings.uv.angle = radians(m.uvRotation[mapping]);
            settings.uv.scale = m.uvScale[mapping].Vector();
            settings.uv.offset = m.uvOffset[mapping].Vector();
            settings.fine = false;
            settings.single = query.minimal;
            settings.hex = m.hexTiling[mapping] != 0;
            settings.hexContrast = m.hexTilingContrast[mapping];
            settings.hexRotation = m.hexTilingRotation[mapping] / 180;
            settings.frameNormal = point.shadingNormal;
            ForEachPlacement(query.hit, point, mapping, settings, base);
            normal = base.MappedNormal();
            diffuse = MixedColor(d.diffuseColor[row], base.texels[0], base.Handle(0),
                                 d.diffuseColorTextureChannel[row],
                                 d.diffuseColorTextureInfluence[row]);
            if (detailed) {
                SurfaceTexels<1> fine = SurfaceTexels<1>(query, normal);
                fine.bindings = query.Bindings(textures, uint(PreviewTexture::DetailAlbedo));
                if (Normals) {
                    fine.normalHandle =
                        query.Texture(textures, uint(PreviewTexture::DetailNormal));
                    fine.normalValue = s.detailNormal[detail].Vector();
                    fine.normalInfluence = s.detailNormalTextureInfluence[detail];
                    fine.normalStrength = s.detailNormalStrength[detail];
                    fine.directX = directX;
                }
                settings.fine = true;
                settings.detail.angle = radians(s.detailRotation[detail]);
                settings.detail.scale = s.detailScale[detail].Vector();
                settings.detail.offset = s.detailOffset[detail].Vector();
                settings.hex = s.detailHexTiling[detail] != 0;
                settings.frameNormal = normal;
                ForEachPlacement(query.hit, point, mapping, settings, fine);
                normal = fine.MappedNormal();
                const float3 overlay =
                    MixedColor(s.detailAlbedo[detail], fine.texels[0], fine.Handle(0),
                               s.detailAlbedoTextureChannel[detail],
                               s.detailAlbedoTextureInfluence[detail]);
                diffuse *= lerp(float3(1), 2 * overlay, s.detailAlbedoStrength[detail]);
            }
        }
        if (dot(normal, point.shadingNormal) > 0) {
            point.shadingNormal = normal;
        }
        return diffuse;
    }

    static Spectrum LoadAlbedo(MaterialQuery query, uint material, inout SurfacePoint point) {
        SurfaceTexels<1> base;
        return Spectrum.FromRgb(Mapped<1, false>(query, material, point, base));
    }

    static PreviewSurface<Spectrum> Load(MaterialQuery query, uint material,
                                         inout SurfacePoint point) {
        SceneData* scene = query.hit.scene;
        const uint row = PrimRow(scene, material, kSlotUsdPreviewSurface);
        UsdPreviewSurfaceData* d = &scene->schemas.usdPreviewSurface;
        SurfaceTexels<kTextureInputs> base;
        const float3 diffuse = Mapped<kTextureInputs, true>(query, material, point, base);
        return PreviewSurface<Spectrum>(
            diffuse,
            MixedColor(d.emissiveColor[row], base.texels[1], base.Handle(1),
                       d.emissiveColorTextureChannel[row], d.emissiveColorTextureInfluence[row]),
            d.useSpecularWorkflow[row] != 0,
            MixedColor(d.specularColor[row], base.texels[2], base.Handle(2),
                       d.specularColorTextureChannel[row], d.specularColorTextureInfluence[row]),
            MixedFloat(d.metallic[row], base.texels[3], base.Handle(3),
                       d.metallicTextureChannel[row], d.metallicTextureInfluence[row]),
            MixedFloat(d.roughness[row], base.texels[4], base.Handle(4),
                       d.roughnessTextureChannel[row], d.roughnessTextureInfluence[row]),
            MixedFloat(d.clearcoat[row], base.texels[5], base.Handle(5),
                       d.clearcoatTextureChannel[row], d.clearcoatTextureInfluence[row]),
            MixedFloat(d.clearcoatRoughness[row], base.texels[6], base.Handle(6),
                       d.clearcoatRoughnessTextureChannel[row],
                       d.clearcoatRoughnessTextureInfluence[row]),
            MixedFloat(d.opacity[row], base.texels[7], base.Handle(7),
                       d.opacityTextureChannel[row], d.opacityTextureInfluence[row]),
            d.ior[row]);
    }

    [Differentiable]
    Spectrum Albedo(SurfacePoint point) {
        return Spectrum.FromRgb(diffuseColor);
    }

    [Differentiable]
    Bsdf Surface(SurfacePoint point) {
        const float alpha = max(roughness * roughness, kMinimumAlpha);
        const float clearcoatAlpha = max(clearcoatRoughness * clearcoatRoughness, kMinimumAlpha);
        if (useSpecularWorkflow) {
            return Bsdf(Spectrum.FromRgb(diffuseColor), Spectrum.FromRgb(specularColor), alpha,
                        clearcoat, clearcoatAlpha);
        }
        const float r = (1 - ior) / (1 + ior);
        return Bsdf(Spectrum.FromRgb(diffuseColor * (1 - metallic)),
                    Spectrum.FromRgb(lerp(float3(r * r), diffuseColor, metallic)), alpha,
                    clearcoat, clearcoatAlpha);
    }

    [Differentiable]
    Spectrum Emission(SurfacePoint point, no_diff float3 outgoing) {
        return Spectrum.FromRgb(emissiveColor);
    }

    [Differentiable]
    float Opacity(SurfacePoint point) {
        return opacity;
    }

    [Differentiable]
    Medium Volume(SurfacePoint point) {
        return Medium();
    }
}
