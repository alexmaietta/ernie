// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

#include "ernie/scene/tables/kernels/materials.h"

using namespace ernie;

MappingRecord LoadMapping(AuthoredData* scene, uint row) {
    TextureMappingAPIData* m = &scene->schemas.textureMappingAPI;
    MappingRecord record;
    record.projection = m.projection[row];
    record.projectionSpace = m.projectionSpace[row];
    record.uvPrimvar = m.uvPrimvar[row];
    record.tileSize = m.tileSize[row];
    record.triplanarBlend = m.triplanarBlend[row];
    record.uvScale = m.uvScale[row];
    record.uvOffset = m.uvOffset[row];
    record.uvRotation = m.uvRotation[row];
    record.hexTiling = m.hexTiling[row];
    record.hexTilingContrast = m.hexTilingContrast[row];
    record.hexTilingRotation = m.hexTilingRotation[row];
    record.padding0 = 0;
    record.padding1 = 0;
    record.padding2 = 0;
    return record;
}

DetailRecord LoadDetail(AuthoredData* scene, uint row) {
    SurfaceDetailAPIData* s = &scene->schemas.surfaceDetailAPI;
    DetailRecord record;
    record.normal = s.normal[row];
    record.normalTextureInfluence = s.normalTextureInfluence[row];
    record.normalStrength = s.normalStrength[row];
    record.normalConvention = s.normalConvention[row];
    record.enableDetail = s.enableDetail[row];
    record.detailNormal = s.detailNormal[row];
    record.detailNormalTextureInfluence = s.detailNormalTextureInfluence[row];
    record.detailNormalStrength = s.detailNormalStrength[row];
    record.detailAlbedo = s.detailAlbedo[row];
    record.detailAlbedoTextureChannel = s.detailAlbedoTextureChannel[row];
    record.detailAlbedoTextureInfluence = s.detailAlbedoTextureInfluence[row];
    record.detailAlbedoStrength = s.detailAlbedoStrength[row];
    record.detailScale = s.detailScale[row];
    record.detailOffset = s.detailOffset[row];
    record.detailRotation = s.detailRotation[row];
    record.detailHexTiling = s.detailHexTiling[row];

    return record;
}

PreviewRecord LoadPreview(AuthoredData* scene, uint prim) {
    UsdPreviewSurfaceData* d = &scene->schemas.usdPreviewSurface;
    const uint row = PrimRow(scene, prim, kSlotUsdPreviewSurface);
    PreviewRecord record;
    record.diffuseColor = d.diffuseColor[row];
    record.diffuseColorTextureChannel = d.diffuseColorTextureChannel[row];
    record.diffuseColorTextureInfluence = d.diffuseColorTextureInfluence[row];
    record.emissiveColor = d.emissiveColor[row];
    record.emissiveColorTextureChannel = d.emissiveColorTextureChannel[row];
    record.emissiveColorTextureInfluence = d.emissiveColorTextureInfluence[row];
    record.useSpecularWorkflow = d.useSpecularWorkflow[row];
    record.specularColor = d.specularColor[row];
    record.specularColorTextureChannel = d.specularColorTextureChannel[row];
    record.specularColorTextureInfluence = d.specularColorTextureInfluence[row];
    record.metallic = d.metallic[row];
    record.metallicTextureChannel = d.metallicTextureChannel[row];
    record.metallicTextureInfluence = d.metallicTextureInfluence[row];
    record.roughness = d.roughness[row];
    record.roughnessTextureChannel = d.roughnessTextureChannel[row];
    record.roughnessTextureInfluence = d.roughnessTextureInfluence[row];
    record.ior = d.ior[row];
    record.clearcoat = d.clearcoat[row];
    record.clearcoatTextureChannel = d.clearcoatTextureChannel[row];
    record.clearcoatTextureInfluence = d.clearcoatTextureInfluence[row];
    record.clearcoatRoughness = d.clearcoatRoughness[row];
    record.clearcoatRoughnessTextureChannel = d.clearcoatRoughnessTextureChannel[row];
    record.clearcoatRoughnessTextureInfluence = d.clearcoatRoughnessTextureInfluence[row];
    record.opacity = d.opacity[row];
    record.opacityTextureChannel = d.opacityTextureChannel[row];
    record.opacityTextureInfluence = d.opacityTextureInfluence[row];

    record.mapping = LoadMapping(scene, ApiRow(scene, prim, kApiTextureMappingAPI));
    record.detail = LoadDetail(scene, ApiRow(scene, prim, kApiSurfaceDetailAPI));
    return record;
}

[shader("compute")]
[numthreads(kMaterialGroup, 1, 1)]
void WriteMaterials(uniform MaterialParams p) {
    const uint prim = LinearThreadIndex();
    AuthoredData* scene = p.scene;
    if (prim >= scene->primCount || (scene->prims.flags[prim] & kPrimAlive) == 0) {
        return;
    }
    const SchemaId schema = SchemaId(scene->prims.schema[prim]);
    if (schema == SchemaId::UsdPreviewSurface) {
        p.previews[PrimRow(scene, prim, kSlotUsdPreviewSurface)] = LoadPreview(scene, prim);
    }
    if (HasApi(scene, prim, kApiNonVisualMaterialAPI)) {
        const uint row = ApiRow(scene, prim, kApiNonVisualMaterialAPI);
        NonVisualMaterialAPIData* n = &scene->schemas.nonVisualMaterialAPI;
        NonVisualRecord record;
        record.lidarReflectance = n.nonvisualLidarReflectance[row];
        record.radarBackscatter = n.nonvisualRadarBackscatter[row];
        record.retroreflective = n.nonvisualRetroreflective[row];
        record.padding0 = 0;
        p.nonVisuals[row] = record;
    }
    if (HasApi(scene, prim, kApiCheckerboardAPI)) {
        const uint row = ApiRow(scene, prim, kApiCheckerboardAPI);
        CheckerboardAPIData* c = &scene->schemas.checkerboardAPI;
        CheckerRecord record;
        record.colorA = c.checkerColorA[row];
        record.colorB = c.checkerColorB[row];
        record.enabled = c.checkerboard[row];
        record.size = c.checkerSize[row];
        record.axis = schema == SchemaId::Plane ? scene->schemas.plane.axis[PrimRow(scene, prim, kSlotPlane)] : 2;
        record.padding0 = 0;
        record.padding1 = 0;
        record.padding2 = 0;
        p.checkers[row] = record;
    }
}
