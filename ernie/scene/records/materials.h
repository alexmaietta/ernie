// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/schema/types.h"

namespace ernie {

struct MappingRecord {
    uint projection;
    uint projectionSpace;
    uint uvPrimvar;
    float tileSize;
    float triplanarBlend;
    Float2 uvScale;
    Float2 uvOffset;
    float uvRotation;
    uint hexTiling;
    float hexTilingContrast;
    float hexTilingRotation;
    uint padding0;
    uint padding1;
    uint padding2;
};

ERNIE_CHECK_SIZE(MappingRecord, 64);

struct DetailRecord {
    Float3 normal;
    float normalTextureInfluence;
    float normalStrength;
    uint normalConvention;
    uint enableDetail;
    Float3 detailNormal;
    float detailNormalTextureInfluence;
    float detailNormalStrength;
    Float3 detailAlbedo;
    uint detailAlbedoTextureChannel;
    float detailAlbedoTextureInfluence;
    float detailAlbedoStrength;
    Float2 detailScale;
    Float2 detailOffset;
    float detailRotation;
    uint detailHexTiling;
};

ERNIE_CHECK_SIZE(DetailRecord, 96);

struct PreviewRecord {
    Float3 diffuseColor;
    uint diffuseColorTextureChannel;
    float diffuseColorTextureInfluence;
    Float3 emissiveColor;
    uint emissiveColorTextureChannel;
    float emissiveColorTextureInfluence;
    uint useSpecularWorkflow;
    Float3 specularColor;
    uint specularColorTextureChannel;
    float specularColorTextureInfluence;
    float metallic;
    uint metallicTextureChannel;
    float metallicTextureInfluence;
    float roughness;
    uint roughnessTextureChannel;
    float roughnessTextureInfluence;
    float ior;
    float clearcoat;
    uint clearcoatTextureChannel;
    float clearcoatTextureInfluence;
    float clearcoatRoughness;
    uint clearcoatRoughnessTextureChannel;
    float clearcoatRoughnessTextureInfluence;
    float opacity;
    uint opacityTextureChannel;
    float opacityTextureInfluence;
    MappingRecord mapping;
    DetailRecord detail;
};

ERNIE_CHECK_SIZE(PreviewRecord, 288);

struct NonVisualRecord {
    float lidarReflectance;
    float radarBackscatter;
    uint retroreflective;
    uint padding0;
};

ERNIE_CHECK_SIZE(NonVisualRecord, 16);

struct CheckerRecord {
    Float3 colorA;
    uint enabled;
    Float3 colorB;
    float size;
    uint axis;
    uint padding0;
    uint padding1;
    uint padding2;
};

ERNIE_CHECK_SIZE(CheckerRecord, 48);

}
