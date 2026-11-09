// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/stage/schema/types.h"

namespace ernie {

static const uint kDefinitionDiffuse = 0;
static const uint kDefinitionPreviewConstant = 1;
static const uint kDefinitionPreviewTextured = 2;
static const uint kDefinitionPlaceholder = 3;

ERNIE_ENUM(TextureChannel){
    Rgb,
    R,
    G,
    B,
    A,
};

ERNIE_ENUM(PreviewInput){
    DiffuseColor,
    EmissiveColor,
    SpecularColor,
    Metallic,
    Roughness,
    Clearcoat,
    ClearcoatRoughness,
    Opacity,
    Occlusion,
    Normal,
    DetailNormal,
    DetailAlbedo,
    Displacement,
};

static const uint kPreviewInputs = 13;

struct MaterialRecord {
    uint definition;
    uint words;
    uint previousDefinition;
    uint previousWords;
};

ERNIE_CHECK_SIZE(MaterialRecord, 16);

struct TextureInput {
    uint64_t handle;
    uint channel;
    uint spectrum;
    Float4 scale;
    Float4 bias;
};

ERNIE_CHECK_SIZE(TextureInput, 48);

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
    float normalStrength;
    Float3 detailNormal;
    float detailNormalStrength;
    Float3 detailAlbedo;
    float detailAlbedoStrength;
    Float2 detailScale;
    Float2 detailOffset;
    float detailRotation;
    uint detailHexTiling;
    uint enableDetail;
    uint padding0;
};

ERNIE_CHECK_SIZE(DetailRecord, 80);

struct PreviewParams {
    Float3 diffuseColor;
    uint useSpecularWorkflow;
    Float3 emissiveColor;
    float metallic;
    Float3 specularColor;
    float roughness;
    float ior;
    float clearcoat;
    float clearcoatRoughness;
    float opacity;
    MappingRecord mapping;
    DetailRecord detail;
    TextureInput inputs[kPreviewInputs];
};

ERNIE_CHECK_SIZE(PreviewParams, 832);

struct RadioRecord {
    uint kind;
    uint valid;
    float rmsHeight;
    float correlationLength;
    Float4 data;
};

ERNIE_CHECK_SIZE(RadioRecord, 32);

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
