// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

namespace ernie {

static const uint kCheckerTextureBinding = 0;
static const uint kNoTextures = 0;

ERNIE_ENUM(PreviewTexture){
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

}
