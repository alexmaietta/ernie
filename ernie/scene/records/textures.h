// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/schema/types.h"

namespace ernie {

static const uint kCheckerTextureBinding = 0;
static const uint kNoTextures = 0;

struct TextureRecord {
    uint textures;
    uint shaping;
    uint width;
    uint height;
};

ERNIE_CHECK_SIZE(TextureRecord, 16);

}
