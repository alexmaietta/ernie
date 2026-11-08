// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/schema/types.h"

namespace ernie {

struct ProductRecord {
    ArrayRef sensors;
    ArrayRef vars;
    uint sensorSchema;
    uint width;
    uint height;
    uint returns;
    float frameRate;
    float startTime;
    float missDistance;
    uint renderMode;
    uint minimalMode;
    uint spectralMode;
    float textureLodBias;
    uint maxAnisotropy;
    uint maxBounces;
    uint castShadows;
    float shadowVisibility;
    uint hemispheric;
    Float3 ambient;
    uint background;
    Float3 ground;
    uint padding0;
    Float3 backgroundColor;
    uint padding1;
};

ERNIE_CHECK_SIZE(ProductRecord, 144);

}
