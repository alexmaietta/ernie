// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/schema/types.h"

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
    uint backgroundKind;
    Float3 backgroundColor;
    uint padding0;
    Float3 backgroundLow;
    uint padding1;
    Float3 backgroundHigh;
    uint padding2;
};

ERNIE_CHECK_SIZE(ProductRecord, 176);

}
