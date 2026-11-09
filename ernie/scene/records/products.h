// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/records/render_vars.h"
#include "ernie/stage/schema/types.h"

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
    uint pixelSamples;
    Float3 backgroundLow;
    uint conform;
    Float3 backgroundHigh;
    float pixelAspect;
};

ERNIE_CHECK_SIZE(ProductRecord, 176);

struct PlannedCapture {
    uint64_t firstPixel;
    uint product;
    uint sensor;
    uint layer;
    uint width;
    uint height;
    uint taxonomy;
    float missDistance;
    uint returns;
};

ERNIE_CHECK_SIZE(PlannedCapture, 40);

}
