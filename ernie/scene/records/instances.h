// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/schema/types.h"

namespace ernie {

struct SurfaceRecord {
    uint64_t points;
    uint64_t indices;
    uint64_t corners;
    uint64_t faces;
    uint64_t normals;
    uint64_t smooth;
    uint normalsInterpolation;
    uint normalsCount;
    uint meshRow;
    uint padding0;
    ArrayRef primvars;
    ArrayRef instancerPrimvars;
    Float3 displayColor;
    uint padding1;
    int64_t instanceId;
};

ERNIE_CHECK_SIZE(SurfaceRecord, 120);

static const float kRenderOriginCell = 64;

struct RenderOrigin {
    Float3 high;
    Float3 low;
};

ERNIE_CHECK_SIZE(RenderOrigin, 24);

#ifdef __SLANG__

float3 RelativeTranslation(RenderOrigin origin, WorldTransform w) {
    return (w.high.Vector() - origin.high.Vector()) + (w.low.Vector() - origin.low.Vector());
}

RenderOrigin SnapOrigin(WorldTransform w) {
    const float3 high = round(w.high.Vector() / kRenderOriginCell) * kRenderOriginCell;
    const float3 rest = (w.high.Vector() - high) + w.low.Vector();
    RenderOrigin origin;
    origin.high = Float3(high);
    origin.low = Float3(round(rest / kRenderOriginCell) * kRenderOriginCell);
    return origin;
}

#endif

}
