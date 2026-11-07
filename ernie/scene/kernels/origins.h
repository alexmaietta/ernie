// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/scene.h"

namespace ernie {

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
