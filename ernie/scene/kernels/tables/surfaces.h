// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/schema/types.h"

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
    uint padding;
};

ERNIE_CHECK_SIZE(SurfaceRecord, 64);

}
