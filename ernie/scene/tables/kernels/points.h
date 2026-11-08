// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/bvh/kernels/bvh.h"
#include "ernie/scene/records/points.h"

namespace ernie {

struct PointsJob {
    PointsRecord record;
    Ptr<BvhBox> boxes;
    Ptr<Float3> bounds;
    uint first;
    uint capacity;
};

ERNIE_CHECK_SIZE(PointsJob, 112);

struct PointsParams {
    Ptr<PointsJob> jobs;
    Ptr<uint> extents;
    Ptr<uint> keys;
    Ptr<uint> values;
    uint jobCount;
    uint total;
};

ERNIE_CHECK_SIZE(PointsParams, 40);

static const uint kPointsGroup = 256;

}
