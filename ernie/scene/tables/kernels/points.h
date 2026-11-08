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
    uint padding0;
    uint padding1;
    uint padding2;
};

ERNIE_CHECK_SIZE(PointsJob, 80);

struct PointsParams {
    Ptr<PointsJob> jobs;
    Ptr<uint> extents;
    uint jobCount;
    uint total;
};

ERNIE_CHECK_SIZE(PointsParams, 24);

static const uint kPointsGroup = 256;

}
