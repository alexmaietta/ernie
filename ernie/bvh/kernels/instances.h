// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/bvh/kernels/bvh.h"

namespace ernie {

static const uint kMaskInstancesThreads = 64;

struct MaskInstancesParams {
    Ptr<InstanceRecord> source;
    Ptr<InstanceRecord> destination;
    Ptr<uint> offsets;
    Ptr<uint> counts;
    uint total;
    uint worlds;
};

ERNIE_CHECK_SIZE(MaskInstancesParams, 40);

static const uint kGatherWorldsThreads = 64;

struct GatherWorldsParams {
    Ptr<InstanceRecord> source;
    Ptr<InstanceRecord> destination;
    Ptr<uint> sourceOffsets;
    Ptr<uint> offsets;
    Ptr<uint> worlds;
    Ptr<uint> counts;
    Ptr<uint> gatheredCounts;
    uint total;
    uint selected;
};

ERNIE_CHECK_SIZE(GatherWorldsParams, 64);

}
