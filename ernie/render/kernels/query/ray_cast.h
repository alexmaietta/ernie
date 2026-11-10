// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

#ifndef __SLANG__
#include "ernie/scene/records/instances.h"
#include "ernie/scene/records/render_tables.h"
#endif

namespace ernie {

struct RayCastParams {
    Ptr<RenderTables> tables;
    Ptr<float> origins;
    Ptr<float> directions;
    Ptr<float> tMins;
    Ptr<float> tMaxes;
    Ptr<uint> worlds;
    Ptr<uint> hits;
    Ptr<float> distances;
    Ptr<uint> prims;
    Ptr<float> normals;
    Ptr<float> positions;
    Ptr<int64_t> instances;
    Ptr<uint> ignored;
    float minDistance;
    float maxDistance;
    uint world;
    uint count;
    uint ignoredCount;
    uint spaceRow;
};

ERNIE_CHECK_SIZE(RayCastParams, 128);

static const uint kRayCastGroup = 64;

}
