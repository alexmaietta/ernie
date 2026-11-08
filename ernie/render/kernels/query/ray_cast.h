// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

#ifndef __SLANG__
#include "ernie/scene/kernels/instance_layout.h"
#include "ernie/scene/kernels/scene.h"
#include "ernie/scene/kernels/tables/render_tables.h"
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
    float minDistance;
    float maxDistance;
    uint world;
    uint count;
};

ERNIE_CHECK_SIZE(RayCastParams, 104);

static const uint kRayCastGroup = 64;

}
