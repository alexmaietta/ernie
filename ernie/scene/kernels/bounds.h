// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/bvh/kernels/bvh.h"
#include "ernie/scene/kernels/instance_layout.h"
#include "ernie/scene/kernels/scene.h"

namespace ernie {

struct BoundsParams {
    Ptr<SceneData> scene;
    Ptr<uint> origins;
    Ptr<uint> extents;
    Ptr<BvhBox> boxes;
    Ptr<InstanceLayout> instances;
    Ptr<uint> instancerExtents;
    Ptr<WorldTransform> transforms;
    Ptr<ResolvedImageable> imageables;
    uint primCount;
    uint slotCount;
    uint instanceSlots;
    uint tlasCount;
};

ERNIE_CHECK_SIZE(BoundsParams, 80);

static const uint kBoundsGroup = 64;

}
