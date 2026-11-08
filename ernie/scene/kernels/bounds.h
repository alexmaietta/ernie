// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/bvh/kernels/bvh.h"
#include "ernie/scene/kernels/authored.h"
#include "ernie/scene/kernels/instance_layout.h"
#include "ernie/scene/records/bounds.h"
#include "ernie/scene/records/meshes.h"
#include "ernie/scene/records/transforms.h"

namespace ernie {

struct BoundsParams {
    Ptr<AuthoredData> scene;
    Ptr<uint> origins;
    Ptr<uint> extents;
    Ptr<BvhBox> boxes;
    Ptr<InstanceLayout> instances;
    Ptr<uint> instancerExtents;
    Ptr<WorldTransform> transforms;
    Ptr<ResolvedImageable> imageables;
    Ptr<MeshRecord> meshes;
    Ptr<BoundableBox> primBounds;
    uint primCount;
    uint slotCount;
    uint instanceSlots;
    uint tlasCount;
};

ERNIE_CHECK_SIZE(BoundsParams, 96);

static const uint kBoundsGroup = 64;

}
