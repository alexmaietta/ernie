// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/bvh/kernels/bvh.h"
#include "ernie/scene/records/bounds.h"
#include "ernie/scene/records/meshes.h"
#include "ernie/scene/records/points.h"
#include "ernie/scene/records/prims.h"
#include "ernie/scene/records/transforms.h"
#include "ernie/scene/tables/kernels/instance_layout.h"

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
    Ptr<PointsRecord> points;
    Ptr<BoundableBox> primBounds;
    uint primCount;
    uint slotCount;
    uint instanceSlots;
    uint tlasCount;
};

ERNIE_CHECK_SIZE(BoundsParams, 104);

static const uint kBoundsGroup = 64;

}
