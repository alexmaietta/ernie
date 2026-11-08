// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/bvh/kernels/bvh.h"
#include "ernie/scene/kernels/instance_layout.h"
#include "ernie/scene/kernels/scene.h"
#include "ernie/scene/kernels/tables/geometry.h"
#include "ernie/scene/kernels/tables/instances.h"

namespace ernie {

struct InstanceParams {
    Ptr<SceneData> scene;
    Ptr<InstanceLayout> layout;
    Ptr<InstanceRecord> instances;
    Ptr<ShapeRecord> shapes;
    Ptr<uint> tlases;
    Ptr<SlotHit> hits;
    Ptr<RenderOrigin> origins;
    Ptr<WorldTransform> transforms;
    Ptr<ResolvedImageable> imageables;
    Ptr<MeshRecord> meshes;
    uint64_t unitBox;
};

ERNIE_CHECK_SIZE(InstanceParams, 88);

struct OriginParams {
    Ptr<SceneData> scene;
    Ptr<InstanceLayout> layout;
    Ptr<uint> products;
    Ptr<RenderOrigin> origins;
    Ptr<uint> viewers;
    Ptr<WorldTransform> transforms;
    uint productCount;
    uint tlasCount;
};

ERNIE_CHECK_SIZE(OriginParams, 56);

static const uint kInstanceGroup = 64;

}
