// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/bvh/kernels/bvh.h"
#include "ernie/scene/kernels/instance_layout.h"
#include "ernie/scene/kernels/scene.h"

namespace ernie {

struct InstanceParams {
    Ptr<SceneData> scene;
    Ptr<InstanceLayout> layout;
    Ptr<InstanceRecord> instances;
    Ptr<ShapeRecord> shapes;
    Ptr<uint> tlases;
    Ptr<SlotHit> hits;
    uint64_t unitBox;
};

ERNIE_CHECK_SIZE(InstanceParams, 56);

struct OriginParams {
    Ptr<SceneData> scene;
    Ptr<InstanceLayout> layout;
    Ptr<uint> products;
    uint productCount;
    uint tlasCount;
};

ERNIE_CHECK_SIZE(OriginParams, 32);

static const uint kInstanceGroup = 64;

}
