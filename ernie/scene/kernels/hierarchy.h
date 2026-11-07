// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/scene.h"
#include "ernie/scene/kernels/tables/transforms.h"

namespace ernie {

struct HierarchyTree {
    Ptr<uint> parent;
    Ptr<uint> xform;
    Ptr<uint> segmentRows;
    Ptr<uint> segmentStart;
    Ptr<uint> segmentCount;
};

ERNIE_CHECK_SIZE(HierarchyTree, 40);

struct ResolveParams {
    HierarchyTree tree;
    Ptr<SceneData> scene;
    Ptr<uint> segments;
    Ptr<WorldTransform> transforms;
    Ptr<ResolvedImageable> imageables;
    uint64_t since;
    uint64_t until;
    uint all;
    uint inherit;
    uint zero;
    uint padding;
};

ERNIE_CHECK_SIZE(ResolveParams, 104);

struct GatherParams {
    Ptr<uint> source;
    Ptr<uint> rows;
    Ptr<uint> out;
    uint words;
    uint count;
};

ERNIE_CHECK_SIZE(GatherParams, 32);

static const uint kHierarchyGroup = 64;
static const uint kMaxHierarchyDepth = 256;

}
