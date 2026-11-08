// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/records/transforms.h"
#include "ernie/scene/tables/kernels/authored.h"

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
    Ptr<AuthoredData> scene;
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

struct MotionParams {
    Ptr<AuthoredData> scene;
    Ptr<MotionRecord> motions;
    uint rows;
    uint padding;
};

ERNIE_CHECK_SIZE(MotionParams, 24);

static const uint kHierarchyGroup = 64;
static const uint kMaxHierarchyDepth = 256;

}
