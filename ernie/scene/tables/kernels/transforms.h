// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/records/prims.h"
#include "ernie/scene/records/transforms.h"

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
    Paged<uint64_t> translateStamps;
    Paged<uint64_t> orientStamps;
    Paged<uint64_t> scaleStamps;
    Paged<uint64_t> resetXformStackStamps;
    Paged<uint64_t> visibilityStamps;
    Paged<uint64_t> purposeStamps;
    Paged<uint64_t> materialBindingStamps;
    uint64_t since;
    uint64_t until;
    uint all;
    uint inherit;
    uint zero;
    uint padding;
};

ERNIE_CHECK_SIZE(ResolveParams, 160);

struct MotionParams {
    Ptr<AuthoredData> scene;
    Ptr<MotionRecord> motions;
    uint rows;
    uint padding;
};

ERNIE_CHECK_SIZE(MotionParams, 24);

struct FollowState {
    Float3 high;
    uint target;
    Float3 low;
    uint settled;
    Float4 orient;
};

ERNIE_CHECK_SIZE(FollowState, 48);

struct FollowParams {
    HierarchyTree tree;
    Ptr<AuthoredData> scene;
    Ptr<WorldTransform> transforms;
    Ptr<uint> rows;
    Ptr<uint> owners;
    Ptr<FollowState> states;
    uint first;
    uint count;
    float dt;
    uint snap;
    uint zero;
    uint padding;
};

ERNIE_CHECK_SIZE(FollowParams, 104);

static const uint kHierarchyGroup = 64;
static const uint kMaxHierarchyDepth = 256;

}
