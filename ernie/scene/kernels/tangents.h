// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/scene.h"

namespace ernie {

struct TangentParams {
    Ptr<Float3> points;
    Ptr<Float3> smoothNormals;
    Ptr<Float3> cornerNormals;
    Ptr<Float2> cornerSt;
    Ptr<int> indices;
    Ptr<uint> triangles;
    Ptr<uint> groupStarts;
    Ptr<uint> groupCorners;
    Ptr<uint> cornerGroups;
    Ptr<Float4> groupTangents;
    Ptr<Float4> tangents;
    uint groupCount;
    uint cornerCount;
};

ERNIE_CHECK_SIZE(TangentParams, 96);

static const uint kTangentsGroup = 64;

}
