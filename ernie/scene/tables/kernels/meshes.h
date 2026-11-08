// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/records/prims.h"
#include "ernie/scene/records/transforms.h"

namespace ernie {

struct NormalsShape {
    Ptr<Float3> points;
    Ptr<uint> triangles;
    Ptr<uint> starts;
    Ptr<uint> adjacency;
    Ptr<Float3> normals;
    Ptr<Float3> bounds;
    uint first;
    uint count;
    uint covered;
    uint padding;
};

ERNIE_CHECK_SIZE(NormalsShape, 64);

struct NormalsParams {
    Ptr<NormalsShape> shapes;
    Ptr<uint> extents;
    uint shapeCount;
    uint total;
};

ERNIE_CHECK_SIZE(NormalsParams, 24);

static const uint kNormalsGroup = 64;

struct TopologyJob {
    Ptr<int> indices;
    Ptr<int> counts;
    Ptr<uint> triangles;
    Ptr<uint> corners;
    Ptr<uint> faces;
    Ptr<uint> starts;
    Ptr<uint> adjacency;
    uint indexCount;
    uint faceCount;
    uint firstFace;
    uint triangleCount;
    uint covered;
    uint firstCorner;
    uint firstPoint;
    uint padding;
};

ERNIE_CHECK_SIZE(TopologyJob, 88);

struct TriangulateParams {
    Ptr<TopologyJob> jobs;
    Ptr<TopologyJob> emitted;
    Ptr<uint> work;
    Ptr<uint> faceStarts;
    Ptr<uint> triangleStarts;
    Ptr<uint> sizes;
    Ptr<uint> keys;
    Ptr<uint> values;
    Ptr<uint> histogram;
    Ptr<uint> pointStarts;
    uint jobCount;
    uint emittedCount;
    uint faceCount;
    uint cornerCount;
    uint pointCount;
    uint padding;
};

ERNIE_CHECK_SIZE(TriangulateParams, 104);

static const uint kTopologyGroup = 64;

}
