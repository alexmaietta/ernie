// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

#include "ernie/scene/tables/kernels/meshes.h"
#include "ernie/scene/tables/kernels/shapes.h"

using namespace ernie;

uint FindShape(NormalsParams p, uint point) {
    uint low = 0;
    uint high = p.shapeCount - 1;
    while (low < high) {
        const uint middle = (low + high + 1) / 2;
        if (p.shapes[middle].first <= point) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    return low;
}

groupshared uint s_extents[6];
groupshared uint s_shape;

[shader("compute")]
[numthreads(kNormalsGroup, 1, 1)]
void SmoothNormals(uniform NormalsParams p, uint thread: SV_GroupIndex) {
    const uint item = LinearThreadIndex();
    if (thread == 0) {
        s_shape = FindShape(p, min(item, p.total - 1));
        for (uint axis = 0; axis < 3; ++axis) {
            s_extents[axis] = ~0u;
            s_extents[3 + axis] = 0;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (item < p.total) {
        const uint index = FindShape(p, item);
        const NormalsShape shape = p.shapes[index];
        const uint point = item - shape.first;
        const float3 position = shape.points[point].Vector();
        for (uint axis = 0; axis < 3; ++axis) {
            const uint ordered = OrderedFloat(position[axis]);
            if (index == s_shape) {
                InterlockedMin(s_extents[axis], ordered);
                InterlockedMax(s_extents[3 + axis], ordered);
            } else {
                InterlockedMin(p.extents[index * 3 + axis], ordered);
                InterlockedMax(p.extents[(p.shapeCount + index) * 3 + axis], ordered);
            }
        }
        float3 sum = float3(0);
        const uint end = point < shape.covered ? shape.starts[point + 1] : 0;
        for (uint at = point < shape.covered ? shape.starts[point] : 0; at < end; ++at) {
            const uint triangle = shape.adjacency[at];
            const float3 a = shape.points[shape.triangles[triangle * 3 + 0]].Vector();
            const float3 b = shape.points[shape.triangles[triangle * 3 + 1]].Vector();
            const float3 c = shape.points[shape.triangles[triangle * 3 + 2]].Vector();
            sum += cross(b - a, c - a);
        }
        const float size = length(sum);
        shape.normals[point] = Float3(size > 0 ? sum / size : float3(0, 0, 1));
    }
    GroupMemoryBarrierWithGroupSync();
    if (thread < 6 && item - thread < p.total) {
        const uint axis = thread % 3;
        if (thread < 3) {
            InterlockedMin(p.extents[s_shape * 3 + axis], s_extents[thread]);
        } else {
            InterlockedMax(p.extents[(p.shapeCount + s_shape) * 3 + axis], s_extents[thread]);
        }
    }
}

[shader("compute")]
[numthreads(kNormalsGroup, 1, 1)]
void FinishShapeBounds(uniform NormalsParams p) {
    const uint index = LinearThreadIndex();
    if (index >= p.shapeCount) {
        return;
    }
    float3 lower;
    float3 upper;
    for (uint axis = 0; axis < 3; ++axis) {
        lower[axis] = FloatFromOrdered(p.extents[index * 3 + axis]);
        upper[axis] = FloatFromOrdered(p.extents[(p.shapeCount + index) * 3 + axis]);
    }
    const NormalsShape shape = p.shapes[index];
    shape.bounds[0] = Float3(lower);
    shape.bounds[1] = Float3(upper);
}

uint FindJob(Ptr<TopologyJob> jobs, uint count, uint item, uint field) {
    uint low = 0;
    uint high = count - 1;
    while (low < high) {
        const uint middle = (low + high + 1) / 2;
        const TopologyJob job = jobs[middle];
        const uint first = field == 0 ? job.firstFace : field == 1 ? job.firstCorner : job.firstPoint;
        if (first <= item) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    return low;
}

[shader("compute")]
[numthreads(kTopologyGroup, 1, 1)]
void FaceSizes(uniform TriangulateParams p) {
    const uint face = LinearThreadIndex();
    if (face >= p.faceCount) {
        return;
    }
    const TopologyJob job = p.jobs[FindJob(p.jobs, p.jobCount, face, 0)];
    p.work[face] = uint(max(job.counts[face - job.firstFace], 0));
}

[shader("compute")]
[numthreads(kTopologyGroup, 1, 1)]
void FaceTriangles(uniform TriangulateParams p) {
    const uint face = LinearThreadIndex();
    if (face >= p.faceCount) {
        return;
    }
    const uint index = FindJob(p.jobs, p.jobCount, face, 0);
    const TopologyJob job = p.jobs[index];
    const int size = job.counts[face - job.firstFace];
    const uint first = p.faceStarts[face] - p.faceStarts[job.firstFace];
    bool valid = size >= 3 && first + uint(size) <= job.indexCount;
    uint covered = 0;
    for (uint at = first; valid && at < first + uint(size); ++at) {
        const int point = job.indices[at];
        valid = point >= 0;
        covered = max(covered, uint(point) + 1);
    }
    p.work[face] = valid ? uint(size) - 2 : 0;
    if (valid) {
        InterlockedMax(p.sizes[index * 2 + 1], covered);
    }
}

[shader("compute")]
[numthreads(kTopologyGroup, 1, 1)]
void TopologySizes(uniform TriangulateParams p) {
    const uint index = LinearThreadIndex();
    if (index >= p.jobCount) {
        return;
    }
    const TopologyJob job = p.jobs[index];
    const uint last = job.firstFace + job.faceCount - 1;
    p.sizes[index * 2] = p.triangleStarts[last] + p.work[last] - p.triangleStarts[job.firstFace];
}

[shader("compute")]
[numthreads(kTopologyGroup, 1, 1)]
void EmitTriangles(uniform TriangulateParams p) {
    const uint face = LinearThreadIndex();
    if (face >= p.faceCount || p.work[face] == 0) {
        return;
    }
    const TopologyJob job = p.jobs[FindJob(p.jobs, p.jobCount, face, 0)];
    const uint first = p.faceStarts[face] - p.faceStarts[job.firstFace];
    const uint firstTriangle = p.triangleStarts[face] - p.triangleStarts[job.firstFace];
    for (uint k = 0; k < p.work[face]; ++k) {
        const uint triangle = firstTriangle + k;
        const uint fan[3] = {first, first + k + 1, first + k + 2};
        for (uint c = 0; c < 3; ++c) {
            const uint corner = triangle * 3 + c;
            const uint point = uint(job.indices[fan[c]]);
            job.triangles[corner] = point;
            job.corners[corner] = fan[c];
            p.keys[job.firstCorner + corner] = job.firstPoint + point;
            p.values[job.firstCorner + corner] = job.firstCorner + corner;
            InterlockedAdd(p.histogram[job.firstPoint + point], 1u);
        }
        job.faces[triangle] = face - job.firstFace;
    }
}

[shader("compute")]
[numthreads(kTopologyGroup, 1, 1)]
void WriteAdjacency(uniform TriangulateParams p) {
    const uint at = LinearThreadIndex();
    if (at >= p.cornerCount) {
        return;
    }
    const TopologyJob job = p.emitted[FindJob(p.emitted, p.emittedCount, at, 1)];
    job.adjacency[at - job.firstCorner] = (p.values[at] - job.firstCorner) / 3;
}

[shader("compute")]
[numthreads(kTopologyGroup, 1, 1)]
void WriteStarts(uniform TriangulateParams p) {
    const uint point = LinearThreadIndex();
    if (point >= p.pointCount) {
        return;
    }
    const TopologyJob job = p.emitted[FindJob(p.emitted, p.emittedCount, point, 2)];
    const uint local = point - job.firstPoint;
    job.starts[local] = p.pointStarts[point] - job.firstCorner;
    if (local + 1 == job.covered) {
        job.starts[job.covered] = job.triangleCount * 3;
    }
}
