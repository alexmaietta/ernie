// SPDX-License-Identifier: BSD-3-Clause
// Vertex normals of a mesh whose points moved on the GPU: each point's normal the sum of the
// normals of the triangles around it, weighted by their areas (the cross products' lengths), from
// a list of the triangles around each point built once with the mesh. No atomics, so the sums
// come out the same every time.
//
// Those of a mesh whose triangles are written on the GPU (a writer's, device indices), around
// whose points no list is known, the `topology*Main` passes give alike: each triangle adds its
// area-weighted normal to its corners' sums in fixed point, so integer atomics make the sums come
// out the same in any order.
//
// Bindings: 0 the points pool, 1 the indices pool, 2 the adjacency pool (per mesh, offsets for
// each point then the triangles), 3 the normals pool, written; for the topology passes, 4 the
// fixed-point sums and 5 the meshes' points and triangles in use (`activeCounts`).
//
// Reference: M. Botsch et al., "Polygon Mesh Processing", A K Peters, 2010, section 3.2 (vertex
// normals as area-weighted face normals).

struct NormalsConstants {
    // x: the mesh's first point (and normal), y: its first index, z: its points, w: its first
    // adjacency entry, or for the topology passes its triangles.
    uint4 mesh;
    // x: the offset of the points' current half in the double-buffered points pool; for the
    // topology passes, y: the mesh's first entry in `sums` (three per point, then its largest
    // triangle's) and z: its slot in `activeCounts` plus 1, 0 for all in use.
    uint4 pool;
}

[[vk::binding(0, 0)]] StructuredBuffer<float> points;
[[vk::binding(1, 0)]] StructuredBuffer<uint> indices;
[[vk::binding(2, 0)]] StructuredBuffer<uint> adjacency;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> normals;
[[vk::binding(4, 0)]] RWStructuredBuffer<int> sums;
[[vk::binding(5, 0)]] StructuredBuffer<int> activeCounts;
[[vk::push_constant]] ConstantBuffer<NormalsConstants> constants;

float3 pointAt(uint i) {
    let at = 3 * (constants.pool.x + constants.mesh.x + i);
    return float3(points[at], points[at + 1], points[at + 2]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void normalsMain(uint3 id : SV_DispatchThreadID) {
    let count = constants.mesh.z;
    let point = id.x;
    if (point >= count) {
        return;
    }
    let base = constants.mesh.w;
    let triangles = base + count + 1;
    var sum = float3(0.0);
    for (uint t = adjacency[base + point]; t < adjacency[base + point + 1]; ++t) {
        let first = constants.mesh.y + 3 * adjacency[triangles + t];
        let a = pointAt(indices[first]);
        let b = pointAt(indices[first + 1]);
        let c = pointAt(indices[first + 2]);
        sum += cross(b - a, c - a);
    }
    let length2 = dot(sum, sum);
    let n = length2 > 0.0 ? sum * rsqrt(length2) : float3(0.0);
    let at = 3 * (constants.mesh.x + point);
    normals[at] = n.x;
    normals[at + 1] = n.y;
    normals[at + 2] = n.z;
}

// The fixed point's one: the largest triangle's normal sums to 2^18, so a point may gather 8192
// such before an int overflows, and triangles 2^-18 of its area still count.
static const float kTopologyOne = 262144.0;

// The mesh's points and triangles in use: all, or as many as its slot of `activeCounts` says.
uint2 topologyInUse() {
    var inUse = uint2(constants.mesh.z, constants.mesh.w);
    if (constants.pool.z != 0) {
        let slot = constants.pool.z - 1;
        inUse = min(inUse, uint2(uint(max(activeCounts[2 * slot], 0)),
                                 uint(max(activeCounts[2 * slot + 1], 0)) / 3));
    }
    return inUse;
}

// Triangle `t`'s corners, and whether it is in use: within the triangles in use, its corners
// within the points in use.
bool topologyTriangle(uint t, out uint3 corners) {
    let inUse = topologyInUse();
    let first = constants.mesh.y + 3 * t;
    corners = uint3(indices[first], indices[first + 1], indices[first + 2]);
    return t < inUse.y && all(corners < inUse.x);
}

float3 topologyFaceNormal(uint3 corners) {
    let a = pointAt(corners.x);
    return cross(pointAt(corners.y) - a, pointAt(corners.z) - a);
}

// The mesh's sums, and its largest triangle's, to 0.
[shader("compute")]
[numthreads(64, 1, 1)]
void topologyClearMain(uint3 id : SV_DispatchThreadID) {
    if (id.x <= 3 * constants.mesh.z) {
        sums[constants.pool.y + id.x] = 0;
    }
}

// The largest triangle's normal length, whose bits order as its value does.
[shader("compute")]
[numthreads(64, 1, 1)]
void topologyLargestMain(uint3 id : SV_DispatchThreadID) {
    uint3 corners;
    if (!topologyTriangle(id.x, corners)) {
        return;
    }
    let length = sqrt(dot(topologyFaceNormal(corners), topologyFaceNormal(corners)));
    if (length > 0.0 && isfinite(length)) {
        InterlockedMax(sums[constants.pool.y + 3 * constants.mesh.z], asint(length));
    }
}

// Each triangle's normal, in fixed point, added to its corners' sums.
[shader("compute")]
[numthreads(64, 1, 1)]
void topologyAddMain(uint3 id : SV_DispatchThreadID) {
    uint3 corners;
    let largest = asfloat(sums[constants.pool.y + 3 * constants.mesh.z]);
    if (!(largest > 0.0) || !topologyTriangle(id.x, corners)) {
        return;
    }
    let n = int3(round(topologyFaceNormal(corners) * (kTopologyOne / largest)));
    for (uint k = 0; k < 3; ++k) {
        let at = constants.pool.y + 3 * corners[k];
        InterlockedAdd(sums[at], n.x);
        InterlockedAdd(sums[at + 1], n.y);
        InterlockedAdd(sums[at + 2], n.z);
    }
}

// Each point's sum, normalised into `normals`; 0 for a point no triangle in use reaches, which
// shades its triangles' own.
[shader("compute")]
[numthreads(64, 1, 1)]
void topologyNormalizeMain(uint3 id : SV_DispatchThreadID) {
    let point = id.x;
    if (point >= constants.mesh.z) {
        return;
    }
    let at = constants.pool.y + 3 * point;
    let sum = float3(float(sums[at]), float(sums[at + 1]), float(sums[at + 2]));
    let length2 = dot(sum, sum);
    let n = length2 > 0.0 ? sum * rsqrt(length2) : float3(0.0);
    let out = 3 * (constants.mesh.x + point);
    normals[out] = n.x;
    normals[out + 1] = n.y;
    normals[out + 2] = n.z;
}
