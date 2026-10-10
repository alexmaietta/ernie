// SPDX-License-Identifier: BSD-3-Clause
// Vertex normals of a mesh whose points moved on the GPU: each point's normal the sum of the
// normals of the triangles around it, weighted by their areas (the cross products' lengths), from
// a list of the triangles around each point built once with the mesh. No atomics, so the sums
// come out the same every time.
//
// Bindings: 0 the points pool, 1 the indices pool, 2 the adjacency pool (per mesh, offsets for
// each point then the triangles), 3 the normals pool, written.
//
// Reference: M. Botsch et al., "Polygon Mesh Processing", A K Peters, 2010, section 3.2 (vertex
// normals as area-weighted face normals).

struct NormalsConstants {
    // x: the mesh's first point (and normal), y: its first index, z: its points, w: its first
    // adjacency entry.
    uint4 mesh;
    // x: the offset of the points' current half in the double-buffered points pool.
    uint4 pool;
}

[[vk::binding(0, 0)]] StructuredBuffer<float> points;
[[vk::binding(1, 0)]] StructuredBuffer<uint> indices;
[[vk::binding(2, 0)]] StructuredBuffer<uint> adjacency;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> normals;
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
