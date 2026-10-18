// SPDX-License-Identifier: BSD-3-Clause
/// @file ray_tracer.slang
/// @brief `IRayTracer`, the one interface passes trace through whatever backend answers it, and
/// its hardware ray-query backend.
// The geometry kinds, generated from GeometryKind in ray_tracer.cpp.
import geometry_kinds;
import shapes;

/// A geometry's record, as the C++ `GeometryRecord`: an instance's custom index finds it.
public struct GeometryRecord {
    public uint kind;
    public float halfLength;
    public uint vertexCount;
    public uint triangleCount;
    public float* vertices;
    public uint* indices;
};

/// What a ray hit: where along it, which instance, primitive and geometry, and the transforms
/// between the instance's frame and the scene's. Surfaces are reconstructed from it.
public struct Hit {
    public bool found;
    public float t;
    /// The instance's row: `InstanceIndex()`.
    public uint instance;
    public uint primitive;
    public uint geometry;
    public float2 barycentrics;
    /// An implicit shape's outward normal, in its own frame.
    public float3 objectNormal;
    public float3x4 objectToWorld;
    public float3x4 worldToObject;
};

/// The one interface every pass traces through, whichever backend answers it.
public interface IRayTracer {
    /// @return The nearest hit along the ray among instances in `mask`.
    Hit TraceClosest(RayDesc ray, uint mask);
    /// @return Whether anything in `mask` blocks the ray, for visibility.
    bool TraceAny(RayDesc ray, uint mask);
}

/// Ray queries against a hardware top level, by its device address.
public struct HardwareRayQuery : IRayTracer {
    public uint64_t scene;
    public GeometryRecord* records;

    /// Runs the query, solving each procedural candidate exactly; `Flags` may end it at the first
    /// hit. NVIDIA's compiler fails on `CandidateObjectRay*`: the object ray comes from the
    /// candidate's matrix instead.
    void Run<let Flags : int>(inout RayQuery<Flags> query, RayDesc ray, uint mask,
                              out ShapeHit nearest) {
        query.TraceRayInline(RaytracingAccelerationStructure(scene), RAY_FLAG_NONE, mask, ray);
        nearest.t = ray.TMax;
        nearest.normal = float3(0, 0, 1);
        while (query.Proceed()) {
            if (query.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE) {
                continue;
            }
            GeometryRecord record = records[query.CandidateInstanceID()];
            float3x4 toObject = query.CandidateWorldToObject3x4();
            float3 o = mul(toObject, float4(query.WorldRayOrigin(), 1));
            float3 d = mul(toObject, float4(query.WorldRayDirection(), 0));
            // Object-space distances along the transformed direction are the ray's own.
            ShapeHit hit = nearest;
            hit.t = query.CommittedRayT();
            if (IntersectShape(record.kind, record.halfLength, o, d, ray.TMin, hit)) {
                query.CommitProceduralPrimitiveHit(hit.t);
                nearest = hit;
            }
        }
    }

    public Hit TraceClosest(RayDesc ray, uint mask) {
        RayQuery<RAY_FLAG_NONE> query;
        ShapeHit nearest;
        Run(query, ray, mask, nearest);
        Hit hit;
        uint status = query.CommittedStatus();
        hit.found = status != COMMITTED_NOTHING;
        hit.t = query.CommittedRayT();
        hit.instance = hit.found ? query.CommittedInstanceIndex() : 0;
        hit.primitive = hit.found ? query.CommittedPrimitiveIndex() : 0;
        hit.geometry = hit.found ? query.CommittedInstanceID() : 0;
        hit.barycentrics = status == COMMITTED_TRIANGLE_HIT ? query.CommittedTriangleBarycentrics()
                                                            : float2(0);
        hit.objectNormal = nearest.normal;
        hit.objectToWorld = query.CommittedObjectToWorld3x4();
        hit.worldToObject = query.CommittedWorldToObject3x4();
        return hit;
    }

    public bool TraceAny(RayDesc ray, uint mask) {
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
        ShapeHit nearest;
        Run(query, ray, mask, nearest);
        return query.CommittedStatus() != COMMITTED_NOTHING;
    }
}

/// @return The surface's geometric normal at a hit, in the scene's frame, facing out of an
/// implicit shape or by a triangle's winding.
public float3 WorldNormal(Hit hit, GeometryRecord* records) {
    float3 normal = hit.objectNormal;
    GeometryRecord record = records[hit.geometry];
    if (record.kind == kTriangles) {
        uint* corners = record.indices + 3 * hit.primitive;
        float3 p[3];
        for (uint i = 0; i < 3; ++i) {
            float* v = record.vertices + 3 * corners[i];
            p[i] = float3(v[0], v[1], v[2]);
        }
        normal = cross(p[1] - p[0], p[2] - p[0]);
    }
    // Normals take the inverse transpose: a row vector times the world-to-object matrix.
    return normalize(mul(normal, (float3x3)hit.worldToObject));
}
