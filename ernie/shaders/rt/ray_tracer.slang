// SPDX-License-Identifier: BSD-3-Clause
/// @file ray_tracer.slang
/// @brief `RayTracer`, what every pass traces the scene through: inline ray queries against the
/// top level, each procedural candidate solved exactly as its unit shape.
///
///     import ray_tracer;
///     RayTracer tracer = RayTracer(parameters.tracer);
///     Hit hit = tracer.TraceClosest(ray, 0xff);
///     if (hit.found) { float3 n = tracer.WorldNormal(hit); }
import geometry;
import shapes;

/// What a pass hands its tracer, as the C++ `TracerParameters`.
public struct TracerParameters {
    /// The top level's device address.
    public uint64_t scene;
    public GeometryRecord* records;
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

public struct RayTracer {
    TracerParameters parameters;

    public __init(TracerParameters parameters) {
        this.parameters = parameters;
    }

    /// @return The nearest hit along the ray among instances in `mask`.
    public Hit TraceClosest(RayDesc ray, uint mask) {
        RayQuery<RAY_FLAG_NONE> query;
        ShapeHit nearest;
        Run(query, ray, mask, nearest);
        Hit hit = {};
        uint status = query.CommittedStatus();
        hit.found = status != COMMITTED_NOTHING;
        hit.t = query.CommittedRayT();
        // A committed instance's values are read only once there is one: lavapipe faults on them
        // after a miss.
        if (hit.found) {
            hit.instance = query.CommittedInstanceIndex();
            hit.primitive = query.CommittedPrimitiveIndex();
            hit.geometry = query.CommittedInstanceID();
            if (status == COMMITTED_TRIANGLE_HIT) {
                hit.barycentrics = query.CommittedTriangleBarycentrics();
            }
            hit.objectNormal = nearest.normal;
            hit.objectToWorld = query.CommittedObjectToWorld3x4();
            hit.worldToObject = query.CommittedWorldToObject3x4();
        }
        return hit;
    }

    /// @return Whether anything in `mask` blocks the ray, for visibility.
    public bool TraceAny(RayDesc ray, uint mask) {
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
        ShapeHit nearest;
        Run(query, ray, mask, nearest);
        return query.CommittedStatus() != COMMITTED_NOTHING;
    }

    /// @return The surface's geometric normal at a hit, in the scene's frame, facing out of an
    /// implicit shape or by a triangle's winding.
    public float3 WorldNormal(Hit hit) {
        float3 normal = hit.objectNormal;
        GeometryRecord record = parameters.records[hit.geometry];
        if (record.kind == kTriangles) {
            float3 p[3];
            Corners(record, hit.primitive, p);
            normal = cross(p[1] - p[0], p[2] - p[0]);
        }
        // Normals take the inverse transpose: a row vector times the world-to-object matrix.
        return normalize(mul(normal, (float3x3)hit.worldToObject));
    }

    /// Runs the query; `Flags` may end it at the first hit. NVIDIA's compiler fails on
    /// `CandidateObjectRay*`: the object ray comes from the candidate's matrix instead.
    void Run<let Flags : int>(inout RayQuery<Flags> query, RayDesc ray, uint mask,
                              out ShapeHit nearest) {
        query.TraceRayInline(RaytracingAccelerationStructure(parameters.scene), RAY_FLAG_NONE,
                             mask, ray);
        nearest.t = ray.TMax;
        nearest.normal = float3(0, 0, 1);
        while (query.Proceed()) {
            if (query.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE) {
                continue;
            }
            GeometryRecord record = parameters.records[query.CandidateInstanceID()];
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
}
