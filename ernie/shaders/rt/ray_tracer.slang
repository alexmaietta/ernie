// SPDX-License-Identifier: BSD-3-Clause
/// @file ray_tracer.slang
/// @brief `RayTracer`, what every pass traces the scene through: inline ray queries against the
/// top level, each procedural candidate solved exactly as its unit shape.
///
///     import ray_tracer;
///     RayTracer tracer = RayTracer(parameters.tracer);
///     Hit hit = tracer.TraceClosest(ray, 0xff);
///     if (hit.found) { float3 n = tracer.WorldNormal(hit); }
///
/// What a pass hands its tracer is shared with C++ (rt/ray_tracer.h).
#ifdef __cplusplus
#pragma once
#include "ernie/shaders/scene/geometry.slang"
namespace ernie::shaders {
#else
import geometry;
import buffer_ref;
import shapes;
#endif

/// What a pass hands its tracer.
struct TracerParameters {
    /// The top level's device address.
    uint64_t scene;
    BufferRef<GeometryRecord> records;
};

#ifdef __cplusplus
}  // namespace ernie::shaders
#else

/// What a ray hit: where along it, which instance, primitive and geometry, and the transform
/// from the scene's frame to the instance's. Surfaces are reconstructed from it.
struct Hit {
    bool found;
    float t;
    /// The instance's row: `InstanceIndex()`.
    uint instance;
    uint primitive;
    uint geometry;
    float2 barycentrics;
    /// An implicit shape's outward normal, in its own frame.
    float3 objectNormal;
    float3x4 worldToObject;
};

struct RayTracer {
    TracerParameters parameters;

    __init(TracerParameters parameters) {
        this.parameters = parameters;
    }

    /// @return The nearest hit along the ray among instances in `mask`.
    Hit TraceClosest(RayDesc ray, uint mask) {
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
            hit.worldToObject = query.CommittedWorldToObject3x4();
        }
        return hit;
    }

    /// @return Whether anything in `mask` blocks the ray, for visibility.
    bool TraceAny(RayDesc ray, uint mask) {
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
        ShapeHit nearest;
        Run(query, ray, mask, nearest);
        return query.CommittedStatus() != COMMITTED_NOTHING;
    }

    /// @return The surface's geometric normal at a hit, in the scene's frame, facing out of an
    /// implicit shape or by a triangle's winding.
    float3 WorldNormal(Hit hit) {
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

#endif
