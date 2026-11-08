// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.trace.itracer;

RaytracingAccelerationStructure Accel(uint64_t address) {
    return spirv_asm {
        result:$$RaytracingAccelerationStructure = OpConvertUToAccelerationStructureKHR $address
    };
}

RayDesc ToRayDesc(Ray ray) {
    RayDesc desc;
    desc.Origin = ray.origin;
    desc.TMin = ray.tMin;
    desc.Direction = ray.direction;
    desc.TMax = ray.tMax;
    return desc;
}

struct HardwareTracer : ITracer {
    static Hit Closest<F : IHitFilter, I : IIntersector>(uint64_t tlas, Ray ray, uint mask,
                                                         F filter, I intersector) {
        return Trace(tlas, ray, mask, filter, intersector, false);
    }

    static bool Any<F : IHitFilter, I : IIntersector>(uint64_t tlas, Ray ray, uint mask, F filter,
                                                      I intersector) {
        return Trace(tlas, ray, mask, filter, intersector, true).IsHit();
    }

    static Hit Trace<F : IHitFilter, I : IIntersector>(uint64_t tlas, Ray ray, uint mask,
                                                       F filter, I intersector, bool any) {
        const uint flags = (filter.Opaque() ? RAY_FLAG_FORCE_OPAQUE : RAY_FLAG_FORCE_NON_OPAQUE) |
                           (any ? RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH : 0) |
                           ((mask & kTraceCullBackFaces) != 0 ? RAY_FLAG_CULL_BACK_FACING_TRIANGLES : 0);
        RayQuery<F.kRayFlags> query;
        query.TraceRayInline(Accel(tlas), flags, mask & kTraceInstanceMask, ToRayDesc(ray));
        float2 attributes = float2(0);
        while (query.Proceed()) {
            Ray local = ray;
            local.origin = query.CandidateObjectRayOrigin();
            local.direction = query.CandidateObjectRayDirection();
            const uint instance = query.CandidateInstanceID();
            const uint geometry = query.CandidateGeometryIndex();
            const uint primitive = query.CandidatePrimitiveIndex();

            if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                Hit candidate = Miss(ray);
                candidate.t = query.CandidateTriangleRayT();
                candidate.barycentrics = query.CandidateTriangleBarycentrics();
                candidate.instance = instance;
                candidate.geometry = geometry;
                candidate.primitive = primitive;
                if (!filter.Accept(local, candidate)) {
                    continue;
                }
                query.CommitNonOpaqueTriangleHit();
            } else {
                const float nearest = query.CommittedStatus() == COMMITTED_NOTHING
                                          ? ray.tMax
                                          : query.CommittedRayT();
                float t = nearest;
                float2 found;
                if (!intersector.Intersect(local, instance, geometry, primitive, t, found) ||
                    t < ray.tMin || t >= nearest) {
                    continue;
                }
                if (!F.AcceptsAll()) {
                    Hit candidate = Miss(ray);
                    candidate.t = t;
                    candidate.barycentrics = found;
                    candidate.instance = instance;
                    candidate.geometry = geometry;
                    candidate.primitive = primitive;
                    candidate.procedural = true;
                    if (!filter.Accept(local, candidate)) {
                        continue;
                    }
                }
                query.CommitProceduralPrimitiveHit(t);
                attributes = found;
            }
            if (any) {
                query.Abort();
            }
        }
        return Committed(query, ray, attributes);
    }

    static Hit Committed<let Flags : uint>(RayQuery<Flags> query, Ray ray, float2 attributes) {
        Hit hit = Miss(ray);
        const uint status = query.CommittedStatus();
        if (status == COMMITTED_NOTHING) {
            return hit;
        }
        hit.t = query.CommittedRayT();
        hit.instance = query.CommittedInstanceID();
        hit.geometry = query.CommittedGeometryIndex();
        hit.primitive = query.CommittedPrimitiveIndex();
        hit.procedural = status == COMMITTED_PROCEDURAL_PRIMITIVE_HIT;
        hit.barycentrics = hit.procedural ? attributes : query.CommittedTriangleBarycentrics();
        return hit;
    }
}
