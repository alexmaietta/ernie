// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.trace.itracer;

#include "ernie/bvh/kernels/bvh.h"

using namespace ernie;

struct TraceQuery<F : IHitFilter, I : IIntersector> {
    Ray ray;
    uint mask;
    F filter;
    I intersector;
}

static const uint kTraceStackSize = 32;
static const uint kRecordEntry = 1u << 31;
static const uint kTopLevel = ~0u;

struct TraceStack {
    uint entries[kTraceStackSize];

    [mutating]
    void Push(inout uint depth, uint node, uint children) {
        if (children != 0 && depth < kTraceStackSize) {
            entries[depth] = node << 8 | children;
            ++depth;
        }
    }

    [mutating]
    void PushRecords(inout uint depth, uint first, uint count) {
        if (count != 0 && depth < kTraceStackSize) {
            entries[depth] = kRecordEntry | first << 2 | count;
            ++depth;
        }
    }

    [mutating]
    void Pop(inout uint depth, uint4* bvhNodes, uint octant, out uint child) {
        const uint entry = entries[depth - 1];
        const uint parent = entry >> 8;
        const uint left = entry & 0xFF;
        if ((left & (left - 1)) == 0) {
            --depth;
        } else {
            entries[depth - 1] = entry & ~(left & -left);
        }
        const uint slot = firstbitlow(left) ^ octant;
        const uint imask = bvhNodes[parent * 5].w >> 24;
        child = bvhNodes[parent * 5 + 1].x + countbits(imask & ((1u << slot) - 1));
    }
}

uint NodeByte(uint low, uint high, uint slot) {
    return ((slot < 4 ? low : high) >> ((slot & 3) * 8)) & 0xFF;
}

float NodeScale(uint exponents, uint axis) {
    return asfloat(((exponents >> (axis * 8)) & 0xFF) << 23);
}

uint SwapSlots(uint mask, uint octant) {
    mask = (octant & 1) != 0 ? ((mask & 0x55) << 1) | ((mask >> 1) & 0x55) : mask;
    mask = (octant & 2) != 0 ? ((mask & 0x33) << 2) | ((mask >> 2) & 0x33) : mask;
    return (octant & 4) != 0 ? ((mask & 0x0F) << 4) | ((mask >> 4) & 0x0F) : mask;
}

float SafeInverse(float d) {
    return 1.0f / (abs(d) < 1e-20f ? (d < 0 ? -1e-20f : 1e-20f) : d);
}

struct BvhLevel {
    uint4* nodes;
    uint4* records;
    uint words;
    uint kind;
    float3 inverse;
    bool3 flip;
    uint octant;

    __init(BvhHeader* bvh, Ray ray) {
        nodes = (uint4*)((uint*)bvh + uint(bvh.nodes / 4));
        records = (uint4*)((uint*)bvh + uint(bvh.leaves / 4));
        words = BvhRecordWords(bvh.kind);
        kind = bvh.kind;
        inverse = float3(SafeInverse(ray.direction.x), SafeInverse(ray.direction.y),
                         SafeInverse(ray.direction.z));
        flip = inverse < 0;
        octant = (flip.x ? 4 : 0) | (flip.y ? 2 : 0) | (flip.z ? 1 : 0);
    }

    void Test(uint node, Ray ray, float far, out uint internal, out uint leaves, out uint4 q1) {
        uint4* at = nodes + node * 5;
        const uint4 q0 = loadAligned<16>(at);
        q1 = loadAligned<16>(at + 1);
        const uint4 q2 = loadAligned<16>(at + 2);
        const uint4 q3 = loadAligned<16>(at + 3);
        const uint4 q4 = loadAligned<16>(at + 4);
        const float3 scale = float3(NodeScale(q0.w, 0), NodeScale(q0.w, 1), NodeScale(q0.w, 2));
        const float3 step = scale * inverse;
        const float3 start = (asfloat(q0.xyz) - ray.origin) * inverse;
        const uint2 nearX = flip.x ? q3.zw : q2.xy;
        const uint2 farX = flip.x ? q2.xy : q3.zw;
        const uint2 nearY = flip.y ? q4.xy : q2.zw;
        const uint2 farY = flip.y ? q2.zw : q4.xy;
        const uint2 nearZ = flip.z ? q4.zw : q3.xy;
        const uint2 farZ = flip.z ? q3.xy : q4.zw;
        const uint imask = q0.w >> 24;
        uint hits = 0;
        [ForceUnroll]
        for (uint slot = 0; slot < 8; ++slot) {
            const float3 enters =
                float3(NodeByte(nearX.x, nearX.y, slot), NodeByte(nearY.x, nearY.y, slot),
                       NodeByte(nearZ.x, nearZ.y, slot)) * step + start;
            const float3 exits =
                float3(NodeByte(farX.x, farX.y, slot), NodeByte(farY.x, farY.y, slot),
                       NodeByte(farZ.x, farZ.y, slot)) * step + start;
            const float enter = max(max(enters.x, enters.y), max(enters.z, ray.tMin));
            const float exit = min(min(exits.x, exits.y), min(exits.z, far));
            const bool used = NodeByte(q1.z, q1.w, slot) != 0;
            hits |= used && enter <= exit ? 1u << slot : 0;
        }
        internal = SwapSlots(hits & imask, octant);
        leaves = SwapSlots(hits & ~imask, octant);
    }
}

BvhInstance LoadInstance(uint4* record) {
    const uint4 r0 = loadAligned<16>(record);
    const uint4 r1 = loadAligned<16>(record + 1);
    const uint4 r2 = loadAligned<16>(record + 2);
    const uint4 r3 = loadAligned<16>(record + 3);
    BvhInstance instance;
    instance.blas = uint64_t(r0.x) | uint64_t(r0.y) << 32;
    instance.toObject = { asfloat(r0.z), asfloat(r0.w), asfloat(r1.x), asfloat(r1.y),
                          asfloat(r1.z), asfloat(r1.w), asfloat(r2.x), asfloat(r2.y),
                          asfloat(r2.z), asfloat(r2.w), asfloat(r3.x), asfloat(r3.y) };
    instance.custom = r3.z;
    instance.index = r3.w;
    return instance;
}

Ray ToObject(BvhInstance instance, Ray ray) {
    const float m[12] = instance.toObject;
    Ray moved = ray;
    moved.origin = float3(dot(float3(m[0], m[1], m[2]), ray.origin) + m[3],
                          dot(float3(m[4], m[5], m[6]), ray.origin) + m[7],
                          dot(float3(m[8], m[9], m[10]), ray.origin) + m[11]);
    moved.direction = float3(dot(float3(m[0], m[1], m[2]), ray.direction),
                             dot(float3(m[4], m[5], m[6]), ray.direction),
                             dot(float3(m[8], m[9], m[10]), ray.direction));
    return moved;
}

struct BlasVisitor<F : IHitFilter, I : IIntersector> {
    TraceQuery<F, I> query;
    uint instance;
    uint facing;
    Hit nearest;
    bool any;

    [mutating]
    bool Visit(uint4* records, uint count, uint kind) {
        for (uint i = 0; i < count; ++i) {
            const bool kept = kind == kBvhBoxes
                                  ? TestBox(((uint*)records)[i * (kBvhBoxBytes / 4) + 3])
                                  : TestTriangle(records + i * (kBvhTriangleBytes / 16));
            if (kept && any) {
                return true;
            }
        }
        return false;
    }

    [mutating]
    bool TestTriangle(uint4* record) {
        const uint4 c = loadAligned<16>(record);
        const uint4 b = loadAligned<16>(record + 1);
        const uint4 a = loadAligned<16>(record + 2);
        const float3 edgeC = asfloat(c.xyz);
        const float3 edgeB = asfloat(b.xyz);
        const TriangleTest test = IntersectTriangle(query.ray, asfloat(a.xyz), edgeB, edgeC);
        if (!test.inside || test.t < query.ray.tMin || test.t >= nearest.t) {
            return false;
        }
        const bool front = test.counterclockwise != ((facing & kInstanceFlipFacing) != 0);
        if (!front && (query.mask & kTraceCullBackFaces) != 0 &&
            (facing & kInstanceFacingCullDisable) == 0) {
            return false;
        }
        const Hit candidate = Candidate(test.t, test.barycentrics, a.w, false);
        if (!F.AcceptsAll() && !query.filter.Accept(query.ray, candidate)) {
            return false;
        }
        nearest = candidate;
        return true;
    }

    [mutating]
    bool TestBox(uint index) {
        float t = nearest.t;
        float2 attributes;
        if (!query.intersector.Intersect(query.ray, instance, 0, index, t, attributes) ||
            t < query.ray.tMin || t >= nearest.t) {
            return false;
        }
        const Hit candidate = Candidate(t, attributes, index, true);
        if (!F.AcceptsAll() && !query.filter.Accept(query.ray, candidate)) {
            return false;
        }
        nearest = candidate;
        return true;
    }

    Hit Candidate(float t, float2 barycentrics, uint primitive, bool procedural) {
        Hit candidate = nearest;
        candidate.t = t;
        candidate.barycentrics = barycentrics;
        candidate.instance = instance;
        candidate.primitive = primitive;
        candidate.procedural = procedural;
        return candidate;
    }
}

struct SoftwareTracer : ITracer {
    static Hit Closest<F : IHitFilter, I : IIntersector>(uint64_t bvh, Ray ray, uint mask,
                                                         F filter, I intersector) {
        return Trace(bvh, ray, mask, filter, intersector, false);
    }

    static bool Any<F : IHitFilter, I : IIntersector>(uint64_t bvh, Ray ray, uint mask,
                                                      F filter, I intersector) {
        return Trace(bvh, ray, mask, filter, intersector, true).IsHit();
    }

    static Hit Trace<F : IHitFilter, I : IIntersector>(uint64_t accel, Ray ray, uint mask,
                                                       F filter, I intersector, bool any) {
        BvhHeader* bvh = (BvhHeader*)accel;
        const TraceQuery<F, I> query = { ray, mask, filter, intersector };
        BlasVisitor<F, I> visitor = { query, 0, 0, Miss(query.ray), any };
        if (bvh.primitives == 0 || bvh.nodeCount == 0) {
            return visitor.nearest;
        }
        BvhLevel level = BvhLevel(bvh, query.ray);
        const bool instanced = bvh.kind == kBvhInstances;
        TraceStack stack;
        uint depth = 0;
        uint floor = instanced ? kTopLevel : 0;
        uint node = 0;
        bool visiting = true;
        while (true) {
            if (visiting) {
                uint internal;
                uint leaves;
                uint4 q1;
                level.Test(node, visitor.query.ray, visitor.nearest.t, internal, leaves, q1);
                stack.Push(depth, node, internal);
                while (leaves != 0) {
                    const uint slot = firstbitlow(leaves) ^ level.octant;
                    leaves &= leaves - 1;
                    const uint meta = NodeByte(q1.z, q1.w, slot);
                    const uint first = q1.y + (meta & 31) * level.words;
                    if (floor == kTopLevel) {
                        stack.PushRecords(depth, first, countbits(meta >> 5));
                    } else if (visitor.Visit(level.records + first, countbits(meta >> 5),
                                             level.kind)) {
                        return visitor.nearest;
                    }
                }
                visiting = false;
            }
            if (instanced && floor != kTopLevel && depth == floor) {
                floor = kTopLevel;
                visitor.query.ray = query.ray;
                level = BvhLevel(bvh, query.ray);
            }
            if (depth == 0) {
                return visitor.nearest;
            }
            const uint top = depth - 1;
            const uint entry = stack.entries[top];
            if ((entry & kRecordEntry) == 0) {
                stack.Pop(depth, level.nodes, level.octant, node);
                visiting = true;
                continue;
            }
            const uint record = (entry & ~kRecordEntry) >> 2;
            if ((entry & 3) > 1) {
                stack.entries[top] = kRecordEntry | (record + level.words) << 2 | ((entry & 3) - 1);
            } else {
                --depth;
            }
            const BvhInstance instance = LoadInstance(level.records + record);
            BvhHeader* blas = (BvhHeader*)instance.blas;
            if (((instance.custom >> 24) & query.mask) == 0 || blas.primitives == 0 ||
                blas.nodeCount == 0) {
                continue;
            }
            visitor.instance = instance.custom & 0xFFFFFF;
            visitor.facing = instance.index >> 24;
            visitor.query.ray = ToObject(instance, query.ray);
            level = BvhLevel(blas, visitor.query.ray);
            floor = depth;
            node = 0;
            visiting = true;
        }
    }
}
