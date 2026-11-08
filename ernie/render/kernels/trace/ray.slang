// SPDX-License-Identifier: BSD-3-Clause

struct Ray : IDifferentiable {
    float3 origin;
    no_diff float tMin;
    float3 direction;
    no_diff float tMax;
    no_diff float time;

    [Differentiable]
    __init(float3 origin, float3 direction, no_diff float tMin, no_diff float tMax,
           no_diff float time) {
        this.origin = origin;
        this.tMin = tMin;
        this.direction = direction;
        this.tMax = tMax;
        this.time = time;
    }
}

static const uint kNoHit = 0xFFFFFFFF;
static const uint kTraceCullBackFaces = 0x100u;
static const uint kTraceInstanceMask = 0xFFu;

struct Hit : IDifferentiable {
    float t;
    float2 barycentrics;
    uint instance;
    uint geometry;
    uint primitive;
    bool procedural;

    bool IsHit() {
        return primitive != kNoHit;
    }
}

Hit Miss(Ray ray) {
    Hit miss;
    miss.t = ray.tMax;
    miss.barycentrics = float2(0);
    miss.instance = 0;
    miss.geometry = 0;
    miss.primitive = kNoHit;
    miss.procedural = false;
    return miss;
}

interface IHitFilter {
    static const uint kRayFlags;

    static bool AcceptsAll();

    bool Opaque();

    bool Accept(Ray ray, Hit candidate);
}

interface IIntersector {
    bool Intersect(Ray ray, uint instance, uint geometry, uint primitive, inout float t,
                   out float2 attributes);

    [Differentiable]
    Hit Reintersect(Ray ray, no_diff Hit found);
}

struct AcceptAll : IHitFilter {
    static const uint kRayFlags = RAY_FLAG_FORCE_OPAQUE;

    static bool AcceptsAll() {
        return true;
    }

    bool Opaque() {
        return true;
    }

    bool Accept(Ray ray, Hit candidate) {
        return true;
    }
}

struct NoIntersector : IIntersector {
    bool Intersect(Ray ray, uint instance, uint geometry, uint primitive, inout float t,
                   out float2 attributes) {
        attributes = float2(0);
        return false;
    }

    [Differentiable]
    Hit Reintersect(Ray ray, no_diff Hit found) {
        return found;
    }
}

float Gamma(uint n) {
    const float epsilon = 0.5f * 1.1920929e-7f;
    return n * epsilon / (1 - n * epsilon);
}

float NextFloatUp(float v) {
    if (isinf(v) && v > 0) {
        return v;
    }
    const uint bits = asuint(v == 0 ? 0.0f : v);
    return asfloat(v >= 0 ? bits + 1 : bits - 1);
}

float NextFloatDown(float v) {
    if (isinf(v) && v < 0) {
        return v;
    }
    const uint bits = asuint(v == 0 ? -0.0f : v);
    return asfloat(v > 0 ? bits - 1 : bits + 1);
}

float3 OffsetRayOrigin(float3 position, float3 offset) {
    float3 origin = position + offset;
    [ForceUnroll]
    for (uint i = 0; i < 3; ++i) {
        if (offset[i] > 0) {
            origin[i] = NextFloatUp(origin[i]);
        } else if (offset[i] < 0) {
            origin[i] = NextFloatDown(origin[i]);
        }
    }
    return origin;
}
