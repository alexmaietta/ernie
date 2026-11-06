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
