// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.shapes.ishape;
import ernie.device.kernels.device;

[Differentiable]
ShapeHit CrossZ(Ray ray) {
    ShapeHit hit;
    hit.t = -ray.origin.z / ray.direction.z;
    hit.normal = float3(0, 0, 1);
    hit.inside = detach(hit.t) >= ray.tMin && isfinite(detach(hit.t));
    return hit;
}

struct Plane : IShape {
    float width;
    float length;

    [Differentiable]
    __init(float width, float length) {
        this.width = width;
        this.length = length;
    }

    float3 Extents() {
        return float3(width / 2, length / 2, 0);
    }

    [Differentiable]
    ShapeHit Intersect(Ray ray) {
        ShapeHit hit = CrossZ(ray);
        const float2 at = detach(ray.origin.xy) + detach(hit.t) * detach(ray.direction.xy);
        hit.inside =
            hit.inside && abs(at.x) <= detach(width) / 2 && abs(at.y) <= detach(length) / 2;
        return hit;
    }

    float3 Normal(float3 position) {
        return float3(0, 0, 1);
    }

    ShapeSt St(float3 position, float3 normal) {
        ShapeSt st;
        st.st = position.xy / float2(width, length) + 0.5f;
        st.dpds = float3(width, 0, 0);
        st.dpdt = float3(0, length, 0);
        return st;
    }
}

static const float kInfinitePlaneExtent = 1e18f;

struct InfinitePlane : IShape {
    float3 Extents() {
        return float3(kInfinitePlaneExtent, kInfinitePlaneExtent, 0);
    }

    [Differentiable]
    ShapeHit Intersect(Ray ray) {
        return CrossZ(ray);
    }

    float3 Normal(float3 position) {
        return float3(0, 0, 1);
    }

    ShapeSt St(float3 position, float3 normal) {
        ShapeSt st;
        st.st = position.xy;
        st.dpds = float3(1, 0, 0);
        st.dpdt = float3(0, 1, 0);
        return st;
    }
}
