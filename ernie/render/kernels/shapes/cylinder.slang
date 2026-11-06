// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.shapes.ishape;
import ernie.device.kernels.device;

struct Cylinder : IShape {
    float radius;
    float height;

    [Differentiable]
    __init(float radius, float height) {
        this.radius = radius;
        this.height = height;
    }

    float3 Extents() {
        return float3(radius, radius, height / 2);
    }

    [Differentiable]
    ShapeHit Intersect(Ray ray) {
        return Intersect(ray, true);
    }

    [Differentiable]
    ShapeHit Intersect(Ray ray, no_diff bool capped) {
        const float half = height / 2;
        const float3 o = ray.origin;
        const float3 d = ray.direction;
        const float a = d.x * d.x + d.y * d.y;
        const float b = o.x * d.x + o.y * d.y;
        const float c = o.x * o.x + o.y * o.y - radius * radius;
        const float discriminant = b * b - a * c;
        const float root = sqrt(max(discriminant, 1e-30f));
        const float q = detach(b) < 0 ? root - b : -root - b;

        float nearest = kInfinity;
        float3 normal = float3(0, 0, 1);
        [ForceUnroll]
        for (uint i = 0; i < 2; ++i) {
            const float t = i == 0 ? q / a : c / q;
            const float z = detach(o.z) + detach(t) * detach(d.z);
            if (detach(a) > 0 && detach(discriminant) >= 0 && detach(t) >= ray.tMin &&
                detach(t) < detach(nearest) && abs(z) <= detach(half)) {
                nearest = t;
                const float3 p = o + t * d;
                normal = float3(p.x, p.y, 0) / radius;
            }
        }
        [ForceUnroll]
        for (uint i = 0; i < 2; ++i) {
            const float side = i == 0 ? 1.0f : -1.0f;
            const float t = (side * half - o.z) / d.z;
            const float2 onCap = detach(o.xy) + detach(t) * detach(d.xy);
            if (capped && detach(t) >= ray.tMin && detach(t) < detach(nearest) &&
                dot(onCap, onCap) <= detach(radius) * detach(radius)) {
                nearest = t;
                normal = float3(0, 0, side);
            }
        }
        ShapeHit hit;
        hit.t = nearest;
        hit.normal = normal;
        hit.inside = isfinite(detach(nearest));
        return hit;
    }

    float3 Normal(float3 position) {
        const float toCap = height / 2 - abs(position.z);
        const float toSide = radius - length(position.xy);
        if (toCap <= toSide) {
            return float3(0, 0, position.z < 0 ? -1 : 1);
        }
        return length(position.xy) > 0 ? normalize(float3(position.xy, 0)) : float3(1, 0, 0);
    }

    ShapeSt St(float3 position, float3 normal) {
        ShapeSt st;
        if (abs(normal.z) > 0.99999f) {
            st.st = position.xy / (2 * radius) + 0.5f;
            st.dpds = float3(2 * radius, 0, 0);
            st.dpdt = float3(0, 2 * radius, 0);
            return st;
        }
        st.st = float2(Around(position), position.z / height + 0.5f);
        st.dpds = 2 * kPi * float3(-position.y, position.x, 0);
        st.dpdt = float3(0, 0, height);
        return st;
    }
}
