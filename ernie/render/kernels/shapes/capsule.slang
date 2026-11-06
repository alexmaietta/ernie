// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.shapes.ishape;
import ernie.device.kernels.device;

struct Capsule : IShape {
    float radius;
    float height;

    [Differentiable]
    __init(float radius, float height) {
        this.radius = radius;
        this.height = height;
    }

    float3 Extents() {
        return float3(radius, radius, height / 2 + radius);
    }

    [Differentiable]
    ShapeHit Intersect(Ray ray) {
        const float half = height / 2;
        const float3 o = ray.origin;
        const float3 d = ray.direction;

        float nearest = kInfinity;
        float3 normal = float3(0, 0, 1);
        const float a = d.x * d.x + d.y * d.y;
        const float b = o.x * d.x + o.y * d.y;
        const float c = o.x * o.x + o.y * o.y - radius * radius;
        const float discriminant = b * b - a * c;
        const float root = sqrt(max(discriminant, 1e-30f));
        const float q = detach(b) < 0 ? root - b : -root - b;
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
        for (uint end = 0; end < 2; ++end) {
            const float side = end == 0 ? 1.0f : -1.0f;
            const float3 center = float3(0, 0, side * half);
            const float3 from = o - center;
            const float sa = dot(d, d);
            const float sb = dot(from, d);
            const float sc = dot(from, from) - radius * radius;
            const float sd = sb * sb - sa * sc;
            const float sroot = sqrt(max(sd, 1e-30f));
            const float sq = detach(sb) < 0 ? sroot - sb : -sroot - sb;
            [ForceUnroll]
            for (uint i = 0; i < 2; ++i) {
                const float t = i == 0 ? sq / sa : sc / sq;
                const float z = detach(o.z) + detach(t) * detach(d.z);
                if (detach(sd) >= 0 && detach(t) >= ray.tMin && detach(t) < detach(nearest) &&
                    side * z >= detach(half)) {
                    nearest = t;
                    normal = (o + t * d - center) / radius;
                }
            }
        }
        ShapeHit hit;
        hit.t = nearest;
        hit.normal = normal;
        hit.inside = isfinite(detach(nearest));
        return hit;
    }

    float3 Normal(float3 position) {
        const float half = height / 2;
        const float3 axis = float3(0, 0, clamp(position.z, -half, half));
        const float3 out = position - axis;
        return dot(out, out) > 0 ? normalize(out) : float3(0, 0, position.z < 0 ? -1 : 1);
    }

    ShapeSt St(float3 position, float3 normal) {
        const float length = height + 2 * radius;
        ShapeSt st;
        st.st = float2(Around(position), (position.z + height / 2 + radius) / length);
        st.dpds = 2 * kPi * float3(-position.y, position.x, 0);
        st.dpdt = float3(0, 0, length);
        return st;
    }
}
