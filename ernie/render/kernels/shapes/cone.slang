// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.shapes.ishape;
import ernie.device.kernels.device;

struct Cone : IShape {
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
        const float half = height / 2;
        const float k = radius / height;
        const float k2 = k * k;
        const float3 o = ray.origin;
        const float3 d = ray.direction;
        const float apart = half - o.z;
        const float a = d.x * d.x + d.y * d.y - k2 * d.z * d.z;
        const float b = o.x * d.x + o.y * d.y + k2 * apart * d.z;
        const float c = o.x * o.x + o.y * o.y - k2 * apart * apart;
        const float discriminant = b * b - a * c;
        const float root = sqrt(max(discriminant, 1e-30f));
        const float q = detach(b) < 0 ? root - b : -root - b;
        const float first = q / a;
        const float second = c / q;
        const float base = (-half - o.z) / d.z;

        float nearest = kInfinity;
        float3 normal = float3(0, 0, -1);
        [ForceUnroll]
        for (uint i = 0; i < 2; ++i) {
            const float t = i == 0 ? first : second;
            const float z = detach(o.z) + detach(t) * detach(d.z);
            if (detach(discriminant) >= 0 && detach(t) >= ray.tMin && detach(t) < detach(nearest) &&
                z >= -detach(half) && z <= detach(half)) {
                nearest = t;
                const float3 p = o + t * d;
                normal = normalize(float3(p.x, p.y, k2 * (half - p.z)));
            }
        }
        const float2 onBase = detach(o.xy) + detach(base) * detach(d.xy);
        if (detach(base) >= ray.tMin && detach(base) < detach(nearest) &&
            dot(onBase, onBase) <= detach(radius) * detach(radius)) {
            nearest = base;
            normal = float3(0, 0, -1);
        }
        ShapeHit hit;
        hit.t = nearest;
        hit.normal = normal;
        hit.inside = isfinite(detach(nearest));
        return hit;
    }

    float3 Normal(float3 position) {
        if (position.z <= -height / 2 + 1e-5f * height) {
            return float3(0, 0, -1);
        }
        const float k = radius / height;
        const float3 gradient = float3(position.xy, k * k * (height / 2 - position.z));
        return dot(gradient, gradient) > 0 ? normalize(gradient) : float3(0, 0, 1);
    }

    ShapeSt St(float3 position, float3 normal) {
        ShapeSt st;
        if (normal.z < -0.99999f && position.z <= -height / 2 + 1e-6f * height) {
            st.st = position.xy / (2 * radius) + 0.5f;
            st.dpds = float3(2 * radius, 0, 0);
            st.dpdt = float3(0, 2 * radius, 0);
            return st;
        }
        const float ring = max(length(position.xy), 1e-6f * radius);
        st.st = float2(Around(position), position.z / height + 0.5f);
        st.dpds = 2 * kPi * float3(-position.y, position.x, 0);
        st.dpdt = float3(-radius * position.x / ring, -radius * position.y / ring, height);
        return st;
    }
}
