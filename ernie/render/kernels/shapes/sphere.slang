// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.shapes.ishape;
import ernie.device.kernels.device;

struct Sphere : IShape {
    float radius;

    [Differentiable]
    __init(float radius) {
        this.radius = radius;
    }

    float3 Extents() {
        return float3(radius);
    }

    [Differentiable]
    ShapeHit Intersect(Ray ray) {
        const float a = dot(ray.direction, ray.direction);
        const float b = dot(ray.origin, ray.direction);
        const float c = dot(ray.origin, ray.origin) - radius * radius;
        const float3 toNearest = ray.origin - (b / a) * ray.direction;
        const float discriminant = a * (radius * radius - dot(toNearest, toNearest));
        const float root = sqrt(max(discriminant, 1e-30f));
        const float q = detach(b) < 0 ? root - b : -root - b;
        const float first = q / a;
        const float second = c / q;
        const bool ordered = detach(first) < detach(second);
        const float nearer = ordered ? first : second;
        const float farther = ordered ? second : first;
        ShapeHit hit;
        hit.t = detach(nearer) < ray.tMin ? farther : nearer;
        hit.normal = (ray.origin + hit.t * ray.direction) / radius;
        hit.inside = detach(discriminant) >= 0 && detach(hit.t) >= ray.tMin;
        return hit;
    }

    float3 Normal(float3 position) {
        return normalize(position);
    }

    ShapeSt St(float3 position, float3 normal) {
        const float ring = max(length(position.xy), 1e-6f * radius);
        ShapeSt st;
        st.st = float2(Around(position), acos(clamp(-position.z / radius, -1.0f, 1.0f)) / kPi);
        st.dpds = 2 * kPi * float3(-position.y, position.x, 0);
        st.dpdt = kPi * float3(-position.z * position.x / ring, -position.z * position.y / ring,
                               ring);
        return st;
    }
}
