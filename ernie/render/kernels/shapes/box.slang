// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.shapes.ishape;

struct Box : IShape {
    float3 halfExtents;

    [Differentiable]
    __init(float3 halfExtents) {
        this.halfExtents = halfExtents;
    }

    float3 Extents() {
        return halfExtents;
    }

    [Differentiable]
    ShapeHit Intersect(Ray ray) {
        const float3 inverse = 1 / ray.direction;
        const float3 a = (-halfExtents - ray.origin) * inverse;
        const float3 b = (halfExtents - ray.origin) * inverse;
        const float3 near = min(a, b);
        const float3 far = max(a, b);
        const float enter = max(max(near.x, near.y), near.z);
        const float leave = min(min(far.x, far.y), far.z);
        ShapeHit hit;
        hit.t = detach(enter) < ray.tMin ? leave : enter;
        hit.normal = no_diff Normal(detach(ray.origin) + detach(hit.t) * detach(ray.direction));
        hit.inside = detach(enter) <= detach(leave) && detach(hit.t) >= ray.tMin &&
                     isfinite(detach(hit.t));
        return hit;
    }

    float3 Normal(float3 position) {
        const float3 out = abs(position) / halfExtents;
        const float3 sign = float3(position.x < 0 ? -1 : 1, position.y < 0 ? -1 : 1,
                                   position.z < 0 ? -1 : 1);
        return out.x >= out.y && out.x >= out.z ? float3(sign.x, 0, 0)
               : out.y >= out.z                 ? float3(0, sign.y, 0)
                                                : float3(0, 0, sign.z);
    }

    ShapeSt St(float3 position, float3 normal) {
        const uint axis = abs(normal.x) > 0.5f ? 0 : abs(normal.y) > 0.5f ? 1 : 2;
        const uint sAxis = (axis + 1) % 3;
        const uint tAxis = (axis + 2) % 3;
        ShapeSt st;
        st.st = float2(position[sAxis] / halfExtents[sAxis], position[tAxis] / halfExtents[tAxis]) *
                    0.5f +
                0.5f;
        st.dpds = float3(0);
        st.dpdt = float3(0);
        st.dpds[sAxis] = 2 * halfExtents[sAxis];
        st.dpdt[tAxis] = 2 * halfExtents[tAxis];
        return st;
    }
}
