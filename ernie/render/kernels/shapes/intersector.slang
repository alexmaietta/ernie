// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.scene.records.render_tables;
__exported import ernie.render.kernels.shapes.box;
__exported import ernie.render.kernels.shapes.capsule;
__exported import ernie.render.kernels.shapes.cone;
__exported import ernie.render.kernels.shapes.cylinder;
__exported import ernie.render.kernels.shapes.plane;
__exported import ernie.render.kernels.shapes.sphere;
__exported import ernie.render.kernels.trace.ray;
import ernie.device.kernels.device;

using namespace ernie;

ShapeHit CenteredShapeHit(ShapeRecord record, Ray ray) {
    switch (ShapeKind(record.kind)) {
    case ShapeKind::Sphere:
        return Sphere(1).Intersect(ray);
    case ShapeKind::Plane:
        return Plane(2, 2).Intersect(ray);
    case ShapeKind::Cylinder:
        return Cylinder(1, 2).Intersect(ray);
    case ShapeKind::Cone:
        return Cone(1, 2).Intersect(ray);
    case ShapeKind::Disk: {
        ShapeHit hit = CrossZ(ray);
        const float2 at = detach(ray.origin.xy) + detach(hit.t) * detach(ray.direction.xy);
        hit.inside = hit.inside && dot(at, at) <= 1;
        return hit;
    }
    case ShapeKind::Tube:
        return Cylinder(1, 2).Intersect(ray, false);
    case ShapeKind::Capsule: {
        const float stretch = record.param;
        Ray local = ray;
        local.origin.z *= stretch;
        local.direction.z *= stretch;
        ShapeHit hit = Capsule(1, 2 * (stretch - 1)).Intersect(local);
        hit.normal.z *= stretch;
        return hit;
    }
    default:
        return Box(float3(1)).Intersect(ray);
    }
}

ShapeHit UnitShapeHit(ShapeRecord record, Ray ray) {
    const float closest = -dot(ray.origin, ray.direction) / dot(ray.direction, ray.direction);
    Ray centered = ray;
    centered.origin = ray.origin + closest * ray.direction;
    centered.tMin = ray.tMin - closest;
    ShapeHit hit = CenteredShapeHit(record, centered);
    hit.t += closest;
    return hit;
}

#if !ERNIE_VULKAN
[noinline]
#endif
ShapeHit IntersectUnitShape(ShapeRecord record, Ray ray) {
    return UnitShapeHit(record, ray);
}

#if !ERNIE_VULKAN
[noinline]
#endif
float UnitShapeDistance(ShapeRecord record, Ray ray) {
    const ShapeHit hit = UnitShapeHit(record, ray);
    return hit.inside ? hit.t : kInfinity;
}

struct ShapeIntersector : IIntersector {
    Ptr<ShapeRecord> shapes;
    uint first;

    bool Intersect(Ray ray, uint instance, uint geometry, uint primitive, inout float t,
                   out float2 attributes) {
        attributes = float2(0);
        const float distance = UnitShapeDistance(shapes[first + instance], ray);
        if (distance >= t) {
            return false;
        }
        t = distance;
        return true;
    }

    [Differentiable]
    [NoDiffThis]
    Hit Reintersect(Ray ray, no_diff Hit found) {
        return found;
    }
}
