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
import ernie.render.kernels.random;
import ernie.render.kernels.surface.primvars;

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

Ray Centered(Ray ray, out float closest) {
    closest = -dot(ray.origin, ray.direction) / dot(ray.direction, ray.direction);
    Ray centered = ray;
    centered.origin = ray.origin + closest * ray.direction;
    centered.tMin = ray.tMin - closest;
    return centered;
}

ShapeHit PointHit(PointsRecord cloud, uint point, Ray ray) {
    ray.origin -= Ptr<Float3>(cloud.points)[point].Vector();
    float closest;
    const Ray centered = Centered(ray, closest);
    const float radius = PointRadius(cloud, point);
    float3 normal = float3(0);
    if (cloud.normalCount > 0) {
        normal = Ptr<Float3>(cloud.normals)[cloud.normalCount == cloud.pointCount ? point : 0].Vector();
    }
    ShapeHit hit;
    if (dot(normal, normal) > 0) {
        hit.normal = normalize(normal);
        const float facing = dot(centered.direction, hit.normal);
        hit.t = -dot(centered.origin, hit.normal) / facing;
        const float3 at = centered.origin + hit.t * centered.direction;
        hit.inside = facing != 0 && hit.t >= centered.tMin && dot(at, at) <= radius * radius;
    } else {
        hit = Sphere(radius).Intersect(centered);
    }
    hit.t += closest;
    return hit;
}

ShapeHit LocalShapeHit(ShapeRecord record, Ptr<PointsRecord> points, uint primitive, Ray ray) {
    if (ShapeKind(record.kind) == ShapeKind::Points) {
        return PointHit(points[record.row], primitive, ray);
    }
    float closest;
    ShapeHit hit = CenteredShapeHit(record, Centered(ray, closest));
    hit.t += closest;
    return hit;
}

#if !ERNIE_VULKAN
[noinline]
#endif
ShapeHit IntersectShape(ShapeRecord record, Ptr<PointsRecord> points, uint primitive, Ray ray) {
    return LocalShapeHit(record, points, primitive, ray);
}

#if !ERNIE_VULKAN
[noinline]
#endif
float ShapeDistance(ShapeRecord record, Ptr<PointsRecord> points, uint primitive, Ray ray) {
    const ShapeHit hit = LocalShapeHit(record, points, primitive, ray);
    return hit.inside ? hit.t : kInfinity;
}

bool PointBlocks(SurfaceRecord surface, uint opacityName, uint point, Ray ray) {
    float opacity = surface.displayOpacity;
    float4 authored;
    if (ReadPrimvar(FindPrimvarIn(surface.primvars, opacityName), SurfaceElement.Point(point),
                    authored)) {
        opacity = authored.x;
    }
    if (opacity >= 1) {
        return true;
    }
    uint seed = Hash(point);
    for (uint axis = 0; axis < 3; ++axis) {
        seed = Hash(seed ^ asuint(ray.origin[axis]));
        seed = Hash(seed ^ asuint(ray.direction[axis]));
    }
    return float(seed >> 8) * (1.0f / 16777216.0f) < opacity;
}

struct ShapeIntersector : IIntersector {
    Ptr<ShapeRecord> shapes;
    Ptr<PointsRecord> points;
    Ptr<SurfaceRecord> surfaces;
    uint first;
    uint opacityName;

    bool Intersect(Ray ray, uint instance, uint geometry, uint primitive, inout float t,
                   out float2 attributes) {
        attributes = float2(0);
        const uint slot = first + instance;
        const ShapeRecord shape = shapes[slot];
        if (ShapeKind(shape.kind) == ShapeKind::Points) {
            return IntersectPoints(points[shape.row], surfaces[slot], primitive, ray, t, attributes);
        }
        const float distance = ShapeDistance(shape, points, primitive, ray);
        if (distance >= t) {
            return false;
        }
        t = distance;
        return true;
    }

    bool IntersectPoints(PointsRecord cloud, SurfaceRecord surface, uint box, Ray ray,
                         inout float t, inout float2 attributes) {
        const uint perBox = max(cloud.perBox, 1u);
        const uint start = box * perBox;
        const uint end = min(start + perBox, cloud.pointCount);
        bool found = false;
        for (uint at = start; at < end; ++at) {
            const uint point = perBox == 1 ? at : Ptr<uint>(cloud.order)[at];
            const ShapeHit hit = PointHit(cloud, point, ray);
            if (!hit.inside || hit.t >= t || !PointBlocks(surface, opacityName, point, ray)) {
                continue;
            }
            t = hit.t;
            attributes = float2(asfloat(point), 0);
            found = true;
        }
        return found;
    }

    [Differentiable]
    [NoDiffThis]
    Hit Reintersect(Ray ray, no_diff Hit found) {
        return found;
    }
}
