// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.query.hit;
__exported import ernie.render.kernels.surface.element;
__exported import ernie.render.kernels.surface.point;
import ernie.render.kernels.basis;

using namespace ernie;

float3 Interpolate(Ptr<Float3> values, uint3 indices, float3 weights) {
    return values[indices.x].Vector() * weights.x + values[indices.y].Vector() * weights.y +
           values[indices.z].Vector() * weights.z;
}

float3 MeshShadingNormal(SurfaceRecord surface, ObjectSpace space, SurfaceElement element) {
    if (surface.normals != 0) {
        const uint3 at = InterpolationIndices(element, Interpolation(surface.normalsInterpolation));
        if (all(at < surface.normalsCount)) {
            return space.NormalToWorld(
                Interpolate(Ptr<Float3>(surface.normals), at, element.weights));
        }
    }
    return space.NormalToWorld(Interpolate(Ptr<Float3>(surface.smooth), element.points, element.weights));
}

SurfacePoint ComputeSurfacePoint(HitQuery query, Ray ray, PrimHit primHit, float spread) {
    SceneData* data = query.scene;
    const uint prim = primHit.prim;
    const ObjectSpace space = query.LoadObjectSpace(primHit);
    SurfacePoint point;
    point.prim = prim;
    point.slot = primHit.slot;
    point.distance = primHit.hit.t;
    const SurfaceRecord surface = query.tables->surfaces[primHit.slot];
    point.element = FindSurfaceElement(data, surface, prim, primHit.hit);
    float3 error;

    if (point.element.mesh) {
        const Ptr<Float3> positions = Ptr<Float3>(surface.points);
        const uint3 corners = point.element.points;
        const float3 a = positions[corners.x].Vector();
        const float3 b = positions[corners.y].Vector();
        const float3 c = positions[corners.z].Vector();
        const float3 weights = point.element.weights;
        const float3 local = weights.x * a + weights.y * b + weights.z * c;
        point.position = space.ToWorld(local);
        const float3 largest = max(max(abs(a), abs(b)), abs(c));
        error = space.ErrorToWorld(local, Gamma(7) * max(largest.x, max(largest.y, largest.z)));
        point.geometricNormal = space.NormalToWorld(cross(b - a, c - a));
        point.shadingNormal = MeshShadingNormal(surface, space, point.element);
    } else {
        const Ray local = space.ToObject(ray);
        const ShapeHit hit = IntersectUnitShape(primHit.shape, local);
        const float3 along = hit.t * local.direction;
        const float3 at = local.origin + along;
        point.position = space.ToWorld(at);
        error = space.ErrorToWorld(at, Gamma(16) * (abs(local.origin) + abs(along)));
        point.geometricNormal = space.NormalToWorld(hit.normal);
        point.shadingNormal = point.geometricNormal;
    }
    if (dot(point.geometricNormal, ray.direction) > 0) {
        point.geometricNormal = -point.geometricNormal;
    }
    if (dot(point.shadingNormal, point.geometricNormal) < 0) {
        point.shadingNormal = -point.shadingNormal;
    }
    point.offset = dot(abs(point.geometricNormal), error);

    point.footprint = spread * primHit.hit.t;
    const float cosine = max(abs(dot(ray.direction, point.geometricNormal)), 0.05f);
    const float3 across =
        ray.direction - dot(ray.direction, point.geometricNormal) * point.geometricNormal;
    point.stretch = length(across) > 1e-6f
                        ? normalize(across) * point.footprint / cosine
                        : Basis(point.geometricNormal).tangent * point.footprint;
    return point;
}
