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

float3 MeshShadingNormal(SceneData* data, ObjectSpace space, uint prim, SurfaceElement element) {
    const uint meshRow = PrimRow(data, prim, kSlotMesh);
    const uint pointsRow = PrimRow(data, prim, kSlotPointBased);
    const ArrayRef authored = data->schemas.pointBased.normals[pointsRow];
    const bool subdivided =
        data->schemas.mesh.subdivisionScheme[meshRow] != uint(SubdivisionScheme::None);
    if (!subdivided && authored.count > 0) {
        const uint3 at = InterpolationIndices(
            element, Interpolation(data->schemas.pointBased.normalsInterpolation[pointsRow]));
        if (all(at < authored.count)) {
            return space.NormalToWorld(
                Interpolate(Ptr<Float3>(authored.address), at, element.weights));
        }
    }
    const Ptr<Float3> smooth = Ptr<Float3>(data->schemas.mesh.smoothNormals[meshRow].address);
    return space.NormalToWorld(Interpolate(smooth, element.points, element.weights));
}

SurfacePoint ComputeSurfacePoint(HitQuery query, Ray ray, PrimHit primHit, float spread) {
    SceneData* data = query.data;
    const uint prim = primHit.prim;
    const ObjectSpace space = query.LoadObjectSpace(primHit);
    SurfacePoint point;
    point.prim = prim;
    point.slot = primHit.slot;
    point.distance = primHit.hit.t;
    point.position = ray.origin + primHit.hit.t * ray.direction;
    point.element = FindSurfaceElement(data, prim, primHit.hit);

    if (point.element.mesh) {
        const uint pointsRow = PrimRow(data, prim, kSlotPointBased);
        const Ptr<Float3> positions =
            Ptr<Float3>(data->schemas.pointBased.points[pointsRow].address);
        const uint3 corners = point.element.points;
        const float3 a = positions[corners.x].Vector();
        const float3 b = positions[corners.y].Vector();
        const float3 c = positions[corners.z].Vector();
        point.geometricNormal = space.NormalToWorld(cross(b - a, c - a));
        point.shadingNormal = MeshShadingNormal(data, space, prim, point.element);
    } else {
        const ShapeHit hit = IntersectUnitShape(primHit.shape, space.ToObject(ray));
        point.geometricNormal = space.NormalToWorld(hit.normal);
        point.shadingNormal = point.geometricNormal;
    }
    if (dot(point.geometricNormal, ray.direction) > 0) {
        point.geometricNormal = -point.geometricNormal;
    }
    if (dot(point.shadingNormal, point.geometricNormal) < 0) {
        point.shadingNormal = -point.shadingNormal;
    }

    point.footprint = spread * primHit.hit.t;
    const float cosine = max(abs(dot(ray.direction, point.geometricNormal)), 0.05f);
    const float3 across =
        ray.direction - dot(ray.direction, point.geometricNormal) * point.geometricNormal;
    point.stretch = length(across) > 1e-6f
                        ? normalize(across) * point.footprint / cosine
                        : Basis(point.geometricNormal).tangent * point.footprint;
    return point;
}
