// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.trace.ray;

struct TriangleCorners : IDifferentiable {
    float3 a;
    float3 b;
    float3 c;
}

interface ITriangles {
    [Differentiable]
    TriangleCorners Corners(uint instance, uint geometry, uint primitive);
}

struct TriangleTest : IDifferentiable {
    float t;
    float2 barycentrics;
    bool inside;
}

[Differentiable]
TriangleTest IntersectTriangle(Ray ray, float3 a, float3 edgeB, float3 edgeC) {
    const float3 p = cross(ray.direction, edgeC);
    const float determinant = dot(edgeB, p);
    const float inverse = 1.0f / determinant;
    const float3 fromA = ray.origin - a;
    const float u = dot(fromA, p) * inverse;
    const float3 q = cross(fromA, edgeB);
    const float v = dot(ray.direction, q) * inverse;
    TriangleTest test;
    test.t = dot(edgeC, q) * inverse;
    test.barycentrics = float2(u, v);
    test.inside = abs(detach(determinant)) >= 1e-12f && detach(u) >= 0 && detach(v) >= 0 &&
                  detach(u) + detach(v) <= 1;
    return test;
}

[Differentiable]
Hit Reintersect<T : ITriangles>(Ray ray, no_diff Hit found, T triangles) {
    if (!found.IsHit()) {
        return found;
    }
    const TriangleCorners corners =
        triangles.Corners(found.instance, found.geometry, found.primitive);
    const TriangleTest test =
        IntersectTriangle(ray, corners.a, corners.b - corners.a, corners.c - corners.a);
    Hit hit = found;
    hit.t = test.t;
    hit.barycentrics = test.barycentrics;
    return hit;
}
