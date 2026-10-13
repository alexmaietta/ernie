// SPDX-License-Identifier: BSD-3-Clause
// The cube sphere's faces (`CubeSphereLod`): the equiangular mapping between a face's coordinates
// (-1 to 1) and unit directions, both ways, for kernels that walk a planet's faces outside its
// draw passes (planet.slang's `faceFrame` and `faceDirection` are the passes' own).
//
// Reference: C. Ronchi, R. Iacono and P. S. Paolucci, "The 'Cubed Sphere': A New Method for the
// Solution of Partial Differential Equations in Spherical Geometry", Journal of Computational
// Physics 124(1):93-114, 1996 (the equiangular gnomonic projection).

public struct CubeFace {
    public float3 normal;
    public float3 u;
    public float3 v;
}

// Face `face`'s outward normal and its coordinates' axes (`CubeSphereLod::FaceDirection`).
public CubeFace cubeFace(uint face) {
    let normals = float3[6](float3(1, 0, 0), float3(0, 1, 0), float3(-1, 0, 0), float3(0, -1, 0),
                            float3(0, 0, 1), float3(0, 0, -1));
    let us = float3[6](float3(0, 1, 0), float3(-1, 0, 0), float3(0, -1, 0), float3(1, 0, 0),
                       float3(0, 1, 0), float3(0, 1, 0));
    let vs = float3[6](float3(0, 0, -1), float3(0, 0, -1), float3(0, 0, -1), float3(0, 0, -1),
                       float3(1, 0, 0), float3(-1, 0, 0));
    CubeFace f;
    f.normal = normals[face];
    f.u = us[face];
    f.v = vs[face];
    return f;
}

// The unit direction of face `face`'s point `st`.
public float3 cubeDirection(uint face, float2 st) {
    let f = cubeFace(face);
    let ab = tan(0.785398163 * st);
    return normalize(f.normal + ab.x * f.u + ab.y * f.v);
}

// The face unit direction `d` lies on, and its point there (`st`, -1 to 1): `cubeDirection`'s
// inverse.
public uint cubeFaceOf(float3 d, out float2 st) {
    let a = abs(d);
    uint face;
    if (a.x >= a.y && a.x >= a.z) {
        face = d.x > 0.0 ? 0u : 2u;
    } else if (a.y >= a.z) {
        face = d.y > 0.0 ? 1u : 3u;
    } else {
        face = d.z > 0.0 ? 4u : 5u;
    }
    let f = cubeFace(face);
    let along = dot(d, f.normal);
    st = clamp(atan(float2(dot(d, f.u), dot(d, f.v)) / along) * 1.273239545, float2(-1.0),
               float2(1.0));
    return face;
}

// The point of the ellipsoid of largest radius `radius` and shape `shape` ((radius / a_i)^2 for
// its semi-axes a_i) along unit direction `d`, `height` above it along its normal, body-fixed.
public float3 ellipsoidPoint(float3 d, float height, float radius, float3 shape) {
    return radius * rsqrt(dot(d * d, shape)) * d + height * normalize(d * shape);
}
