// SPDX-License-Identifier: BSD-3-Clause
/// @file shapes.slang
/// @brief The unit shapes intersected exactly in their own frame, against a ray o + t d whose
/// direction need not be unit length. The sphere's roots follow Haines et al., "Precision
/// Improvements for Ray/Sphere Intersection", Ray Tracing Gems ch. 7, 2019: the discriminant from
/// the ray's closest approach, and the smaller root from the larger by Vieta's formula.
import geometry;

/// The nearest hit along a ray, and the shape's outward normal there in its own frame.
public struct ShapeHit {
    public float t;
    public float3 normal;
};

/// Keeps `t` if it is in [tMin, hit.t) and nearer than what `hit` holds.
bool Take(float t, float3 normal, float tMin, inout ShapeHit hit) {
    if (t >= tMin && t < hit.t) {
        hit.t = t;
        hit.normal = normal;
        return true;
    }
    return false;
}

/// The roots of a t^2 + 2 b t + c with the discriminant given, in order.
float2 Roots(float a, float b, float c, float discriminant) {
    float q = -b - (b >= 0 ? 1 : -1) * sqrt(discriminant);
    float t0 = q / a;
    float t1 = c / q;
    return t0 < t1 ? float2(t0, t1) : float2(t1, t0);
}

bool Sphere(float3 o, float3 d, float3 centre, float radius, float tMin, inout ShapeHit hit) {
    float3 f = o - centre;
    float a = dot(d, d);
    float b = dot(f, d);
    float3 l = f - (b / a) * d;
    float discriminant = a * (radius * radius - dot(l, l));
    if (discriminant < 0) {
        return false;
    }
    float2 t = Roots(a, b, dot(f, f) - radius * radius, discriminant);
    bool found = false;
    for (uint i = 0; i < 2; ++i) {
        found = Take(t[i], (o + t[i] * d - centre) / radius, tMin, hit) || found;
    }
    return found;
}

bool Cube(float3 o, float3 d, float tMin, inout ShapeHit hit) {
    float3 inverse = 1 / d;
    float3 t0 = (-1 - o) * inverse;
    float3 t1 = (1 - o) * inverse;
    float3 near = min(t0, t1);
    float3 far = max(t0, t1);
    float entry = max(max(near.x, near.y), near.z);
    float exit = min(min(far.x, far.y), far.z);
    if (entry > exit) {
        return false;
    }
    // The face entered, or the one left from inside.
    bool inside = entry < tMin;
    float t = inside ? exit : entry;
    float3 at = inside ? far : near;
    float3 axis = at.x == t ? float3(1, 0, 0) : at.y == t ? float3(0, 1, 0) : float3(0, 0, 1);
    return Take(t, axis * sign(o + t * d), tMin, hit);
}

/// The disc z = height, x^2 + y^2 <= 1, facing `facing`.
bool Disc(float3 o, float3 d, float height, float facing, float tMin, inout ShapeHit hit) {
    float t = (height - o.z) / d.z;
    float2 at = o.xy + t * d.xy;
    return dot(at, at) <= 1 && Take(t, float3(0, 0, facing), tMin, hit);
}

/// The unit cylinder's side, with z in [low, high].
bool Side(float3 o, float3 d, float low, float high, float tMin, inout ShapeHit hit) {
    float a = dot(d.xy, d.xy);
    float b = dot(o.xy, d.xy);
    float c = dot(o.xy, o.xy) - 1;
    float discriminant = b * b - a * c;
    if (a == 0 || discriminant < 0) {
        return false;
    }
    float2 t = Roots(a, b, c, discriminant);
    bool found = false;
    for (uint i = 0; i < 2; ++i) {
        float3 p = o + t[i] * d;
        if (p.z >= low && p.z <= high) {
            found = Take(t[i], float3(p.xy, 0), tMin, hit) || found;
        }
    }
    return found;
}

/// Apex at z = 1, a base of radius 1 at z = -1: x^2 + y^2 = k^2 (1 - z)^2 with k = 1/2.
bool Cone(float3 o, float3 d, float tMin, inout ShapeHit hit) {
    const float k2 = 0.25;
    float w = 1 - o.z;
    float a = dot(d.xy, d.xy) - k2 * d.z * d.z;
    float b = dot(o.xy, d.xy) + k2 * w * d.z;
    float c = dot(o.xy, o.xy) - k2 * w * w;
    bool found = false;
    float discriminant = b * b - a * c;
    if (a != 0 && discriminant >= 0) {
        float2 t = Roots(a, b, c, discriminant);
        for (uint i = 0; i < 2; ++i) {
            float3 p = o + t[i] * d;
            if (p.z >= -1 && p.z <= 1) {
                found = Take(t[i], float3(p.xy, k2 * (1 - p.z)), tMin, hit) || found;
            }
        }
    }
    return Disc(o, d, -1, -1, tMin, hit) || found;
}

/// A unit-radius capsule about the segment z in [-halfLength, halfLength].
bool Capsule(float3 o, float3 d, float halfLength, float tMin, inout ShapeHit hit) {
    bool found = Side(o, d, -halfLength, halfLength, tMin, hit);
    found = Sphere(o, d, float3(0, 0, halfLength), 1, tMin, hit) || found;
    return Sphere(o, d, float3(0, 0, -halfLength), 1, tMin, hit) || found;
}

/// The unit square in z = 0, facing +z.
bool Plane(float3 o, float3 d, float tMin, inout ShapeHit hit) {
    float t = -o.z / d.z;
    float2 at = o.xy + t * d.xy;
    return all(abs(at) <= 1) && Take(t, float3(0, 0, 1), tMin, hit);
}

/// @return Whether the ray meets a unit shape nearer than `hit.t` and from `tMin`; the hit then
/// holds the distance and the outward normal, both in the shape's frame.
public bool IntersectShape(uint kind, float halfLength, float3 o, float3 d, float tMin,
                           inout ShapeHit hit) {
    // Each hit is taken only when nearer, so a shape tried in pieces keeps its nearest.
    if (kind == kSphere) {
        return Sphere(o, d, float3(0), 1, tMin, hit);
    }
    if (kind == kCube) {
        return Cube(o, d, tMin, hit);
    }
    if (kind == kCylinder) {
        bool found = Side(o, d, -1, 1, tMin, hit);
        found = Disc(o, d, 1, 1, tMin, hit) || found;
        return Disc(o, d, -1, -1, tMin, hit) || found;
    }
    if (kind == kCone) {
        return Cone(o, d, tMin, hit);
    }
    if (kind == kCapsule) {
        return Capsule(o, d, halfLength, tMin, hit);
    }
    if (kind == kPlane) {
        return Plane(o, d, tMin, hit);
    }
    return false;
}
