// SPDX-License-Identifier: BSD-3-Clause
// The camera projection of pushbroom cameras, loaded as module `projection` in place of
// projection_pinhole.slang: a detector line along the camera's x axis, at normalised image y
// `line`, read out row by row as the camera moves from the strip's first pose (top row) to its
// last (bottom row), its position along the line between them and its rotation along the great
// arc. A world point is seen by the row whose pose puts it in the line's plane of view, found by
// the secant method (exact in one step for straight, unturning motion); it projects there as
// through a pinhole across the row, with that pose's view depth for perspective-correct
// interpolation and the usual [0, 1] depth.
//
// The strip arrives in place of the view-projection matrix (`PushbroomStrip::Packed`): columns
// (first position, near plane), (last position, far plane), first rotation and last rotation as
// quaternions (x, y, z, w), camera to world. `strip` holds (2 f / width, 2 cx / width - 1, line,
// unused). Straight edges of triangles stay straight in the image while the projection bends
// them slightly, so large triangles land approximately.
//
// Reference: R. Gupta and R. I. Hartley, "Linear Pushbroom Cameras", IEEE TPAMI 19(9):963-975,
// 1997.

import math;

public static const bool kPushbroomCamera = true;

struct Strip {
    float3 from;
    float3 to;
    float4 fromRotation;
    float4 toRotation;
    float near;
    float far;
}

Strip stripOf(float4x4 packed) {
    // The matrix's columns as stored (glm's column-major order).
    let columns = transpose(packed);
    Strip strip;
    strip.from = columns[0].xyz;
    strip.near = columns[0].w;
    strip.to = columns[1].xyz;
    strip.far = columns[1].w;
    strip.fromRotation = columns[2];
    strip.toRotation = columns[3];
    return strip;
}

float3 unrotate(float4 q, float3 v) {
    return rotate(float4(-q.xyz, q.w), v);
}

// The rotation `s` of the way along the strip (slerp; the CPU puts both on one side).
float4 rotationAt(Strip strip, float s) {
    let cosine = clamp(dot(strip.fromRotation, strip.toRotation), -1.0, 1.0);
    let angle = acos(cosine);
    if (angle < 1e-5) {
        return normalize(lerp(strip.fromRotation, strip.toRotation, s));
    }
    return (sin((1.0 - s) * angle) * strip.fromRotation + sin(s * angle) * strip.toRotation) /
           sin(angle);
}

// How far `world` is in front of the line's plane of view at `s` of the way along the strip; the
// plane holds the camera's x axis and the line's direction (0, line, -1).
float planeOffset(Strip strip, float line, float3 world, float s) {
    let normal = rotate(rotationAt(strip, s), float3(0.0, 1.0, line));
    return dot(normal, world - lerp(strip.from, strip.to, s));
}

// The share of the way along the strip at which `world` is in the line's plane of view.
float scanTime(Strip strip, float line, float3 world) {
    var s0 = 0.0;
    var s1 = 1.0;
    var g0 = planeOffset(strip, line, world, s0);
    var g1 = planeOffset(strip, line, world, s1);
    for (int i = 0; i < 4; ++i) {
        let slope = g1 - g0;
        if (abs(slope) < 1e-20) {
            break;
        }
        let s = s1 - g1 * (s1 - s0) / slope;
        s0 = s1;
        g0 = g1;
        s1 = s;
        g1 = planeOffset(strip, line, world, s1);
    }
    return s1;
}

public float4 cameraProject(float4x4 viewProj, float4 intrinsics, float3 world) {
    let strip = stripOf(viewProj);
    let s = scanTime(strip, intrinsics.z, world);
    let local = unrotate(rotationAt(strip, s), world - lerp(strip.from, strip.to, s));
    let depth = -local.z;
    let x = local.x / depth * intrinsics.x + intrinsics.y;
    let y = 1.0 - 2.0 * s;
    // Reverse-Z, as the cameras' projections: 1 at the near plane, 0 at the far.
    return float4(x * depth, y * depth, strip.near * (strip.far - depth) / (strip.far - strip.near),
                  depth);
}

// The ray of the pixel at `ndc`: its row's pose and the line's direction across it.
void rayAt(Strip strip, float4 intrinsics, float2 ndc, out float3 origin, out float3 direction,
           out float4 rotation) {
    let s = 0.5 - 0.5 * ndc.y;
    rotation = rotationAt(strip, s);
    origin = lerp(strip.from, strip.to, s);
    let x = (ndc.x - intrinsics.y) / intrinsics.x;
    direction = rotate(rotation, float3(x, intrinsics.z, -1.0));
}

public float3 cameraPoint(float4x4 invViewProj, float4 intrinsics, float2 ndc, float depth) {
    let strip = stripOf(invViewProj);
    float3 origin;
    float3 direction;
    float4 rotation;
    rayAt(strip, intrinsics, ndc, origin, direction, rotation);
    // The view depth of [0, 1] depth, as the projection mapped it.
    let viewDepth = strip.near * strip.far / (strip.near + depth * (strip.far - strip.near));
    return origin + direction * viewDepth;
}

// The ray through `ndc`: from its row's camera position, along the line's direction there (one
// unit of view depth long).
public void cameraRayThrough(float4x4 invViewProj, float4 intrinsics, float2 ndc,
                             out float3 origin, out float3 direction) {
    float4 rotation;
    rayAt(stripOf(invViewProj), intrinsics, ndc, origin, direction, rotation);
}

public float3 cameraRay(float4x4 invViewProj, float4 intrinsics, float2 ndc) {
    let strip = stripOf(invViewProj);
    float3 origin;
    float3 direction;
    float4 rotation;
    rayAt(strip, intrinsics, ndc, origin, direction, rotation);
    return normalize(direction);
}
