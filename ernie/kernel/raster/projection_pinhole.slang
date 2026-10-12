// SPDX-License-Identifier: BSD-3-Clause
// The camera projection of frame cameras, loaded as module `projection` (projection_pushbroom.slang
// is the pushbroom's, with the same functions): a view-projection matrix and its inverse. The
// renderer compiles its passes with one of them, so a frame camera's passes run only this.
//
// `strip` is unused here: the pushbroom's intrinsics.

public static const bool kPushbroomCamera = false;

// Clip coordinates of world point `world`.
public float4 cameraProject(float4x4 viewProj, float4 strip, float3 world) {
    return mul(viewProj, float4(world, 1.0));
}

// The world point at normalised device coordinates `ndc` and [0, 1] depth `depth`: reverse-Z, 1 on
// the near plane falling away from the camera, to 0 at the far plane or, with an infinite one, at
// infinity, where no point is.
public float3 cameraPoint(float4x4 invViewProj, float4 strip, float2 ndc, float depth) {
    let clip = mul(invViewProj, float4(ndc, depth, 1.0));
    return clip.xyz / clip.w;
}

// The ray through `ndc`: from its point on the near plane (depth 1), along the ray away from the
// camera; `direction` is not of unit length (it reaches depth 1/2, twice the near plane's distance
// with an infinite far plane, which has no point at depth 0).
public void cameraRayThrough(float4x4 invViewProj, float4 strip, float2 ndc, out float3 origin,
                             out float3 direction) {
    let near = mul(invViewProj, float4(ndc, 1.0, 1.0));
    let beyond = mul(invViewProj, float4(ndc, 0.5, 1.0));
    origin = near.xyz / near.w;
    direction = beyond.xyz / beyond.w - origin;
}

// The unit direction of the ray through `ndc`, away from the camera.
public float3 cameraRay(float4x4 invViewProj, float4 strip, float2 ndc) {
    float3 origin;
    float3 direction;
    cameraRayThrough(invViewProj, strip, ndc, origin, direction);
    return normalize(direction);
}
