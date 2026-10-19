// SPDX-License-Identifier: BSD-3-Clause
/// @file lens.slang
/// @brief Lens models: a pixel to the ray it sees in the camera's frame (inverse projection), and
/// a direction to the pixel it lands on (forward projection). The camera looks down -z with +y up;
/// pixel coordinates run right and down from the image's top-left corner. Every model shares the
/// pixel intrinsics, as OpenCV's camera matrix holds them, and reads its own coefficients from
/// `coefficients`, so a new model is a new kind, not a new record. The record is shared with C++
/// (rt/camera_pass.h).
///
///     LensRay ray = Unproject(lens, float2(pixel) + 0.5);
#ifdef __cplusplus
#pragma once
#include "ernie/shaders/shared.h"
namespace ernie::shaders {
#else
import buffer_ref;
#endif

/// Perspective through a pinhole (UsdGeomCamera's perspective projection).
static const uint kPinhole = 0;
/// Parallel rays (telecentric, UsdGeomCamera's orthographic projection): the focal lengths are
/// pixels a metre, and rays start on the camera's z = 0 plane.
static const uint kOrthographic = 1;

/// A lens: the model and its intrinsics.
struct Lens {
    uint kind;
    /// Focal lengths in pixels (x, y), and the principal point in pixels from the top-left corner.
    float2 focal;
    float2 principal;
    /// x's change per unit of y in the image plane, OpenCV's skew over fx.
    float skew;
    /// The model's own coefficients, null for one with none.
    BufferRef<float> coefficients;
};

#ifdef __cplusplus
}  // namespace ernie::shaders
#else

/// A ray in the camera's frame: unit direction.
struct LensRay : IDifferentiable {
    float3 origin;
    float3 direction;
};

/// @return The point on the z = -1 image plane a pixel coordinate maps to, before distortion.
[Differentiable]
float2 ToImagePlane(no_diff Lens lens, float2 pixel) {
    // Normalized coordinates with y down, as OpenCV's: u = fx (x + skew y) + cx.
    float2 offset = (pixel - lens.principal) / lens.focal;
    return float2(offset.x - lens.skew * offset.y, -offset.y);
}

/// @return The ray a pixel coordinate sees.
[Differentiable]
LensRay Unproject(no_diff Lens lens, float2 pixel) {
    float2 plane = ToImagePlane(lens, pixel);
    LensRay ray;
    if (lens.kind == kOrthographic) {
        ray.origin = float3(plane, 0);
        ray.direction = float3(0, 0, -1);
    } else {
        ray.origin = float3(0);
        ray.direction = normalize(float3(plane, -1));
    }
    return ray;
}

/// @return The pixel coordinate a point in the camera's frame lands on, and whether it is in front
/// of the lens.
[Differentiable]
float2 Project(no_diff Lens lens, float3 point, out bool visible) {
    float2 plane = point.xy;
    visible = lens.kind == kOrthographic || point.z < 0;
    if (lens.kind != kOrthographic) {
        plane = point.xy / -point.z;
    }
    float down = -plane.y;
    return lens.principal + float2(plane.x + lens.skew * down, down) * lens.focal;
}

#endif
