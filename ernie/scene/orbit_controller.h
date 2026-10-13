// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/camera.h"

#include <glm/glm.hpp>

namespace ernie {

/// Interactive orbit control of a `CameraParams` around a target
/// point in a z-up world, for inspecting terrain. It holds no camera
/// of its own: `Apply` writes its pose, field of view and clipping
/// range into one.
///
/// Yaw rotates around the vertical axis (0 looks north, positive
/// turns the view clockwise seen from above); pitch is the elevation
/// angle of the eye above the target's horizontal plane.
///
/// The target and distance are double precision, as the scene's
/// transforms are, so the view orbits as smoothly light-years from
/// the origin as at it (the renderer draws relative to a floating
/// origin near the eye).
class OrbitController {
  public:
    glm::dvec3 target{0.0};
    float yaw = 0.0f;  ///< Radians.
    float pitch =
        0.6f;  ///< Radians, clamped to [kMinPitch, kMaxPitch].
    double distance =
        100.0;  ///< World units from the eye to the target.
    float verticalFov = 0.785398f;  ///< Radians.

    /// Pitch stops short of straight down and up, where the view's up
    /// vector degenerates; views may look up from below the ground.
    static constexpr float kMinPitch = -1.55f;
    static constexpr float kMaxPitch = 1.55f;

    /// Eye position in world coordinates.
    glm::dvec3 Eye() const;

    /// Camera-to-world transform of the eye looking at the target, +z
    /// up.
    glm::dmat4 Transform() const;

    /// Clipping planes scaled to the orbit distance.
    double NearPlane() const {
        return distance * 0.01;
    }

    double FarPlane() const {
        return distance * 100.0;
    }

    /// Sets `camera`'s transform, vertical field of view (through its
    /// aperture, keeping its focal length) and clipping range.
    void Apply(CameraParams& camera) const;

    /// A default camera of `width` x `height` with this pose, field
    /// of view and clipping range.
    CameraParams MakeCamera(std::uint32_t width,
                            std::uint32_t height) const;

    /// Rotates by `deltaYaw` and `deltaPitch` radians, clamping
    /// pitch.
    void Orbit(float deltaYaw, float deltaPitch);

    /// Moves the target parallel to the ground by a screen-space drag
    /// of (`dx`, `dy`) pixels in a viewport `viewportHeight` pixels
    /// tall, so the terrain follows the cursor at the target depth.
    void Pan(float dx, float dy, float viewportHeight);

    /// Moves the target across the screen by a drag of (`dx`, `dy`)
    /// pixels: sideways as `Pan`, and up and down along world z.
    void PanVertical(float dx, float dy, float viewportHeight);

    /// Zooms by `steps` scroll steps; positive moves closer. Distance
    /// stays within [kMinDistance, kMaxDistance].
    void Zoom(float steps);

    /// Aims at the centre of the axis-aligned box [`boundsMin`,
    /// `boundsMax`] from a distance at which its bounding sphere fits
    /// the vertical field of view with a small margin.
    void Frame(const glm::dvec3& boundsMin,
               const glm::dvec3& boundsMax);

    /// Aims at the centre of the box [`boundsMin`, `boundsMax`] from
    /// the current direction, at the distance where all its corners
    /// fit a view `aspect` (width over height) wide with a small
    /// margin: a flat or long box fills the view rather than its
    /// bounding sphere.
    void FrameView(const glm::dvec3& boundsMin,
                   const glm::dvec3& boundsMax, double aspect);

    /// The orbit distance's range: a millimetre to about 100,000
    /// light-years.
    static constexpr double kMinDistance = 1e-3;
    static constexpr double kMaxDistance = 1e21;
};

}  // namespace ernie
