// SPDX-License-Identifier: BSD-3-Clause
/// @file orbit_camera.h
/// @brief The viewer's camera: an eye orbiting a target point in a z-up world.
#pragma once

#include "ernie/geometry/bounds.h"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace ernie::app {

/// @brief An orbit around a target, in double precision as the scene's transforms are, so it
/// orbits as smoothly far from the origin as at it. Yaw turns about +z (0 looks along +y, positive
/// turns clockwise seen from above); pitch is the eye's elevation above the target's horizontal
/// plane.
struct OrbitCamera {
    glm::dvec3 target{0.0};
    /// Radians.
    double yaw = 0.0;
    /// Radians, within [kMinPitch, kMaxPitch].
    double pitch = 0.6;
    /// Metres from the eye to the target, within [kMinDistance, kMaxDistance].
    double distance = 10.0;
    /// Radians.
    double verticalFov = 0.785398;

    /// Short of straight down and up, where the view's up vector degenerates.
    static constexpr double kMinPitch = -1.55;
    static constexpr double kMaxPitch = 1.55;
    /// A millimetre to about 100,000 light-years.
    static constexpr double kMinDistance = 1e-3;
    static constexpr double kMaxDistance = 1e21;

    glm::dvec3 Eye() const;
    /// @return The camera-to-world transform of the eye looking at the target, +z up.
    glm::dmat4 Transform() const;

    /// @brief Turns by yaw and pitch radians, clamping pitch.
    void Orbit(double deltaYaw, double deltaPitch);
    /// @brief Moves the target parallel to the ground by a drag of (`dx`, `dy`) pixels in a view
    /// `height` pixels tall, so the ground at the target follows the cursor.
    void Pan(double dx, double dy, double height);
    /// @brief Moves the target across the view: sideways as `Pan`, and up and down along +z.
    void PanVertical(double dx, double dy, double height);
    /// @brief Moves closer by `steps` scroll steps; negative moves away.
    void Zoom(double steps);
    /// @brief Aims at the box's centre from the current direction, at a distance where its
    /// bounding sphere fits the vertical field of view with a small margin. An empty box leaves
    /// the camera as it is.
    void Frame(const Box& box);
};

}  // namespace ernie::app
