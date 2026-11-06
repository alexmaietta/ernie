// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/viewer/viewport/orbit_camera.h"

#include <algorithm>
#include <cmath>

namespace ernie::app {

Double3 OrbitCamera::Eye() const {
    const double across = std::cos(pitch) * distance;
    return {target.x + std::sin(yaw) * across, target.y - std::cos(yaw) * across,
            target.z + std::sin(pitch) * distance};
}

Quatd OrbitCamera::Orient() const {
    return LookAt(Eye(), target);
}

void OrbitCamera::Orbit(double deltaYaw, double deltaPitch) {
    yaw += deltaYaw;
    pitch = std::clamp(pitch + deltaPitch, -kMaxPitch, kMaxPitch);
}

void OrbitCamera::Pan(double right, double ahead) {
    target.x += std::cos(yaw) * right - std::sin(yaw) * ahead;
    target.y += std::sin(yaw) * right + std::cos(yaw) * ahead;
}

void OrbitCamera::PanVertical(double right, double raised) {
    target.x += std::cos(yaw) * right;
    target.y += std::sin(yaw) * right;
    target.z += raised;
}

void OrbitCamera::Zoom(double steps) {
    distance = std::clamp(distance * std::pow(0.9, steps), 1e-3, 1e12);
}

void OrbitCamera::LookFrom(const Double3& eye, const Quatd& orient,
                           const std::optional<Double3>& focus) {
    const Double3 forward = Rotate(orient, {0, 0, -1});
    if (focus) {
        const double ahead = (focus->x - eye.x) * forward.x + (focus->y - eye.y) * forward.y +
                             (focus->z - eye.z) * forward.z;
        if (ahead > 0) {
            distance = ahead;
        }
    }
    target = {eye.x + forward.x * distance, eye.y + forward.y * distance,
              eye.z + forward.z * distance};
    yaw = std::atan2(-forward.x, forward.y);
    pitch = std::asin(std::clamp(-forward.z, -1.0, 1.0));
}

}
