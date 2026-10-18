// SPDX-License-Identifier: BSD-3-Clause
/// @file orbit_camera.cpp
/// @brief Orbiting, panning, zooming and framing.
#include "ernie/app/orbit_camera.h"

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace ernie::app {
namespace {

/// @return Metres per pixel at the target's depth in a view `height` pixels tall.
double MetresPerPixel(const OrbitCamera& camera, double height) {
    return 2.0 * camera.distance * std::tan(0.5 * camera.verticalFov) / std::max(height, 1.0);
}

}  // namespace

glm::dvec3 OrbitCamera::Eye() const {
    const double horizontal = std::cos(pitch) * distance;
    return target + glm::dvec3(std::sin(yaw) * horizontal, -std::cos(yaw) * horizontal,
                               std::sin(pitch) * distance);
}

glm::dmat4 OrbitCamera::Transform() const {
    return glm::inverse(glm::lookAtRH(Eye(), target, glm::dvec3(0.0, 0.0, 1.0)));
}

void OrbitCamera::Orbit(double deltaYaw, double deltaPitch) {
    yaw += deltaYaw;
    pitch = std::clamp(pitch + deltaPitch, kMinPitch, kMaxPitch);
}

void OrbitCamera::Pan(double dx, double dy, double height) {
    const glm::dvec3 right(std::cos(yaw), std::sin(yaw), 0.0);
    const glm::dvec3 forward(-std::sin(yaw), std::cos(yaw), 0.0);
    target += (-dx * right + dy * forward) * MetresPerPixel(*this, height);
}

void OrbitCamera::PanVertical(double dx, double dy, double height) {
    const glm::dvec3 right(std::cos(yaw), std::sin(yaw), 0.0);
    target += (-dx * right + dy * glm::dvec3(0.0, 0.0, 1.0)) * MetresPerPixel(*this, height);
}

void OrbitCamera::Zoom(double steps) {
    distance = std::clamp(distance * std::pow(0.9, steps), kMinDistance, kMaxDistance);
}

void OrbitCamera::Frame(const Box& box) {
    if (box.Empty()) {
        return;
    }
    target = box.Center();
    const double radius = std::max(0.5 * glm::length(box.max - box.min), kMinDistance);
    distance = std::clamp(1.1 * radius / std::sin(0.5 * verticalFov), kMinDistance, kMaxDistance);
}

}  // namespace ernie::app
