// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/orbit_controller.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace ernie {

glm::dvec3 OrbitController::Eye() const {
    const double horizontal = std::cos(double(pitch)) * distance;
    return target + glm::dvec3(std::sin(double(yaw)) * horizontal,
                               -std::cos(double(yaw)) * horizontal,
                               std::sin(double(pitch)) * distance);
}

glm::dmat4 OrbitController::Transform() const {
    return glm::inverse(
        glm::lookAtRH(Eye(), target, glm::dvec3(0.0, 0.0, 1.0)));
}

void OrbitController::Apply(CameraParams& camera) const {
    camera.transform = Transform();
    camera.SetVerticalFov(verticalFov);
    camera.nearPlane = NearPlane();
    camera.farPlane = FarPlane();
}

CameraParams OrbitController::MakeCamera(std::uint32_t width,
                                         std::uint32_t height) const {
    CameraParams camera;
    camera.width = width;
    camera.height = height;
    Apply(camera);
    return camera;
}

void OrbitController::Orbit(float deltaYaw, float deltaPitch) {
    yaw += deltaYaw;
    pitch = std::clamp(pitch + deltaPitch, kMinPitch, kMaxPitch);
}

void OrbitController::Pan(float dx, float dy, float viewportHeight) {
    const double worldPerPixel =
        2.0 * distance * std::tan(0.5 * double(verticalFov)) /
        std::max(double(viewportHeight), 1.0);
    const glm::dvec3 right(std::cos(double(yaw)), std::sin(double(yaw)),
                           0.0);
    const glm::dvec3 forward(-std::sin(double(yaw)),
                             std::cos(double(yaw)), 0.0);
    target += (-double(dx) * right + double(dy) * forward) * worldPerPixel;
}

void OrbitController::PanVertical(float dx, float dy,
                                  float viewportHeight) {
    const double worldPerPixel =
        2.0 * distance * std::tan(0.5 * double(verticalFov)) /
        std::max(double(viewportHeight), 1.0);
    const glm::dvec3 right(std::cos(double(yaw)), std::sin(double(yaw)),
                           0.0);
    target += (-double(dx) * right +
               double(dy) * glm::dvec3(0.0, 0.0, 1.0)) *
              worldPerPixel;
}

void OrbitController::Zoom(float steps) {
    distance = std::clamp(distance * std::pow(0.9, double(steps)),
                          kMinDistance, kMaxDistance);
}

void OrbitController::Frame(const glm::dvec3& boundsMin,
                            const glm::dvec3& boundsMax) {
    target = 0.5 * (boundsMin + boundsMax);
    const double radius = std::max(
        0.5 * glm::length(boundsMax - boundsMin), kMinDistance);
    distance =
        std::clamp(1.1 * radius / std::sin(0.5 * double(verticalFov)),
                   kMinDistance, kMaxDistance);
}

}  // namespace ernie
