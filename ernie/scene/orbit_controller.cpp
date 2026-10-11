// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/orbit_controller.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace ernie {

namespace {

constexpr float kMinDistance = 1e-3f;
constexpr float kMaxDistance = 1e7f;

}  // namespace

glm::vec3 OrbitController::Eye() const {
    const float horizontal = std::cos(pitch) * distance;
    return target + glm::vec3(std::sin(yaw) * horizontal,
                              -std::cos(yaw) * horizontal,
                              std::sin(pitch) * distance);
}

glm::dmat4 OrbitController::Transform() const {
    return glm::inverse(glm::dmat4(
        glm::lookAtRH(Eye(), target, glm::vec3(0.0f, 0.0f, 1.0f))));
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
    const float worldPerPixel = 2.0f * distance *
                                std::tan(0.5f * verticalFov) /
                                std::max(viewportHeight, 1.0f);
    const glm::vec3 right(std::cos(yaw), std::sin(yaw), 0.0f);
    const glm::vec3 forward(-std::sin(yaw), std::cos(yaw), 0.0f);
    target += (-dx * right + dy * forward) * worldPerPixel;
}

void OrbitController::PanVertical(float dx, float dy,
                                  float viewportHeight) {
    const float worldPerPixel = 2.0f * distance *
                                std::tan(0.5f * verticalFov) /
                                std::max(viewportHeight, 1.0f);
    const glm::vec3 right(std::cos(yaw), std::sin(yaw), 0.0f);
    target += (-dx * right + dy * glm::vec3(0.0f, 0.0f, 1.0f)) *
              worldPerPixel;
}

void OrbitController::Zoom(float steps) {
    distance = std::clamp(distance * std::pow(0.9f, steps),
                          kMinDistance, kMaxDistance);
}

void OrbitController::Frame(const glm::vec3& boundsMin,
                            const glm::vec3& boundsMax) {
    target = 0.5f * (boundsMin + boundsMax);
    const float radius = std::max(
        0.5f * glm::length(boundsMax - boundsMin), kMinDistance);
    distance =
        std::clamp(1.1f * radius / std::sin(0.5f * verticalFov),
                   kMinDistance, kMaxDistance);
}

}  // namespace ernie
