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
    const glm::dvec3 right(std::cos(double(yaw)),
                           std::sin(double(yaw)), 0.0);
    const glm::dvec3 forward(-std::sin(double(yaw)),
                             std::cos(double(yaw)), 0.0);
    target +=
        (-double(dx) * right + double(dy) * forward) * worldPerPixel;
}

void OrbitController::PanVertical(float dx, float dy,
                                  float viewportHeight) {
    const double worldPerPixel =
        2.0 * distance * std::tan(0.5 * double(verticalFov)) /
        std::max(double(viewportHeight), 1.0);
    const glm::dvec3 right(std::cos(double(yaw)),
                           std::sin(double(yaw)), 0.0);
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

void OrbitController::FrameView(const glm::dvec3& boundsMin,
                                const glm::dvec3& boundsMax,
                                double aspect) {
    target = 0.5 * (boundsMin + boundsMax);
    // The view's axes from the current yaw and pitch.
    const double horizontal = std::cos(double(pitch));
    const glm::dvec3 back(std::sin(double(yaw)) * horizontal,
                          -std::cos(double(yaw)) * horizontal,
                          std::sin(double(pitch)));
    const glm::dvec3 forward = -back;
    const glm::dvec3 right = glm::normalize(
        glm::cross(forward, glm::dvec3(0.0, 0.0, 1.0)));
    const glm::dvec3 up = glm::cross(right, forward);
    // A margin of 5% of the view on each side.
    const double tanV = std::tan(0.5 * double(verticalFov)) / 1.1;
    const double tanH = tanV * std::max(aspect, 1e-3);
    // The eye at target - forward * d sees a corner at depth d + z;
    // each corner needs d at least |x| / tanH - z and |y| / tanV - z,
    // and in front of the near plane (1% of d).
    double need = kMinDistance;
    for (int corner = 0; corner < 8; ++corner) {
        const glm::dvec3 c =
            glm::dvec3((corner & 1) ? boundsMax.x : boundsMin.x,
                       (corner & 2) ? boundsMax.y : boundsMin.y,
                       (corner & 4) ? boundsMax.z : boundsMin.z) -
            target;
        const double x = glm::dot(c, right);
        const double y = glm::dot(c, up);
        const double z = glm::dot(c, forward);
        need = std::max({need, std::abs(x) / tanH - z,
                         std::abs(y) / tanV - z, -z / 0.98});
    }
    distance = std::clamp(need, kMinDistance, kMaxDistance);
}

}  // namespace ernie
