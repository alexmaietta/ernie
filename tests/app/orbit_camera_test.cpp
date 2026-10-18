// SPDX-License-Identifier: BSD-3-Clause
/// @file orbit_camera_test.cpp
/// @brief Tests of the viewer's orbit camera.
#include "ernie/app/ui.h"

#include <gtest/gtest.h>

#include <cmath>

namespace ernie {
namespace {

using app::OrbitCamera;

TEST(OrbitCamera, LooksAlongPlusYAtNoYaw) {
    const OrbitCamera camera{.target = {1.0, 2.0, 3.0}, .yaw = 0.0, .pitch = 0.0, .distance = 5.0};
    const glm::dvec3 eye = camera.Eye();
    EXPECT_NEAR(eye.x, 1.0, 1e-12);
    EXPECT_NEAR(eye.y, -3.0, 1e-12);
    EXPECT_NEAR(eye.z, 3.0, 1e-12);
    const glm::dvec3 forward = -glm::dvec3(camera.Transform()[2]);
    EXPECT_NEAR(forward.y, 1.0, 1e-12) << "cameras look down their -z";
}

TEST(OrbitCamera, ClampsPitchAndDistance) {
    OrbitCamera camera;
    camera.Orbit(0.0, 10.0);
    EXPECT_EQ(camera.pitch, OrbitCamera::kMaxPitch);
    camera.Zoom(1e6);
    EXPECT_EQ(camera.distance, OrbitCamera::kMinDistance);
    camera.Zoom(-1e6);
    EXPECT_EQ(camera.distance, OrbitCamera::kMaxDistance);
}

TEST(OrbitCamera, PansAlongTheGroundAtTheTargetsDepth) {
    OrbitCamera camera{.yaw = 0.0, .distance = 10.0};
    const double height = 2.0 * 10.0 * std::tan(0.5 * camera.verticalFov);
    // A drag of the view's whole height moves that many metres.
    camera.Pan(0.0, 100.0, 100.0);
    EXPECT_NEAR(camera.target.y, height, 1e-9);
    EXPECT_NEAR(camera.target.z, 0.0, 1e-12);
    camera.PanVertical(0.0, 100.0, 100.0);
    EXPECT_NEAR(camera.target.z, height, 1e-9);
}

TEST(OrbitCamera, FramingAnEmptyBoxMovesNothing) {
    OrbitCamera camera;
    const OrbitCamera before = camera;
    camera.Frame(Box());
    EXPECT_EQ(camera.target, before.target);
    EXPECT_EQ(camera.distance, before.distance);
}

}  // namespace
}  // namespace ernie
