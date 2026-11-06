// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/viewer/viewport/orbit_camera.h"

#include <cmath>
#include <numbers>

#include <gtest/gtest.h>

#include "ernie/scene/rotation.h"

namespace ernie::app {
namespace {

void ExpectNear(Double3 actual, Double3 expected, double tolerance = 1e-9) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
}

Double3 Forward(const OrbitCamera& camera) {
    return Rotate(camera.Orient(), {0, 0, -1});
}

TEST(OrbitCameraTest, LooksAtItsTargetFromItsOrbit) {
    OrbitCamera camera{.target = {1, 2, 3}, .yaw = 0, .pitch = 0, .distance = 4};
    ExpectNear(camera.Eye(), {1, -2, 3});
    ExpectNear(Forward(camera), {0, 1, 0});

    camera.Orbit(std::numbers::pi / 2, 0);
    ExpectNear(camera.Eye(), {5, 2, 3});
    ExpectNear(Forward(camera), {-1, 0, 0});

    camera.Orbit(0, 10);
    EXPECT_DOUBLE_EQ(camera.pitch, OrbitCamera::kMaxPitch);
    camera.Orbit(0, -20);
    EXPECT_DOUBLE_EQ(camera.pitch, -OrbitCamera::kMaxPitch);
}

TEST(OrbitCameraTest, PansZoomsAndLooksFromAnyPose) {
    OrbitCamera camera{.distance = 10};
    camera.Pan(1, 2);
    ExpectNear(camera.target, {1, 2, 0});
    camera.PanVertical(0, 3);
    ExpectNear(camera.target, {1, 2, 3});
    camera.Zoom(1);
    EXPECT_NEAR(camera.distance, 9, 1e-9);
    camera.Zoom(-1e6);
    EXPECT_LE(camera.distance, 1e12);
    camera.Zoom(1e6);
    EXPECT_GE(camera.distance, 1e-3);

    const Double3 eye{3, -4, 2};
    camera.LookFrom(eye, LookAt(eye, {0, 0, 1}), Double3{0, 0, 1});
    ExpectNear(camera.Eye(), eye, 1e-6);
    ExpectNear(camera.target, {0, 0, 1}, 1e-6);
}

}
}
