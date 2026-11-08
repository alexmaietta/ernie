// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/viewer/viewport/gizmo.h"

#include <optional>

#include <gtest/gtest.h>

namespace ernie::app {
namespace {

GizmoCamera Camera(bool orthographic, float scale) {
    GizmoCamera camera;
    camera.frame.x = {scale, 0, 0};
    camera.frame.y = {0, 0, scale};
    camera.frame.z = {0, -scale, 0};
    camera.frame.high = {1, -10, 2};
    camera.focal = 35;
    camera.width = 36;
    camera.height = 24;
    camera.right = 1.5f;
    camera.up = -0.5f;
    camera.orthographic = orthographic;
    camera.min = {100, 50};
    camera.size = {640, 360};
    return camera;
}

void ExpectRoundTrip(const GizmoCamera& camera) {
    for (const double u : {0.0, 0.25, 0.5, 0.9}) {
        for (const double v : {0.0, 0.4, 1.0}) {
            const GizmoRay ray = CameraRay(camera, u, v);
            const Double3 point{ray.origin.x + 3 * ray.direction.x,
                                ray.origin.y + 3 * ray.direction.y,
                                ray.origin.z + 3 * ray.direction.z};
            const std::optional<ImVec2> screen = ToScreen(camera, point);
            ASSERT_TRUE(screen);
            EXPECT_NEAR(screen->x, camera.min.x + u * camera.size.x, 1e-2);
            EXPECT_NEAR(screen->y, camera.min.y + v * camera.size.y, 1e-2);
        }
    }
}

TEST(GizmoTest, ProjectsCameraRaysBackToTheirPixels) {
    ExpectRoundTrip(Camera(false, 1));
    ExpectRoundTrip(Camera(true, 1));
}

TEST(GizmoTest, ProjectsThroughScaledCameraFrames) {
    ExpectRoundTrip(Camera(false, 2.5f));
    ExpectRoundTrip(Camera(true, 0.5f));
}

TEST(GizmoTest, HidesPointsBehindPerspectiveCameras) {
    const GizmoCamera camera = Camera(false, 1);
    EXPECT_FALSE(ToScreen(camera, {1, -20, 2}));
    EXPECT_FALSE(ToScreen(camera, {1, -10, 2}));
    EXPECT_TRUE(ToScreen(Camera(true, 1), {1, -20, 2}));
}

}
}
