// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/orbit_controller.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cmath>

using ernie::OrbitController;

namespace {

glm::vec3 Project(const glm::mat4& viewProjection,
                  const glm::vec3& point) {
    const glm::vec4 clip = viewProjection * glm::vec4(point, 1.0f);
    return glm::vec3(clip) / clip.w;
}

}  // namespace

TEST(OrbitController, EyeSitsAtDistanceAndPitchAboveTarget) {
    OrbitController camera;
    camera.target = {10.0f, 20.0f, 5.0f};
    camera.distance = 50.0f;
    camera.pitch = 0.5f;
    const glm::vec3 offset = camera.Eye() - camera.target;
    EXPECT_NEAR(glm::length(offset), 50.0f, 1e-3f);
    EXPECT_NEAR(offset.z, 50.0f * std::sin(0.5f), 1e-3f);
}

TEST(OrbitController, ZeroYawLooksNorth) {
    OrbitController camera;
    const glm::vec3 offset = camera.Eye() - camera.target;
    EXPECT_NEAR(offset.x, 0.0f, 1e-4f);
    EXPECT_LT(offset.y, 0.0f);
}

TEST(OrbitController, ViewPlacesTargetStraightAhead) {
    OrbitController camera;
    camera.target = {3.0f, -4.0f, 1.0f};
    camera.distance = 25.0f;
    camera.yaw = 1.2f;
    const glm::vec4 inView =
        glm::mat4(glm::inverse(camera.Transform())) *
        glm::vec4(camera.target, 1.0f);
    EXPECT_NEAR(inView.x, 0.0f, 1e-3f);
    EXPECT_NEAR(inView.y, 0.0f, 1e-3f);
    EXPECT_NEAR(inView.z, -25.0f, 1e-3f);
}

// Reverse-Z: 1 at the near plane, 0 at the far.
TEST(OrbitController, ProjectionUsesZeroToOneDepth) {
    OrbitController camera;
    camera.distance = 10.0f;
    const glm::mat4 viewProjection =
        camera.MakeCamera(1, 1).RasterView().ViewProjection();
    const glm::vec3 forward =
        glm::normalize(camera.target - camera.Eye());
    EXPECT_NEAR(Project(viewProjection,
                        glm::vec3(camera.Eye()) + forward * 0.1f)
                    .z,
                1.0f, 1e-4f);
    EXPECT_NEAR(Project(viewProjection,
                        glm::vec3(camera.Eye()) + forward * 1000.0f)
                    .z,
                0.0f, 1e-4f);
}

TEST(OrbitController, OrbitClampsPitch) {
    OrbitController camera;
    camera.Orbit(0.3f, 10.0f);
    EXPECT_FLOAT_EQ(camera.pitch, OrbitController::kMaxPitch);
    EXPECT_FLOAT_EQ(camera.yaw, 0.3f);
    camera.Orbit(0.0f, -10.0f);
    EXPECT_FLOAT_EQ(camera.pitch, OrbitController::kMinPitch);
}

TEST(OrbitController, ZoomMovesCloserAndClamps) {
    OrbitController camera;
    camera.distance = 100.0f;
    camera.Zoom(1.0f);
    EXPECT_FLOAT_EQ(camera.distance, 90.0f);
    camera.Zoom(-1.0f);
    EXPECT_NEAR(camera.distance, 100.0f, 1e-3f);
    camera.Zoom(1000.0f);
    EXPECT_FLOAT_EQ(camera.distance, 1e-3f);
}

TEST(OrbitController, PanFollowsTheCursorAtTargetDepth) {
    OrbitController camera;
    camera.distance = 100.0f;
    camera.pitch = OrbitController::kMaxPitch;
    const float viewportHeight = 500.0f;
    const glm::vec3 before = camera.target;
    camera.Pan(50.0f, 0.0f, viewportHeight);
    const float expected = 50.0f * 2.0f * 100.0f *
                           std::tan(0.5f * camera.verticalFov) /
                           viewportHeight;
    EXPECT_NEAR(glm::length(glm::vec3(camera.target) - before),
                expected, 1e-3f);
    EXPECT_LT(camera.target.x, before.x);
    EXPECT_FLOAT_EQ(camera.target.z, before.z);
}

TEST(OrbitController, VerticalPanMovesAlongZ) {
    OrbitController camera;
    camera.distance = 100.0f;
    const float perPixel =
        2.0f * 100.0f * std::tan(0.5f * camera.verticalFov) / 500.0f;
    camera.PanVertical(0.0f, 20.0f, 500.0f);
    EXPECT_NEAR(camera.target.z, 20.0f * perPixel, 1e-4f)
        << "dragging down raises the view";
    EXPECT_FLOAT_EQ(camera.target.x, 0.0f);
    EXPECT_FLOAT_EQ(camera.target.y, 0.0f);
}

TEST(OrbitController, FrameFitsTheBoxInView) {
    OrbitController camera;
    camera.yaw = 0.7f;
    const glm::vec3 boundsMin(-500.0f, -300.0f, -40.0f);
    const glm::vec3 boundsMax(500.0f, 300.0f, 60.0f);
    camera.Frame(boundsMin, boundsMax);
    EXPECT_NEAR(camera.target.z, 10.0f, 1e-3f);

    const glm::mat4 viewProjection =
        camera.MakeCamera(1, 1).RasterView().ViewProjection();
    for (int i = 0; i < 8; ++i) {
        const glm::vec3 corner((i & 1) ? boundsMax.x : boundsMin.x,
                               (i & 2) ? boundsMax.y : boundsMin.y,
                               (i & 4) ? boundsMax.z : boundsMin.z);
        const glm::vec3 ndc = Project(viewProjection, corner);
        EXPECT_LE(std::fabs(ndc.x), 1.0f) << i;
        EXPECT_LE(std::fabs(ndc.y), 1.0f) << i;
        EXPECT_GE(ndc.z, 0.0f) << i;
        EXPECT_LE(ndc.z, 1.0f) << i;
    }
}

TEST(OrbitController, AppliesItsPoseFieldOfViewAndClippingToACamera) {
    OrbitController orbit;
    orbit.target = {1.0f, 2.0f, 3.0f};
    orbit.distance = 40.0f;
    orbit.verticalFov = 0.5f;
    ernie::CameraParams camera;
    camera.focalLengthMm = 35.0;
    camera.exposureStops = 2.0;
    orbit.Apply(camera);
    EXPECT_NEAR(camera.VerticalFov(), 0.5, 1e-6);
    EXPECT_DOUBLE_EQ(camera.focalLengthMm, 35.0)
        << "the focal length is kept";
    EXPECT_DOUBLE_EQ(camera.exposureStops, 2.0)
        << "and so is everything else";
    EXPECT_NEAR(camera.nearPlane, 0.4, 1e-6);
    EXPECT_NEAR(camera.farPlane, 4000.0, 1e-3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(camera.Eye()[i], orbit.Eye()[i], 1e-4);
    }
    const glm::dvec3 forward =
        glm::normalize(glm::dvec3(orbit.target - orbit.Eye()));
    EXPECT_NEAR(glm::dot(camera.Forward(), forward), 1.0, 1e-6);
}

TEST(OrbitController, OrbitsAsExactlyAcrossTheSolarSystem) {
    // At Saturn's distance, 1.4e12 m, where a float's step is some
    // 130 km (a double's, 0.2 mm): the view still holds a 10 m orbit
    // to a few millimetres, frames a metre-sized box there and pans
    // by a pixel's worth.
    const glm::dvec3 far(1.4e12, -2.0e11, 3.0e10);
    OrbitController orbit;
    orbit.target = far;
    orbit.distance = 10.0;
    orbit.yaw = 0.7f;
    orbit.pitch = 0.3f;
    EXPECT_NEAR(glm::length(orbit.Eye() - far), 10.0, 5e-3);
    const glm::dmat4 pose = orbit.Transform();
    EXPECT_NEAR(glm::length(glm::dvec3(pose[3]) - orbit.Eye()), 0.0,
                5e-3);

    orbit.Frame(far - glm::dvec3(0.5), far + glm::dvec3(0.5));
    EXPECT_NEAR(glm::length(orbit.target - far), 0.0, 1e-3)
        << "framed on the box's centre";
    EXPECT_LT(orbit.distance, 10.0);

    const glm::dvec3 before = orbit.target;
    orbit.Pan(1.0f, 0.0f, 1000.0f);
    const double moved = glm::length(orbit.target - before);
    EXPECT_GT(moved, 0.0) << "a pixel's pan moves it";
    EXPECT_LT(moved, 0.01);
}

TEST(OrbitController, ZoomsOutFarEnoughToFrameASolarSystem) {
    OrbitController orbit;
    orbit.Frame(glm::dvec3(-4.5e12), glm::dvec3(4.5e12));
    EXPECT_GT(orbit.distance, 1e13) << "Neptune's orbit, whole";
    for (int i = 0; i < 2000; ++i) {
        orbit.Zoom(-1.0f);
    }
    EXPECT_DOUBLE_EQ(orbit.distance, OrbitController::kMaxDistance);
}
