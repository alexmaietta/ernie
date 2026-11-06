// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/scene/rotation.h"
#include "ernie/scene/scene.h"
#include "support/devices.h"

namespace ernie {
namespace {

template <class Test> void ForEachScene(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        if (device.Adapter().backend == device::Backend::Hip) {
            return;
        }
        const Ref<Scene> scene = Scene::Create(device);
        test(*scene);
    });
}

void ExpectNear(Float3 found, Float3 expected) {
    EXPECT_NEAR(found.x, expected.x, 1e-4f);
    EXPECT_NEAR(found.y, expected.y, 1e-4f);
    EXPECT_NEAR(found.z, expected.z, 1e-4f);
}

void ExpectBox(const std::optional<Box>& box, Float3 lower, Float3 upper) {
    ASSERT_TRUE(box);
    ExpectNear(box->lower, lower);
    ExpectNear(box->upper, upper);
}

void ExpectPrim(const Prim& prim, Float3 lower, Float3 upper) {
    ExpectNear(prim.Get<Float3>("worldMin"), lower);
    ExpectNear(prim.Get<Float3>("worldMax"), upper);
}

const std::vector<Float3> kQuad = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};

TEST(BoundsTest, ShapesAndMeshesBoundThemselvesAndTheirWorld) {
    ForEachScene([](Scene& scene) {
        const Prim ball = scene.Define<Sphere>("/Ball", {.translate = {1, 0, 0}, .radius = 2});
        const Prim box = scene.Define<Cube>(
            "/Box", {.translate = {0, 0, 5}, .orient = FromAxisAngle({0, 0, 1}, 45), .size = 2});
        const Prim quad = scene.Define<Mesh>("/Quad", {.translate = {10, 0, 0},
                                                       .points = kQuad,
                                                       .faceVertexIndices = {0, 1, 2, 3},
                                                       .faceVertexCounts = {4}});
        scene.Publish().Wait();
        ExpectPrim(ball, {-1, -2, -2}, {3, 2, 2});
        const float diagonal = std::sqrt(2.0f);
        ExpectPrim(box, {-diagonal, -diagonal, 4}, {diagonal, diagonal, 6});
        ExpectPrim(quad, {10, 0, 0}, {11, 1, 0});
        ExpectBox(scene.Bounds(), {-diagonal, -2, -2}, {11, 2, 6});
    });
}

TEST(BoundsTest, InfinitePlanesLeaveTheBoundsAlone) {
    ForEachScene([](Scene& scene) {
        const Prim ground = scene.Define<Plane>("/Ground", {.width = 0, .length = 0});
        const Prim floor = scene.Define<Plane>("/Floor", {.width = 4, .length = 2});
        scene.Define<Sphere>("/Ball", {.translate = {0, 0, 3}});
        scene.Publish().Wait();
        ExpectPrim(ground, {0, 0, 0}, {0, 0, 0});
        ExpectPrim(floor, {-2, -1, 0}, {2, 1, 0});
        ExpectBox(scene.Bounds(), {-2, -1, 0}, {2, 1, 4});
    });
}

TEST(BoundsTest, WorldsAreBoundRelativeToTheirOriginWithoutGlobals) {
    ForEachScene([](Scene& scene) {
        scene.Define<World>("/env_0", {.translate = {100, 0, 0}});
        scene.Define<Sphere>("/env_0/Ball", {.translate = {1, 0, 0}, .radius = 1});
        scene.Define<Sphere>("/env_0/Hidden",
                             {.visibility = Visibility::Invisible, .translate = {50, 0, 0}});
        scene.Clone(*scene.Find("/env_0"), std::vector<std::string>{"/env_1"});
        scene.Define<Sphere>("/Sun", {.translate = {0, 0, 1000}});
        scene.Define<World>("/empty");
        scene.Publish().Wait();
        ExpectBox(scene.Bounds(0), {0, -1, -1}, {2, 1, 1});
        ExpectBox(scene.Bounds(1), {0, -1, -1}, {2, 1, 1});
        EXPECT_FALSE(scene.Bounds(2));
        ExpectBox(scene.Bounds(), {-1, -1, 999}, {1, 1, 1001});
        ExpectPrim(*scene.Find("/env_0/Hidden"), {49, -1, -1}, {51, 1, 1});
        EXPECT_THROW(scene.Bounds(3), Error);
    });
}

TEST(BoundsTest, MovingPrimsAndPointsUpdateTheirBoxes) {
    ForEachScene([](Scene& scene) {
        const Prim ball = scene.Define<Sphere>("/Ball");
        const Prim quad = scene.Define<Mesh>(
            "/Quad", {.points = kQuad, .faceVertexIndices = {0, 1, 2, 3}, .faceVertexCounts = {4}});
        scene.Publish().Wait();
        ExpectBox(scene.Bounds(), {-1, -1, -1}, {1, 1, 1});
        ball.Set("translate", Double3{0, 0, -5});
        quad.Set(Mesh::kPoints, {{0, 0, 0}, {3, 0, 0}, {3, 3, 0}, {0, 3, 0}});
        scene.Publish().Wait();
        ExpectPrim(ball, {-1, -1, -6}, {1, 1, -4});
        ExpectPrim(quad, {0, 0, 0}, {3, 3, 0});
        ExpectBox(scene.Bounds(), {-1, -1, -6}, {3, 3, 0});
        scene.Remove(ball);
        scene.Publish().Wait();
        ExpectBox(scene.Bounds(), {0, 0, 0}, {3, 3, 0});
    });
}

}
}
