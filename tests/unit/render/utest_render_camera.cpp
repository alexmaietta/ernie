// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/render/render_output.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/scene.h"
#include "support/devices.h"

namespace ernie {
namespace {

using device::Device;
using test::ForEachDevice;

constexpr uint32_t kSize = 8;
constexpr float kWallDistance = 5;
const std::vector<std::string> kProducts = {"/Render/View"};

Ref<Scene> FacingWall(Device& device, const Camera::Desc& camera) {
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<Plane>("/Wall", {.translate = {0, 0, -kWallDistance}, .width = 0, .length = 0});
    scene->Define<Camera>("/Cam", camera);
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Depth", {.sourceName = RenderVarSource::DistanceToCamera});
    scene->Define<RenderVar>("/Render/Plane",
                             {.sourceName = RenderVarSource::DistanceToImagePlane});
    scene->Define<RenderProduct>("/Render/View",
                                 {.camera = {"/Cam"},
                                  .resolution = {kSize, kSize},
                                  .orderedVars = {"/Render/Depth", "/Render/Plane"}});
    return scene;
}

std::vector<float> Distances(Renderer& renderer, const char* source = "DistanceToCamera") {
    RenderOutput output = renderer.Step(kProducts, 0);
    output.Wait();
    return output["/Render/View"].frames.at(0).Source(source).tensor.ToHost<float>();
}

std::vector<float> WallDistances(Device& device, Float2 clippingRange) {
    Renderer renderer(FacingWall(device, {.focalLength = 10, .clippingRange = clippingRange}));
    return Distances(renderer);
}

TEST(CameraTest, FarClipsByDistanceAlongTheRay) {
    ForEachDevice([](Device& device) {
        const std::vector<float> depth = WallDistances(device, {1, 6});
        ASSERT_EQ(depth.size(), kSize * kSize);
        EXPECT_NEAR(depth[kSize / 2 * kSize + kSize / 2], kWallDistance, 0.1);
        EXPECT_TRUE(std::isinf(depth[0]));
    });
}

TEST(CameraTest, NearClipsByDistanceAlongTheRay) {
    ForEachDevice([](Device& device) {
        const std::vector<float> depth = WallDistances(device, {6, 100});
        ASSERT_EQ(depth.size(), kSize * kSize);
        EXPECT_TRUE(std::isinf(depth[kSize / 2 * kSize + kSize / 2]));
        EXPECT_GT(depth[0], 6);
    });
}

TEST(CameraTest, PerspectiveImagePlaneDistanceIsDepthAlongTheAxis) {
    ForEachDevice([](Device& device) {
        Renderer renderer(FacingWall(device, {.focalLength = 10}));
        const std::vector<float> depth = Distances(renderer, "DistanceToImagePlane");
        ASSERT_EQ(depth.size(), kSize * kSize);
        EXPECT_NEAR(depth[0], kWallDistance, 0.01);
        EXPECT_GT(Distances(renderer)[0], kWallDistance + 1);
    });
}

TEST(CameraTest, OrthographicRaysRunParallelToTheAxis) {
    ForEachDevice([](Device& device) {
        Renderer renderer(FacingWall(device, {.projection = Projection::Orthographic}));
        const std::vector<float> depth = Distances(renderer);
        ASSERT_EQ(depth.size(), kSize * kSize);
        EXPECT_NEAR(depth[0], kWallDistance, 0.01);
        EXPECT_NEAR(depth[kSize * kSize - 1], kWallDistance, 0.01);
        EXPECT_NEAR(Distances(renderer, "DistanceToImagePlane")[0], kWallDistance, 0.01);
    });
}

TEST(CameraTest, DepthOfFieldRaysMeetAtTheFocusDistance) {
    ForEachDevice([](Device& device) {
        Renderer renderer(FacingWall(device, {.fStop = 1, .focusDistance = kWallDistance}));
        const std::vector<float> depth = Distances(renderer, "DistanceToImagePlane");
        ASSERT_EQ(depth.size(), kSize * kSize);
        EXPECT_NEAR(depth[kSize / 2 * kSize + kSize / 2], kWallDistance, 0.01);
    });
}

TEST(CameraTest, MovingTheCameraUpdatesItsRays) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = FacingWall(device, {});
        Renderer renderer(scene);
        const uint32_t center = kSize / 2 * kSize + kSize / 2;
        EXPECT_NEAR(Distances(renderer)[center], kWallDistance, 0.05);
        scene->Select("/Cam").Get<Double3>("translate").Set(Double3{0, 0, 1});
        EXPECT_NEAR(Distances(renderer)[center], kWallDistance + 1, 0.05);
    });
}

}  // namespace
}  // namespace ernie
