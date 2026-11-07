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

std::vector<float> WallDistances(Device& device, Float2 clippingRange) {
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<Plane>("/Wall", {.translate = {0, 0, -kWallDistance}, .width = 0, .length = 0});
    scene->Define<Camera>("/Cam", {.focalLength = 10, .clippingRange = clippingRange});
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Depth", {.sourceName = RenderVarSource::DistanceToCamera});
    scene->Define<RenderProduct>(
        "/Render/View",
        {.camera = {"/Cam"}, .resolution = {kSize, kSize}, .orderedVars = {"/Render/Depth"}});
    Renderer renderer(scene);
    RenderOutput output = renderer.Step(kProducts, 0);
    output.Wait();
    return output["/Render/View"].frames.at(0).Source("DistanceToCamera").tensor.ToHost<float>();
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

}  // namespace
}  // namespace ernie
