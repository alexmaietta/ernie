// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/render/renderer.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/render/render_output.h"
#include "ernie/scene/rotation.h"
#include "ernie/scene/scene.h"
#include "ernie/util/error.h"
#include "support/devices.h"
#include "support/support.h"

namespace ernie {
namespace {

using device::Device;
using test::ForEachDevice;

constexpr uint32_t kWidth = 32;
constexpr uint32_t kHeight = 24;
const std::vector<std::string> kProducts = {"/Render/View"};

Ref<Scene> TwoWorlds(Device& device, RenderMode mode) {
    const Ref<Scene> scene = Scene::Create(device);
    for (const char* world : {"/env_0", "/env_1"}) {
        scene->Define<World>(world);
        scene->Define<Camera>(std::string(world) + "/Cam",
                              {.translate = {0, -5, 0}, .orient = LookAt({0, -5, 0}, {0, 0, 0})});
    }
    scene->Define<Sphere>("/env_0/Ball", {.radius = 1});
    scene->Define<Sphere>("/env_1/Ball", {.translate = {0, 0, 50}, .radius = 1});
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Color", {.sourceName = RenderVarSource::LdrColor});
    scene->Define<RenderVar>("/Render/Depth", {.sourceName = RenderVarSource::DistanceToCamera});
    scene->Define<RenderProduct>("/Render/View",
                                 {.camera = {"/env_0/Cam", "/env_1/Cam"},
                                  .resolution = {kWidth, kHeight},
                                  .orderedVars = {"/Render/Color", "/Render/Depth"}});
    scene->Select("/Render/View").Get<RenderMode>("renderMode").Set(mode);
    return scene;
}

struct Views {
    std::vector<float> depth;
    std::vector<uint8_t> color;

    float Depth(uint32_t camera, uint32_t x, uint32_t y) const {
        return depth[(camera * kHeight + y) * kWidth + x];
    }

    uint8_t Alpha(uint32_t camera, uint32_t x, uint32_t y) const {
        return color[((camera * kHeight + y) * kWidth + x) * 4 + 3];
    }
};

Views Render(Renderer& renderer, uint32_t steps = 1) {
    RenderOutput output;
    for (uint32_t i = 0; i < steps; ++i) {
        output = renderer.Step(kProducts, 1.0 / 30);
    }
    output.Wait();
    const FrameOutput& frame = output["/Render/View"].frames.at(0);
    return {frame.Source("DistanceToCamera").tensor.ToHost<float>(),
            frame.Source("LdrColor").tensor.ToHost<uint8_t>()};
}

void ExpectEachWorldSeesOnlyItsOwnBall(RenderMode mode) {
    ForEachDevice([&](Device& device) {
        const Ref<Scene> scene = TwoWorlds(device, mode);
        Renderer renderer(scene);
        const Views views = Render(renderer, 2);
        ASSERT_EQ(views.depth.size(), 2u * kWidth * kHeight);
        EXPECT_NEAR(views.Depth(0, kWidth / 2, kHeight / 2), 4, 0.02);
        EXPECT_EQ(views.Alpha(0, kWidth / 2, kHeight / 2), 255);
        EXPECT_TRUE(std::isinf(views.Depth(0, 0, 0)));
        EXPECT_EQ(views.Alpha(0, 0, 0), 0);
        EXPECT_TRUE(std::isinf(views.Depth(1, kWidth / 2, kHeight / 2)));

        scene->Select("/env_1/Ball").Get<Double3>("translate").Set(Double3{0, 0, 0});
        EXPECT_NEAR(Render(renderer).Depth(1, kWidth / 2, kHeight / 2), 4, 0.02);
    });
}

TEST(RendererTest, MinimalModeShowsEachWorldOnlyItsOwnPrims) {
    ExpectEachWorldSeesOnlyItsOwnBall(RenderMode::Minimal);
}

TEST(RendererTest, RealTimeModeShowsEachWorldOnlyItsOwnPrims) {
    ExpectEachWorldSeesOnlyItsOwnBall(RenderMode::RealTime);
}

TEST(RendererTest, StepsAdvanceTimeUntilReset) {
    ForEachDevice([](Device& device) {
        Renderer renderer(TwoWorlds(device, RenderMode::Minimal));
        EXPECT_EQ(renderer.Time(), 0);
        RenderOutput output = renderer.Step(kProducts, 0.25);
        output = renderer.Step(kProducts, 0.25);
        output.Wait();
        EXPECT_TRUE(output.Ready());
        EXPECT_DOUBLE_EQ(renderer.Time(), 0.5);
        EXPECT_DOUBLE_EQ(output.simulationStartTime, 0.25);
        EXPECT_DOUBLE_EQ(output.simulationEndTime, 0.5);
        EXPECT_TRUE(output.Contains("/Render/View"));
        EXPECT_EQ(output.Size(), 1u);
        renderer.Reset(3);
        EXPECT_DOUBLE_EQ(renderer.Time(), 3);
        EXPECT_THROW(renderer.Step(std::vector<std::string>{"/Render/Nothing"}, 0.1), Error);
    });
}

TEST(RendererTest, WritesFramesAsImages) {
    ForEachDevice([](Device& device) {
        Renderer renderer(TwoWorlds(device, RenderMode::Minimal));
        RenderOutput output = renderer.Step(kProducts, 0.1);
        output.Wait();
        const FrameOutput& frame = output["/Render/View"].frames.at(0);
        const test::TemporaryDirectory directory;
        const std::vector<std::filesystem::path> written = WriteFrame(directory.Path(), frame);
        EXPECT_FALSE(written.empty());
        for (const std::filesystem::path& path : written) {
            EXPECT_TRUE(std::filesystem::exists(path)) << path;
        }

        ImageWriter writer(2);
        const std::vector<std::filesystem::path> queued =
            writer.WriteFrame(directory.Path() / "async", frame);
        writer.Wait();
        EXPECT_EQ(writer.Pending(), 0u);
        EXPECT_EQ(queued.size(), written.size());
        for (const std::filesystem::path& path : queued) {
            EXPECT_TRUE(std::filesystem::exists(path)) << path;
        }
        EXPECT_THROW(frame.Source("Normal"), Error);
    });
}

}
}
