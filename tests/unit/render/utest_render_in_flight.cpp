// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/render/render_output.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/rotation.h"
#include "ernie/scene/scene.h"
#include "support/devices.h"

namespace ernie {
namespace {

using device::Device;
using test::ForEachDevice;

constexpr uint32_t kWidth = 9;
constexpr uint32_t kHeight = 7;
const std::vector<std::string> kProducts = {"/Render/View"};

Ref<Scene> BallScene(Device& device) {
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<Camera>("/Cam",
                          {.translate = {0, -5, 0}, .orient = LookAt({0, -5, 0}, {0, 0, 0})});
    scene->Define<Sphere>("/Ball", {.radius = 1});
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Depth", {.sourceName = "DistanceToCamera"});
    scene->Define<RenderProduct>(
        "/Render/View",
        {.camera = {"/Cam"}, .resolution = {kWidth, kHeight}, .orderedVars = {"/Render/Depth"}});
    scene->Select("/Render/View").Get<RenderMode>("renderMode").Set(RenderMode::Minimal);
    return scene;
}

float CenterDepth(RenderOutput& output) {
    output.Wait();
    const std::vector<float> depth =
        output["/Render/View"].frames.at(0).Source("DistanceToCamera").tensor.ToHost<float>();
    return depth[(kHeight / 2) * kWidth + kWidth / 2];
}

void MoveBall(Scene& scene, double y) {
    scene.Select("/Ball").Get<Double3>("translate").Set(Double3{0, y, 0});
}

TEST(InFlightTest, EachFrameSeesTheSceneItWasSteppedWith) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = BallScene(device);
        Renderer renderer(scene);
        std::vector<RenderOutput> outputs;
        for (const double y : {0.0, 1.0, 2.0}) {
            MoveBall(*scene, y);
            outputs.push_back(renderer.Step(kProducts, 1.0 / 30));
        }
        EXPECT_NEAR(CenterDepth(outputs[0]), 4, 1e-3);
        EXPECT_NEAR(CenterDepth(outputs[1]), 5, 1e-3);
        EXPECT_NEAR(CenterDepth(outputs[2]), 6, 1e-3);
    });
}

TEST(InFlightTest, RelayoutsWhileAFrameIsInFlightLeaveBothFramesRight) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = BallScene(device);
        Renderer renderer(scene);
        RenderOutput before = renderer.Step(kProducts, 1.0 / 30);
        scene->Define<World>("/Env0");
        scene->Define<Sphere>("/Env0/Ball", {.radius = 1});
        scene->Define<Sphere>("/Near", {.translate = {0, -2, 0}, .radius = 0.5});
        RenderOutput after = renderer.Step(kProducts, 1.0 / 30);
        scene->Remove(*scene->Find("/Near"));
        RenderOutput removed = renderer.Step(kProducts, 1.0 / 30);
        EXPECT_NEAR(CenterDepth(before), 4, 1e-3);
        EXPECT_NEAR(CenterDepth(after), 2.5, 1e-3);
        EXPECT_NEAR(CenterDepth(removed), 4, 1e-3);
    });
}

TEST(InFlightTest, ManyPublishesBetweenFramesKeepOnlyTheLatest) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = BallScene(device);
        Renderer renderer(scene);
        RenderOutput first = renderer.Step(kProducts, 1.0 / 30);
        EXPECT_NEAR(CenterDepth(first), 4, 1e-3);
        for (int step = 0; step < 8; ++step) {
            MoveBall(*scene, step % 2 == 0 ? 3.0 : -1.0);
            scene->Publish();
        }
        MoveBall(*scene, 1.0);
        RenderOutput output = renderer.Step(kProducts, 1.0 / 30);
        EXPECT_NEAR(CenterDepth(output), 5, 1e-3);
    });
}

}
}
