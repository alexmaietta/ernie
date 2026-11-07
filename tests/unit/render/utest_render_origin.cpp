// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
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

constexpr uint32_t kWidth = 15;
constexpr uint32_t kHeight = 11;
constexpr double kAu = 1.495978707e11;
const std::vector<std::string> kProducts = {"/Render/View"};

void DefineProduct(Scene& scene, const std::vector<std::string>& cameras, RenderMode mode) {
    scene.Define<Scope>("/Render");
    scene.Define<RenderVar>("/Render/Color", {.sourceName = RenderVarSource::HdrColor});
    scene.Define<RenderVar>("/Render/Depth", {.sourceName = RenderVarSource::DistanceToCamera});
    scene.Define<RenderProduct>("/Render/View",
                                {.camera = cameras,
                                 .resolution = {kWidth, kHeight},
                                 .orderedVars = {"/Render/Color", "/Render/Depth"}});
    scene.Select("/Render/View").Get<RenderMode>("renderMode").Set(mode);
}

void DefineCamera(Scene& scene, const std::string& path, Double3 eye, Double3 target) {
    scene.Define<Camera>(path, {.translate = eye, .orient = LookAt(eye, target, {0, 0, 1})});
}

struct Frames {
    std::vector<float> color;
    std::vector<float> depth;

    float Depth(uint32_t camera) const {
        return depth[(camera * kHeight + kHeight / 2) * kWidth + kWidth / 2];
    }

    float Red(uint32_t camera) const {
        return color[((camera * kHeight + kHeight / 2) * kWidth + kWidth / 2) * 4];
    }
};

Frames Render(Renderer& renderer) {
    RenderOutput output = renderer.Step(kProducts, 1.0 / 30);
    output.Wait();
    const FrameOutput& frame = output["/Render/View"].frames.at(0);
    return {frame.Source("HdrColor").tensor.ToHost<float>(),
            frame.Source("DistanceToCamera").tensor.ToHost<float>()};
}

TEST(RenderOriginTest, EachWorldRendersPreciselyNearItsFarAwayCamera) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<World>("/env_0");
        scene->Define<World>("/env_1");
        DefineCamera(*scene, "/env_0/Cam", {kAu, 0, 0}, {kAu, 1, 0});
        scene->Define<Sphere>("/env_0/Pebble", {.translate = {kAu, 2, 0}, .radius = 0.01});
        DefineCamera(*scene, "/env_1/Cam", {-0.5 * kAu, 3e8, 0}, {-0.5 * kAu, 3e8 + 1, 0});
        scene->Define<Sphere>("/env_1/Pebble",
                              {.translate = {-0.5 * kAu, 3e8 + 1.5, 0}, .radius = 0.1});
        DefineProduct(*scene, {"/env_0/Cam", "/env_1/Cam"}, RenderMode::Minimal);
        Renderer renderer(scene);
        const Frames frames = Render(renderer);
        EXPECT_NEAR(frames.Depth(0), 1.99, 1e-4);
        EXPECT_NEAR(frames.Depth(1), 1.4, 1e-4);
    });
}

TEST(RenderOriginTest, FollowsTheCameraAsItMovesAndSwitches) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        DefineCamera(*scene, "/Cam", {kAu, 0, 0}, {kAu, 1, 0});
        DefineCamera(*scene, "/Other", {-kAu, 0, 5e9}, {-kAu, 1, 5e9});
        scene->Define<Sphere>("/Near", {.translate = {kAu, 2, 0}, .radius = 0.5});
        scene->Define<Sphere>("/Ahead", {.translate = {kAu + 5000, 3, 0}, .radius = 0.5});
        scene->Define<Sphere>("/Away", {.translate = {-kAu, 1.3, 5e9}, .radius = 0.05});
        DefineProduct(*scene, {"/Cam"}, RenderMode::Minimal);
        Renderer renderer(scene);
        EXPECT_NEAR(Render(renderer).Depth(0), 1.5, 1e-4);

        scene->Select("/Cam").Get<Double3>("translate").Set(Double3{kAu + 5000, 0, 0});
        EXPECT_NEAR(Render(renderer).Depth(0), 2.5, 1e-4);

        scene->Select("/Render/View").Get("camera").SetTargets(0, {"/Other"});
        EXPECT_NEAR(Render(renderer).Depth(0), 1.25, 1e-4);
    });
}

Frames RenderLitGround(Device& device, double x) {
    const Ref<Scene> scene = Scene::Create(device);
    DefineCamera(*scene, "/Cam", {x, -3, 1}, {x, 0, 0});
    scene->Define<Mesh>("/Ground", {.translate = {x, 0, 0},
                                    .points = {{-5, -5, 0}, {5, -5, 0}, {5, 5, 0}, {-5, 5, 0}},
                                    .faceVertexIndices = {0, 1, 2, 3},
                                    .faceVertexCounts = {4}});
    scene->Define<SphereLight>("/Lamp",
                               {.translate = {x + 1, 0, 1}, .intensity = 100, .radius = 0.1});
    DefineProduct(*scene, {"/Cam"}, RenderMode::RealTime);
    Renderer renderer(scene);
    return Render(renderer);
}

TEST(RenderOriginTest, LightsSitWhereTheSurfacesDoFarFromTheOrigin) {
    ForEachDevice([](Device& device) {
        const Frames near = RenderLitGround(device, 0);
        const Frames far = RenderLitGround(device, kAu);
        EXPECT_GT(near.Red(0), 0);
        EXPECT_NEAR(far.Red(0), near.Red(0), 1e-3 * near.Red(0));
        EXPECT_NEAR(far.Depth(0), near.Depth(0), 1e-4);
    });
}

}
}
