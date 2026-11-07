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

constexpr uint32_t kWidth = 32;
constexpr uint32_t kHeight = 24;
constexpr double kFar = 10000;
constexpr double kAu = 1.495978707e11;
const std::vector<std::string> kProducts = {"/Render/View"};

struct Frame {
    std::vector<float> color;
    std::vector<float> normal;
    std::vector<float> depth;

    float Red(uint32_t x, uint32_t y) const {
        return color[(y * kWidth + x) * 4];
    }
};

Ref<Scene> SunScene(Device& device, Double3 eye, Double3 target, Double3 toward,
                    RenderMode mode = RenderMode::Minimal, float focalLength = 50) {
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<Camera>(
        "/Cam",
        {.translate = eye, .orient = LookAt(eye, target, {0, 1, 0}), .focalLength = focalLength});
    scene->Define<DistantLight>("/Sun", {.orient = LookAlong({-toward.x, -toward.y, -toward.z})});
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Color", {.sourceName = RenderVarSource::HdrColor});
    scene->Define<RenderVar>("/Render/Normal", {.sourceName = RenderVarSource::Normal});
    scene->Define<RenderVar>("/Render/Depth", {.sourceName = RenderVarSource::DistanceToCamera});
    scene->Define<RenderProduct>(
        "/Render/View", {.camera = {"/Cam"},
                         .resolution = {kWidth, kHeight},
                         .orderedVars = {"/Render/Color", "/Render/Normal", "/Render/Depth"}});
    scene->Select("/Render/View").Get<RenderMode>("renderMode").Set(mode);
    return scene;
}

Frame Render(Renderer& renderer) {
    RenderOutput output = renderer.Step(kProducts, 1.0 / 30);
    output.Wait();
    const FrameOutput& frame = output["/Render/View"].frames.at(0);
    return {frame.Source("HdrColor").tensor.ToHost<float>(),
            frame.Source("Normal").tensor.ToHost<float>(),
            frame.Source("DistanceToCamera").tensor.ToHost<float>()};
}

void DefineGround(Scene& scene, bool mesh, Double3 at) {
    if (mesh) {
        scene.Define<Mesh>("/Ground",
                           {.translate = at,
                            .points = {{-50, -50, 0}, {50, -50, 0}, {50, 50, 0}, {-50, 50, 0}},
                            .faceVertexIndices = {0, 1, 2, 3},
                            .faceVertexCounts = {4}});
    } else {
        scene.Define<Cube>("/Ground", {.translate = {at.x, at.y, at.z - 50}, .size = 100});
    }
}

void ExpectShortWallShadowsTheGroundBehindIt(bool mesh, RenderMode mode = RenderMode::Minimal,
                                             double distance = kFar) {
    ForEachDevice([&](Device& device) {
        const Double3 behind = {distance - 0.1, 0, 0};
        const double elevation = std::sqrt(0.5);
        const Ref<Scene> scene =
            SunScene(device, {behind.x, 0, 2000}, behind, {elevation, 0, elevation}, mode, 30000);
        DefineGround(*scene, mesh, {distance, 0, 0});
        scene->Define<Cube>("/Wall", {.translate = {distance, 0, 0.05}, .size = 0.1});
        Renderer renderer(scene);
        EXPECT_EQ(Render(renderer).Red(kWidth / 2, kHeight / 2), 0);

        scene->Select("/Wall").Get<Double3>("translate").Set(Double3{distance, 20, 0.05});
        EXPECT_GT(Render(renderer).Red(kWidth / 2, kHeight / 2), 0);
    });
}

TEST(RayOriginTest, ShortWallsShadowMeshGroundSeenFromAfar) {
    ExpectShortWallShadowsTheGroundBehindIt(true);
}

TEST(RayOriginTest, ShortWallsShadowShapeGroundSeenFromAfar) {
    ExpectShortWallShadowsTheGroundBehindIt(false);
}

TEST(RayOriginTest, ShortWallsShadowGroundInRealTimeMode) {
    ExpectShortWallShadowsTheGroundBehindIt(true, RenderMode::RealTime);
}

TEST(RayOriginTest, ShortWallsShadowGroundAnAstronomicalUnitAway) {
    ExpectShortWallShadowsTheGroundBehindIt(true, RenderMode::Minimal, kAu);
    ExpectShortWallShadowsTheGroundBehindIt(false, RenderMode::RealTime, kAu);
}

struct Sunlit {
    uint32_t facing = 0;
    uint32_t dark = 0;
};

Sunlit CountSunlit(const Frame& frame, Double3 toward) {
    Sunlit count;
    for (uint32_t pixel = 0; pixel < kWidth * kHeight; ++pixel) {
        if (!std::isfinite(frame.depth[pixel])) {
            continue;
        }
        const float* n = &frame.normal[pixel * 3];
        if (n[0] * toward.x + n[1] * toward.y + n[2] * toward.z > 0.02) {
            ++count.facing;
            count.dark += frame.color[pixel * 4] <= 0 ? 1 : 0;
        }
    }
    return count;
}

TEST(RayOriginTest, GrazingSunDoesNotShadowFarSurfacesOnThemselves) {
    ForEachDevice([](Device& device) {
        const double elevation = 3.0 * 3.14159265358979 / 180;
        const Double3 toward = {std::cos(elevation), 0, std::sin(elevation)};
        const Double3 center = {kFar, 0, 0};

        const Ref<Scene> ground = SunScene(device, {kFar - 4, -3, 2}, center, toward);
        DefineGround(*ground, true, center);
        ground->Select("/Ground").Get<Quatd>("orient").Set(FromAxisAngle({1, 1, 0}, 0.7));
        Renderer flat(ground);
        const Sunlit plane = CountSunlit(Render(flat), toward);
        EXPECT_EQ(plane.facing, kWidth * kHeight);
        EXPECT_EQ(plane.dark, 0u);

        const Ref<Scene> shapes = SunScene(device, {kFar, -20, 300}, {kFar, -20, 0}, toward);
        shapes->Define<Sphere>("/Ball", {.translate = center, .radius = 50});
        shapes->Define<Cube>(
            "/Box",
            {.translate = {kFar, -56, 0}, .orient = FromAxisAngle({1, 2, 3}, 20), .size = 6});
        Renderer round(shapes);
        const Sunlit ball = CountSunlit(Render(round), toward);
        EXPECT_GT(ball.facing, kWidth * kHeight / 4);
        EXPECT_EQ(ball.dark, 0u);
    });
}

}
}
