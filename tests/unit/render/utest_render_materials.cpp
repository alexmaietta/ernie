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

constexpr uint32_t kSide = 8;

Ref<Scene> LookingDown(Device& device, MinimalMode mode) {
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<Camera>("/Cam", {.translate = {0, 0, 5},
                                   .orient = LookAt({0, 0, 5}, {0, 0, 0}, {0, 1, 0}),
                                   .focalLength = 30});
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Color", {.sourceName = RenderVarSource::HdrColor});
    scene->Define<RenderProduct>(
        "/Render/View",
        {.camera = {"/Cam"}, .resolution = {kSide, kSide}, .orderedVars = {"/Render/Color"}});
    scene->Find("/Render/View")->Set(MinimalAPI::kMode, mode);
    return scene;
}

std::vector<Float3> Colors(Renderer& renderer) {
    RenderOutput output = renderer.Step(std::vector<std::string>{"/Render/View"}, 0);
    output.Wait();
    const std::vector<float> color =
        output["/Render/View"].frames.at(0).Source("HdrColor").tensor.ToHost<float>();
    std::vector<Float3> pixels;
    for (size_t at = 0; at + 3 < color.size(); at += 4) {
        pixels.push_back({color[at], color[at + 1], color[at + 2]});
    }
    return pixels;
}

TEST(MaterialsTest, CheckerboardsAlternateTheirColorsAndFollowEdits) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = LookingDown(device, MinimalMode::TextureDiffuse);
        const Prim plane = scene->Define<Plane>("/Floor", {.width = 40, .length = 40});
        plane.Set(CheckerboardAPI::kCheckerboard, true);
        plane.Set(CheckerboardAPI::kCheckerColorA, Float3{1, 0, 0});
        plane.Set(CheckerboardAPI::kCheckerColorB, Float3{0, 0, 1});
        plane.Set(CheckerboardAPI::kCheckerSize, 0.5f);
        Renderer renderer(scene);
        const auto count = [](const std::vector<Float3>& pixels, bool red) {
            uint32_t found = 0;
            for (const Float3& pixel : pixels) {
                found += (red ? pixel.x > pixel.z : pixel.z > pixel.x) ? 1 : 0;
            }
            return found;
        };
        std::vector<Float3> pixels = Colors(renderer);
        EXPECT_GT(count(pixels, true), 0u);
        EXPECT_GT(count(pixels, false), 0u);

        plane.Set(CheckerboardAPI::kCheckerboard, false);
        pixels = Colors(renderer);
        EXPECT_EQ(count(pixels, false), 0u);
    });
}

TEST(MaterialsTest, PreviewSurfaceEditsReachTheImage) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = LookingDown(device, MinimalMode::Full);
        scene->Define<Scope>("/Looks");
        const Prim look =
            scene->Define<UsdPreviewSurface>("/Looks/Paint", {.diffuseColor = {1, 0, 0}});
        scene->Define<Mesh>("/Quad", {.materialBinding = {"/Looks/Paint"},
                                      .points = {{-9, -9, 0}, {9, -9, 0}, {9, 9, 0}, {-9, 9, 0}},
                                      .faceVertexIndices = {0, 1, 2, 3},
                                      .faceVertexCounts = {4},
                                      .subdivisionScheme = SubdivisionScheme::None});
        Renderer renderer(scene);
        const uint32_t center = (kSide / 2) * kSide + kSide / 2;
        Float3 pixel = Colors(renderer)[center];
        EXPECT_GT(pixel.x, pixel.y);
        look.Set(UsdPreviewSurface::kDiffuseColor, Float3{0, 1, 0});
        pixel = Colors(renderer)[center];
        EXPECT_GT(pixel.y, pixel.x);
    });
}

TEST(MaterialsTest, NonvisualReflectanceScalesLidarIntensity) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<Scope>("/Looks");
        const Prim look = scene->Define<UsdPreviewSurface>("/Looks/Wall");
        scene->Apply<NonVisualMaterialAPI>(look, {.nonvisualLidarReflectance = 1});
        scene->Define<Plane>("/Wall", {.translate = {0, 0, -5}, .width = 0, .length = 0})
            .Set(MaterialBindingAPI::kMaterialBinding, std::vector<std::string>{"/Looks/Wall"});
        scene->Define<Lidar>(
            "/Lidar",
            {.scanType = ScanType::SolidState, .firingsPerScan = 1, .emitterElevations = {0}});
        scene->Define<Scope>("/Render");
        scene->Define<RenderVar>("/Render/Points", {.sourceName = RenderVarSource::PointCloud,
                                                    .channels = {"Intensity"}});
        scene->Define<RenderProduct>("/Render/Scan",
                                     {.camera = {"/Lidar"}, .orderedVars = {"/Render/Points"}});
        Renderer renderer(scene);
        const auto intensity = [&] {
            RenderOutput output = renderer.Step(std::vector<std::string>{"/Render/Scan"}, 0);
            output.Wait();
            const std::vector<float> values = output["/Render/Scan"]
                                                  .frames.at(0)
                                                  .Source("PointCloud")
                                                  .channels.at("Intensity")
                                                  .ToHost<float>();
            return values.empty() ? 0.0f : values.front();
        };
        const float full = intensity();
        ASSERT_GT(full, 0);
        look.Set(NonVisualMaterialAPI::kNonvisualLidarReflectance, 0.25f);
        EXPECT_NEAR(intensity() / full, 0.25f, 1e-3f);
    });
}

}
}
