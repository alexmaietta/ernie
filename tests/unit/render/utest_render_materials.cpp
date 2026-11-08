// SPDX-License-Identifier: BSD-3-Clause
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/io/image.h"
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
        scene->DefineMaterial("/Looks/Paint", {.inputs = {{"diffuseColor", Float3{1, 0, 0}}}});
        const Prim look = *scene->Find("/Looks/Paint/Surface");
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

TEST(MaterialsTest, TexturesConnectedThroughUsdUVTextureReachTheImage) {
    const std::filesystem::path red =
        std::filesystem::temp_directory_path() / "ernie_render_materials_red.png";
    WritePng(red, Image(4, 4, Rgba{1, 0, 0, 1}));
    ForEachDevice([&](Device& device) {
        const Ref<Scene> scene = LookingDown(device, MinimalMode::TextureDiffuse);
        scene->Define<Scope>("/Looks");
        scene->DefineMaterial("/Looks/Paint",
                              {.inputs = {{"diffuseColor", Float3{0, 0, 1}}},
                               .textures = {{.input = "diffuseColor", .file = red.string()}}});
        const std::vector<Float2> st = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        scene->Define<Mesh>("/Quad",
                            {.materialBinding = {"/Looks/Paint"},
                             .primvars = {Primvar::Create("st", Interpolation::Vertex, st)},
                             .points = {{-9, -9, 0}, {9, -9, 0}, {9, 9, 0}, {-9, 9, 0}},
                             .faceVertexIndices = {0, 1, 2, 3},
                             .faceVertexCounts = {4},
                             .subdivisionScheme = SubdivisionScheme::None});
        Renderer renderer(scene);
        const uint32_t center = (kSide / 2) * kSide + kSide / 2;
        Float3 pixel = Colors(renderer)[center];
        for (int attempt = 0; attempt < 500 && pixel.x <= pixel.z; ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
            pixel = Colors(renderer)[center];
        }
        EXPECT_GT(pixel.x, pixel.z);
        scene->Find("/Looks/Paint/Surface")->Disconnect("diffuseColor");
        pixel = Colors(renderer)[center];
        EXPECT_GT(pixel.z, pixel.x);
    });
}

TEST(MaterialsTest, NonvisualReflectanceScalesLidarIntensity) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<Scope>("/Looks");
        const Prim look = scene->DefineMaterial("/Looks/Wall");
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

TEST(MaterialsTest, ShadowLinksLetExcludedBlockersPassLight) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = LookingDown(device, MinimalMode::Full);
        scene->Find("/Cam")->Set(Xformable::kTranslate, Double3{0, 0, 1});
        scene->Find("/Render/View")->Set(RenderProduct::kRenderMode, RenderMode::RealTime);
        const Prim sun = scene->Define<DistantLight>("/Sun", {.orient = LookAlong({0, 0, -1})});
        scene->Define<Cube>("/Blocker", {.translate = {0, 0, 3}, .size = 4});
        scene->Define<Mesh>("/Ground", {.points = {{-9, -9, 0}, {9, -9, 0}, {9, 9, 0}, {-9, 9, 0}},
                                        .faceVertexIndices = {0, 1, 2, 3},
                                        .faceVertexCounts = {4},
                                        .subdivisionScheme = SubdivisionScheme::None});
        Renderer renderer(scene);
        const uint32_t center = (kSide / 2) * kSide + kSide / 2;
        const float shadowed = Colors(renderer)[center].x;
        sun.Set(CollectionAPI::kCollectionExcludes, "shadowLink", {"/Blocker"});
        const float lit = Colors(renderer)[center].x;
        EXPECT_GT(lit, 2 * shadowed + 1e-3f);
    });
}

TEST(MaterialsTest, BackgroundsFollowTheirPrimAndItsEdits) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = LookingDown(device, MinimalMode::DisplayColor);
        const Prim fill = scene->Define<ColorBackground>("/Fill", {.color = {0, 1, 0}});
        scene->Find("/Render/View")
            ->Set(BackgroundAPI::kBackground, std::vector<std::string>{"/Fill"});
        Renderer renderer(scene);
        Float3 pixel = Colors(renderer)[0];
        EXPECT_GT(pixel.y, 0.5f);
        EXPECT_LT(pixel.x, 0.1f);
        fill.Set(ColorBackground::kColor, Float3{1, 0, 0});
        pixel = Colors(renderer)[0];
        EXPECT_GT(pixel.x, 0.5f);
        EXPECT_LT(pixel.y, 0.1f);
    });
}

}
}
