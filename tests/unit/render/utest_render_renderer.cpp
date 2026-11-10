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
    scene->Define<RenderVar>("/Render/Color", {.sourceName = "LdrColor"});
    scene->Define<RenderVar>("/Render/Depth", {.sourceName = "DistanceToCamera"});
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
        const Ref<Scene> scene = TwoWorlds(device, RenderMode::Minimal);
        Renderer renderer(scene);
        EXPECT_EQ(scene->Time(), 0);
        RenderOutput output = renderer.Step(kProducts, 0.25);
        output = renderer.Step(kProducts, 0.25);
        output.Wait();
        EXPECT_TRUE(output.Ready());
        EXPECT_DOUBLE_EQ(scene->Time(), 0.5);
        EXPECT_DOUBLE_EQ(output.simulationStartTime, 0.25);
        EXPECT_DOUBLE_EQ(output.simulationEndTime, 0.5);
        EXPECT_TRUE(output.Contains("/Render/View"));
        EXPECT_EQ(output.Size(), 1u);
        scene->SetTime(3);
        EXPECT_DOUBLE_EQ(scene->Time(), 3);
        EXPECT_THROW(renderer.Step("/Render/Nothing", 0.1), Error);
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

std::vector<Float3> Roof(float ridge) {
    return {{-1, -1, 0}, {0, -1, ridge}, {1, -1, 0}, {-1, 1, 0}, {0, 1, ridge}, {1, 1, 0}};
}

Float3 CenterNormal(Renderer& renderer, uint32_t side) {
    RenderOutput output = renderer.Step("/Render/Roof", 1.0 / 30);
    output.Wait();
    const std::vector<float> normals =
        output["/Render/Roof"].frames.at(0).Source("Normal").tensor.ToHost<float>();
    const uint32_t center = ((side / 2) * side + side / 2) * 3;
    return {normals[center], normals[center + 1], normals[center + 2]};
}

TEST(RendererTest, MinimalModeLightsEachWorldWithItsOwnDistantLightFirst) {
    ForEachDevice([](Device& device) {
        constexpr uint32_t kSide = 8;
        const Ref<Scene> scene = Scene::Create(device);
        const Prim sun = scene->Define<DistantLight>("/Sun");
        for (const char* world : {"/env_0", "/env_1"}) {
            scene->Define<World>(world);
            scene->Define<Camera>(std::string(world) + "/Cam",
                                  {.translate = {0, 0, 5},
                                   .orient = LookAt({0, 0, 5}, {0, 0, 0}, {0, 1, 0}),
                                   .focalLength = 400});
            scene->Define<Mesh>(std::string(world) + "/Ground",
                                {.points = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}},
                                 .faceVertexIndices = {0, 1, 2, 3},
                                 .faceVertexCounts = {4}});
        }
        scene->Define<DistantLight>("/env_1/Under", {.orient = FromAxisAngle({1, 0, 0}, 180)});
        scene->Define<Scope>("/Render");
        scene->Define<RenderVar>("/Render/Color", {.sourceName = "HdrColor"});
        scene->Define<RenderProduct>("/Render/View", {.camera = {"/env_0/Cam", "/env_1/Cam"},
                                                      .resolution = {kSide, kSide},
                                                      .orderedVars = {"/Render/Color"}});
        scene->Find("/Render/View")->Set(MinimalAPI::kMode, MinimalMode::DisplayColor);
        Renderer renderer(scene);
        const auto red = [&](uint32_t camera) {
            RenderOutput output = renderer.Step(kProducts, 1.0 / 30);
            output.Wait();
            const std::vector<float> colors =
                output["/Render/View"].frames.at(0).Source("HdrColor").tensor.ToHost<float>();
            return colors[((camera * kSide + kSide / 2) * kSide + kSide / 2) * 4];
        };
        const float above = red(0);
        const float under = red(1);
        EXPECT_GT(above, under);

        sun.Set(LightAPI::kIntensity, 2 * sun.Get(LightAPI::kIntensity));
        EXPECT_NEAR(red(0), 2 * above, 1e-3f * above);
        EXPECT_FLOAT_EQ(red(1), under);
    });
}

TEST(RendererTest, MinimalPinholeRaysMatchTheGenericCameraPath) {
    ForEachDevice([](Device& device) {
        constexpr uint32_t kWide = 24;
        constexpr uint32_t kTall = 16;
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<DistantLight>("/Sun", {.orient = LookAlong({1, -1, -2})});
        scene->Define<Mesh>("/Ground", {.points = {{-6, -6, 0}, {6, -6, 0}, {6, 6, 0}, {-6, 6, 0}},
                                        .faceVertexIndices = {0, 1, 2, 3},
                                        .faceVertexCounts = {4}});
        scene->Define<Sphere>("/Ball", {.translate = {0.5, 0, 1}, .radius = 1});
        scene->Define<Cube>("/Box", {.translate = {-2, 1, 0.5}, .size = 1});
        for (const char* path : {"/Pinhole", "/Lens"}) {
            const Prim camera = scene->Define<Camera>(
                path, {.translate = {1, -7, 4}, .orient = LookAt({1, -7, 4}, {0, 0, 0.5})});
            camera.Set(Camera::kHorizontalApertureOffset, 1.5f);
            camera.Set(Camera::kClippingRange, Float2{2, 40});
        }
        scene->Find("/Lens")->Set(LensDistortionAPI::kDistortionModel,
                                  DistortionModel::BrownConrady);
        scene->Define<Scope>("/Render");
        scene->Define<RenderVar>("/Render/Color", {.sourceName = "HdrColor"});
        scene->Define<RenderVar>("/Render/Depth", {.sourceName = "DistanceToCamera"});
        scene->Define<RenderVar>("/Render/Normal", {.sourceName = "Normal"});
        scene->Define<RenderProduct>(
            "/Render/View", {.camera = {"/Pinhole", "/Lens"},
                             .resolution = {kWide, kTall},
                             .orderedVars = {"/Render/Color", "/Render/Depth", "/Render/Normal"}});
        scene->Find("/Render/View")->Set(MinimalAPI::kMode, MinimalMode::DisplayColor);
        Renderer renderer(scene);
        const auto compare = [&]() -> uint32_t {
            RenderOutput output = renderer.Step(kProducts, 1.0 / 30);
            output.Wait();
            const FrameOutput& frame = output["/Render/View"].frames.at(0);
            const std::vector<float> color = frame.Source("HdrColor").tensor.ToHost<float>();
            const std::vector<float> depth =
                frame.Source("DistanceToCamera").tensor.ToHost<float>();
            const std::vector<float> normal = frame.Source("Normal").tensor.ToHost<float>();
            const uint32_t pixels = kWide * kTall;
            uint32_t hits = 0;
            uint32_t misses = 0;
            for (uint32_t i = 0; i < pixels; ++i) {
                const float pinhole = depth[i];
                const float lens = depth[pixels + i];
                EXPECT_EQ(std::isinf(pinhole), std::isinf(lens)) << i;
                if (std::isinf(pinhole) || std::isinf(lens)) {
                    ++misses;
                    continue;
                }
                ++hits;
                EXPECT_NEAR(pinhole, lens, 1e-4f * pinhole) << i;
                for (uint32_t c = 0; c < 3; ++c) {
                    EXPECT_NEAR(normal[i * 3 + c], normal[(pixels + i) * 3 + c], 1e-3f) << i;
                    EXPECT_NEAR(color[i * 4 + c], color[(pixels + i) * 4 + c],
                                1e-3f * std::max(color[i * 4 + c], 1.0f))
                        << i;
                }
            }
            EXPECT_GT(hits, pixels / 2);
            return misses;
        };
        compare();
        for (const char* path : {"/Pinhole", "/Lens"}) {
            scene->Find(path)->Set(Camera::kClippingPlanes, std::vector<Float4>{{1, 0, 0, 0.5f}});
        }
        EXPECT_GT(compare(), 0u);
    });
}

TEST(RendererTest, MeshesShadeWithSmoothNormalsThatFollowDeformation) {
    ForEachDevice([](Device& device) {
        constexpr uint32_t kSide = 9;
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<Camera>("/Cam", {.translate = {-0.5, 0, 5},
                                       .orient = LookAt({-0.5, 0, 5}, {-0.5, 0, 0}, {0, 1, 0}),
                                       .focalLength = 400});
        scene->Define<Mesh>("/Roof", {.points = Roof(1),
                                      .faceVertexIndices = {0, 1, 4, 3, 1, 2, 5, 4},
                                      .faceVertexCounts = {4, 4}});
        scene->Define<Scope>("/Render");
        scene->Define<RenderVar>("/Render/Normal", {.sourceName = "Normal"});
        scene->Define<RenderProduct>(
            "/Render/Roof",
            {.camera = {"/Cam"}, .resolution = {kSide, kSide}, .orderedVars = {"/Render/Normal"}});
        Renderer renderer(scene);
        const Float3 smooth = CenterNormal(renderer, kSide);
        EXPECT_NEAR(smooth.x, -0.5257f, 2e-3f);
        EXPECT_NEAR(smooth.z, 0.8507f, 2e-3f);

        scene->Find("/Roof")->Set(Mesh::kPoints, Roof(2));
        const Float3 steeper = CenterNormal(renderer, kSide);
        EXPECT_NEAR(steeper.x, -0.7497f, 2e-3f);
        EXPECT_NEAR(steeper.z, 0.6618f, 2e-3f);
    });
}

struct Lit {
    Float3 normal;
    float red;
};

Lit RenderLitQuad(Device& device, Float3 tilt, RenderMode mode) {
    constexpr uint32_t kSide = 9;
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<Camera>("/Cam", {.translate = {0, 0, 5},
                                   .orient = LookAt({0, 0, 5}, {0, 0, 0}, {0, 1, 0}),
                                   .focalLength = 400});
    scene->Define<DistantLight>("/Sun", {.orient = LookAlong({0, -1, -1})});
    scene->Define<Scope>("/Looks");
    const Prim material = scene->Define<Material>("/Looks/Bent");
    const Prim surface = scene->Define<Shader>("/Looks/Bent/Surface", {.id = "UsdPreviewSurface"});
    surface.Set(UsdPreviewSurface::kNormal, tilt);
    material.Connect("surface", surface, "surface");
    const std::vector<Float2> st = {{0, 0}, {0, 1}, {1, 1}, {1, 0}};
    scene->Define<Mesh>("/Quad", {.materialBinding = {"/Looks/Bent"},
                                  .primvars = {Primvar::Create("st", Interpolation::Vertex, st)},
                                  .points = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}},
                                  .faceVertexIndices = {0, 1, 2, 3},
                                  .faceVertexCounts = {4},
                                  .subdivisionScheme = SubdivisionScheme::None});
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Normal", {.sourceName = "Normal"});
    scene->Define<RenderVar>("/Render/Color", {.sourceName = "HdrColor"});
    scene->Define<RenderProduct>("/Render/Roof",
                                 {.camera = {"/Cam"},
                                  .resolution = {kSide, kSide},
                                  .orderedVars = {"/Render/Normal", "/Render/Color"}});
    scene->Select("/Render/Roof").Get<RenderMode>("renderMode").Set(mode);
    if (mode == RenderMode::Minimal) {
        scene->Find("/Render/Roof")->Set(MinimalAPI::kMode, MinimalMode::Full);
    }
    Renderer renderer(scene);
    RenderOutput output = renderer.Step("/Render/Roof", 1.0 / 30);
    output.Wait();
    const FrameOutput& frame = output["/Render/Roof"].frames.at(0);
    const std::vector<float> normals = frame.Source("Normal").tensor.ToHost<float>();
    const std::vector<float> colors = frame.Source("HdrColor").tensor.ToHost<float>();
    const uint32_t center = (kSide / 2) * kSide + kSide / 2;
    return {{normals[center * 3], normals[center * 3 + 1], normals[center * 3 + 2]},
            colors[center * 4]};
}

TEST(RendererTest, NormalMapsLightTheSurfaceButLeaveTheNormalOutputGeometric) {
    ForEachDevice([](Device& device) {
        for (const RenderMode mode : {RenderMode::RealTime, RenderMode::Minimal}) {
            const Lit flat = RenderLitQuad(device, {0, 0, 1}, mode);
            const Lit bent = RenderLitQuad(device, {0.5f, 0, 1}, mode);
            for (const Lit& lit : {flat, bent}) {
                EXPECT_NEAR(lit.normal.x, 0, 1e-4f);
                EXPECT_NEAR(lit.normal.y, 0, 1e-4f);
                EXPECT_NEAR(lit.normal.z, 1, 1e-4f);
            }
            EXPECT_GT(bent.red, 1.1f * flat.red) << int(mode);
        }
    });
}

TEST(RendererTest, NormalsHaveThreeChannelsAndMissesAreZero) {
    ForEachDevice([](Device& device) {
        constexpr uint32_t kSide = 9;
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<Camera>("/Cam", {.translate = {0, 0, 5},
                                       .orient = LookAt({0, 0, 5}, {0, 0, 0}, {0, 1, 0}),
                                       .focalLength = 10});
        scene->Define<Mesh>("/Quad", {.points = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}},
                                      .faceVertexIndices = {0, 1, 2, 3},
                                      .faceVertexCounts = {4},
                                      .subdivisionScheme = SubdivisionScheme::None});
        scene->Define<Scope>("/Render");
        scene->Define<RenderVar>("/Render/Normal", {.sourceName = "Normal"});
        scene->Define<RenderProduct>(
            "/Render/Quad",
            {.camera = {"/Cam"}, .resolution = {kSide, kSide}, .orderedVars = {"/Render/Normal"}});
        Renderer renderer(scene);
        RenderOutput output = renderer.Step("/Render/Quad", 1.0 / 30);
        output.Wait();
        const device::Tensor& tensor = output["/Render/Quad"].frames.at(0).Source("Normal").tensor;
        EXPECT_EQ(std::vector<int64_t>(tensor.Shape().begin(), tensor.Shape().end()),
                  (std::vector<int64_t>{1, kSide, kSide, 3}));
        const std::vector<float> normals = tensor.ToHost<float>();
        const uint32_t center = ((kSide / 2) * kSide + kSide / 2) * 3;
        EXPECT_NEAR(normals[center + 2], 1, 1e-4f);
        EXPECT_EQ(std::vector<float>(normals.begin(), normals.begin() + 3),
                  (std::vector<float>{0, 0, 0}));
    });
}

TEST(RendererTest, AuthoredNormalsAndTheSubdivisionSchemeReachTheNormalOutput) {
    ForEachDevice([](Device& device) {
        constexpr uint32_t kSide = 9;
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<Camera>("/Cam", {.translate = {0, 0, 5},
                                       .orient = LookAt({0, 0, 5}, {0, 0, 0}, {0, 1, 0}),
                                       .focalLength = 400});
        const float s = std::sqrt(0.5f);
        const Prim quad = scene->Define<Mesh>(
            "/Roof", {.points = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}},
                      .normals = {{s, 0, s}, {s, 0, s}, {s, 0, s}, {s, 0, s}},
                      .faceVertexIndices = {0, 1, 2, 3},
                      .faceVertexCounts = {4},
                      .subdivisionScheme = SubdivisionScheme::None});
        scene->Define<Scope>("/Render");
        scene->Define<RenderVar>("/Render/Normal", {.sourceName = "Normal"});
        scene->Define<RenderProduct>(
            "/Render/Roof",
            {.camera = {"/Cam"}, .resolution = {kSide, kSide}, .orderedVars = {"/Render/Normal"}});
        Renderer renderer(scene);
        EXPECT_NEAR(CenterNormal(renderer, kSide).x, s, 1e-3f);

        quad.Set(Mesh::kNormals, std::vector<Float3>{{0, s, s}, {0, s, s}, {0, s, s}, {0, s, s}});
        const Float3 turned = CenterNormal(renderer, kSide);
        EXPECT_NEAR(turned.x, 0, 1e-3f);
        EXPECT_NEAR(turned.y, s, 1e-3f);

        quad.Set(Mesh::kSubdivisionScheme, SubdivisionScheme::CatmullClark);
        EXPECT_NEAR(CenterNormal(renderer, kSide).z, 1, 1e-3f);
    });
}

}
}
