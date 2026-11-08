// SPDX-License-Identifier: BSD-3-Clause
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/io/image.h"
#include "ernie/render/render_output.h"
#include "ernie/render/renderer.h"
#include "ernie/render/shaders.h"
#include "ernie/scene/rotation.h"
#include "ernie/scene/scene.h"
#include "ernie/util/error.h"
#include "support/devices.h"

namespace ernie {
namespace {

using device::Device;
using test::ForEachDevice;

constexpr uint32_t kSide = 8;

constexpr const char* kTint = R"(
import ernie.material;

struct TestTint {
    [Color] float3 tint = float3(0, 1, 0);
    [Input(0, 2)] float gain = 1;
    bool dark = false;

    void Material(inout SurfaceInputs m, ShadeContext c) {
        m.baseColor = SpectralValue.Rgb(dark ? float3(0) : tint * gain);
        m.specularWeight = 0;
    }
}
)";

std::filesystem::path WriteFile(std::string_view name, std::string_view text) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    std::ofstream(path) << text;
    return path;
}

Ref<Scene> LookingDown(Device& device, bool wait) {
    const Ref<Scene> scene = Scene::Create(device, {.waitForKernels = wait});
    scene->Define<Camera>("/Cam", {.translate = {0, 0, 5},
                                   .orient = LookAt({0, 0, 5}, {0, 0, 0}, {0, 1, 0}),
                                   .focalLength = 30});
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Color", {.sourceName = "HdrColor"});
    scene->Define<RenderProduct>(
        "/Render/View",
        {.camera = {"/Cam"}, .resolution = {kSide, kSide}, .orderedVars = {"/Render/Color"}});
    scene->Find("/Render/View")->Set(MinimalAPI::kMode, MinimalMode::TextureDiffuse);
    scene->Define<Scope>("/Looks");
    scene->Define<Mesh>(
        "/Quad",
        {.materialBinding = {"/Looks/Paint"},
         .primvars = {Primvar::Create("st", Interpolation::Vertex,
                                      std::vector<Float2>{{0, 0}, {1, 0}, {1, 1}, {0, 1}})},
         .points = {{-9, -9, 0}, {9, -9, 0}, {9, 9, 0}, {-9, 9, 0}},
         .faceVertexIndices = {0, 1, 2, 3},
         .faceVertexCounts = {4},
         .subdivisionScheme = SubdivisionScheme::None});
    return scene;
}

Float3 Center(Renderer& renderer) {
    RenderOutput output = renderer.Step(std::vector<std::string>{"/Render/View"}, 0);
    output.Wait();
    const std::vector<float> color =
        output["/Render/View"].frames.at(0).Source("HdrColor").tensor.ToHost<float>();
    const size_t at = 4 * ((kSide / 2) * kSide + kSide / 2);
    return {color[at], color[at + 1], color[at + 2]};
}

Prim Paint(Scene& scene, std::string_view id) {
    const Prim material = scene.Define<Material>("/Looks/Paint");
    const Prim surface = scene.Define<Shader>("/Looks/Paint/Surface", {.id = std::string(id)});
    material.Connect("surface", surface, "surface");
    return surface;
}

TEST(MaterialNodeTest, LoadedNodesHaveTheirInputsAndShadeTheImage) {
    const std::filesystem::path file = WriteFile("ernie_render_nodes_tint.slang", kTint);
    ForEachDevice([&](Device& device) {
        const Ref<Scene> scene = LookingDown(device, true);
        EXPECT_EQ(LoadShaders(*scene, file), std::vector<std::string>{"TestTint"});
        const Prim surface = Paint(*scene, "TestTint");
        EXPECT_EQ(surface.Get<Float3>("tint"), (Float3{0, 1, 0}));
        EXPECT_EQ(surface.Get<float>("gain"), 1.0f);
        EXPECT_FALSE(surface.Get<bool>("dark"));
        Renderer renderer(scene);
        Float3 pixel = Center(renderer);
        EXPECT_GT(pixel.y, pixel.x);
        surface.Set("tint", Float3{0, 0, 1});
        pixel = Center(renderer);
        EXPECT_GT(pixel.z, pixel.y);
        surface.Set("dark", true);
        pixel = Center(renderer);
        EXPECT_LT(pixel.z, 0.01f);
    });
}

bool Placeholder(Float3 pixel) {
    const bool magenta = pixel.x > 0.5f && pixel.z > 0.5f && pixel.y < 0.1f;
    const bool dark = pixel.x < 0.1f && pixel.y < 0.1f && pixel.z < 0.1f;
    return magenta || dark;
}

TEST(MaterialNodeTest, NewMaterialsShowThePlaceholderAndChangedOnesTheirPreviousLook) {
    const std::string unique =
        std::format("\n// {}\n", std::chrono::steady_clock::now().time_since_epoch().count());
    const std::filesystem::path file = WriteFile(
        "ernie_render_nodes_fresh.slang",
        std::regex_replace(std::string(kTint), std::regex("TestTint"), "TestFresh") + unique);
    const std::filesystem::path blue =
        std::filesystem::temp_directory_path() / "ernie_render_nodes_blue.png";
    WritePng(blue, Image(4, 4, Rgba{0, 0, 1, 1}));
    const auto settle = [](Renderer& renderer, bool (*done)(Float3)) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        Float3 pixel = Center(renderer);
        while (!done(pixel) && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            pixel = Center(renderer);
        }
        return pixel;
    };
    ForEachDevice([&](Device& device) {
        const Ref<Scene> scene = LookingDown(device, false);
        LoadShaders(*scene, file);
        const Prim surface = Paint(*scene, "TestFresh");
        Renderer renderer(scene);
        const Float3 first = Center(renderer);
        EXPECT_TRUE(Placeholder(first)) << first.x << " " << first.y << " " << first.z;
        const auto green = [](Float3 pixel) { return pixel.y > 0.5f && pixel.x < 0.1f; };
        const Float3 linked = settle(renderer, green);
        EXPECT_TRUE(green(linked)) << linked.x << " " << linked.y << " " << linked.z;

        const Prim texture = scene->Define<Shader>("/Looks/Paint/Blue", {.id = "UsdUVTexture"});
        texture.Set(UsdUVTexture::kFile, blue.string());
        surface.Connect("tint", texture, "rgb");
        const Float3 previous = Center(renderer);
        EXPECT_TRUE(green(previous)) << previous.x << " " << previous.y << " " << previous.z;
        const auto textured = [](Float3 pixel) { return pixel.z > 0.5f && pixel.y < 0.1f; };
        const Float3 sampled = settle(renderer, textured);
        EXPECT_TRUE(textured(sampled)) << sampled.x << " " << sampled.y << " " << sampled.z;
    });
}

TEST(MaterialNodeTest, FilesWithoutNodesOrWithBadInputsAreReported) {
    const std::filesystem::path empty = WriteFile("ernie_render_nodes_empty.slang", R"(
import ernie.material;
float Helper(float x) { return x; }
)");
    const std::filesystem::path bad = WriteFile("ernie_render_nodes_bad.slang", R"(
import ernie.material;
struct TestBad {
    float3x3 basis;
    void Material(inout SurfaceInputs m, ShadeContext c) {}
}
)");
    const std::filesystem::path broken =
        WriteFile("ernie_render_nodes_broken.slang", "struct TestBroken { void Material( }");
    ForEachDevice([&](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        EXPECT_TRUE(LoadShaders(*scene, empty).empty());
        EXPECT_THROW(LoadShaders(*scene, bad), Error);
        EXPECT_THROW(LoadShaders(*scene, broken), Error);
        EXPECT_THROW(LoadShaders(*scene, "/no/such/file.slang"), Error);
    });
}

}
}
