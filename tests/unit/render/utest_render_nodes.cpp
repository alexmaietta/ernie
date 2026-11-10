// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
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
#include "support/support.h"

namespace ernie {
namespace {

using device::Device;
using test::ForEachDevice;

const std::filesystem::path& Scratch() {
    static const test::TemporaryDirectory directory;
    return directory.Path();
}

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
    const std::filesystem::path path = Scratch() / name;
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
    RenderOutput output = renderer.Step("/Render/View", 0);
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
    const std::filesystem::path blue = Scratch() / "ernie_render_nodes_blue.png";
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

TEST(MaterialNodeTest, ShadersDefinedBySourceAssetLoadTheirFile) {
    const std::filesystem::path file =
        WriteFile("ernie_render_nodes_asset.slang",
                  std::regex_replace(std::string(kTint), std::regex("TestTint"), "TestAsset"));
    ForEachDevice([&](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim asset = scene->Define<Shader>(
            "/Asset", {.implementationSource = ImplementationSource::SourceAsset,
                       .slangSourceAsset = file.string(),
                       .slangSourceAssetSubIdentifier = "TestAsset"});
        EXPECT_EQ(Info(asset.Schema()).name, "TestAsset");
        EXPECT_EQ(asset.Get(Shader::kId), "TestAsset");
        EXPECT_EQ(asset.Get<float>("gain"), 1.0f);
        const Prim byId = scene->Define<Shader>("/ById", {.id = "TestAsset"});
        EXPECT_EQ(byId.Get(Shader::kImplementationSource), ImplementationSource::SourceAsset);
        EXPECT_EQ(byId.Get(Shader::kSlangSourceAsset),
                  std::filesystem::weakly_canonical(file).string());
        EXPECT_EQ(byId.Get(Shader::kSlangSourceAssetSubIdentifier), "TestAsset");
        EXPECT_THROW(scene->Define<Shader>(
                         "/Missing", {.implementationSource = ImplementationSource::SourceAsset,
                                      .slangSourceAsset = file.string(),
                                      .slangSourceAssetSubIdentifier = "NoSuchNode"}),
                     Error);
        EXPECT_THROW(scene->Define<Shader>(
                         "/NoFile", {.implementationSource = ImplementationSource::SourceAsset,
                                     .slangSourceAsset = "/no/such/file.slang",
                                     .slangSourceAssetSubIdentifier = "TestNoFile"}),
                     Error);
        EXPECT_FALSE(scene->Find("/Missing"));
    });
}

TEST(MaterialNodeTest, ErrorsNameTheUsersFileAndLine) {
    const std::filesystem::path broken = WriteFile("ernie_render_nodes_lines.slang", R"(
import ernie.material;

struct TestLines {
    void Material(inout SurfaceInputs m, ShadeContext c) {
        m.roughness = NoSuchFunction();
    }
}
)");
    ForEachDevice([&](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        try {
            LoadShaders(*scene, broken);
            ADD_FAILURE() << "loading a broken file succeeded";
        } catch (const Error& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find(std::filesystem::weakly_canonical(broken).string() + ":6"),
                      std::string::npos)
                << message;
            EXPECT_EQ(message.find("ernie/nodes"), std::string::npos) << message;
        }
    });
}

constexpr const char* kPatterns = R"(
import ernie.material;

[Pattern]
float3 TestSwap(ShadeContext c, float3 color, [Input(0, 1)] float amount = 1) {
    return lerp(color, color.zyx, amount);
}

[Pattern]
float3 TestLoop(float3 color) {
    return color;
}

struct TestBase {
    [Color] float3 tint = float3(0, 1, 0);

    void Material(inout SurfaceInputs m, ShadeContext c) {
        m.baseColor = SpectralValue.Rgb(tint);
        m.specularWeight = 0;
    }
}
)";

TEST(MaterialNodeTest, PatternsFeedMaterialInputsAndTheirValuesEditInPlace) {
    const std::filesystem::path file = WriteFile("ernie_render_nodes_patterns.slang", kPatterns);
    const std::filesystem::path red = Scratch() / "ernie_render_nodes_pattern_red.png";
    WritePng(red, Image(4, 4, Rgba{1, 0, 0, 1}));
    ForEachDevice([&](Device& device) {
        const Ref<Scene> scene = LookingDown(device, true);
        const std::vector<std::string> nodes = LoadShaders(*scene, file);
        EXPECT_EQ(nodes.size(), 3u);
        const Prim surface = Paint(*scene, "TestBase");
        const Prim swap = scene->Define<Shader>("/Looks/Paint/Swap", {.id = "TestSwap"});
        EXPECT_EQ(swap.Get<float>("amount"), 1.0f);
        const Prim texture = scene->Define<Shader>("/Looks/Paint/Red", {.id = "UsdUVTexture"});
        texture.Set(UsdUVTexture::kFile, red.string());
        swap.Connect("color", texture, "rgb");
        surface.Connect("tint", swap, "result");
        Renderer renderer(scene);
        const auto blue = [](Float3 pixel) { return pixel.z > 0.5f && pixel.x < 0.1f; };
        Float3 pixel = Center(renderer);
        for (int attempt = 0; attempt < 500 && !blue(pixel); ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
            pixel = Center(renderer);
        }
        EXPECT_TRUE(blue(pixel)) << pixel.x << " " << pixel.y << " " << pixel.z;
        swap.Set("amount", 0.0f);
        pixel = Center(renderer);
        EXPECT_GT(pixel.x, 0.5f);
        EXPECT_LT(pixel.z, 0.1f);

        const Prim first = scene->Define<Shader>("/Looks/Paint/First", {.id = "TestLoop"});
        const Prim second = scene->Define<Shader>("/Looks/Paint/Second", {.id = "TestLoop"});
        first.Connect("color", second, "result");
        second.Connect("color", first, "result");
        surface.Connect("tint", first, "result");
        pixel = Center(renderer);
        EXPECT_LT(std::max({pixel.x, pixel.y, pixel.z}), 0.05f);
    });
}

TEST(MaterialNodeTest, SurfacesBuildTheirOwnClosures) {
    const std::filesystem::path file = WriteFile("ernie_render_nodes_surface.slang", R"(
import ernie.material;

struct TestOrange {
    [Color] float3 tint = float3(1, 0.5, 0);
    float glow = 0;

    void Shade<Spectrum : ISpectrum>(ShadeContext c, inout OpticalClosure<Spectrum> s) {
        const Spectrum color = SpectralValue.Rgb(tint).Evaluate<Spectrum>();
        s.AddLobe(Lobe<Spectrum>.DiffuseReflection(color, 1));
        s.albedo = color;
        s.emission = color.Scale(glow);
    }
}
)");
    ForEachDevice([&](Device& device) {
        const Ref<Scene> scene = LookingDown(device, true);
        EXPECT_EQ(LoadShaders(*scene, file), std::vector<std::string>{"TestOrange"});
        const Prim surface = Paint(*scene, "TestOrange");
        Renderer renderer(scene);
        const auto orange = [](Float3 pixel) {
            return pixel.x > 0.5f && pixel.y > 0.2f * pixel.x && pixel.y < 0.8f * pixel.x &&
                   pixel.z < 0.05f;
        };
        Float3 pixel = Center(renderer);
        EXPECT_TRUE(orange(pixel)) << pixel.x << " " << pixel.y << " " << pixel.z;
        scene->Find("/Render/View")->Set(MinimalAPI::kMode, MinimalMode::Full);
        pixel = Center(renderer);
        EXPECT_TRUE(orange(pixel)) << pixel.x << " " << pixel.y << " " << pixel.z;
        const Float3 lit = pixel;
        surface.Set("glow", 4.0f);
        pixel = Center(renderer);
        EXPECT_GT(pixel.x, lit.x + 1);
    });
}

std::string HotNode(std::string_view color, std::string_view inputs) {
    return std::format(R"(
import ernie.material;

struct TestHot {{
    [Color] float3 tint = float3(0, 1, 0);
{}
    void Material(inout SurfaceInputs m, ShadeContext c) {{
        m.baseColor = SpectralValue.Rgb({});
        m.specularWeight = 0;
    }}
}}
)",
                       inputs, color);
}

TEST(MaterialNodeTest, ReloadedFilesChangeTheImageAndKeepInputValues) {
    const std::filesystem::path file =
        WriteFile("ernie_render_nodes_hot.slang", HotNode("tint", ""));
    ForEachDevice([&](Device& device) {
        const Ref<Scene> scene = LookingDown(device, true);
        LoadShaders(*scene, file);
        EXPECT_EQ(LoadShaders(*scene, file), std::vector<std::string>{"TestHot"});
        const Prim surface = Paint(*scene, "TestHot");
        surface.Set("tint", Float3{1, 0, 0});
        Renderer renderer(scene);
        Float3 pixel = Center(renderer);
        EXPECT_GT(pixel.x, 0.5f);
        EXPECT_TRUE(ReloadShaders(*scene).empty());

        WriteFile("ernie_render_nodes_hot.slang", HotNode("tint.zyx", ""));
        EXPECT_EQ(ReloadShaders(*scene), std::vector<std::string>{"TestHot"});
        pixel = Center(renderer);
        EXPECT_GT(pixel.z, 0.5f);
        EXPECT_LT(pixel.x, 0.1f);

        WriteFile("ernie_render_nodes_hot.slang",
                  HotNode("tint.zyx * gain", "    float gain = 0.5;\n"));
        EXPECT_EQ(ReloadShaders(*scene), std::vector<std::string>{"TestHot"});
        EXPECT_EQ(surface.Get<Float3>("tint"), (Float3{1, 0, 0}));
        EXPECT_EQ(surface.Get<float>("gain"), 0.5f);
        const Float3 half = Center(renderer);
        EXPECT_NEAR(half.z, pixel.z * 0.5f, 0.05f);

        WriteFile("ernie_render_nodes_hot.slang", "struct TestHot { void Material( }");
        EXPECT_TRUE(ReloadShaders(*scene).empty());
        EXPECT_NEAR(Center(renderer).z, half.z, 1e-3f);
    });
}

constexpr const char* kLambert = R"(
import ernie.primitive;

[ClosurePrimitive]
struct TestLambert : IClosurePrimitive {
    static bool Glossy() {
        return false;
    }

    [Differentiable]
    static Spectrum Estimate<Spectrum : ISpectrum>(Spectrum tint, float alpha,
                                                   no_diff float4 params,
                                                   no_diff float3 outgoing) {
        return tint.Scale(params.x);
    }

    [Differentiable]
    static Spectrum Evaluate<Spectrum : ISpectrum>(Spectrum tint, float alpha,
                                                   no_diff float4 params, no_diff float3 outgoing,
                                                   no_diff float3 incoming) {
        return tint.Scale(params.x / kPi);
    }

    static float Pdf(float alpha, float4 params, float3 outgoing, float3 incoming) {
        return incoming.z / kPi;
    }

    static float3 Direction(float alpha, float4 params, float3 outgoing, float2 u) {
        return CosineHemisphere(u);
    }
}
)";

constexpr const char* kLobes = R"(
import ernie.material;

struct TestBuiltinLobe {
    void Shade<Spectrum : ISpectrum>(ShadeContext c, inout OpticalClosure<Spectrum> s) {
        const Spectrum color = SpectralValue.Rgb(float3(0.2, 0.6, 0.4)).Evaluate<Spectrum>();
        s.AddLobe(Lobe<Spectrum>.DiffuseReflection(color, 1));
        s.albedo = color;
    }
}

struct TestPrimitiveLobe {
    float scale = 1;

    void Shade<Spectrum : ISpectrum>(ShadeContext c, inout OpticalClosure<Spectrum> s) {
        const Spectrum color = SpectralValue.Rgb(float3(0.2, 0.6, 0.4)).Evaluate<Spectrum>();
        s.AddLobe(Lobe<Spectrum>.Primitive(kLobeTestLambert, color, 1, float4(scale, 0, 0, 0), 1));
        s.albedo = color;
    }
}
)";

TEST(MaterialNodeTest, ClosurePrimitivesShadeLikeTheLobesTheyReproduce) {
    const std::filesystem::path primitive = WriteFile("ernie_render_nodes_lambert.slang", kLambert);
    const std::filesystem::path lobes = WriteFile("ernie_render_nodes_lobes.slang", kLobes);
    ForEachDevice([&](Device& device) {
        const Ref<Scene> scene = LookingDown(device, true);
        scene->Find("/Render/View")->Set(MinimalAPI::kMode, MinimalMode::Full);
        EXPECT_EQ(LoadShaders(*scene, primitive), std::vector<std::string>{"TestLambert"});
        EXPECT_EQ(LoadShaders(*scene, lobes).size(), 2u);
        const Prim surface = Paint(*scene, "TestBuiltinLobe");
        Renderer renderer(scene);
        const Float3 builtin = Center(renderer);
        EXPECT_GT(builtin.y, 0.05f);
        surface.Set(Shader::kId, "TestPrimitiveLobe");
        const Float3 custom = Center(renderer);
        EXPECT_NEAR(custom.x, builtin.x, 1e-3f);
        EXPECT_NEAR(custom.y, builtin.y, 1e-3f);
        EXPECT_NEAR(custom.z, builtin.z, 1e-3f);
        surface.Set("scale", 0.5f);
        EXPECT_NEAR(Center(renderer).y, builtin.y * 0.5f, 1e-3f);
    });
}

TEST(MaterialNodeTest, PrimitiveFilesStayApartFromNodesAndMustImplementTheInterface) {
    const std::filesystem::path mixed = WriteFile(
        "ernie_render_nodes_mixed.slang",
        std::regex_replace(std::string(kLambert), std::regex("ernie.primitive"), "ernie.material"));
    const std::filesystem::path partial = WriteFile("ernie_render_nodes_partial.slang", R"(
import ernie.primitive;

[ClosurePrimitive]
struct TestPartial {
    static bool Glossy() {
        return false;
    }
}
)");
    ForEachDevice([&](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        EXPECT_THROW(LoadShaders(*scene, mixed), Error);
        try {
            LoadShaders(*scene, partial);
            ADD_FAILURE() << "loading a primitive without the interface succeeded";
        } catch (const Error& error) {
            EXPECT_NE(std::string(error.what()).find("ernie_render_nodes_partial.slang"),
                      std::string::npos)
                << error.what();
        }
    });
}

TEST(MaterialNodeTest, FilesWithoutNodesOrWithBadInputsAreReported) {
    const std::filesystem::path empty = WriteFile("ernie_render_nodes_empty.slang", R"(
import ernie.material;
float Helper(float x) { return x; }
struct Plain { float x; }
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

TEST(MaterialNodeTest, FilesAreSortedByTheNodesTheyDefineNotByTheirText) {
    const std::filesystem::path mentioned = WriteFile("ernie_render_nodes_mentioned.slang", R"(
// Unlike a pipeline node, this file doesn't import ernie.pipeline;
import ernie.material;

struct TestMentioned {
    void Material(inout SurfaceInputs m, ShadeContext c) {}
}
)");
    const std::filesystem::path mixed = WriteFile("ernie_render_nodes_mixed.slang", R"(
import ernie.material;
import ernie.pipeline;

using namespace ernie;

struct TestMixedSurface {
    void Material(inout SurfaceInputs m, ShadeContext c) {}
}

struct TestMixedStage : IPointNode {
    Digital Apply(Digital digital, PointContext context) {
        return digital;
    }
}
)");
    ForEachDevice([&](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        EXPECT_EQ(LoadShaders(*scene, mentioned), std::vector<std::string>{"TestMentioned"});
        try {
            LoadShaders(*scene, mixed);
            ADD_FAILURE() << "a file with both kinds of node loaded";
        } catch (const Error& error) {
            EXPECT_NE(std::string(error.what()).find("both pipeline nodes"), std::string::npos)
                << error.what();
        }
    });
}

}
}
