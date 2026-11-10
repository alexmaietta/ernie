// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "bench.h"
#include "ernie/io/image.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/rotation.h"
#include "ernie/scene/scene.h"

namespace ernie {
namespace {

constexpr uint32_t kGroundCells = 32;
constexpr uint32_t kBalls = 8;
constexpr uint32_t kTextureSide = 512;
constexpr uint32_t kWarmup = 5;
constexpr size_t kInflight = 2;
const std::vector<std::string> kProducts = {"/Render/Product"};

struct Mode {
    const char* name;
    RenderMode render;
    MinimalMode minimal;
    SpectralMode spectral;
    const char* var;
    bool shadows;
};

const Mode kModes[] = {
    {"Depth", RenderMode::Minimal, MinimalMode::Data, SpectralMode::Rgb, "/Render/Depth", false},
    {"DisplayColor/shadows:0", RenderMode::Minimal, MinimalMode::DisplayColor, SpectralMode::Rgb,
     "/Render/Color", false},
    {"DisplayColor/shadows:1", RenderMode::Minimal, MinimalMode::DisplayColor, SpectralMode::Rgb,
     "/Render/Color", true},
    {"TextureDiffuse/shadows:0", RenderMode::Minimal, MinimalMode::TextureDiffuse,
     SpectralMode::Rgb, "/Render/Color", false},
    {"TextureDiffuse/shadows:1", RenderMode::Minimal, MinimalMode::TextureDiffuse,
     SpectralMode::Rgb, "/Render/Color", true},
    {"Full/shadows:0", RenderMode::Minimal, MinimalMode::Full, SpectralMode::Rgb, "/Render/Color",
     false},
    {"Full/shadows:1", RenderMode::Minimal, MinimalMode::Full, SpectralMode::Rgb, "/Render/Color",
     true},
    {"RealTime", RenderMode::RealTime, MinimalMode::Full, SpectralMode::Rgb, "/Render/Color", true},
    {"RealTimeBands", RenderMode::RealTime, MinimalMode::Full, SpectralMode::Bands, "/Render/Color",
     true},
};

struct Textures {
    std::filesystem::path directory;
    std::string albedo;
    std::string normal;
    std::string roughness;

    Textures() {
        directory = std::filesystem::temp_directory_path() / "ernie_bench_throughput";
        std::filesystem::create_directories(directory);
        Image color(kTextureSide, kTextureSide);
        Image bumps(kTextureSide, kTextureSide);
        Image rough(kTextureSide, kTextureSide);
        for (uint32_t y = 0; y < kTextureSide; ++y) {
            for (uint32_t x = 0; x < kTextureSide; ++x) {
                const float u = float(x) / kTextureSide;
                const float v = float(y) / kTextureSide;
                const bool odd = ((x / 32) + (y / 32)) % 2 != 0;
                color.Pixel(x, y) = odd ? Rgba{0.8f, 0.3f, 0.2f, 1} : Rgba{0.2f, 0.5f, 0.8f, 1};
                bumps.Pixel(x, y) = {0.5f + 0.3f * std::sin(40 * u), 0.5f + 0.3f * std::cos(40 * v),
                                     1, 1};
                const float r = 0.3f + 0.4f * std::abs(std::sin(10 * (u + v)));
                rough.Pixel(x, y) = {r, r, r, 1};
            }
        }
        albedo = (directory / "albedo.png").string();
        normal = (directory / "normal.png").string();
        roughness = (directory / "roughness.png").string();
        WritePng(albedo, color);
        WritePng(normal, bumps);
        WritePng(roughness, rough);
    }

    ~Textures() {
        std::filesystem::remove_all(directory);
    }
};

Mesh::Desc Ground() {
    Mesh::Desc desc;
    const uint32_t side = kGroundCells + 1;
    std::vector<Float2> st;
    for (uint32_t y = 0; y < side; ++y) {
        for (uint32_t x = 0; x < side; ++x) {
            const float u = float(x) / kGroundCells;
            const float v = float(y) / kGroundCells;
            desc.points.push_back(
                {(u - 0.5f) * 20, (v - 0.5f) * 20, 0.3f * std::sin(u * 17) * std::cos(v * 13)});
            st.push_back({4 * u, 4 * v});
        }
    }
    for (uint32_t y = 0; y < kGroundCells; ++y) {
        for (uint32_t x = 0; x < kGroundCells; ++x) {
            const auto a = int32_t(y * side + x);
            desc.faceVertexIndices.insert(desc.faceVertexIndices.end(),
                                          {a, a + 1, a + int32_t(side) + 1, a + int32_t(side)});
            desc.faceVertexCounts.push_back(4);
        }
    }
    desc.primvars.push_back(Primvar::Create("st", Interpolation::Vertex, st));
    desc.materialBinding = {"/Looks/Ground"};
    return desc;
}

struct Setup {
    Ref<Scene> scene;
    Renderer renderer;
    uint64_t pixels = 0;
};

Setup Prepare(const Mode& mode, uint32_t cameras, uint32_t worlds, uint32_t side) {
    static const Textures textures;
    const Ref<Scene> scene = Scene::Create(bench::BenchDevice());
    scene->Define<Scope>("/Looks");
    const auto look = [&](const std::string& path,
                          std::initializer_list<std::pair<std::string, std::string>> files) {
        const Prim material = scene->Define<Material>(path);
        const Prim surface = scene->Define<Shader>(path + "/Surface", {.id = "UsdPreviewSurface"});
        material.Connect("surface", surface, "surface");
        const Prim placement =
            scene->Define<Shader>(path + "/Placement", {.id = "TexturePlacement"});
        for (const auto& [input, file] : files) {
            const Prim texture = scene->Define<Shader>(path + "/" + input, {.id = "UsdUVTexture"});
            texture.Set(UsdUVTexture::kFile, file);
            texture.Set(UsdUVTexture::kSourceColorSpace,
                        input == "diffuseColor" ? TextureColorSpace::SRGB : TextureColorSpace::Raw);
            if (input == "normal") {
                texture.Set(UsdUVTexture::kScale, Float4{2, 2, 2, 1});
                texture.Set(UsdUVTexture::kBias, Float4{-1, -1, -1, 0});
            }
            texture.Connect("st", placement, "st");
            surface.Connect(input, texture, "rgb");
        }
        return placement;
    };
    look("/Looks/Ground", {{"diffuseColor", textures.albedo},
                           {"normal", textures.normal},
                           {"roughness", textures.roughness}});
    look("/Looks/Hex", {{"diffuseColor", textures.albedo}}).Set(TexturePlacement::kHexTiling, true);
    const Prim triplanar =
        look("/Looks/Triplanar", {{"diffuseColor", textures.albedo}, {"normal", textures.normal}});
    triplanar.Set(TexturePlacement::kProjection, TextureProjection::Triplanar);
    triplanar.Set(TexturePlacement::kTileSize, 2.0f);
    scene->Define<World>("/env_0");
    scene->Define<Mesh>("/env_0/Ground", Ground());
    for (uint32_t i = 0; i < kBalls; ++i) {
        const double angle = i * 0.785;
        scene->Define<Sphere>(fmt::format("/env_0/Ball_{}", i),
                              {.materialBinding = {"/Looks/Hex"},
                               .translate = {3 * std::cos(angle), 3 * std::sin(angle), 0.6},
                               .radius = 0.5f});
    }
    scene->Define<Cube>(
        "/env_0/Box",
        {.materialBinding = {"/Looks/Triplanar"}, .translate = {0, 0, 0.75}, .size = 1.5});
    for (uint32_t c = 0; c < cameras; ++c) {
        const double angle = 2 * 3.14159265358979 * c / cameras;
        const Double3 eye = {7 * std::cos(angle), 7 * std::sin(angle), 4};
        scene->Define<Camera>(fmt::format("/env_0/Cam_{}", c),
                              {.translate = eye, .orient = LookAt(eye, {0, 0, 0.5})});
    }
    scene->Clone(*scene->Find("/env_0"), worlds - 1);
    scene->Define<DistantLight>("/Sun", {.orient = FromAzimuthElevation(60, 50)});
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Color", {.sourceName = "LdrColor"});
    scene->Define<RenderVar>("/Render/Depth", {.sourceName = "DistanceToCamera"});
    std::vector<std::string> views;
    for (uint32_t w = 0; w < worlds; ++w) {
        for (uint32_t c = 0; c < cameras; ++c) {
            views.push_back(fmt::format("/env_{}/Cam_{}", w, c));
        }
    }
    scene->Define<RenderProduct>("/Render/Product", {.mode = mode.minimal,
                                                     .castShadows = mode.shadows,
                                                     .renderMode = mode.render,
                                                     .spectralMode = mode.spectral,
                                                     .camera = views,
                                                     .resolution = {side, side},
                                                     .orderedVars = {mode.var}});
    Setup setup{scene, Renderer(scene), uint64_t(cameras) * worlds * side * side};
    for (uint32_t i = 0; i < kWarmup; ++i) {
        setup.renderer.Step(kProducts, 1.0 / 60).Wait();
    }
    return setup;
}

void Throughput(benchmark::State& state, const Mode& mode) {
    const auto cameras = uint32_t(state.range(0));
    const auto worlds = uint32_t(state.range(1));
    const auto side = uint32_t(state.range(2));
    Setup& setup = bench::Cached<Setup>(
        fmt::format("throughput/{}/{}/{}/{}", mode.name, cameras, worlds, side),
        [&] { return Prepare(mode, cameras, worlds, side); });
    std::deque<RenderOutput> flight;
    for (auto _ : state) {
        flight.push_back(setup.renderer.Step(kProducts, 1.0 / 60));
        if (flight.size() >= kInflight) {
            flight.front().Wait();
            flight.pop_front();
        }
    }
    for (const RenderOutput& output : flight) {
        output.Wait();
    }
    state.counters["views"] = double(cameras) * worlds;
    state.counters["px/s"] = benchmark::Counter(double(state.iterations()) * double(setup.pixels),
                                                benchmark::Counter::kIsRate);
}

const bool kCore = bench::Core("^Render/Throughput/(Depth|DisplayColor/shadows:1|Full/shadows:1)/"
                               "cameras:(1/worlds:1/side:512|1/worlds:64|16/worlds:16)/");

const bool kRegistered = [] {
    for (const Mode& mode : kModes) {
        benchmark::RegisterBenchmark(fmt::format("Render/Throughput/{}", mode.name), Throughput,
                                     mode)
            ->ArgNames({"cameras", "worlds", "side"})
            ->Args({1, 1, 512})
            ->Args({16, 1, 128})
            ->Args({64, 1, 128})
            ->Args({1, 16, 128})
            ->Args({1, 64, 128})
            ->Args({1, 256, 128})
            ->Args({4, 16, 128})
            ->Args({16, 16, 128})
            ->Args({16, 64, 128})
            ->UseRealTime()
            ->Unit(benchmark::kMillisecond);
    }
    return true;
}();

}
}
