// SPDX-License-Identifier: BSD-3-Clause
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "bench.h"
#include "ernie/io/image.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/rotation.h"
#include "ernie/scene/scene.h"
#include "ernie/util/profiler.h"

namespace ernie {
namespace {

constexpr uint32_t kWarmup = 3;
constexpr size_t kInflight = 2;
constexpr size_t kHistory = 32;
constexpr double kTau = 6.283185307179586;

using Clock = std::chrono::steady_clock;

double Since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

device::Traversal SelectTraversal(int64_t hardware) {
    return hardware != 0 ? device::Traversal::Hardware : device::Traversal::Software;
}

Mesh::Desc Tile(uint32_t cells, float size, float phase, bool st) {
    Mesh::Desc desc;
    const uint32_t side = cells + 1;
    std::vector<Float2> coordinates;
    for (uint32_t y = 0; y < side; ++y) {
        for (uint32_t x = 0; x < side; ++x) {
            const float u = float(x) / float(cells);
            const float v = float(y) / float(cells);
            desc.points.push_back({(u - 0.5f) * size, (v - 0.5f) * size,
                                   0.03f * size * std::sin(u * 17 + phase) * std::cos(v * 13)});
            coordinates.push_back({4 * u, 4 * v});
        }
    }
    for (uint32_t y = 0; y < cells; ++y) {
        for (uint32_t x = 0; x < cells; ++x) {
            const auto a = int32_t(y * side + x);
            desc.faceVertexIndices.insert(desc.faceVertexIndices.end(),
                                          {a, a + 1, a + int32_t(side) + 1, a + int32_t(side)});
            desc.faceVertexCounts.push_back(4);
        }
    }
    if (st) {
        desc.primvars.push_back(Primvar::Create("st", Interpolation::Vertex, coordinates));
    }
    desc.displayColor = {0.6f, 0.55f, 0.5f};
    return desc;
}

void Props(Scene& scene, const std::string& world) {
    scene.Define<Mesh>(world + "/Ground", Tile(32, 20, 0, false));
    for (uint32_t i = 0; i < 8; ++i) {
        const double angle = kTau * i / 8;
        scene.Define<Sphere>(
            fmt::format("{}/Ball_{}", world, i),
            {.translate = {3 * std::cos(angle), 3 * std::sin(angle), 0.6},
             .displayColor = {0.2f + 0.1f * float(i), 0.4f, 0.8f - 0.1f * float(i)},
             .radius = 0.5f});
    }
    scene.Define<Cube>(
        world + "/Box",
        {.translate = {0, 0, 0.75}, .displayColor = {0.8f, 0.3f, 0.2f}, .size = 1.5});
}

void Look(Scene& scene, const std::string& path, double angle, double distance, double height,
          Double3 at = {0, 0, 0.5}) {
    const Double3 eye = {distance * std::cos(angle), distance * std::sin(angle), height};
    scene.Define<Camera>(path, {.translate = eye, .orient = LookAt(eye, at)});
}

void Vars(Scene& scene) {
    scene.Define<Scope>("/Render");
    scene.Define<RenderVar>("/Render/Color", {.sourceName = "LdrColor"});
    scene.Define<RenderVar>("/Render/Depth", {.sourceName = "DistanceToCamera"});
}

void Replicate(Scene& scene, uint32_t worlds) {
    std::vector<std::string> targets;
    for (uint32_t w = 1; w < worlds; ++w) {
        targets.push_back(fmt::format("/env_{}", w));
    }
    if (!targets.empty()) {
        scene.Clone(*scene.Find("/env_0"), targets);
    }
}

void Prime(device::Device& device, RenderMode renderMode = RenderMode::Minimal) {
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<World>("/env_0");
    Props(*scene, "/env_0");
    Look(*scene, "/env_0/Cam", 0.3, 7, 4);
    scene->Define<DistantLight>("/Sun", {.orient = FromAzimuthElevation(60, 50)});
    Vars(*scene);
    scene->Define<RenderProduct>("/Render/Product", {.mode = MinimalMode::DisplayColor,
                                                     .renderMode = renderMode,
                                                     .camera = {"/env_0/Cam"},
                                                     .resolution = {64, 64},
                                                     .orderedVars = {"/Render/Color"}});
    Renderer renderer(scene);
    const std::vector<std::string> products = {"/Render/Product"};
    renderer.Step(products, 1.0 / 60).Wait();
}

Ref<Scene> Primed(device::Traversal traversal) {
    static std::set<device::Device*> primed;
    device::Device& device = bench::BenchDevice(traversal);
    if (primed.insert(&device).second) {
        Prime(device);
    }
    return Scene::Create(device);
}

struct Setup {
    Ref<Scene> scene;
    Renderer renderer;
    std::vector<std::string> products;
    uint64_t pixels = 0;
    double firstMs = 0;
    double settledMs = 0;
    double deviceMb = 0;
};

Setup Finish(const Ref<Scene>& scene, std::vector<std::string> products, uint64_t pixels) {
    const Clock::time_point start = Clock::now();
    Setup setup{scene, Renderer(scene), std::move(products), pixels};
    setup.renderer.Step(setup.products, 1.0 / 60).Wait();
    setup.firstMs = Since(start);
    scene->Publish({.wait = true}).Wait();
    setup.settledMs = Since(start);
    for (uint32_t i = 0; i < kWarmup; ++i) {
        setup.renderer.Step(setup.products, 1.0 / 60).Wait();
    }
    setup.deviceMb = double(scene->GetDevice().MemoryUsage().deviceBytes) / (1 << 20);
    return setup;
}

void Run(benchmark::State& state, const std::string& key, const std::function<Setup()>& make) {
    Setup* setup = nullptr;
    try {
        setup = &bench::Cached<Setup>(key, make);
    } catch (const UnsupportedError& error) {
        state.SkipWithMessage(error.what());
        return;
    }
    profiler::SetHistory(kHistory);
    profiler::Clear();
    profiler::BeginFrame();
    std::deque<RenderOutput> flight;
    for (auto _ : state) {
        flight.push_back(setup->renderer.Step(setup->products, 1.0 / 60));
        if (flight.size() >= kInflight) {
            flight.front().Wait();
            flight.pop_front();
        }
        profiler::BeginFrame();
    }
    for (const RenderOutput& output : flight) {
        output.Wait();
    }
    profiler::BeginFrame();
    double step = 0;
    double render = 0;
    for (const profiler::ReportTrack& track : profiler::Summarize(profiler::Frames()).tracks) {
        for (const profiler::ReportNode& zone : track.zones) {
            if (!track.gpu) {
                step += zone.name == "Render/Step" ? zone.ms : 0;
            } else if (zone.name.starts_with("Render/")) {
                render += zone.ms;
            }
        }
    }
    profiler::SetHistory(2);
    state.counters["px/s"] = benchmark::Counter(double(state.iterations()) * double(setup->pixels),
                                                benchmark::Counter::kIsRate);
    state.counters["cpu ms/step"] = step;
    state.counters["gpu render ms"] = render;
    state.counters["first frame ms"] = setup->firstMs;
    state.counters["settled ms"] = setup->settledMs;
    state.counters["device MB"] = setup->deviceMb;
}

void Cameras(benchmark::State& state) {
    const auto cameras = uint32_t(state.range(0));
    const auto side = uint32_t(state.range(1));
    const device::Traversal traversal = SelectTraversal(state.range(2));
    Run(state, fmt::format("cameras/{}/{}/{}", cameras, side, int(traversal)), [&] {
        const Ref<Scene> scene = Primed(traversal);
        scene->Define<World>("/env_0");
        Props(*scene, "/env_0");
        std::vector<std::string> views;
        for (uint32_t c = 0; c < cameras; ++c) {
            views.push_back(fmt::format("/env_0/Cam_{}", c));
            Look(*scene, views.back(), kTau * c / cameras, 6 + 2 * double(c % 3),
                 2 + double(c % 5));
        }
        scene->Define<DistantLight>("/Sun", {.orient = FromAzimuthElevation(60, 50)});
        Vars(*scene);
        scene->Define<RenderProduct>("/Render/Product", {.mode = MinimalMode::DisplayColor,
                                                         .camera = views,
                                                         .resolution = {side, side},
                                                         .orderedVars = {"/Render/Color"}});
        return Finish(scene, {"/Render/Product"}, uint64_t(cameras) * side * side);
    });
}

struct Group {
    uint32_t cameras;
    uint32_t width;
    uint32_t height;
    MinimalMode mode;
    const char* var;
};

struct Mix {
    const char* name;
    std::vector<Group> groups;
};

const Mix kMixes[] = {
    {"uniform", {{4, 148, 147, MinimalMode::DisplayColor, "/Render/Color"}}},
    {"mixed",
     {{1, 32, 32, MinimalMode::DisplayColor, "/Render/Color"},
      {1, 64, 64, MinimalMode::DisplayColor, "/Render/Color"},
      {1, 128, 128, MinimalMode::DisplayColor, "/Render/Color"},
      {1, 256, 256, MinimalMode::DisplayColor, "/Render/Color"}}},
    {"skewed",
     {{21, 32, 32, MinimalMode::DisplayColor, "/Render/Color"},
      {1, 256, 256, MinimalMode::DisplayColor, "/Render/Color"}}},
    {"rl",
     {{1, 128, 128, MinimalMode::DisplayColor, "/Render/Color"},
      {1, 64, 64, MinimalMode::Data, "/Render/Depth"}}},
};

void Mixed(benchmark::State& state, const Mix& mix) {
    const auto worlds = uint32_t(state.range(0));
    const device::Traversal traversal = SelectTraversal(state.range(1));
    Run(state, fmt::format("mixed/{}/{}/{}", mix.name, worlds, int(traversal)), [&] {
        const Ref<Scene> scene = Primed(traversal);
        scene->Define<World>("/env_0");
        Props(*scene, "/env_0");
        uint32_t cameras = 0;
        for (const Group& group : mix.groups) {
            cameras += group.cameras;
        }
        for (uint32_t c = 0; c < cameras; ++c) {
            Look(*scene, fmt::format("/env_0/Cam_{}", c), kTau * c / cameras, 7, 4);
        }
        Replicate(*scene, worlds);
        scene->Define<DistantLight>("/Sun", {.orient = FromAzimuthElevation(60, 50)});
        Vars(*scene);
        std::vector<std::string> products;
        uint64_t pixels = 0;
        uint32_t first = 0;
        for (const Group& group : mix.groups) {
            std::vector<std::string> views;
            for (uint32_t w = 0; w < worlds; ++w) {
                for (uint32_t c = first; c < first + group.cameras; ++c) {
                    views.push_back(fmt::format("/env_{}/Cam_{}", w, c));
                }
            }
            first += group.cameras;
            products.push_back(fmt::format("/Render/Product_{}", products.size()));
            scene->Define<RenderProduct>(products.back(),
                                         {.mode = group.mode,
                                          .camera = views,
                                          .resolution = {group.width, group.height},
                                          .orderedVars = {group.var}});
            pixels += uint64_t(views.size()) * group.width * group.height;
        }
        return Finish(scene, products, pixels);
    });
}

void Lights(benchmark::State& state) {
    const auto lights = uint32_t(state.range(0));
    const device::Traversal traversal = SelectTraversal(state.range(1));
    Run(state, fmt::format("lights/{}/{}", lights, int(traversal)), [&] {
        const Ref<Scene> scene = Primed(traversal);
        scene->Define<World>("/env_0");
        Props(*scene, "/env_0");
        Look(*scene, "/env_0/Cam", 0.3, 7, 4);
        for (uint32_t i = 0; i < lights; ++i) {
            scene->Define<DistantLight>(
                fmt::format("/Sun_{}", i),
                {.orient = FromAzimuthElevation(360.0 * i / lights, 30 + 30 * double(i % 2)),
                 .intensity = 1.0f / float(lights)});
        }
        Vars(*scene);
        scene->Define<RenderProduct>("/Render/Product", {.renderMode = RenderMode::RealTime,
                                                         .camera = {"/env_0/Cam"},
                                                         .resolution = {512, 512},
                                                         .orderedVars = {"/Render/Color"}});
        return Finish(scene, {"/Render/Product"}, 512 * 512);
    });
}

constexpr uint32_t kTiles = 4;

void SceneSize(benchmark::State& state) {
    const auto triangles = uint64_t(state.range(0));
    const bool instanced = state.range(1) != 0;
    const device::Traversal traversal = SelectTraversal(state.range(2));
    const auto cells = uint32_t(std::lround(std::sqrt(double(triangles) / (kTiles * kTiles * 2))));
    state.counters["triangles"] = double(kTiles * kTiles) * cells * cells * 2;
    Run(state, fmt::format("size/{}/{}/{}", triangles, instanced, int(traversal)), [&] {
        const Ref<Scene> scene = Primed(traversal);
        scene->Define<World>("/env_0");
        std::vector<std::string> tiles;
        std::vector<Double3> offsets;
        for (uint32_t y = 0; y < kTiles; ++y) {
            for (uint32_t x = 0; x < kTiles; ++x) {
                tiles.push_back(fmt::format("/env_0/Tile_{}", tiles.size()));
                offsets.push_back({(double(x) - 1.5) * 10, (double(y) - 1.5) * 10, 0});
            }
        }
        if (instanced) {
            scene->Define<Mesh>(tiles[0], Tile(cells, 10, 0, false));
            scene->Clone(*scene->Find(tiles[0]), Span<const std::string>(tiles).subspan(1));
        } else {
            for (uint32_t i = 0; i < tiles.size(); ++i) {
                scene->Define<Mesh>(tiles[i], Tile(cells, 10, float(i), false));
            }
        }
        scene->Select(tiles).Get<Double3>("translate").Set(Span<const Double3>(offsets));
        Look(*scene, "/env_0/Cam", -1.2, 30, 22, {0, 0, 0});
        scene->Define<DistantLight>("/Sun", {.orient = FromAzimuthElevation(60, 50)});
        Vars(*scene);
        scene->Define<RenderProduct>("/Render/Product", {.mode = MinimalMode::DisplayColor,
                                                         .camera = {"/env_0/Cam"},
                                                         .resolution = {512, 512},
                                                         .orderedVars = {"/Render/Color"}});
        return Finish(scene, {"/Render/Product"}, 512 * 512);
    });
}

constexpr uint32_t kObjects = 32;
constexpr uint32_t kTextureSide = 128;

const std::filesystem::path& TextureDirectory() {
    static const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "ernie_bench_materials";
    return directory;
}

std::string TextureFile(uint32_t index) {
    const std::filesystem::path path = TextureDirectory() / fmt::format("albedo_{}.png", index);
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(TextureDirectory());
        const float hue = float(index % 97) / 97;
        Image image(kTextureSide, kTextureSide);
        for (uint32_t y = 0; y < kTextureSide; ++y) {
            for (uint32_t x = 0; x < kTextureSide; ++x) {
                const bool odd = ((x / 16) + (y / 16)) % 2 != 0;
                image.Pixel(x, y) = odd ? Rgba{hue, 0.4f, 1 - hue, 1} : Rgba{0.9f, 0.9f, 0.85f, 1};
            }
        }
        WritePng(path.string(), image);
    }
    return path.string();
}

void Materials(benchmark::State& state) {
    const auto materials = uint32_t(state.range(0));
    const bool textured = state.range(1) != 0;
    const device::Traversal traversal = SelectTraversal(state.range(2));
    Run(state, fmt::format("materials/{}/{}/{}", materials, textured, int(traversal)), [&] {
        const Ref<Scene> scene = Primed(traversal);
        scene->Define<Scope>("/Looks");
        for (uint32_t m = 0; m < materials; ++m) {
            const float hue = float(m % 97) / 97;
            const std::string path = std::format("/Looks/Look_{}", m);
            const Prim material = scene->Define<Material>(path);
            const Prim surface =
                scene->Define<Shader>(path + "/Surface", {.id = "UsdPreviewSurface"});
            surface.Set(UsdPreviewSurface::kDiffuseColor, Float3{hue, 0.4f, 1 - hue});
            surface.Set(UsdPreviewSurface::kRoughness, 0.3f + 0.4f * hue);
            material.Connect("surface", surface, "surface");
            if (textured) {
                const Prim texture =
                    scene->Define<Shader>(path + "/Albedo", {.id = "UsdUVTexture"});
                texture.Set(UsdUVTexture::kFile, TextureFile(m));
                texture.Set(UsdUVTexture::kSourceColorSpace, TextureColorSpace::SRGB);
                surface.Connect("diffuseColor", texture, "rgb");
            }
        }
        scene->Define<World>("/env_0");
        scene->Define<Mesh>("/env_0/Ground", Tile(32, 40, 0, true));
        for (uint32_t i = 0; i < kObjects * kObjects; ++i) {
            scene->Define<Cube>(fmt::format("/env_0/Box_{}", i),
                                {.materialBinding = {fmt::format("/Looks/Look_{}", i % materials)},
                                 .translate = {(double(i % kObjects) - 15.5) * 1.1,
                                               (double(i / kObjects) - 15.5) * 1.1, 0.5},
                                 .size = 0.8});
        }
        Look(*scene, "/env_0/Cam", -1.3, 26, 20, {0, 0, 0});
        scene->Define<DistantLight>("/Sun", {.orient = FromAzimuthElevation(60, 50)});
        Vars(*scene);
        scene->Define<RenderProduct>("/Render/Product", {.mode = MinimalMode::Full,
                                                         .camera = {"/env_0/Cam"},
                                                         .resolution = {512, 512},
                                                         .orderedVars = {"/Render/Color"}});
        return Finish(scene, {"/Render/Product"}, 512 * 512);
    });
}

void ColdStart(benchmark::State& state) {
    const bool warm = state.range(0) != 0;
    const RenderMode renderMode = state.range(1) != 0 ? RenderMode::RealTime : RenderMode::Minimal;
    const std::filesystem::path root = std::filesystem::temp_directory_path() / "ernie_bench_cold";
    double deviceMs = 0;
    double sceneMs = 0;
    double rendererMs = 0;
    double frameMs = 0;
    uint32_t runs = 0;
    for (auto _ : state) {
        const std::filesystem::path cache = root / (warm ? "warm" : fmt::format("cold_{}", runs));
        std::filesystem::remove_all(cache);
        if (warm) {
            device::DeviceDesc desc = bench::BenchDesc();
            desc.cacheDirectory = root / "warm";
            Prime(*device::Device::Create(desc), renderMode);
            device::WaitForBuilds();
        }
        Clock::time_point start = Clock::now();
        device::DeviceDesc desc = bench::BenchDesc();
        desc.cacheDirectory = warm ? root / "warm" : cache;
        const Ref<device::Device> device = device::Device::Create(desc);
        deviceMs += Since(start);
        start = Clock::now();
        const Ref<Scene> scene = Scene::Create(*device);
        scene->Define<World>("/env_0");
        Props(*scene, "/env_0");
        Look(*scene, "/env_0/Cam", 0.3, 7, 4);
        scene->Define<DistantLight>("/Sun", {.orient = FromAzimuthElevation(60, 50)});
        Vars(*scene);
        scene->Define<RenderProduct>("/Render/Product", {.mode = MinimalMode::DisplayColor,
                                                         .renderMode = renderMode,
                                                         .camera = {"/env_0/Cam"},
                                                         .resolution = {256, 256},
                                                         .orderedVars = {"/Render/Color"}});
        sceneMs += Since(start);
        start = Clock::now();
        Renderer renderer(scene);
        rendererMs += Since(start);
        start = Clock::now();
        const std::vector<std::string> products = {"/Render/Product"};
        renderer.Step(products, 1.0 / 60).Wait();
        frameMs += Since(start);
        ++runs;
    }
    std::filesystem::remove_all(root);
    state.counters["device ms"] = deviceMs / runs;
    state.counters["scene ms"] = sceneMs / runs;
    state.counters["renderer ms"] = rendererMs / runs;
    state.counters["first frame ms"] = frameMs / runs;
    state.counters["total ms"] = (deviceMs + sceneMs + rendererMs + frameMs) / runs;
}

const bool kCore = bench::Core("^Render/Mixed/(uniform|mixed|skewed|rl)/worlds:256/") &&
                   bench::Core("^Render/Cameras/cameras:256/side:64/");

const bool kRegistered = [] {
    benchmark::RegisterBenchmark("Render/Cameras", Cameras)
        ->ArgNames({"cameras", "side", "hardware"})
        ->ArgsProduct({{1, 16, 64, 256, 1024}, {64, 128}, {0, 1}})
        ->UseRealTime()
        ->Unit(benchmark::kMillisecond);
    for (const Mix& mix : kMixes) {
        benchmark::RegisterBenchmark(fmt::format("Render/Mixed/{}", mix.name), Mixed, mix)
            ->ArgNames({"worlds", "hardware"})
            ->ArgsProduct({{64, 256, 1024}, {0, 1}})
            ->UseRealTime()
            ->Unit(benchmark::kMillisecond);
    }
    benchmark::RegisterBenchmark("Render/Lights/RealTime", Lights)
        ->ArgNames({"lights", "hardware"})
        ->ArgsProduct({{1, 4, 16, 64}, {0, 1}})
        ->UseRealTime()
        ->Unit(benchmark::kMillisecond);
    benchmark::RegisterBenchmark("Render/SceneSize", SceneSize)
        ->ArgNames({"triangles", "instanced", "hardware"})
        ->ArgsProduct({{10'000, 100'000, 1'000'000, 10'000'000}, {0, 1}, {0, 1}})
        ->UseRealTime()
        ->Unit(benchmark::kMillisecond);
    benchmark::RegisterBenchmark("Render/Materials/Full", Materials)
        ->ArgNames({"materials", "textured", "hardware"})
        ->ArgsProduct({{1, 16, 256, 1024}, {0, 1}, {0, 1}})
        ->UseRealTime()
        ->Unit(benchmark::kMillisecond);
    benchmark::RegisterBenchmark("Render/ColdStart", ColdStart)
        ->ArgNames({"cache", "realtime"})
        ->ArgsProduct({{0, 1}, {0, 1}})
        ->Iterations(1)
        ->UseRealTime()
        ->Unit(benchmark::kMillisecond);
    return true;
}();

}
}
