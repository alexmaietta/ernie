// SPDX-License-Identifier: BSD-3-Clause
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "bench.h"
#include "ernie/io/image.h"
#include "ernie/render/renderer.h"
#include "ernie/render/shaders.h"
#include "ernie/scene/rotation.h"
#include "ernie/scene/scene.h"
#include "ernie/util/profiler.h"

namespace ernie {
namespace {

constexpr uint32_t kWarmup = 3;
constexpr size_t kHistory = 32;
constexpr size_t kInflight = 2;
constexpr double kMinTime = 0.3;

using Clock = std::chrono::steady_clock;

double Since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

const std::filesystem::path& Directory() {
    static const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "ernie_bench_material_nodes";
    return directory;
}

std::string Texture() {
    const std::filesystem::path path = Directory() / "albedo.png";
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(Directory());
        Image image(128, 128);
        for (uint32_t y = 0; y < 128; ++y) {
            for (uint32_t x = 0; x < 128; ++x) {
                const bool odd = ((x / 16) + (y / 16)) % 2 != 0;
                image.Pixel(x, y) = odd ? Rgba{0.7f, 0.3f, 0.2f, 1} : Rgba{0.9f, 0.9f, 0.85f, 1};
            }
        }
        WritePng(path.string(), image);
    }
    return path.string();
}

std::filesystem::path NodeFile(uint32_t index) {
    const std::filesystem::path path = Directory() / std::format("bench_node_{}.slang", index);
    std::filesystem::create_directories(Directory());
    std::ofstream(path) << std::format(R"(
import ernie.material;

struct BenchNode{0} {{
    [Color] float3 tint = float3(0.{1}, 0.5, 0.3);
    float roughness = 0.4;
    float metal = 0;

    void Material(inout SurfaceInputs m, ShadeContext c) {{
        m.baseColor = SpectralValue.Rgb(tint * (1 + 0.0{1} * c.point.position.x));
        m.roughness = roughness;
        m.metallic = metal;
    }}
}}
)",
                                       index, index % 10);
    return path;
}

enum class Look : int64_t { Diffuse, Preview, Textured, Node, NodeTextured };

struct Setup {
    Ref<Scene> scene;
    std::vector<std::string> shaders;
    std::vector<std::string> cubes;
    std::optional<Attribute<float>> roughness;
    std::optional<Attribute<float>> temperatures;
    uint64_t step = 0;
};

void DefineLook(Scene& scene, Look look, const std::string& path, uint32_t index,
                uint32_t definitions) {
    const Prim material = scene.Define<Material>(path);
    if (look == Look::Diffuse) {
        return;
    }
    const float hue = float(index % 97) / 97;
    if (look == Look::Preview || look == Look::Textured) {
        const Prim surface = scene.Define<Shader>(path + "/Surface", {.id = "UsdPreviewSurface"});
        surface.Set(UsdPreviewSurface::kDiffuseColor, Float3{hue, 0.4f, 1 - hue});
        surface.Set(UsdPreviewSurface::kRoughness, 0.3f + 0.4f * hue);
        material.Connect("surface", surface, "surface");
        if (look == Look::Textured) {
            const Prim texture = scene.Define<Shader>(path + "/Albedo", {.id = "UsdUVTexture"});
            texture.Set(UsdUVTexture::kFile, Texture());
            texture.Set(UsdUVTexture::kSourceColorSpace, TextureColorSpace::SRGB);
            surface.Connect("diffuseColor", texture, "rgb");
        }
        return;
    }
    const uint32_t node = index % std::max(definitions, 1u);
    const Prim surface =
        scene.Define<Shader>(path + "/Surface", {.id = std::format("BenchNode{}", node)});
    surface.Set("tint", Float3{hue, 0.4f, 1 - hue});
    material.Connect("surface", surface, "surface");
    if (look == Look::NodeTextured) {
        const Prim texture = scene.Define<Shader>(path + "/Albedo", {.id = "UsdUVTexture"});
        texture.Set(UsdUVTexture::kFile, Texture());
        texture.Set(UsdUVTexture::kSourceColorSpace, TextureColorSpace::SRGB);
        surface.Connect("tint", texture, "rgb");
    }
}

Setup Build(device::Device& device, Look look, uint32_t materials, uint32_t definitions,
            bool thermal, bool infrared, uint32_t objects, bool render) {
    const Ref<Scene> scene = Scene::Create(device, {.waitForKernels = true});
    if (look == Look::Node || look == Look::NodeTextured) {
        for (uint32_t d = 0; d < definitions; ++d) {
            LoadShaders(*scene, NodeFile(d));
        }
    }
    Setup setup{scene};
    scene->Define<Scope>("/Looks");
    for (uint32_t m = 0; m < materials; ++m) {
        const std::string path = std::format("/Looks/Look_{}", m);
        DefineLook(*scene, look, path, m, definitions);
        setup.shaders.push_back(path + "/Surface");
    }
    const uint32_t side = uint32_t(std::ceil(std::sqrt(double(objects))));
    std::vector<Float2> st = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    scene->Define<Mesh>("/Ground",
                        {.materialBinding = {"/Looks/Look_0"},
                         .primvars = {Primvar::Create("st", Interpolation::Vertex, st)},
                         .points = {{-40, -40, 0}, {40, -40, 0}, {40, 40, 0}, {-40, 40, 0}},
                         .faceVertexIndices = {0, 1, 2, 3},
                         .faceVertexCounts = {4}});
    for (uint32_t i = 0; i < objects; ++i) {
        const std::string path = std::format("/Box_{}", i);
        scene->Define<Sphere>(path,
                              {.materialBinding = {std::format("/Looks/Look_{}", i % materials)},
                               .translate = {(double(i % side) - side / 2.0) * 1.1,
                                             (double(i / side) - side / 2.0) * 1.1, 0.5},
                               .radius = 0.45f});
        if (thermal) {
            scene->Apply<ThermalAPI>(*scene->Find(path), {.thermalTemperature = 300});
        }
        setup.cubes.push_back(path);
    }
    const double distance = 0.9 * side + 4;
    const Double3 eye = {-distance * 0.6, -distance * 0.6, distance * 0.7};
    const Prim camera =
        scene->Define<Camera>("/Cam", {.translate = eye, .orient = LookAt(eye, {0, 0, 0})});
    if (infrared) {
        camera.Set(Camera::kDetectorSpectralRange, Float2{400, 1000});
    }
    scene->Define<DistantLight>("/Sun", {.orient = FromAzimuthElevation(60, 50)});
    scene->Define<SphereLight>("/Lamp", {.translate = {2, 2, 4}, .radius = 0.5f});
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Color", {.sourceName = "LdrColor"});
    scene->Define<RenderProduct>("/Render/View",
                                 {.renderMode = render ? RenderMode::RealTime : RenderMode::Minimal,
                                  .camera = {"/Cam"},
                                  .resolution = {512, 512},
                                  .orderedVars = {"/Render/Color"}});
    if (look == Look::Preview || look == Look::Textured) {
        setup.roughness.emplace(scene->Select(setup.shaders).Get<float>("roughness"));
    }
    if (look == Look::Node || look == Look::NodeTextured) {
        setup.roughness.emplace(scene->Select(setup.shaders).Get<float>("roughness"));
    }
    if (thermal) {
        setup.temperatures.emplace(scene->Select(setup.cubes).Get<float>("thermalTemperature"));
    }
    scene->Publish({.wait = true}).Wait();
    return setup;
}

void Report(benchmark::State& state) {
    std::map<std::string, double> cpu;
    std::map<std::string, double> gpu;
    double total = 0;
    const std::function<void(const profiler::ReportNode&, bool)> collect =
        [&](const profiler::ReportNode& zone, bool device) {
            if (device) {
                if (zone.name.starts_with("Materials/") || zone.name.starts_with("Render/")) {
                    gpu[zone.name] += zone.ms;
                }
                return;
            }
            if (zone.name.starts_with("Materials/")) {
                cpu[zone.name] += zone.ms;
            }
            for (const profiler::ReportNode& child : zone.children) {
                collect(child, false);
            }
        };
    for (const profiler::ReportTrack& track : profiler::Summarize(profiler::Frames()).tracks) {
        for (const profiler::ReportNode& zone : track.zones) {
            total += track.gpu ? 0 : zone.ms;
            collect(zone, track.gpu);
        }
    }
    if (std::getenv("ERNIE_BENCH_ZONES") != nullptr) {
        std::printf("%s\n%s\n", state.name().c_str(),
                    profiler::ReportText(profiler::Summarize(profiler::Frames())).c_str());
    }
    state.counters["cpu ms"] = total;
    for (const auto& [name, ms] : cpu) {
        state.counters["cpu " + name] = ms;
    }
    for (const auto& [name, ms] : gpu) {
        state.counters["gpu " + name] = ms;
    }
}

struct Edit {
    const char* name;
    std::function<void(Setup&)> apply;
};

const Edit kEdits[] = {
    {"idle", [](Setup&) {}},
    {"value-one",
     [](Setup& s) {
         s.scene->Find(s.shaders.front())
             ->Set(UsdPreviewSurface::kRoughness, 0.1f + 0.1f * float(s.step % 7));
     }},
    {"value-all",
     [](Setup& s) {
         s.roughness->Set(std::vector<float>(s.shaders.size(), 0.1f + 0.1f * float(s.step % 7)));
     }},
    {"temperature",
     [](Setup& s) {
         s.temperatures->Set(std::vector<float>(s.cubes.size(), 290 + float(s.step % 7)));
     }},
    {"move",
     [](Setup& s) {
         s.scene->Find(s.cubes.front())
             ->Set(Xformable::kTranslate, Double3{0, 0, 0.5 + 0.01 * double(s.step % 7)});
     }},
    {"spawn",
     [](Setup& s) {
         if (const auto found = s.scene->Find("/Spawned")) {
             s.scene->Remove(*found);
         } else {
             s.scene->Define<Cube>("/Spawned", {.translate = {0, 0, 3}, .size = 0.5});
         }
     }},
    {"reconnect",
     [](Setup& s) {
         const Prim material = *s.scene->Find("/Looks/Look_0");
         if (s.step % 2 == 0) {
             material.Disconnect("surface");
         } else {
             material.Connect("surface", *s.scene->Find(s.shaders.front()), "surface");
         }
     }},
};

void Publish(benchmark::State& state, const Edit& edit) {
    const auto materials = uint32_t(state.range(0));
    const bool thermal = state.range(1) != 0;
    const bool infrared = state.range(2) != 0;
    const std::string_view name = edit.name;
    if (name == "temperature" && !thermal) {
        state.SkipWithMessage("needs thermal");
        return;
    }
    Setup& setup =
        bench::Cached<Setup>(std::format("publish/{}/{}/{}", materials, thermal, infrared), [&] {
            return Build(bench::BenchDevice(), Look::Preview, materials, 1, thermal, infrared,
                         materials, false);
        });
    for (uint32_t i = 0; i < kWarmup; ++i) {
        ++setup.step;
        edit.apply(setup);
        setup.scene->Publish().Wait();
    }
    profiler::SetHistory(kHistory);
    profiler::Clear();
    profiler::BeginFrame();
    for (auto _ : state) {
        ++setup.step;
        edit.apply(setup);
        setup.scene->Publish().Wait();
        profiler::BeginFrame();
    }
    profiler::BeginFrame();
    Report(state);
    profiler::SetHistory(2);
}

struct Rendered {
    Setup setup;
    Renderer renderer;
    double firstMs = 0;
};

void Shade(benchmark::State& state) {
    const auto look = Look(state.range(0));
    const auto definitions = uint32_t(state.range(1));
    const auto materials = uint32_t(state.range(2));
    const bool render = state.range(3) != 0;
    if ((look != Look::Node && look != Look::NodeTextured) && definitions != 1) {
        state.SkipWithMessage("definitions only vary for nodes");
        return;
    }
    Rendered& rendered = bench::Cached<Rendered>(
        std::format("shade/{}/{}/{}/{}", int(look), definitions, materials, render), [&] {
            Setup setup = Build(bench::BenchDevice(), look, materials, definitions, false, false,
                                256, render);
            const Clock::time_point start = Clock::now();
            Renderer renderer(setup.scene);
            renderer.Step(std::vector<std::string>{"/Render/View"}, 1.0 / 60).Wait();
            const double first = Since(start);
            for (uint32_t i = 0; i < kWarmup; ++i) {
                renderer.Step(std::vector<std::string>{"/Render/View"}, 1.0 / 60).Wait();
            }
            return Rendered{std::move(setup), std::move(renderer), first};
        });
    profiler::SetHistory(kHistory);
    profiler::Clear();
    profiler::BeginFrame();
    std::deque<RenderOutput> flight;
    const std::vector<std::string> products = {"/Render/View"};
    const Prim camera = *rendered.setup.scene->Find("/Cam");
    const Double3 eye = camera.Get(Xformable::kTranslate);
    uint64_t step = 0;
    for (auto _ : state) {
        camera.Set(Xformable::kTranslate,
                   Double3{eye.x + 0.001 * double(++step % 2), eye.y, eye.z});
        flight.push_back(rendered.renderer.Step(products, 1.0 / 60));
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
    Report(state);
    profiler::SetHistory(2);
    state.counters["first frame ms"] = rendered.firstMs;
}

const bool kRegistered = [] {
    for (const Edit& edit : kEdits) {
        for (const int64_t materials : {16, 1024, 8192}) {
            for (const int64_t thermal : {0, 1}) {
                for (const int64_t infrared : {0, 1}) {
                    benchmark::RegisterBenchmark(std::format("Material/Publish/{}", edit.name),
                                                 Publish, edit)
                        ->ArgNames({"materials", "thermal", "infrared"})
                        ->Args({materials, thermal, infrared})
                        ->MinTime(kMinTime)
                        ->UseRealTime()
                        ->Unit(benchmark::kMillisecond);
                }
            }
        }
    }
    benchmark::RegisterBenchmark("Material/Shade", Shade)
        ->ArgNames({"look", "definitions", "materials", "realtime"})
        ->ArgsProduct({{0, 1, 2, 3, 4}, {1}, {16}, {0, 1}})
        ->MinTime(1.0)
        ->UseRealTime()
        ->Unit(benchmark::kMillisecond);
    benchmark::RegisterBenchmark("Material/Shade", Shade)
        ->ArgNames({"look", "definitions", "materials", "realtime"})
        ->ArgsProduct({{3}, {2, 4, 8}, {16}, {0, 1}})
        ->MinTime(1.0)
        ->UseRealTime()
        ->Unit(benchmark::kMillisecond);
    return true;
}();

}
}
