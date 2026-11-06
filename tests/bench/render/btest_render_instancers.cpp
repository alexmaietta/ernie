// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
#include <deque>
#include <format>
#include <random>
#include <string>
#include <vector>

#include "bench.h"
#include "ernie/device/device.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/rotation.h"
#include "ernie/scene/scene.h"
#include "ernie/util/profiler.h"

namespace ernie {
namespace {

constexpr uint32_t kSide = 128;
constexpr uint32_t kWarmup = 3;
constexpr size_t kInflight = 2;
constexpr size_t kHistory = 32;
const std::vector<std::string> kProducts = {"/Render/Product"};

enum class Layout { Grid, Scattered, Clustered };
enum class Family { Count, Nested, Prototypes, Animated, Lamps };

struct Shape {
    Family family = Family::Count;
    Layout layout = Layout::Grid;
    uint32_t count = 1000;
    uint32_t inner = 1;
    uint32_t prototypes = 1;
    double extent = 100;
};

struct Setup {
    Ref<Scene> scene;
    Renderer renderer;
    Prim instancer;
    std::vector<std::vector<Float3>> frames;
    uint32_t step = 0;
};

std::vector<Float3> Positions(const Shape& shape, uint32_t count, uint32_t seed) {
    std::vector<Float3> positions;
    positions.reserve(count);
    const float half = float(shape.extent) / 2;
    if (shape.layout == Layout::Grid) {
        const auto across = uint32_t(std::ceil(std::sqrt(double(count))));
        const float spacing = float(shape.extent) / float(across);
        for (uint32_t i = 0; i < count; ++i) {
            positions.push_back(
                {float(i % across) * spacing - half, float(i / across) * spacing - half, 0});
        }
        return positions;
    }
    std::mt19937 random(seed);
    const float spread = shape.layout == Layout::Clustered ? half * 0.02f : half;
    std::uniform_real_distribution<float> place(-spread, spread);
    for (uint32_t i = 0; i < count; ++i) {
        positions.push_back({place(random), place(random), 0});
    }
    return positions;
}

std::vector<Float4> Orientations(uint32_t count, uint32_t seed) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> turn(0, 6.2831853f);
    std::vector<Float4> orientations;
    orientations.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const float angle = turn(random) / 2;
        orientations.push_back({0, 0, std::sin(angle), std::cos(angle)});
    }
    return orientations;
}

std::vector<std::string> DefinePrototypes(Scene& scene, const std::string& parent,
                                          uint32_t prototypes) {
    std::vector<std::string> roots;
    scene.Define<Scope>(parent);
    for (uint32_t p = 0; p < prototypes; ++p) {
        const std::string root = std::format("{}/Proto_{}", parent, p);
        scene.Define<Xform>(root);
        const float tint = float(p % 8) / 7;
        if (p % 2 == 0) {
            scene.Define<Cube>(
                root + "/Body",
                {.translate = {0, 0, 0.5}, .displayColor = {tint, 0.5f, 1 - tint}, .size = 1});
        } else {
            scene.Define<Sphere>(
                root + "/Body",
                {.translate = {0, 0, 0.5}, .displayColor = {1 - tint, tint, 0.5f}, .radius = 0.5});
        }
        roots.push_back(root);
    }
    return roots;
}

Setup Prepare(const Shape& shape, device::Traversal traversal) {
    const Ref<Scene> scene = Scene::Create(bench::BenchDevice(traversal));
    const uint32_t outer = shape.family == Family::Nested ? shape.count / shape.inner : shape.count;
    std::vector<std::string> roots;
    if (shape.family == Family::Nested) {
        roots = {"/Forest/Protos/Grove"};
    } else if (shape.family == Family::Lamps) {
        roots = {"/Forest/Protos/Lamp"};
    } else {
        for (uint32_t p = 0; p < shape.prototypes; ++p) {
            roots.push_back(std::format("/Forest/Protos/Proto_{}", p));
        }
    }
    PointInstancer::Desc desc;
    desc.prototypes = roots;
    desc.positions = Positions(shape, outer, 7);
    desc.orientations = Orientations(outer, 11);
    std::mt19937 random(17);
    std::uniform_int_distribution<int32_t> pick(0, int32_t(roots.size()) - 1);
    for (uint32_t i = 0; i < outer; ++i) {
        desc.protoIndices.push_back(pick(random));
    }
    const Prim instancer = scene->Define<PointInstancer>("/Forest", desc);
    if (shape.family == Family::Nested) {
        Shape inner = shape;
        inner.extent = 4;
        scene->Define<Scope>("/Forest/Protos");
        scene->Define<Xform>("/Forest/Protos/Grove");
        PointInstancer::Desc grove;
        grove.prototypes = {"/Forest/Protos/Grove/Trees/Protos/Proto_0"};
        grove.positions = Positions(inner, shape.inner, 13);
        grove.protoIndices.assign(shape.inner, 0);
        scene->Define<PointInstancer>("/Forest/Protos/Grove/Trees", grove);
        DefinePrototypes(*scene, "/Forest/Protos/Grove/Trees/Protos", 1);
    } else if (shape.family == Family::Lamps) {
        scene->Define<Scope>("/Forest/Protos");
        scene->Define<Xform>("/Forest/Protos/Lamp");
        scene->Define<Cube>(
            "/Forest/Protos/Lamp/Post",
            {.translate = {0, 0, 0.5}, .displayColor = {0.5f, 0.5f, 0.5f}, .size = 0.2f});
        scene->Define<SphereLight>("/Forest/Protos/Lamp/Bulb",
                                   {.translate = {0, 0, 1.2}, .intensity = 50, .radius = 0.1f});
    } else {
        DefinePrototypes(*scene, "/Forest/Protos", shape.prototypes);
    }
    scene->Define<Plane>("/Ground", {.displayColor = {0.4f, 0.45f, 0.35f},
                                     .width = float(shape.extent) * 1.2f,
                                     .length = float(shape.extent) * 1.2f});
    const double height = shape.extent * 0.6;
    scene->Define<Camera>("/Eye", {.translate = {0, -shape.extent * 0.6, height},
                                   .orient = LookAt({0, -shape.extent * 0.6, height}, {0, 0, 0})});
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Color", {.sourceName = RenderVarSource::LdrColor});
    scene->Define<RenderProduct>(
        "/Render/Product",
        {.camera = {"/Eye"}, .resolution = {kSide, kSide}, .orderedVars = {"/Render/Color"}});
    if (shape.family == Family::Lamps) {
        scene->Select("/Render/Product").Get<RenderMode>("renderMode").Set(RenderMode::RealTime);
    }
    Setup setup{scene, Renderer(scene), instancer, {}, 0};
    if (shape.family == Family::Animated) {
        for (uint32_t frame = 0; frame < 2; ++frame) {
            std::vector<Float3> moved = desc.positions;
            for (Float3& position : moved) {
                position.z = 0.2f * float(frame);
            }
            setup.frames.push_back(std::move(moved));
        }
    }
    for (uint32_t i = 0; i < kWarmup; ++i) {
        setup.renderer.Step(kProducts, 1.0 / 60).Wait();
    }
    return setup;
}

const char* Name(Layout layout) {
    switch (layout) {
    case Layout::Grid:
        return "grid";
    case Layout::Scattered:
        return "scattered";
    case Layout::Clustered:
        return "clustered";
    }
    return "";
}

void Run(benchmark::State& state, const Shape& shape, bool hardware) {
    const device::Traversal traversal =
        hardware ? device::Traversal::Hardware : device::Traversal::Software;
    Setup* setup = nullptr;
    try {
        setup = &bench::Cached<Setup>(std::format("{}/{}/{}/{}/{}/{}/{}", int(shape.family),
                                                  Name(shape.layout), shape.count, shape.inner,
                                                  shape.prototypes, shape.extent, hardware),
                                      [&] { return Prepare(shape, traversal); });
    } catch (const UnsupportedError& error) {
        state.SkipWithMessage(error.what());
        return;
    }
    profiler::SetHistory(kHistory);
    profiler::Clear();
    profiler::BeginFrame();
    std::deque<RenderOutput> flight;
    for (auto _ : state) {
        if (!setup->frames.empty()) {
            setup->instancer.Set(PointInstancer::kPositions,
                                 setup->frames[setup->step++ % setup->frames.size()]);
        }
        flight.push_back(setup->renderer.Step(kProducts, 1.0 / 60));
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
    double bvh = 0;
    double instances = 0;
    double other = 0;
    for (const profiler::ReportTrack& track : profiler::Summarize(profiler::Frames()).tracks) {
        for (const profiler::ReportNode& zone : track.zones) {
            if (!track.gpu) {
                step += zone.name == "Render/Step" ? zone.ms : 0;
            } else if (zone.name.starts_with("Bvh/")) {
                bvh += zone.ms;
            } else if (zone.name.starts_with("Instances/")) {
                instances += zone.ms;
            } else if (zone.name.starts_with("Render/")) {
                render += zone.ms;
            } else {
                other += zone.ms;
            }
        }
    }
    profiler::SetHistory(2);
    state.counters["copies"] = double(shape.count);
    state.counters["copies/s"] =
        benchmark::Counter(double(state.iterations()) * shape.count, benchmark::Counter::kIsRate);
    state.counters["cpu ms/step"] = step;
    state.counters["gpu render ms"] = render;
    state.counters["gpu bvh ms"] = bvh;
    state.counters["gpu instances ms"] = instances;
    state.counters["gpu other ms"] = other;
}

void Count(benchmark::State& state, Layout layout, double extent) {
    Run(state, {.layout = layout, .count = uint32_t(state.range(0)), .extent = extent},
        state.range(1) != 0);
}

void Nested(benchmark::State& state) {
    Run(state,
        {.family = Family::Nested,
         .count = uint32_t(state.range(0)),
         .inner = uint32_t(state.range(1))},
        state.range(2) != 0);
}

void Prototypes(benchmark::State& state) {
    Run(state,
        {.family = Family::Prototypes,
         .count = uint32_t(state.range(0)),
         .prototypes = uint32_t(state.range(1))},
        state.range(2) != 0);
}

void Animated(benchmark::State& state) {
    Run(state, {.family = Family::Animated, .count = uint32_t(state.range(0))},
        state.range(1) != 0);
}

void Lamps(benchmark::State& state) {
    Run(state, {.family = Family::Lamps, .count = uint32_t(state.range(0))}, state.range(1) != 0);
}

const bool kCore = bench::Core("^Render/Instancers/Count/grid/copies:(10000|1000000)/") &&
                   bench::Core("^Render/Instancers/Nested/copies:1000000/") &&
                   bench::Core("^Render/Instancers/Animated/copies:100000/");

const bool kRegistered = [] {
    const std::vector<int64_t> counts = {1000, 10000, 100000, 1000000};
    const std::vector<int64_t> hardware = {0, 1};
    benchmark::RegisterBenchmark("Render/Instancers/Count/grid", Count, Layout::Grid, 100.0)
        ->ArgsProduct({counts, hardware})
        ->ArgNames({"copies", "hardware"})
        ->Unit(benchmark::kMillisecond)
        ->UseRealTime();
    benchmark::RegisterBenchmark("Render/Instancers/Count/scattered", Count, Layout::Scattered,
                                 100.0)
        ->ArgsProduct({counts, hardware})
        ->ArgNames({"copies", "hardware"})
        ->Unit(benchmark::kMillisecond)
        ->UseRealTime();
    benchmark::RegisterBenchmark("Render/Instancers/Count/scattered-10km", Count, Layout::Scattered,
                                 10000.0)
        ->ArgsProduct({counts, hardware})
        ->ArgNames({"copies", "hardware"})
        ->Unit(benchmark::kMillisecond)
        ->UseRealTime();
    benchmark::RegisterBenchmark("Render/Instancers/Count/clustered", Count, Layout::Clustered,
                                 100.0)
        ->ArgsProduct({counts, hardware})
        ->ArgNames({"copies", "hardware"})
        ->Unit(benchmark::kMillisecond)
        ->UseRealTime();
    benchmark::RegisterBenchmark("Render/Instancers/Nested", Nested)
        ->ArgsProduct({{10000, 1000000}, {10, 100, 1000}, hardware})
        ->ArgNames({"copies", "inner", "hardware"})
        ->Unit(benchmark::kMillisecond)
        ->UseRealTime();
    benchmark::RegisterBenchmark("Render/Instancers/Prototypes", Prototypes)
        ->ArgsProduct({{100000}, {1, 16, 256}, hardware})
        ->ArgNames({"copies", "prototypes", "hardware"})
        ->Unit(benchmark::kMillisecond)
        ->UseRealTime();
    benchmark::RegisterBenchmark("Render/Instancers/Animated", Animated)
        ->ArgsProduct({{10000, 100000, 1000000}, hardware})
        ->ArgNames({"copies", "hardware"})
        ->Unit(benchmark::kMillisecond)
        ->UseRealTime();
    benchmark::RegisterBenchmark("Render/Instancers/Lamps", Lamps)
        ->ArgsProduct({{16, 1000, 100000}, hardware})
        ->ArgNames({"copies", "hardware"})
        ->Unit(benchmark::kMillisecond)
        ->UseRealTime();
    return true;
}();

}
}
