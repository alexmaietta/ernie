// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "bench.h"
#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/rotation.h"
#include "ernie/scene/scene.h"
#include "ernie/util/profiler.h"

namespace ernie {
namespace {

constexpr uint32_t kSide = 64;
constexpr uint32_t kWarmup = 5;
constexpr uint32_t kFrames = 2;
constexpr size_t kInflight = 2;
constexpr size_t kHistory = 32;
constexpr double kTau = 6.283185307179586;
const std::vector<std::string> kProducts = {"/Render/Product"};

struct Mode {
    const char* name;
    MinimalMode minimal;
    bool shadows;
    bool render;
    const char* var;
};

const Mode kUpdate = {"Update", MinimalMode::Data, false, false, ""};
const Mode kDepth = {"Depth", MinimalMode::Data, false, true, "/Render/Depth"};
const Mode kColor = {"DisplayColor/shadows:0", MinimalMode::DisplayColor, false, true,
                     "/Render/Color"};
const Mode kShadowed = {"DisplayColor/shadows:1", MinimalMode::DisplayColor, true, true,
                        "/Render/Color"};

enum class Motion { Static, Rigid, Deforming, Remesh, Respawn };

const char* Name(Motion motion) {
    switch (motion) {
    case Motion::Static:
        return "static";
    case Motion::Rigid:
        return "rigid";
    case Motion::Deforming:
        return "deforming";
    case Motion::Remesh:
        return "remesh";
    case Motion::Respawn:
        return "respawn";
    }
    return "";
}

const std::pair<const char*, const char*> kVars[] = {
    {"LdrColor", "/Render/Color"},
    {"DistanceToCamera", "/Render/Depth"},
    {"Normal", "/Render/Normal"},
    {"DiffuseAlbedo", "/Render/Albedo"},
    {"NonStableInstanceSegmentation", "/Render/Instances"},
    {"HdrColor", "/Render/Hdr"},
    {"DistanceToImagePlane", "/Render/Plane"},
    {"StableIdSegmentation", "/Render/StableIds"},
};

struct Shape {
    uint32_t worlds = 1;
    uint32_t cameras = 1;
    uint32_t cells = 32;
    uint32_t balls = 8;
    uint32_t outputs = 0;
};

std::vector<Float3> Wave(uint32_t cells, double phase) {
    std::vector<Float3> points;
    const uint32_t side = cells + 1;
    for (uint32_t y = 0; y < side; ++y) {
        for (uint32_t x = 0; x < side; ++x) {
            const float u = float(x) / float(cells);
            const float v = float(y) / float(cells);
            points.push_back({(u - 0.5f) * 20, (v - 0.5f) * 20,
                              0.3f * std::sin(u * 17 + float(phase)) * std::cos(v * 13)});
        }
    }
    return points;
}

struct Topology {
    std::vector<int32_t> indices;
    std::vector<int32_t> counts;
};

Topology Grid(uint32_t cells, bool triangles) {
    Topology topology;
    const auto side = int32_t(cells + 1);
    for (uint32_t y = 0; y < cells; ++y) {
        for (uint32_t x = 0; x < cells; ++x) {
            const auto a = int32_t(y) * side + int32_t(x);
            if (triangles) {
                topology.indices.insert(topology.indices.end(),
                                        {a, a + 1, a + side + 1, a, a + side + 1, a + side});
                topology.counts.insert(topology.counts.end(), {3, 3});
            } else {
                topology.indices.insert(topology.indices.end(), {a, a + 1, a + side + 1, a + side});
                topology.counts.push_back(4);
            }
        }
    }
    return topology;
}

std::vector<Double3> Spots(uint32_t balls) {
    const auto across = uint32_t(std::ceil(std::sqrt(double(balls))));
    const double spacing = 14.0 / across;
    std::vector<Double3> spots;
    for (uint32_t i = 0; i < balls; ++i) {
        spots.push_back({(double(i % across) - 0.5 * (across - 1)) * spacing,
                         (double(i / across) - 0.5 * (across - 1)) * spacing, 0.6});
    }
    return spots;
}

float Radius(uint32_t balls) {
    return float(std::min(0.5, 0.35 * 14.0 / std::ceil(std::sqrt(double(balls)))));
}

struct Setup {
    Ref<Scene> scene;
    Renderer renderer;
    std::vector<std::string> products;
    Shape shape;
    ArrayAttribute<Float3> points;
    ArrayAttribute<int32_t> indices;
    ArrayAttribute<int32_t> counts;
    Attribute<Double3> translate;
    Ref<device::Buffer> frames;
    uint64_t frameBytes = 0;
    Topology topologies[2];
    std::vector<std::string> respawned;
    uint64_t pixels = 0;
    uint64_t step = 0;
};

Ref<device::Buffer> Frames(device::Device& device, const Mapping& mapping,
                           const std::vector<std::byte>& frames) {
    const Span<const int64_t> shape = mapping.Values().Shape();
    const Span<const int64_t> strides = mapping.Values().Strides();
    int64_t expected = 1;
    for (size_t axis = shape.size(); axis-- > 0;) {
        if (strides[axis] != expected) {
            throw Error("the mapped values aren't contiguous");
        }
        expected *= shape[axis];
    }
    return device.CreateBuffer({.label = "motion frames"}, std::span<const std::byte>(frames));
}

template <class T> void Append(std::vector<std::byte>& bytes, const std::vector<T>& items) {
    const auto* data = reinterpret_cast<const std::byte*>(items.data());
    bytes.insert(bytes.end(), data, data + items.size() * sizeof(T));
}

Setup Prepare(const Mode& mode, Motion motion, const Shape& shape, device::Traversal traversal) {
    const Ref<Scene> scene = Scene::Create(bench::BenchDevice(traversal));
    scene->Define<World>("/env_0");
    Mesh::Desc ground;
    ground.points = Wave(shape.cells, 0);
    const Topology quads = Grid(shape.cells, false);
    ground.faceVertexIndices = quads.indices;
    ground.faceVertexCounts = quads.counts;
    ground.displayColor = {0.6f, 0.55f, 0.5f};
    scene->Define<Mesh>("/env_0/Ground", ground);
    const std::vector<Double3> spots = Spots(shape.balls);
    for (uint32_t i = 0; i < shape.balls; ++i) {
        scene->Define<Sphere>(
            fmt::format("/env_0/Ball_{}", i),
            {.translate = spots[i],
             .displayColor = {0.2f + 0.6f * float(i % 8) / 7, 0.4f, 0.8f - 0.6f * float(i % 8) / 7},
             .radius = Radius(shape.balls)});
    }
    scene->Define<Cube>(
        "/env_0/Box", {.translate = {0, 0, 0.75}, .displayColor = {0.8f, 0.3f, 0.2f}, .size = 1.5});
    for (uint32_t c = 0; c < shape.cameras; ++c) {
        const double angle = kTau * c / shape.cameras;
        const Double3 eye = {7 * std::cos(angle), 7 * std::sin(angle), 4};
        scene->Define<Camera>(fmt::format("/env_0/Cam_{}", c),
                              {.translate = eye, .orient = LookAt(eye, {0, 0, 0.5})});
    }
    scene->Clone(*scene->Find("/env_0"), shape.worlds - 1);
    scene->Define<DistantLight>("/Sun", {.orient = FromAzimuthElevation(60, 50)});
    scene->Define<Scope>("/Render");
    for (const auto& [source, path] : kVars) {
        scene->Define<RenderVar>(path, {.sourceName = source});
    }
    std::vector<std::string> views;
    std::vector<std::string> grounds;
    std::vector<std::string> balls;
    std::vector<std::string> respawned;
    for (uint32_t w = 0; w < shape.worlds; ++w) {
        grounds.push_back(fmt::format("/env_{}/Ground", w));
        respawned.push_back(fmt::format("/env_{}/Ball_0", w));
        for (uint32_t c = 0; c < shape.cameras; ++c) {
            views.push_back(fmt::format("/env_{}/Cam_{}", w, c));
        }
        for (uint32_t i = 0; i < shape.balls; ++i) {
            balls.push_back(fmt::format("/env_{}/Ball_{}", w, i));
        }
    }
    std::vector<std::string> vars = {mode.var};
    if (shape.outputs > 0) {
        vars.clear();
        for (uint32_t i = 0; i < shape.outputs; ++i) {
            vars.push_back(kVars[i].second);
        }
    }
    if (mode.render) {
        scene->Define<RenderProduct>("/Render/Product", {.mode = mode.minimal,
                                                         .castShadows = mode.shadows,
                                                         .camera = views,
                                                         .resolution = {kSide, kSide},
                                                         .orderedVars = vars});
    }
    const View surfaces = scene->Select(grounds);
    Setup setup{scene,
                Renderer(scene),
                mode.render ? kProducts : std::vector<std::string>{},
                shape,
                surfaces.GetArray<Float3>("points"),
                surfaces.GetArray<int32_t>("faceVertexIndices"),
                surfaces.GetArray<int32_t>("faceVertexCounts"),
                scene->Select(balls).Get<Double3>("translate"),
                nullptr,
                0,
                {quads, Grid(shape.cells, true)},
                respawned,
                uint64_t(shape.worlds) * shape.cameras * kSide * kSide};
    device::Device& device = scene->GetDevice();
    std::vector<std::byte> frames;
    if (motion == Motion::Deforming) {
        for (uint32_t f = 0; f < kFrames; ++f) {
            const std::vector<Float3> wave = Wave(shape.cells, 0.05 * double(f + 1));
            for (uint32_t w = 0; w < shape.worlds; ++w) {
                Append(frames, wave);
            }
        }
        Mapping mapping = setup.points.Map(MapAccess::Write);
        setup.frames = Frames(device, mapping, frames);
        mapping.Close();
    } else if (motion == Motion::Rigid) {
        for (uint32_t f = 0; f < kFrames; ++f) {
            std::vector<Double3> moved;
            for (uint32_t w = 0; w < shape.worlds; ++w) {
                for (uint32_t i = 0; i < shape.balls; ++i) {
                    const double phase = 0.05 * double(f + 1) + 0.1 * double(i % 8);
                    moved.push_back({spots[i].x + 0.3 * std::sin(phase),
                                     spots[i].y + 0.3 * std::cos(phase), spots[i].z});
                }
            }
            Append(frames, moved);
        }
        Mapping mapping = setup.translate.Map(MapAccess::Write);
        setup.frames = Frames(device, mapping, frames);
        mapping.Close();
    }
    setup.frameBytes = frames.size() / kFrames;
    for (uint32_t i = 0; i < kWarmup; ++i) {
        setup.renderer.Step(setup.products, 1.0 / 60).Wait();
    }
    return setup;
}

void Copy(Setup& setup, UntypedAttribute& attribute) {
    Mapping mapping = attribute.Map(MapAccess::Write);
    const device::Tensor& values = mapping.Values();
    device::Queue& queue = setup.scene->GetDevice().GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    encoder.Copy(setup.frames->Slice(setup.step % kFrames * setup.frameBytes, setup.frameBytes),
                 values.GetBuffer().Slice(values.Offset(), setup.frameBytes));
    queue.Submit(encoder.Finish());
    mapping.Close();
}

void Move(Setup& setup, Motion motion) {
    ++setup.step;
    switch (motion) {
    case Motion::Static:
        break;
    case Motion::Rigid:
        Copy(setup, setup.translate);
        break;
    case Motion::Deforming:
        Copy(setup, setup.points);
        break;
    case Motion::Remesh: {
        const Topology& topology = setup.topologies[setup.step % 2];
        for (uint32_t i = 0; i < setup.shape.worlds; ++i) {
            setup.indices.Set(i, topology.indices);
            setup.counts.Set(i, topology.counts);
        }
        break;
    }
    case Motion::Respawn: {
        for (const std::string& path : setup.respawned) {
            setup.scene->Remove(*setup.scene->Find(path));
        }
        const double phase = 0.05 * double(setup.step);
        const Double3 spot = Spots(setup.shape.balls)[0];
        setup.scene->Define<Sphere>(
            Span<const std::string>(setup.respawned),
            {.translate = {spot.x + 0.3 * std::sin(phase), spot.y + 0.3 * std::cos(phase), spot.z},
             .displayColor = {0.2f, 0.4f, 0.8f},
             .radius = Radius(setup.shape.balls)});
        break;
    }
    }
}

void Run(benchmark::State& state, const Mode& mode, Motion motion, const Shape& shape,
         bool hardware) {
    const device::Traversal traversal =
        hardware ? device::Traversal::Hardware : device::Traversal::Software;
    Setup* setup = nullptr;
    try {
        setup = &bench::Cached<Setup>(
            fmt::format("{}/{}/{}/{}/{}/{}/{}/{}", mode.name, Name(motion), shape.worlds,
                        shape.cameras, shape.cells, shape.balls, shape.outputs, hardware),
            [&] { return Prepare(mode, motion, shape, traversal); });
    } catch (const UnsupportedError& error) {
        state.SkipWithMessage(error.what());
        return;
    }
    profiler::SetHistory(kHistory);
    profiler::Clear();
    profiler::BeginFrame();
    std::deque<RenderOutput> flight;
    for (auto _ : state) {
        Move(*setup, motion);
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
    double bvh = 0;
    double geometry = 0;
    double other = 0;
    for (const profiler::ReportTrack& track : profiler::Summarize(profiler::Frames()).tracks) {
        for (const profiler::ReportNode& zone : track.zones) {
            if (!track.gpu) {
                step += zone.name == "Render/Step" ? zone.ms : 0;
            } else if (zone.name.starts_with("Bvh/")) {
                bvh += zone.ms;
            } else if (zone.name.starts_with("Geometry/")) {
                geometry += zone.ms;
            } else if (zone.name.starts_with("Render/")) {
                render += zone.ms;
            } else {
                other += zone.ms;
            }
        }
    }
    profiler::SetHistory(2);
    const double steps = double(state.iterations());
    state.counters["views"] = double(shape.worlds) * shape.cameras;
    state.counters["worlds/s"] =
        benchmark::Counter(steps * double(shape.worlds), benchmark::Counter::kIsRate);
    if (mode.render) {
        state.counters["px/s"] =
            benchmark::Counter(steps * double(setup->pixels), benchmark::Counter::kIsRate);
    }
    state.counters["cpu ms/step"] = step;
    state.counters["gpu render ms"] = render;
    state.counters["gpu bvh ms"] = bvh;
    state.counters["gpu geometry ms"] = geometry;
    state.counters["gpu other ms"] = other;
}

void Scaling(benchmark::State& state, const Mode& mode, Motion motion) {
    Run(state, mode, motion,
        {.worlds = uint32_t(state.range(0)), .cameras = uint32_t(state.range(1))},
        state.range(2) != 0);
}

void Complexity(benchmark::State& state, Motion motion) {
    Run(state, kShadowed, motion,
        {.worlds = uint32_t(state.range(2)),
         .cells = uint32_t(state.range(0)),
         .balls = uint32_t(state.range(1))},
        state.range(3) != 0);
}

void Outputs(benchmark::State& state) {
    Run(state, kShadowed, Motion::Static,
        {.worlds = uint32_t(state.range(1)), .outputs = uint32_t(state.range(0))},
        state.range(2) != 0);
}

const bool kCore =
    bench::Core(
        "^Render/Scaling/DisplayColor/shadows:1/motion:(static|deforming)/worlds:(16|256|1024)/"
        "cameras:1/") &&
    bench::Core("^Render/Outputs/DisplayColor/shadows:1/outputs:(1|8)/worlds:1024/");

const bool kRegistered = [] {
    const std::pair<const Mode*, Motion> scaled[] = {
        {&kDepth, Motion::Static},       {&kColor, Motion::Static},
        {&kShadowed, Motion::Static},    {&kUpdate, Motion::Rigid},
        {&kShadowed, Motion::Rigid},     {&kUpdate, Motion::Deforming},
        {&kShadowed, Motion::Deforming}, {&kUpdate, Motion::Remesh},
        {&kShadowed, Motion::Remesh},    {&kUpdate, Motion::Respawn},
        {&kShadowed, Motion::Respawn},
    };
    const int64_t layouts[][2] = {{1, 1},    {16, 1}, {256, 1}, {1024, 1},
                                  {4096, 1}, {1, 64}, {16, 16}, {256, 4}};
    for (const auto& [mode, motion] : scaled) {
        benchmark::internal::Benchmark* registered = benchmark::RegisterBenchmark(
            fmt::format("Render/Scaling/{}/motion:{}", mode->name, Name(motion)), Scaling, *mode,
            motion);
        registered->ArgNames({"worlds", "cameras", "hardware"});
        for (const auto& [worlds, cameras] : layouts) {
            for (const int64_t hardware : {0, 1}) {
                registered->Args({worlds, cameras, hardware});
            }
        }
        registered->UseRealTime()->Unit(benchmark::kMillisecond);
    }
    for (const Motion motion : {Motion::Static, Motion::Deforming}) {
        benchmark::RegisterBenchmark(
            fmt::format("Render/Complexity/{}/motion:{}", kShadowed.name, Name(motion)), Complexity,
            motion)
            ->ArgNames({"cells", "balls", "worlds", "hardware"})
            ->ArgsProduct({{32, 128, 256}, {8, 64, 512}, {16, 256}, {0, 1}})
            ->UseRealTime()
            ->Unit(benchmark::kMillisecond);
    }
    benchmark::RegisterBenchmark(fmt::format("Render/Outputs/{}", kShadowed.name), Outputs)
        ->ArgNames({"outputs", "worlds", "hardware"})
        ->ArgsProduct({{1, 2, 4, 8}, {16, 1024}, {0, 1}})
        ->UseRealTime()
        ->Unit(benchmark::kMillisecond);
    return true;
}();

}
}
