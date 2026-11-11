// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "bench.h"
#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/util/error.h"
#include "ernie/util/profiler.h"

namespace ernie {
namespace {

constexpr uint32_t kProps = 16;
constexpr uint32_t kCells = 32;
constexpr uint32_t kFrames = 2;
constexpr uint32_t kWarmup = 3;
constexpr size_t kHistory = 32;
constexpr double kMinTime = 0.2;
constexpr int64_t kEditedPlans = 20;

std::vector<Float3> Wave(double phase) {
    std::vector<Float3> points;
    for (uint32_t y = 0; y <= kCells; ++y) {
        for (uint32_t x = 0; x <= kCells; ++x) {
            const float u = float(x) / kCells;
            const float v = float(y) / kCells;
            points.push_back({u * 4 - 2, v * 4 - 2, 0.2f * std::sin(u * 9 + float(phase))});
        }
    }
    return points;
}

Mesh::Desc Sheet() {
    Mesh::Desc desc;
    desc.points = Wave(0);
    const auto side = int32_t(kCells + 1);
    for (uint32_t y = 0; y < kCells; ++y) {
        for (uint32_t x = 0; x < kCells; ++x) {
            const int32_t a = int32_t(y) * side + int32_t(x);
            desc.faceVertexIndices.insert(desc.faceVertexIndices.end(),
                                          {a, a + 1, a + side + 1, a + side});
            desc.faceVertexCounts.push_back(4);
        }
    }
    return desc;
}

struct Setup {
    Ref<Scene> scene;
    uint32_t worlds = 1;
    Attribute<Double3> translate;
    Attribute<Double3> tenth;
    Attribute<Float3> colors;
    ArrayAttribute<Float3> points;
    Ref<device::Buffer> poseFrames;
    Ref<device::Buffer> pointFrames;
    uint64_t poseBytes = 0;
    uint64_t pointBytes = 0;
    std::vector<std::string> spawned;
    uint64_t step = 0;
};

Setup Prepare(uint32_t worlds, bool dome, device::Traversal traversal = device::Traversal::Auto) {
    const Ref<Scene> scene = Scene::Create(bench::BenchDevice(traversal));
    scene->Define<Scope>("/Looks");
    const Prim material = scene->Define<Material>("/Looks/Paint");
    const Prim surface = scene->Define<Shader>("/Looks/Paint/Surface", {.id = "UsdPreviewSurface"});
    surface.Set(UsdPreviewSurface::kDiffuseColor, Float3{0.8f, 0.2f, 0.1f});
    material.Connect("surface", surface, "surface");
    scene->Define<World>("/env_0");
    std::vector<std::string> props;
    for (uint32_t i = 0; i < kProps; ++i) {
        const Prim prop = scene->Define<Sphere>(
            fmt::format("/env_0/Prop_{}", i),
            {.materialBinding = {"/Looks/Paint"}, .translate = {double(i % 4), double(i / 4), 0}});
        scene->Apply<SemanticsLabelsAPI>(prop, "class", {.semanticsLabels = {"prop"}});
    }
    scene->Define<Mesh>("/env_0/Sheet", Sheet());
    scene->Define<Camera>("/env_0/Cam", {.translate = {0, -6, 3}});
    scene->Define<SphereLight>("/env_0/Lamp", {.translate = {0, 0, 3}});
    std::vector<std::string> targets;
    for (uint32_t w = 1; w < worlds; ++w) {
        targets.push_back(fmt::format("/env_{}", w));
    }
    if (!targets.empty()) {
        scene->Clone(*scene->Find("/env_0"), targets);
    }
    scene->Define<DistantLight>("/Sun");
    if (dome) {
        scene->Define<DomeLight>("/Sky");
    }
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Color", {.sourceName = "LdrColor"});
    scene->Define<RenderProduct>(
        "/Render/View",
        {.camera = {"/env_0/Cam"}, .resolution = {64, 64}, .orderedVars = {"/Render/Color"}});

    std::vector<std::string> moved;
    std::vector<std::string> sheets;
    for (uint32_t w = 0; w < worlds; ++w) {
        sheets.push_back(fmt::format("/env_{}/Sheet", w));
        for (uint32_t i = 0; i < kProps; ++i) {
            moved.push_back(fmt::format("/env_{}/Prop_{}", w, i));
        }
    }
    std::vector<std::string> some;
    for (uint32_t w = 0; w < worlds; w += 10) {
        for (uint32_t i = 0; i < kProps; ++i) {
            some.push_back(fmt::format("/env_{}/Prop_{}", w, i));
        }
    }
    Setup setup{scene,
                worlds,
                scene->Select(moved).Get<Double3>("translate"),
                scene->Select(some).Get<Double3>("translate"),
                scene->Select(moved).Get<Float3>("displayColor"),
                scene->Select(sheets).GetArray<Float3>("points")};
    for (uint32_t w = 0; w < worlds; ++w) {
        setup.spawned.push_back(fmt::format("/env_{}/Spawned", w));
    }
    setup.points.Set(0, Wave(0.5));
    scene->Publish({.wait = true}).Wait();

    std::vector<Double3> poses;
    std::vector<Float3> waves;
    for (uint32_t f = 0; f < kFrames; ++f) {
        for (uint32_t p = 0; p < worlds * kProps; ++p) {
            poses.push_back({double(p % 4) + 0.1 * f, double(p / 4 % 4), 0.2 * f});
        }
        const std::vector<Float3> wave = Wave(0.3 * (f + 1));
        for (uint32_t w = 0; w < worlds; ++w) {
            waves.insert(waves.end(), wave.begin(), wave.end());
        }
    }
    device::Device& device = scene->GetDevice();
    setup.poseFrames =
        device.CreateBuffer({.label = "pose frames"}, std::span<const Double3>(poses));
    setup.pointFrames =
        device.CreateBuffer({.label = "point frames"}, std::span<const Float3>(waves));
    setup.poseBytes = poses.size() * sizeof(Double3) / kFrames;
    setup.pointBytes = waves.size() * sizeof(Float3) / kFrames;
    for (uint32_t i = 0; i < kWarmup; ++i) {
        scene->Publish({.dt = 1.0 / 60}).Wait();
    }
    return setup;
}

void Upload(Setup& setup, UntypedAttribute& attribute, const Ref<device::Buffer>& frames,
            uint64_t bytes) {
    Mapping mapping = attribute.Map(MapAccess::Write);
    const device::Tensor& values = mapping.Values();
    device::Queue& queue = setup.scene->GetDevice().GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    encoder.Copy(frames->Slice(setup.step % kFrames * bytes, bytes),
                 values.GetBuffer().Slice(values.Offset(), bytes));
    queue.Submit(encoder.Finish());
    mapping.Close();
}

struct Edit {
    const char* name;
    std::function<void(Setup&)> apply;
};

const Edit kEdits[] = {
    {"idle", [](Setup&) {}},
    {"clock", [](Setup&) {}},
    {"poses-host",
     [](Setup& s) {
         std::vector<Double3> poses(size_t(s.worlds) * kProps);
         for (size_t p = 0; p < poses.size(); ++p) {
             poses[p] = {double(p % 4) + 0.01 * double(s.step % 7), double(p / 4 % 4), 0};
         }
         s.translate.Set(poses);
     }},
    {"poses-device", [](Setup& s) { Upload(s, s.translate, s.poseFrames, s.poseBytes); }},
    {"pose-one",
     [](Setup& s) {
         s.scene->Find("/env_0/Prop_0")
             ->Set(Xformable::kTranslate, Double3{0.01 * double(s.step % 7), 0, 0});
     }},
    {"poses-tenth",
     [](Setup& s) {
         std::vector<Double3> poses(size_t(s.tenth.Size()));
         for (size_t p = 0; p < poses.size(); ++p) {
             poses[p] = {double(p % 4) + 0.01 * double(s.step % 7), double(p / 4 % 4), 0};
         }
         s.tenth.Set(poses);
     }},
    {"points-device", [](Setup& s) { Upload(s, s.points, s.pointFrames, s.pointBytes); }},
    {"colors",
     [](Setup& s) {
         const float shade = float(s.step % 7) / 7;
         s.colors.Set(std::vector<Float3>(size_t(s.worlds) * kProps, Float3{shade, 0.5f, 0.5f}));
     }},
    {"camera",
     [](Setup& s) {
         s.scene->Find("/env_0/Cam")
             ->Set(Xformable::kTranslate, Double3{0.01 * double(s.step % 7), -6, 3});
     }},
    {"material",
     [](Setup& s) {
         s.scene->Find("/Looks/Paint/Surface")
             ->Set(UsdPreviewSurface::kRoughness, 0.1f + 0.1f * float(s.step % 7));
     }},
    {"lamp",
     [](Setup& s) {
         s.scene->Find("/env_0/Lamp")->Set(LightAPI::kIntensity, 1.0f + float(s.step % 7));
     }},
    {"product",
     [](Setup& s) {
         const uint32_t side = 64 + 8 * uint32_t(s.step % 2);
         s.scene->Find("/Render/View")->Set(RenderProduct::kResolution, Uint2{side, side});
     }},
    {"labels",
     [](Setup& s) {
         s.scene->Find("/env_0/Prop_0")
             ->Set(SemanticsLabelsAPI::kSemanticsLabels, "class",
                   {s.step % 2 == 0 ? "rock" : "prop"});
     }},
    {"spawn",
     [](Setup& s) {
         if (!s.scene->Find(s.spawned.front())) {
             s.scene->Define<Cube>(Span<const std::string>(s.spawned), {.size = 0.5});
         } else {
             for (const std::string& path : s.spawned) {
                 s.scene->Remove(*s.scene->Find(path));
             }
         }
     }},
};

void Report(benchmark::State& state) {
    std::map<std::string, double> cpu;
    std::map<std::string, double> gpu;
    double total = 0;
    const std::function<void(const profiler::ReportNode&, bool)> collect =
        [&](const profiler::ReportNode& zone, bool device) {
            const std::string group = zone.name.substr(0, zone.name.find('/'));
            if (device) {
                gpu[group] += zone.ms;
                return;
            }
            if (zone.name.ends_with("/Run")) {
                cpu[group] += zone.ms;
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
    state.counters["cpu ms"] = total;
    if (std::getenv("ERNIE_BENCH_ZONES") != nullptr) {
        std::printf("%s\n%s\n", state.name().c_str(),
                    profiler::ReportText(profiler::Summarize(profiler::Frames())).c_str());
    }
    for (const auto& [group, ms] : cpu) {
        if (ms >= 0.005) {
            state.counters["cpu " + group] = ms;
        }
    }
    for (const auto& [group, ms] : gpu) {
        if (ms >= 0.005) {
            state.counters["gpu " + group] = ms;
        }
    }
}

void Publish(benchmark::State& state, const Edit& edit) {
    const auto worlds = uint32_t(state.range(0));
    const bool dome = state.range(1) != 0;
    const device::Traversal traversal =
        state.range(2) != 0 ? device::Traversal::Hardware : device::Traversal::Software;
    Setup* found = nullptr;
    try {
        found = &bench::Cached<Setup>(fmt::format("publish/{}/{}/{}", worlds, dome, int(traversal)),
                                      [&] { return Prepare(worlds, dome, traversal); });
    } catch (const UnsupportedError& error) {
        state.SkipWithMessage(error.what());
        return;
    }
    Setup& setup = *found;
    const bool clocked = std::string_view(edit.name) == "clock";
    profiler::SetHistory(kHistory);
    profiler::Clear();
    profiler::BeginFrame();
    for (auto _ : state) {
        ++setup.step;
        edit.apply(setup);
        setup.scene->Publish({.dt = clocked ? std::optional<double>(1.0 / 60) : std::nullopt})
            .Wait();
        profiler::BeginFrame();
    }
    profiler::BeginFrame();
    Report(state);
    profiler::SetHistory(2);
    state.counters["worlds/s"] =
        benchmark::Counter(double(state.iterations()) * worlds, benchmark::Counter::kIsRate);
}

void Plan(benchmark::State& state) {
    const auto products = uint32_t(state.range(0));
    const bool edited = state.range(1) != 0;
    Setup& setup = bench::Cached<Setup>(fmt::format("plan/{}", products), [&] {
        Setup made = Prepare(products, false);
        for (uint32_t p = 1; p < products; ++p) {
            made.scene->Define<RenderProduct>(fmt::format("/Render/View_{}", p),
                                              {.camera = {fmt::format("/env_{}/Cam", p)},
                                               .resolution = {64, 64},
                                               .orderedVars = {"/Render/Color"}});
        }
        made.scene->Publish({.wait = true}).Wait();
        return made;
    });
    std::vector<std::string> paths = {"/Render/View"};
    for (uint32_t p = 1; p < products; ++p) {
        paths.push_back(fmt::format("/Render/View_{}", p));
    }
    ProductPlans& plans = SceneImpl::Get(*setup.scene).GetPlans();
    for (auto _ : state) {
        if (edited) {
            state.PauseTiming();
            const uint32_t side = 64 + 8 * uint32_t(++setup.step % 2);
            setup.scene->Find("/Render/View")->Set(RenderProduct::kResolution, Uint2{side, side});
            setup.scene->Publish().Wait();
            state.ResumeTiming();
        }
        benchmark::DoNotOptimize(plans.Plan(paths));
    }
    state.counters["products/s"] =
        benchmark::Counter(double(state.iterations()) * products, benchmark::Counter::kIsRate);
}

void Schedule(benchmark::State& state) {
    const auto products = uint32_t(state.range(0));
    Setup& setup = bench::Cached<Setup>(fmt::format("plan/{}", products), [&] {
        Setup made = Prepare(products, false);
        for (uint32_t p = 1; p < products; ++p) {
            made.scene->Define<RenderProduct>(fmt::format("/Render/View_{}", p),
                                              {.camera = {fmt::format("/env_{}/Cam", p)},
                                               .resolution = {64, 64},
                                               .orderedVars = {"/Render/Color"}});
        }
        made.scene->Publish({.wait = true}).Wait();
        return made;
    });
    CaptureSchedule& captures = SceneImpl::Get(*setup.scene).GetCaptures();
    for (auto _ : state) {
        captures.Schedule();
    }
    state.counters["products/s"] =
        benchmark::Counter(double(state.iterations()) * products, benchmark::Counter::kIsRate);
}

const bool kCore =
    bench::Core("^Scene/Publish/(idle|poses-device|points-device|camera|spawn)/worlds:(1|1024)/"
                "dome:0/hardware:1") &&
    bench::Core("^Scene/Plan/products:64/");

const bool kRegistered = [] {
    for (const int64_t hardware : {1, 0}) {
        for (const int64_t worlds : {1, 64, 1024, 4096}) {
            for (const int64_t dome : {0, 1}) {
                for (const Edit& edit : kEdits) {
                    const std::string_view name = edit.name;
                    if (hardware == 0 &&
                        (dome != 0 || (name != "poses-device" && name != "points-device" &&
                                       name != "pose-one"))) {
                        continue;
                    }
                    benchmark::RegisterBenchmark(fmt::format("Scene/Publish/{}", edit.name),
                                                 Publish, edit)
                        ->ArgNames({"worlds", "dome", "hardware"})
                        ->Args({worlds, dome, hardware})
                        ->MinTime(kMinTime)
                        ->UseRealTime()
                        ->Unit(benchmark::kMillisecond);
                }
            }
        }
    }
    for (const int64_t products : {1, 64, 256}) {
        benchmark::RegisterBenchmark("Scene/Plan", Plan)
            ->ArgNames({"products", "edited"})
            ->Args({products, 0})
            ->MinTime(kMinTime)
            ->Unit(benchmark::kMicrosecond);
        benchmark::RegisterBenchmark("Scene/Plan", Plan)
            ->ArgNames({"products", "edited"})
            ->Args({products, 1})
            ->Iterations(kEditedPlans)
            ->Unit(benchmark::kMicrosecond);
        benchmark::RegisterBenchmark("Scene/Schedule", Schedule)
            ->ArgNames({"products"})
            ->Args({products})
            ->MinTime(kMinTime)
            ->Unit(benchmark::kMicrosecond);
    }
    return true;
}();

}
}
