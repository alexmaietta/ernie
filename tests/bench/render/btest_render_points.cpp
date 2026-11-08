// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
#include <deque>
#include <format>
#include <optional>
#include <random>
#include <string>
#include <vector>

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
constexpr uint32_t kWarmup = 3;
constexpr uint32_t kFrames = 2;
constexpr size_t kInflight = 2;
constexpr size_t kHistory = 32;
constexpr float kExtent = 10;
constexpr uint64_t kMaxPoints = 16u << 20;
const std::vector<std::string> kProducts = {"/Render/Product"};

enum class Layout { Box, Splash, Bed };
enum class Motion { Stationary, Moving, Uploaded, Recounted, Resized };
enum class Family { Points, Spheres, Instancer };

const char* Name(Layout layout) {
    switch (layout) {
    case Layout::Box:
        return "box";
    case Layout::Splash:
        return "splash";
    case Layout::Bed:
        return "bed";
    }
    return "";
}

const char* Name(Motion motion) {
    switch (motion) {
    case Motion::Stationary:
        return "stationary";
    case Motion::Moving:
        return "moving";
    case Motion::Uploaded:
        return "uploaded";
    case Motion::Recounted:
        return "recounted";
    case Motion::Resized:
        return "resized";
    }
    return "";
}

const char* Name(Family family) {
    switch (family) {
    case Family::Points:
        return "points";
    case Family::Spheres:
        return "spheres";
    case Family::Instancer:
        return "instancer";
    }
    return "";
}

struct Shape {
    Family family = Family::Points;
    Layout layout = Layout::Box;
    Motion motion = Motion::Stationary;
    uint32_t worlds = 1;
    uint32_t points = 1000;
};

struct Cloud {
    std::vector<Float3> points;
    std::vector<float> widths;
};

Cloud Scatter(Layout layout, uint32_t count, uint32_t seed) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> unit(0, 1);
    std::normal_distribution<float> normal(0, 1);
    const float half = kExtent / 2;
    Cloud cloud;
    cloud.points.reserve(count);
    float width = 0.5f * kExtent / std::cbrt(float(std::max(count, 1u)));
    if (layout == Layout::Box) {
        for (uint32_t i = 0; i < count; ++i) {
            cloud.points.push_back({(unit(random) - 0.5f) * kExtent,
                                    (unit(random) - 0.5f) * kExtent, unit(random) * kExtent});
        }
    } else if (layout == Layout::Splash) {
        constexpr uint32_t kBlobs = 8;
        std::vector<Float3> blobs;
        for (uint32_t b = 0; b < kBlobs; ++b) {
            blobs.push_back({(unit(random) - 0.5f) * half, (unit(random) - 0.5f) * half,
                             half * (0.5f + unit(random))});
        }
        const float spread = kExtent * 0.04f;
        for (uint32_t i = 0; i < count; ++i) {
            const Float3& blob = blobs[i % kBlobs];
            cloud.points.push_back({blob.x + normal(random) * spread,
                                    blob.y + normal(random) * spread,
                                    blob.z + normal(random) * spread});
        }
        width *= 0.25f;
    } else {
        for (uint32_t i = 0; i < count; ++i) {
            cloud.points.push_back({(unit(random) - 0.5f) * kExtent,
                                    (unit(random) - 0.5f) * kExtent,
                                    unit(random) * kExtent * 0.02f});
        }
        width = 1.2f * kExtent / std::sqrt(float(std::max(count, 1u)));
    }
    std::uniform_real_distribution<float> vary(0.5f, 1.5f);
    for (uint32_t i = 0; i < count; ++i) {
        cloud.widths.push_back(width * vary(random));
    }
    return cloud;
}

std::vector<Float3> Shifted(const std::vector<Float3>& points, uint32_t frame) {
    std::vector<Float3> shifted = points;
    const float phase = 0.3f * float(frame + 1);
    for (size_t i = 0; i < shifted.size(); ++i) {
        const float turn = phase + float(i % 64) * 0.1f;
        shifted[i].x += 0.02f * kExtent * std::sin(turn);
        shifted[i].z += 0.02f * kExtent * std::cos(turn);
    }
    return shifted;
}

template <class T> void Append(std::vector<std::byte>& bytes, const std::vector<T>& items) {
    const auto* data = reinterpret_cast<const std::byte*>(items.data());
    bytes.insert(bytes.end(), data, data + items.size() * sizeof(T));
}

struct Setup {
    Ref<Scene> scene;
    Renderer renderer;
    Shape shape;
    std::optional<View> clouds;
    std::optional<View> eyes;
    std::optional<UntypedAttribute> moved;
    Ref<device::Buffer> frames;
    uint64_t frameBytes = 0;
    Cloud cloud;
    uint64_t pixels = 0;
    double bytesPerPoint = 0;
    uint64_t step = 0;
};

std::string Path(const Shape& shape, uint32_t world) {
    return shape.worlds == 1 ? std::string("/Cloud") : std::format("/env_{}/Cloud", world);
}

void DefineCloud(Scene& scene, const Shape& shape, const std::string& path, const Cloud& cloud) {
    if (shape.family == Family::Points) {
        scene.Define<Points>(
            path,
            {.displayColor = {0.7f, 0.6f, 0.4f}, .points = cloud.points, .widths = cloud.widths});
        return;
    }
    if (shape.family == Family::Instancer) {
        PointInstancer::Desc desc;
        desc.prototypes = {path + "/Protos/Grain"};
        desc.positions = cloud.points;
        desc.scales.reserve(cloud.widths.size());
        for (const float width : cloud.widths) {
            desc.scales.push_back({width, width, width});
        }
        desc.protoIndices.assign(cloud.points.size(), 0);
        scene.Define<PointInstancer>(path, desc);
        scene.Define<Scope>(path + "/Protos");
        scene.Define<Sphere>(path + "/Protos/Grain",
                             {.displayColor = {0.7f, 0.6f, 0.4f}, .radius = 0.5});
        return;
    }
    scene.Define<Xform>(path);
    std::vector<std::string> grains;
    grains.reserve(cloud.points.size());
    for (size_t i = 0; i < cloud.points.size(); ++i) {
        grains.push_back(std::format("{}/Grain_{}", path, i));
    }
    scene.Define<Sphere>(Span<const std::string>(grains), {.displayColor = {0.7f, 0.6f, 0.4f}});
    std::vector<Double3> translates;
    std::vector<float> radii;
    for (size_t i = 0; i < cloud.points.size(); ++i) {
        translates.push_back({cloud.points[i].x, cloud.points[i].y, cloud.points[i].z});
        radii.push_back(cloud.widths[i] / 2);
    }
    const View view = scene.Select(grains);
    view.Get<Double3>("translate").Set(translates);
    view.Get<float>("radius").Set(radii);
}

UntypedAttribute Moved(Scene& scene, const Shape& shape, const View& clouds) {
    switch (shape.family) {
    case Family::Points:
        if (shape.motion == Motion::Resized) {
            return clouds.GetArray<float>("widths");
        }
        return clouds.GetArray<Float3>("points");
    case Family::Instancer:
        return clouds.GetArray<Float3>("positions");
    case Family::Spheres:
        break;
    }
    std::vector<std::string> grains;
    for (uint32_t i = 0; i < shape.points; ++i) {
        grains.push_back(std::format("{}/Grain_{}", Path(shape, 0), i));
    }
    return scene.Select(grains).Get<Double3>("translate");
}

Double3 Eye(uint64_t step) {
    return {0.02 * kExtent * double(step % 2), -1.6 * kExtent, 0.8 * kExtent};
}

constexpr Double3 kTarget = {0, 0, 0.3 * kExtent};

Setup Prepare(const Shape& shape, device::Traversal traversal) {
    device::Device& device = bench::BenchDevice(traversal);
    const uint64_t before = device.MemoryUsage().deviceBytes;
    const Ref<Scene> scene = Scene::Create(device);
    Setup setup{scene, Renderer(scene), shape};
    setup.cloud = Scatter(shape.layout, shape.points, 7);
    if (shape.worlds > 1) {
        scene->Define<World>("/env_0");
    }
    DefineCloud(*scene, shape, Path(shape, 0), setup.cloud);
    const std::string camera = shape.worlds == 1 ? "/Eye" : "/env_0/Eye";
    scene->Define<Camera>(camera, {.translate = Eye(0), .orient = LookAt(Eye(0), kTarget)});
    std::vector<std::string> clouds = {Path(shape, 0)};
    std::vector<std::string> views = {camera};
    if (shape.worlds > 1) {
        std::vector<std::string> targets;
        for (uint32_t w = 1; w < shape.worlds; ++w) {
            targets.push_back(std::format("/env_{}", w));
            clouds.push_back(Path(shape, w));
            views.push_back(std::format("/env_{}/Eye", w));
        }
        scene->Clone(*scene->Find("/env_0"), targets);
    }
    setup.clouds = scene->Select(clouds);
    setup.eyes = scene->Select(views);
    scene->Define<DistantLight>("/Sun", {.orient = FromAzimuthElevation(60, 50)});
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Color", {.sourceName = RenderVarSource::LdrColor});
    scene->Define<RenderProduct>("/Render/Product", {.mode = MinimalMode::DisplayColor,
                                                     .castShadows = true,
                                                     .camera = views,
                                                     .resolution = {kSide, kSide},
                                                     .orderedVars = {"/Render/Color"}});
    setup.pixels = uint64_t(shape.worlds) * kSide * kSide;
    if (shape.motion == Motion::Moving || shape.motion == Motion::Resized) {
        std::vector<std::byte> frames;
        for (uint32_t f = 0; f < kFrames; ++f) {
            for (uint32_t w = 0; w < shape.worlds; ++w) {
                if (shape.motion == Motion::Resized) {
                    std::vector<float> widths = setup.cloud.widths;
                    for (float& width : widths) {
                        width *= 0.8f + 0.4f * float(f);
                    }
                    Append(frames, widths);
                } else if (shape.family == Family::Spheres) {
                    std::vector<Double3> translates;
                    for (const Float3& point : Shifted(setup.cloud.points, f)) {
                        translates.push_back({point.x, point.y, point.z});
                    }
                    Append(frames, translates);
                } else {
                    Append(frames, Shifted(setup.cloud.points, f));
                }
            }
        }
        setup.frames =
            device.CreateBuffer({.label = "points frames"}, std::span<const std::byte>(frames));
        setup.frameBytes = frames.size() / kFrames;
        setup.moved = Moved(*scene, shape, *setup.clouds);
        Mapping mapping = setup.moved->Map(MapAccess::Write);
        const Span<const int64_t> extents = mapping.Values().Shape();
        const Span<const int64_t> strides = mapping.Values().Strides();
        int64_t expected = 1;
        for (size_t axis = extents.size(); axis-- > 0;) {
            if (strides[axis] != expected) {
                throw Error("the mapped values aren't contiguous");
            }
            expected *= extents[axis];
        }
        mapping.Close();
    }
    for (uint32_t i = 0; i < kWarmup; ++i) {
        setup.renderer.Step(kProducts, 1.0 / 60).Wait();
    }
    const uint64_t frameBytes = setup.frames ? setup.frames->Bytes() : 0;
    setup.bytesPerPoint = double(device.MemoryUsage().deviceBytes - before - frameBytes) /
                          (double(shape.worlds) * shape.points);
    return setup;
}

void Move(Setup& setup) {
    ++setup.step;
    setup.eyes->Get<Double3>("translate").Set(Eye(setup.step));
    switch (setup.shape.motion) {
    case Motion::Stationary:
        break;
    case Motion::Moving:
    case Motion::Resized: {
        Mapping mapping = setup.moved->Map(MapAccess::Write);
        const device::Tensor& values = mapping.Values();
        device::Queue& queue = setup.scene->GetDevice().GetQueue();
        device::CommandEncoder encoder = queue.CreateEncoder();
        encoder.Copy(setup.frames->Slice(setup.step % kFrames * setup.frameBytes, setup.frameBytes),
                     values.GetBuffer().Slice(values.Offset(), setup.frameBytes));
        queue.Submit(encoder.Finish());
        mapping.Close();
        break;
    }
    case Motion::Uploaded: {
        const std::vector<Float3> points = Shifted(setup.cloud.points, uint32_t(setup.step % 2));
        ArrayAttribute<Float3> attribute = setup.clouds->GetArray<Float3>("points");
        for (uint32_t w = 0; w < setup.shape.worlds; ++w) {
            attribute.Set(w, points);
        }
        break;
    }
    case Motion::Recounted: {
        const size_t count =
            setup.step % 2 == 0 ? setup.cloud.points.size() : setup.cloud.points.size() * 9 / 10;
        const std::vector<Float3> points(setup.cloud.points.begin(),
                                         setup.cloud.points.begin() + ptrdiff_t(count));
        const std::vector<float> widths(setup.cloud.widths.begin(),
                                        setup.cloud.widths.begin() + ptrdiff_t(count));
        ArrayAttribute<Float3> pointsAttribute = setup.clouds->GetArray<Float3>("points");
        ArrayAttribute<float> widthsAttribute = setup.clouds->GetArray<float>("widths");
        for (uint32_t w = 0; w < setup.shape.worlds; ++w) {
            pointsAttribute.Set(w, points);
            widthsAttribute.Set(w, widths);
        }
        break;
    }
    }
}

void Run(benchmark::State& state, const Shape& shape, bool hardware) {
    const device::Traversal traversal =
        hardware ? device::Traversal::Hardware : device::Traversal::Software;
    Setup* setup = nullptr;
    try {
        setup = &bench::Cached<Setup>(std::format("{}/{}/{}/{}/{}/{}", Name(shape.family),
                                                  Name(shape.layout), Name(shape.motion),
                                                  shape.worlds, shape.points, hardware),
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
        Move(*setup);
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
    double boxes = 0;
    double blas = 0;
    double tlas = 0;
    double render = 0;
    double other = 0;
    for (const profiler::ReportTrack& track : profiler::Summarize(profiler::Frames()).tracks) {
        for (const profiler::ReportNode& zone : track.zones) {
            if (!track.gpu) {
                step += zone.name == "Render/Step" ? zone.ms : 0;
            } else if (zone.name == "Points/Boxes") {
                boxes += zone.ms;
            } else if (zone.name == "Points/Blas") {
                blas += zone.ms;
            } else if (zone.name.starts_with("Bvh/") || zone.name.starts_with("Instances/")) {
                tlas += zone.ms;
            } else if (zone.name.starts_with("Render/")) {
                render += zone.ms;
            } else {
                other += zone.ms;
            }
        }
    }
    profiler::SetHistory(2);
    const double steps = double(state.iterations());
    const double points = double(shape.worlds) * shape.points;
    state.counters["points"] = points;
    state.counters["points/s"] = benchmark::Counter(steps * points, benchmark::Counter::kIsRate);
    state.counters["px/s"] =
        benchmark::Counter(steps * double(setup->pixels), benchmark::Counter::kIsRate);
    state.counters["B/point"] = setup->bytesPerPoint;
    state.counters["cpu ms/step"] = step;
    state.counters["gpu boxes ms"] = boxes;
    state.counters["gpu blas ms"] = blas;
    state.counters["gpu tlas ms"] = tlas;
    state.counters["gpu render ms"] = render;
    state.counters["gpu other ms"] = other;
}

void Single(benchmark::State& state, Family family, Layout layout, Motion motion) {
    Run(state,
        {.family = family, .layout = layout, .motion = motion, .points = uint32_t(state.range(0))},
        state.range(1) != 0);
}

void Batched(benchmark::State& state, Motion motion) {
    Run(state,
        {.motion = motion, .worlds = uint32_t(state.range(0)), .points = uint32_t(state.range(1))},
        state.range(2) != 0);
}

const bool kCore =
    bench::Core("^Render/Points/World/box/motion:(stationary|moving)/points:(100000|1000000)/") &&
    bench::Core("^Render/Points/Worlds/box/motion:(stationary|moving)/worlds:1024/points:10000/");

const bool kRegistered = [] {
    const Motion motions[] = {Motion::Stationary, Motion::Moving, Motion::Uploaded,
                              Motion::Recounted, Motion::Resized};
    const std::vector<int64_t> hardware = {0, 1};
    for (const Layout layout : {Layout::Box, Layout::Splash, Layout::Bed}) {
        for (const Motion motion : motions) {
            if (layout != Layout::Box && motion != Motion::Stationary && motion != Motion::Moving) {
                continue;
            }
            benchmark::RegisterBenchmark(
                std::format("Render/Points/World/{}/motion:{}", Name(layout), Name(motion)), Single,
                Family::Points, layout, motion)
                ->ArgsProduct({{1000, 10000, 100000, 1000000, 4000000, 16000000}, hardware})
                ->ArgNames({"points", "hardware"})
                ->Unit(benchmark::kMillisecond)
                ->UseRealTime();
        }
    }
    for (const Motion motion : motions) {
        benchmark::internal::Benchmark* registered = benchmark::RegisterBenchmark(
            std::format("Render/Points/Worlds/box/motion:{}", Name(motion)), Batched, motion);
        registered->ArgNames({"worlds", "points", "hardware"});
        for (const int64_t worlds : {16, 256, 1024, 4096}) {
            for (const int64_t points : {1000, 10000, 50000}) {
                if (uint64_t(worlds * points) > kMaxPoints) {
                    continue;
                }
                for (const int64_t traversal : hardware) {
                    registered->Args({worlds, points, traversal});
                }
            }
        }
        registered->Unit(benchmark::kMillisecond)->UseRealTime();
    }
    const std::pair<Family, std::vector<int64_t>> baselines[] = {
        {Family::Spheres, {1000, 10000, 100000}},
        {Family::Instancer, {1000, 10000, 100000, 1000000}},
    };
    for (const auto& [family, counts] : baselines) {
        for (const Motion motion : {Motion::Stationary, Motion::Moving}) {
            benchmark::RegisterBenchmark(
                std::format("Render/Points/Baseline/{}/motion:{}", Name(family), Name(motion)),
                Single, family, Layout::Box, motion)
                ->ArgsProduct({counts, hardware})
                ->ArgNames({"points", "hardware"})
                ->Unit(benchmark::kMillisecond)
                ->UseRealTime();
        }
    }
    return true;
}();

}
}
