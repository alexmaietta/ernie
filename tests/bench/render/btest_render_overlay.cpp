// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "bench.h"
#include "ernie/device/queue.h"
#include "ernie/render/overlay/overlay_impl.h"
#include "ernie/scene/rotation.h"
#include "ernie/scene/scene.h"
#include "ernie/util/profiler.h"

namespace ernie {
namespace {

enum class Source : int64_t {
    HostLines,
    DeviceLines,
    WorldLines,
    FlatWorldLines,
    Markers,
    Boxes,
    Spheres,
    Frames,
    Arrows,
    Capsules,
    Trails,
    Labels,
    Numbers,
    ScreenNumbers,
    DashedLines,
    HaloLabels,
    Triangles,
    Images,
    PrimBounds,
    Outline,
};

constexpr uint32_t kTrailSamples = 32;

struct Shape {
    uint32_t views;
    uint32_t width;
    uint32_t height;
};

const Shape kShapes[] = {{1, 1920, 1080}, {64, 640, 480}, {1024, 128, 128}};

struct Setup {
    Ref<Scene> scene;
    std::shared_ptr<void> released;
    Snapshot snapshot;
    Ref<device::Buffer> overlay;
    Ref<device::Buffer> depth;
    Ref<device::Buffer> primIds;
    std::vector<OverlayTarget> targets;
    std::vector<OverlayContents> overlays;
    std::unique_ptr<OverlayPass> pass;
};

std::vector<Float3> Endpoints(uint32_t count) {
    std::mt19937 random(7);
    std::uniform_real_distribution<float> across(-2, 2);
    std::uniform_real_distribution<float> along(-1, 3);
    std::uniform_real_distribution<float> up(-1.5f, 1.5f);
    std::uniform_real_distribution<float> step(-0.15f, 0.15f);
    std::vector<Float3> endpoints;
    for (uint32_t line = 0; line < count; ++line) {
        const Float3 a{across(random), along(random), up(random)};
        endpoints.push_back(a);
        endpoints.push_back({a.x + step(random), a.y + step(random), a.z + step(random)});
    }
    return endpoints;
}

Overlay Draw(device::Device& device, Source source, uint32_t count,
             const std::vector<uint32_t>& worlds, const std::vector<uint32_t>& prims) {
    const std::vector<Float3> endpoints = Endpoints(count);
    Overlay overlay;
    if (source == Source::HostLines || source == Source::DashedLines) {
        const float dash = source == Source::DashedLines ? 6.0f : 0.0f;
        overlay.Lines(endpoints, {.color = {1, 0.3f, 0, 1}, .width = 2, .dash = dash});
        return overlay;
    }
    if (source == Source::Labels || source == Source::HaloLabels) {
        const OverlayStyle style = source == Source::Labels
                                       ? OverlayStyle{.color = {1, 1, 0.4f, 1}, .size = 12}
                                       : OverlayStyle{.color = {1, 1, 0.4f, 1},
                                                      .size = 12,
                                                      .background = {0, 0, 0, 0},
                                                      .halo = {0, 0, 0, 0.8f}};
        for (uint32_t at = 0; at < count; ++at) {
            overlay.Label(endpoints[2 * at], fmt::format("body {}", at), style);
        }
        return overlay;
    }
    if (source == Source::Triangles) {
        for (uint32_t at = 0; at < count; ++at) {
            const Float3 a = endpoints[2 * at];
            const Float3 b = endpoints[2 * at + 1];
            overlay.Triangle(a, b, {a.x, a.y, a.z + 0.15f}, {.color = {0.2f, 0.6f, 1, 0.6f}});
        }
        return overlay;
    }
    if (source == Source::Images) {
        std::vector<uint32_t> texels(64 * 64);
        for (size_t at = 0; at < texels.size(); ++at) {
            texels[at] = 0xFF000000u | uint32_t(at * 2654435761u >> 8);
        }
        const device::Tensor pixels(device.CreateBuffer({.label = "bench image"}, texels), 0,
                                    device::DType::UInt8, {64, 64, 4});
        for (uint32_t at = 0; at < count; ++at) {
            overlay.Image(endpoints[2 * at], {0.4f, 0, 0}, {0, 0, -0.3f}, pixels);
        }
        return overlay;
    }
    if (source == Source::PrimBounds) {
        overlay.Prims(prims, OverlayPrimShape::Bounds, {.color = {1, 0.6f, 0.1f, 1}});
        return overlay;
    }
    if (source == Source::Outline) {
        overlay.Outline(std::vector<uint32_t>{1}, {.color = {1, 0.6f, 0.1f, 1}, .width = 2});
        return overlay;
    }
    if (source == Source::Markers) {
        std::vector<Float3> points;
        for (uint32_t at = 0; at < count; ++at) {
            points.push_back(endpoints[2 * at]);
        }
        overlay.Markers(points, {.color = {0, 0.8f, 1, 1}, .size = 6});
        return overlay;
    }
    std::vector<float> points;
    std::vector<float> ends;
    for (uint32_t at = 0; at < count; ++at) {
        const Float3 a = endpoints[2 * at];
        const Float3 b = endpoints[2 * at + 1];
        points.insert(points.end(), {a.x, a.y, a.z});
        ends.insert(ends.end(), {b.x, b.y, b.z});
    }
    const auto tensor = [&](const auto& values, device::DType type, std::vector<int64_t> shape) {
        return device::Tensor(device.CreateBuffer({.label = "bench overlay"}, values), 0, type,
                              std::move(shape));
    };
    const auto rows = int64_t(count);
    const std::vector<float> small(size_t(count), 0.05f);
    const OverlayStyle shaped = {.color = {0.3f, 1, 0.4f, 1}, .width = 2};
    switch (source) {
    case Source::Boxes:
        overlay.Add(OverlayBoxes{.points = tensor(points, device::DType::Float32, {rows, 3}),
                                 .halfExtents = tensor(std::vector<float>(points.size(), 0.05f),
                                                       device::DType::Float32, {rows, 3}),
                                 .style = shaped});
        return overlay;
    case Source::Spheres:
        overlay.Add(OverlaySpheres{.points = tensor(points, device::DType::Float32, {rows, 3}),
                                   .radii = tensor(small, device::DType::Float32, {rows}),
                                   .style = shaped});
        return overlay;
    case Source::Frames:
        overlay.Add(OverlayFrames{.points = tensor(points, device::DType::Float32, {rows, 3}),
                                  .length = 0.1f,
                                  .style = shaped});
        return overlay;
    case Source::Arrows:
        overlay.Add(OverlayArrows{.points = tensor(points, device::DType::Float32, {rows, 3}),
                                  .ends = tensor(ends, device::DType::Float32, {rows, 3}),
                                  .style = shaped});
        return overlay;
    case Source::Capsules:
        overlay.Add(OverlayCapsules{.points = tensor(points, device::DType::Float32, {rows, 3}),
                                    .radii = tensor(small, device::DType::Float32, {rows}),
                                    .halfHeights = tensor(small, device::DType::Float32, {rows}),
                                    .style = shaped});
        return overlay;
    case Source::Trails: {
        std::vector<float> samples;
        for (uint32_t at = 0; at < count; ++at) {
            const Float3 a = endpoints[2 * at];
            const Float3 b = endpoints[2 * at + 1];
            for (uint32_t sample = 0; sample < kTrailSamples; ++sample) {
                const float t = float(sample) / float(kTrailSamples - 1);
                samples.insert(samples.end(), {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
                                               a.z + (b.z - a.z) * t + 0.05f * std::sin(6 * t)});
            }
        }
        overlay.Add(OverlayTrails{
            .points = tensor(samples, device::DType::Float32, {rows, int64_t(kTrailSamples), 3}),
            .fade = 0.2f,
            .style = shaped});
        return overlay;
    }
    case Source::Numbers:
    case Source::ScreenNumbers: {
        std::vector<float> values;
        for (uint32_t at = 0; at < count; ++at) {
            values.push_back(float(at) * 0.37f - 50);
        }
        OverlayNumbers numbers;
        numbers.values = tensor(values, device::DType::Float32, {rows});
        numbers.format.prefix = "v=";
        numbers.format.suffix = " m/s";
        numbers.format.decimals = 2;
        numbers.style = {.color = {1, 1, 1, 1}, .size = 12};
        if (source == Source::Numbers) {
            numbers.points = tensor(points, device::DType::Float32, {rows, 3});
            if (worlds.size() > 1) {
                std::vector<uint32_t> owners(count);
                for (uint32_t at = 0; at < count; ++at) {
                    owners[at] = worlds[at % worlds.size()];
                }
                numbers.worlds = tensor(owners, device::DType::UInt32, {rows});
            }
        } else {
            numbers.placement = ScreenPlacement{.at = {0, 0}, .offset = {8, 8}};
        }
        overlay.Add(std::move(numbers));
        return overlay;
    }
    default:
        break;
    }
    if (source == Source::WorldLines) {
        const auto views = int64_t(worlds.size());
        const int64_t perWorld = count / views;
        std::vector<float> slices(points.size());
        std::vector<float> sliceEnds(ends.size());
        for (int64_t view = 0; view < views; ++view) {
            const size_t from = size_t(view * perWorld * 3);
            const size_t to = size_t(worlds[size_t(view)]) * perWorld * 3;
            std::copy_n(points.begin() + from, perWorld * 3, slices.begin() + to);
            std::copy_n(ends.begin() + from, perWorld * 3, sliceEnds.begin() + to);
        }
        overlay.Add(
            OverlayLines{.points = tensor(slices, device::DType::Float32, {views, perWorld, 3}),
                         .ends = tensor(sliceEnds, device::DType::Float32, {views, perWorld, 3}),
                         .style = {.color = {1, 0.3f, 0, 1}, .width = 2}});
        return overlay;
    }
    if (source == Source::FlatWorldLines) {
        const auto views = uint32_t(worlds.size());
        std::vector<uint32_t> owners(count);
        for (uint32_t line = 0; line < count; ++line) {
            owners[line] = worlds[line % views];
        }
        overlay.Add(
            OverlayLines{.points = tensor(points, device::DType::Float32, {int64_t(count), 3}),
                         .ends = tensor(ends, device::DType::Float32, {int64_t(count), 3}),
                         .worlds = tensor(owners, device::DType::UInt32, {int64_t(count)}),
                         .style = {.color = {1, 0.3f, 0, 1}, .width = 2}});
        return overlay;
    }
    overlay.Add(OverlayLines{
        .points = tensor(points, device::DType::Float32, {int64_t(count), 3}),
        .ends = tensor(ends, device::DType::Float32, {int64_t(count), 3}),
        .itemCount = tensor(std::vector<uint32_t>{count / 2}, device::DType::UInt32, {1}),
        .style = {.color = {1, 0.3f, 0, 1}, .width = 2}});
    return overlay;
}

Setup Prepare(Source source, uint32_t count, const Shape& shape) {
    device::Device& device = bench::BenchDevice();
    Setup setup;
    setup.scene = Scene::Create(device);
    std::vector<uint32_t> cameras;
    std::vector<uint32_t> worlds;
    std::vector<uint32_t> prims;
    if (source == Source::PrimBounds) {
        const std::vector<Float3> centers = Endpoints(count);
        for (uint32_t at = 0; at < count; ++at) {
            const std::string path = fmt::format("/Cube_{}", at);
            const Float3 center = centers[2 * at];
            setup.scene->Define<Cube>(path,
                                      {.translate = {center.x, center.y, center.z}, .size = 0.1});
            prims.push_back(setup.scene->Find(path)->Index());
        }
    }
    for (uint32_t view = 0; view < shape.views; ++view) {
        std::string path = "/Cam";
        if (shape.views > 1) {
            const std::string world = fmt::format("/env_{}", view);
            setup.scene->Define<World>(world, {});
            path = world + "/Cam";
        }
        setup.scene->Define<Camera>(path,
                                    {.translate = {0, -5, 0}, .orient = LookAlong({0, 1, 0})});
        cameras.push_back(setup.scene->Find(path)->Index());
        worlds.push_back(setup.scene->Find(path)->World());
    }
    setup.scene->Publish({.wait = true});
    setup.snapshot = setup.scene->Acquire();
    setup.released =
        std::shared_ptr<void>(nullptr, [scene = setup.scene, snapshot = setup.snapshot](void*) {
            scene->Release(snapshot);
        });

    const uint64_t pixels = uint64_t(shape.views) * shape.width * shape.height;
    setup.overlay = device.CreateBuffer({.bytes = pixels * 4, .label = "bench overlay image"});
    setup.depth =
        device.CreateBuffer({.label = "bench depth"},
                            std::vector<float>(pixels, std::numeric_limits<float>::infinity()));
    if (source == Source::Outline) {
        std::vector<uint32_t> ids(pixels);
        for (uint64_t at = 0; at < pixels; ++at) {
            const uint64_t x = at % shape.width;
            const uint64_t y = at / shape.width % shape.height;
            ids[at] = (x / 96 + y / 96) % 2;
        }
        setup.primIds = device.CreateBuffer({.label = "bench prim ids"}, ids);
    }
    for (uint32_t view = 0; view < shape.views; ++view) {
        setup.targets.push_back({.camera = cameras[view],
                                 .layer = view,
                                 .width = shape.width,
                                 .height = shape.height,
                                 .missDistance = std::numeric_limits<float>::infinity(),
                                 .overlay = setup.overlay->Address(),
                                 .depth = setup.depth->Address(),
                                 .primIds = setup.primIds ? setup.primIds->Address() : 0});
    }
    setup.overlays.push_back(OverlayPass::Share(Draw(device, source, count, worlds, prims)));
    setup.pass = std::make_unique<OverlayPass>(device);
    for (int warm = 0; warm < 3; ++warm) {
        device::CommandEncoder encoder = device.GetQueue().CreateEncoder();
        setup.pass->Record(encoder, setup.snapshot, setup.scene->WorldCount(), setup.targets,
                           setup.overlays);
        device.GetQueue().Submit(encoder.Finish()).Wait();
    }
    return setup;
}

void DrawOverlays(benchmark::State& state, Source source) {
    const auto count = uint32_t(state.range(0));
    const Shape& shape = kShapes[state.range(1)];
    Setup& setup =
        bench::Cached<Setup>(fmt::format("{}/{}/{}", int64_t(source), count, state.range(1)),
                             [&] { return Prepare(source, count, shape); });
    bench::TimeGpu(state, [&](device::CommandEncoder& encoder) {
        setup.pass->Record(encoder, setup.snapshot, setup.scene->WorldCount(), setup.targets,
                           setup.overlays);
    });
    device::Queue& queue = bench::BenchDevice().GetQueue();
    profiler::BeginFrame();
    device::CommandEncoder encoder = queue.CreateEncoder();
    setup.pass->Record(encoder, setup.snapshot, setup.scene->WorldCount(), setup.targets,
                       setup.overlays);
    queue.Submit(encoder.Finish()).Wait();
    profiler::BeginFrame();
    const std::vector<profiler::Frame> frames = profiler::Frames();
    const std::string_view prefix = "Render/Overlays/";
    const std::function<void(const profiler::ReportNode&)> collect =
        [&](const profiler::ReportNode& node) {
            if (node.name.starts_with(prefix)) {
                state.counters[node.name.substr(prefix.size()) + " ms"] += node.ms;
            }
            for (const profiler::ReportNode& child : node.children) {
                collect(child);
            }
        };
    if (!frames.empty()) {
        for (const profiler::ReportTrack& track : profiler::Summarize({frames.back()}).tracks) {
            if (track.gpu) {
                for (const profiler::ReportNode& zone : track.zones) {
                    collect(zone);
                }
            }
        }
    }
    state.counters["views"] = shape.views;
    state.counters["pairs"] = setup.pass->PairCapacity();
    state.SetItemsProcessed(
        int64_t(state.iterations()) * count *
        (source == Source::WorldLines || source == Source::FlatWorldLines ? 1 : shape.views));
}

void Composite(benchmark::State& state) {
    device::Device& device = bench::BenchDevice();
    const auto width = state.range(0);
    const auto height = state.range(1);

    struct Images {
        device::Tensor color;
        device::Tensor overlay;
    };

    Images& images = bench::Cached<Images>(fmt::format("{}x{}", width, height), [&] {
        const std::vector<int64_t> shape = {1, height, width, 4};
        std::vector<uint32_t> covered(uint64_t(width) * height);
        for (size_t at = 0; at < covered.size(); ++at) {
            covered[at] = at % 7 == 0 ? 0x800000FFu : 0;
        }
        return Images{
            device::Tensor(device.CreateBuffer({.label = "bench color"},
                                               std::vector<uint32_t>(covered.size(), 0xFF336699u)),
                           0, device::DType::UInt8, shape),
            device::Tensor(device.CreateBuffer({.label = "bench overlay"}, covered), 0,
                           device::DType::UInt8, shape)};
    });
    for (auto _ : state) {
        CompositeOverlay(images.color, images.overlay).ToHostBytes();
    }
    state.SetItemsProcessed(int64_t(state.iterations()) * width * height);
}

BENCHMARK_CAPTURE(DrawOverlays, HostLines, Source::HostLines)
    ->Name("Render/Overlays/HostLines")
    ->ArgNames({"lines", "shape"})
    ->Args({1'000, 0})
    ->Args({100'000, 0})
    ->Args({1'000, 1})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, DeviceLines, Source::DeviceLines)
    ->Name("Render/Overlays/DeviceLines")
    ->ArgNames({"lines", "shape"})
    ->Args({100'000, 0})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, WorldLines, Source::WorldLines)
    ->Name("Render/Overlays/WorldLines")
    ->ArgNames({"lines", "shape"})
    ->Args({102'400, 1})
    ->Args({102'400, 2})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, FlatWorldLines, Source::FlatWorldLines)
    ->Name("Render/Overlays/FlatWorldLines")
    ->ArgNames({"lines", "shape"})
    ->Args({102'400, 1})
    ->Args({102'400, 2})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, Markers, Source::Markers)
    ->Name("Render/Overlays/Markers")
    ->ArgNames({"markers", "shape"})
    ->Args({100'000, 0})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, Boxes, Source::Boxes)
    ->Name("Render/Overlays/Boxes")
    ->ArgNames({"boxes", "shape"})
    ->Args({10'000, 0})
    ->Args({10'000, 1})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, Spheres, Source::Spheres)
    ->Name("Render/Overlays/Spheres")
    ->ArgNames({"spheres", "shape"})
    ->Args({10'000, 0})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, Frames, Source::Frames)
    ->Name("Render/Overlays/Frames")
    ->ArgNames({"frames", "shape"})
    ->Args({10'000, 0})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, Arrows, Source::Arrows)
    ->Name("Render/Overlays/Arrows")
    ->ArgNames({"arrows", "shape"})
    ->Args({10'000, 0})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, Capsules, Source::Capsules)
    ->Name("Render/Overlays/Capsules")
    ->ArgNames({"capsules", "shape"})
    ->Args({10'000, 0})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, Trails, Source::Trails)
    ->Name("Render/Overlays/Trails")
    ->ArgNames({"trails", "shape"})
    ->Args({1'000, 0})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, Labels, Source::Labels)
    ->Name("Render/Overlays/Labels")
    ->ArgNames({"labels", "shape"})
    ->Args({100, 0})
    ->Args({1'000, 0})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, Numbers, Source::Numbers)
    ->Name("Render/Overlays/Numbers")
    ->ArgNames({"numbers", "shape"})
    ->Args({1'000, 0})
    ->Args({1'024, 2})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, ScreenNumbers, Source::ScreenNumbers)
    ->Name("Render/Overlays/ScreenNumbers")
    ->ArgNames({"numbers", "shape"})
    ->Args({32, 0})
    ->Args({4, 2})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, DashedLines, Source::DashedLines)
    ->Name("Render/Overlays/DashedLines")
    ->ArgNames({"lines", "shape"})
    ->Args({100'000, 0})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, HaloLabels, Source::HaloLabels)
    ->Name("Render/Overlays/HaloLabels")
    ->ArgNames({"labels", "shape"})
    ->Args({1'000, 0})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, Triangles, Source::Triangles)
    ->Name("Render/Overlays/Triangles")
    ->ArgNames({"triangles", "shape"})
    ->Args({10'000, 0})
    ->Args({10'000, 1})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, Images, Source::Images)
    ->Name("Render/Overlays/Images")
    ->ArgNames({"images", "shape"})
    ->Args({100, 0})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, PrimBounds, Source::PrimBounds)
    ->Name("Render/Overlays/PrimBounds")
    ->ArgNames({"prims", "shape"})
    ->Args({1'000, 0})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(DrawOverlays, Outline, Source::Outline)
    ->Name("Render/Overlays/Outline")
    ->ArgNames({"items", "shape"})
    ->Args({1, 0})
    ->Args({1, 1})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(Composite)
    ->Name("Render/Overlays/Composite")
    ->ArgNames({"width", "height"})
    ->Args({1920, 1080})
    ->Unit(benchmark::kMicrosecond);

}
}
