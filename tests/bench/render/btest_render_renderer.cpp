// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
#include <deque>
#include <format>
#include <string>
#include <vector>

#include "bench.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/rotation.h"
#include "ernie/scene/scene.h"

namespace ernie {
namespace {

constexpr uint32_t kGroundCells = 128;
constexpr uint32_t kWarmup = 10;
const std::vector<std::string> kProducts = {"/Render/Product"};

Mesh::Desc Ground() {
    Mesh::Desc desc;
    const uint32_t side = kGroundCells + 1;
    for (uint32_t y = 0; y < side; ++y) {
        for (uint32_t x = 0; x < side; ++x) {
            const float u = float(x) / kGroundCells - 0.5f;
            const float v = float(y) / kGroundCells - 0.5f;
            desc.points.push_back({u * 20, v * 20, 0.3f * std::sin(u * 17) * std::cos(v * 13)});
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
    desc.displayColor = {0.6f, 0.55f, 0.5f};
    return desc;
}

struct Setup {
    Ref<Scene> scene;
    Renderer renderer;
    uint64_t pixels = 0;
};

Setup Prepare(uint32_t width, uint32_t height, RenderMode mode) {
    const Ref<Scene> scene = Scene::Create(bench::BenchDevice());
    scene->Define<World>("/env_0");
    scene->Define<Mesh>("/env_0/Ground", Ground());
    for (int i = 0; i < 8; ++i) {
        const double angle = i * 0.785;
        scene->Define<Sphere>(
            std::format("/env_0/Ball_{}", i),
            {.translate = {3 * std::cos(angle), 3 * std::sin(angle), 0.6},
             .displayColor = {0.2f + 0.1f * float(i), 0.4f, 0.8f - 0.1f * float(i)},
             .radius = 0.5f});
    }
    scene->Define<Cube>(
        "/env_0/Box", {.translate = {0, 0, 0.75}, .displayColor = {0.8f, 0.3f, 0.2f}, .size = 1.5});
    scene->Define<Camera>("/env_0/Cam", {.translate = {0, 0, 9}});
    scene->Define<DistantLight>("/Sun", {.orient = FromAzimuthElevation(60, 50)});
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Color", {.sourceName = RenderVarSource::LdrColor});
    scene->Define<RenderVar>("/Render/Depth", {.sourceName = RenderVarSource::DistanceToCamera});
    scene->Define<RenderProduct>("/Render/Product",
                                 {.camera = {"/env_0/Cam"},
                                  .resolution = {width, height},
                                  .orderedVars = {"/Render/Color", "/Render/Depth"}});
    scene->Select("/Render/Product").Get<RenderMode>("renderMode").Set(mode);
    Setup setup{scene, Renderer(scene), uint64_t(width) * height};
    for (uint32_t i = 0; i < kWarmup; ++i) {
        setup.renderer.Step(kProducts, 1.0 / 60).Wait();
    }
    return setup;
}

void Steps(benchmark::State& state, Setup& setup, size_t inflight) {
    std::deque<RenderOutput> flight;
    for (auto _ : state) {
        flight.push_back(setup.renderer.Step(kProducts, 1.0 / 60));
        if (flight.size() >= inflight) {
            flight.front().Wait();
            flight.pop_front();
        }
    }
    for (const RenderOutput& output : flight) {
        output.Wait();
    }
    state.counters["FPS"] =
        benchmark::Counter(double(state.iterations()), benchmark::Counter::kIsRate);
    state.counters["Mpx/s"] = benchmark::Counter(
        double(state.iterations()) * double(setup.pixels) * 1e-6, benchmark::Counter::kIsRate);
}

void View(benchmark::State& state) {
    const auto height = uint32_t(state.range(1));
    const auto mode = state.range(2) != 0 ? RenderMode::RealTime : RenderMode::Minimal;
    Setup& setup = bench::Cached<Setup>(std::format("view/{}/{}", height, int(mode)),
                                        [&] { return Prepare(height * 16 / 9, height, mode); });
    Steps(state, setup, size_t(state.range(0)));
}

const bool kCore = bench::Core("^Render/View/inflight:3/height:1080/");

BENCHMARK(View)
    ->Name("Render/View")
    ->ArgNames({"inflight", "height", "realtime"})
    ->ArgsProduct({{1, 3}, {360, 1080, 2160}, {0, 1}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

}
}
