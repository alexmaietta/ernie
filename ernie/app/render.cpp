// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/render.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <iterator>
#include <numbers>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "ernie/ernie.h"

namespace ernie::app {
namespace {

constexpr double kStep = 1.0 / 60;

std::string WorldPath(uint32_t world) {
    return world == 0 ? std::string("/World") : fmt::format("/World_{}", world);
}

RenderVarSource ParseSource(std::string_view name) {
    const auto token = std::ranges::find(kRenderVarSourceTokens, name);
    if (token == std::end(kRenderVarSourceTokens)) {
        throw Error("\"{}\" is no AOV; try LdrColor, HdrColor, DistanceToCamera or Normal", name);
    }
    return RenderVarSource(token - std::begin(kRenderVarSourceTokens));
}

RenderMode ParsePreset(std::string_view preset) {
    if (preset.empty() || preset == "real-time") {
        return RenderMode::RealTime;
    }
    if (preset == "minimal") {
        return RenderMode::Minimal;
    }
    throw Error("--preset takes real-time or minimal, not \"{}\"", preset);
}

int64_t Pixels(const RenderOutput& output, const std::string& product) {
    const ProductOutput& rendered = output[product];
    if (rendered.frames.empty() || rendered.frames.front().renderVars.empty()) {
        return 0;
    }
    const Span<const int64_t> shape =
        rendered.frames.front().renderVars.begin()->second.tensor.Shape();
    return shape.size() >= 3 ? shape[0] * shape[1] * shape[2] : 0;
}

void PrintProfile() {
    const profiler::Report report = profiler::Summarize(profiler::Frames());
    std::fputs(profiler::ReportText(report).c_str(), stderr);
}

}

Ref<Scene> DefaultScene(device::Device& device, uint32_t worlds, bool waitForKernels) {
    const Ref<Scene> scene = Scene::Create(device, {.waitForKernels = waitForKernels});
    if (worlds > 1) {
        scene->Define<World>("/World");
    } else {
        scene->Define<Xform>("/World");
    }
    scene->Define<Scope>("/World/Looks");
    const std::pair<const char*, Float3> looks[] = {{"Red", {0.8f, 0.3f, 0.2f}},
                                                    {"Blue", {0.2f, 0.5f, 0.8f}},
                                                    {"Green", {0.3f, 0.7f, 0.3f}},
                                                    {"Yellow", {0.9f, 0.7f, 0.2f}},
                                                    {"Purple", {0.6f, 0.4f, 0.8f}}};
    for (const auto& [name, color] : looks) {
        scene->DefineMaterial(fmt::format("/World/Looks/{}", name),
                              {.inputs = {{"diffuseColor", color}}});
    }
    scene->Define<Plane>("/World/Ground", {.checkerboard = true,
                                           .checkerColorA = {0.6f, 0.6f, 0.6f},
                                           .checkerColorB = {0.3f, 0.3f, 0.3f},
                                           .width = 0,
                                           .length = 0});
    scene->Define<Cube>("/World/Cube",
                        {.materialBinding = {"/World/Looks/Red"}, .translate = {-6, 0, 1}});
    scene->Define<Sphere>("/World/Sphere",
                          {.materialBinding = {"/World/Looks/Blue"}, .translate = {-3, 0, 1}});
    scene->Define<Cylinder>("/World/Cylinder",
                            {.materialBinding = {"/World/Looks/Green"}, .translate = {0, 0, 1}});
    scene->Define<Cone>("/World/Cone",
                        {.materialBinding = {"/World/Looks/Yellow"}, .translate = {3, 0, 1}});
    scene->Define<Capsule>("/World/Capsule",
                           {.materialBinding = {"/World/Looks/Purple"}, .translate = {6, 0, 1}});
    scene->Define<DistantLight>("/World/Sun", {.orient = FromAzimuthElevation(146, 44)});

    const double halfFov = std::numbers::pi / 8;
    const double yaw = 0;
    const double pitch = 0.15;
    const double distance = 1.1 * 5.5 / std::sin(halfFov);
    const Double3 target{0, 0, 1};
    const double across = std::cos(pitch) * distance;
    const Double3 eye{target.x + std::sin(yaw) * across, target.y - std::cos(yaw) * across,
                      target.z + std::sin(pitch) * distance};
    scene->Define<Camera>("/World/Camera", {.translate = eye,
                                            .orient = LookAt(eye, target),
                                            .focalLength = float(12 / std::tan(halfFov)),
                                            .horizontalAperture = 24.0f * 16 / 9,
                                            .verticalAperture = 24});

    std::vector<std::string> clones;
    for (uint32_t world = 1; world < worlds; ++world) {
        clones.push_back(WorldPath(world));
    }
    if (!clones.empty()) {
        scene->Clone(*scene->Find("/World"), Span<const std::string>(clones));
    }
    return scene;
}

std::string ChooseProduct(Scene& scene, const Command& command) {
    if (!command.product.empty()) {
        const std::optional<Prim> product = scene.Find(command.product);
        if (!product || !product->IsA(SchemaId::RenderProduct)) {
            throw Error("The scene has no render product {}", command.product);
        }
        return command.product;
    }
    const bool viewer = command.kind == Command::Kind::Viewer;
    if (viewer && command.camera.empty() && command.preset.empty() && command.aovs.empty()) {
        return {};
    }

    std::vector<std::string> cameras;
    if (!command.camera.empty()) {
        const std::optional<Prim> camera = scene.Find(command.camera);
        if (!camera || !camera->IsA(SchemaId::Camera)) {
            throw Error("The scene has no camera {}", command.camera);
        }
        cameras.push_back(command.camera);
    } else {
        for (uint32_t world = 0; world < command.worlds; ++world) {
            cameras.push_back(WorldPath(world) + "/Camera");
        }
    }

    scene.Define<Scope>("/Render");
    std::vector<std::string> vars;
    std::vector<std::string> names = {"LdrColor"};
    names.insert(names.end(), command.aovs.begin(), command.aovs.end());
    for (const std::string& name : names) {
        const std::string path = "/Render/" + name;
        if (std::ranges::find(vars, path) == vars.end()) {
            scene.Define<RenderVar>(path, {.sourceName = ParseSource(name)});
            vars.push_back(path);
        }
    }
    const std::string product = "/Render/Product";
    scene.Define<RenderProduct>(product, {.renderMode = ParsePreset(command.preset),
                                          .camera = cameras,
                                          .resolution = {command.width, command.height},
                                          .orderedVars = vars});
    return product;
}

int RenderFrames(const Ref<Scene>& scene, const std::string& product, const Command& command) {
    Renderer renderer(scene);
    profiler::SetEnabled(command.profile);
    const std::string products[] = {product};
    const uint32_t frames = std::max(command.frames, 1u);
    RenderOutput output;
    for (uint32_t frame = 0; frame < frames; ++frame) {
        profiler::BeginFrame();
        output = renderer.Step(products, kStep);
    }
    output.Wait();
    if (!command.output.empty()) {
        for (const std::filesystem::path& written :
             WriteFrame(command.output, output[product].frames.back())) {
            Log(LogLevel::Info, "wrote {}", written.string());
        }
    }
    if (command.profile) {
        PrintProfile();
    }
    return 0;
}

int Benchmark(const Ref<Scene>& scene, const std::string& product, const Command& command) {
    using Clock = std::chrono::steady_clock;
    Renderer renderer(scene);
    const std::string products[] = {product};
    const uint32_t steps = command.frames > 0 ? command.frames : 100;
    int64_t pixels = 0;
    for (uint32_t step = 0; step < std::max(command.warmup, 1u); ++step) {
        const RenderOutput output = renderer.Step(products, kStep);
        output.Wait();
        pixels = Pixels(output, product);
    }

    profiler::SetEnabled(command.profile);
    const Clock::time_point serialStart = Clock::now();
    for (uint32_t step = 0; step < steps; ++step) {
        profiler::BeginFrame();
        renderer.Step(products, kStep).Wait();
    }
    const double latencyMs =
        std::chrono::duration<double, std::milli>(Clock::now() - serialStart).count() / steps;

    std::deque<RenderOutput> inFlight;
    const Clock::time_point pipelinedStart = Clock::now();
    for (uint32_t step = 0; step < steps; ++step) {
        profiler::BeginFrame();
        inFlight.push_back(renderer.Step(products, kStep));
        if (inFlight.size() == 3) {
            inFlight.front().Wait();
            inFlight.pop_front();
        }
    }
    for (const RenderOutput& output : inFlight) {
        output.Wait();
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - pipelinedStart).count();
    const double fps = steps / seconds;
    const double megapixels = double(pixels) * fps / 1e6;

    if (command.json) {
        std::printf("{\"product\": \"%s\", \"worlds\": %u, \"pixels\": %lld, \"warmup\": %u, "
                    "\"steps\": %u, \"latency_ms\": %.4f, \"fps\": %.2f, "
                    "\"megapixels_per_second\": %.2f}\n",
                    product.c_str(), command.worlds, static_cast<long long>(pixels), command.warmup,
                    steps, latencyMs, fps, megapixels);
    } else {
        std::printf("%s, %lld pixels a step\n", product.c_str(), static_cast<long long>(pixels));
        std::printf("  one step at a time  %.3f ms\n", latencyMs);
        std::printf("  three in flight     %.1f steps a second, %.1f megapixels a second\n", fps,
                    megapixels);
    }
    std::fflush(stdout);
    if (command.profile) {
        PrintProfile();
    }
    return 0;
}

}
