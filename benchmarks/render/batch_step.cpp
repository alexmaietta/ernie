// SPDX-License-Identifier: BSD-3-Clause
/// @file batch_step.cpp
/// @brief Batched rendering as reinforcement learning uses it, after Madrona's procedural benchmark
/// scenes: thousands of worlds, each a few shapes over a shared ground, a 64 x 64 camera a world
/// rendering colour and depth, every view in one dispatch. Reports views a second.
#include "benchmarks/bench.h"
#include "ernie/device/context.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/schema/camera.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/planet.h"
#include "ernie/scene/schema/render.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "ernie/util/profiler.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace ernie::bench {
namespace {

constexpr int kShapesPerWorld = 8;

/// @brief A scene of `worlds` worlds side by side, each with shapes scattered in front of its
/// camera, and the products rendering every camera's colour and depth at `size` pixels.
void BuildWorlds(Scene& scene, int worlds, glm::uvec2 size) {
    GroundPlane::Define(scene, "/Ground");
    const std::vector<Path> vars{Path("/Render/Vars/Colour"), Path("/Render/Vars/Depth")};
    RenderVar::Define(scene, vars[0]).SetSourceName(std::string(render_vars::kLdrColor));
    RenderVar::Define(scene, vars[1]).SetSourceName(std::string(render_vars::kDistanceToCamera));
    std::mt19937 random(7);
    std::uniform_real_distribution<double> spread(-2.0, 2.0);
    std::vector<Path> products;
    for (int world = 0; world < worlds; ++world) {
        const std::string root = "/World" + std::to_string(world);
        Xform::Define(scene, root).SetWorld(world);
        for (int shape = 0; shape < kShapesPerWorld; ++shape) {
            const std::string path = root + "/Shape" + std::to_string(shape);
            if (shape % 2 == 0) {
                Sphere::Define(scene, path).SetRadius(0.4);
            } else {
                Cube::Define(scene, path).SetSize(0.7);
            }
            SetXform(scene, path,
                     {.translate = glm::dvec3(spread(random), 5.0 + spread(random),
                                              0.5 + 0.2 * spread(random))});
        }
        const std::string camera = root + "/Camera";
        Camera::Define(scene, camera).SetHorizontalAperture(0.036).SetVerticalAperture(0.036);
        scene.Set(camera, tokens::kTransform,
                  LookAtTransform({0.0, 0.0, 1.5}, {0.0, 5.0, 0.5}, {0.0, 0.0, 1.0}));
        products.emplace_back("/Render/View" + std::to_string(world));
        RenderProduct::Define(scene, products.back())
            .SetCamera(camera)
            .SetResolution(glm::dvec2(size))
            .SetOrderedVars(vars);
    }
    RenderSettings::Define(scene, "/Render/Settings").SetProducts(products);
}

Timing StepWorlds(int worlds, glm::uvec2 size, double targetMs) {
    std::unique_ptr<Renderer> renderer;
    try {
        renderer = std::make_unique<Renderer>();
    } catch (const std::exception& error) {
        std::printf("skipped: %s\n", error.what());
        return {0.0, targetMs};
    }
    Scene scene;
    BuildWorlds(scene, worlds, size);
    // The first step builds every bottom level; later ones rebuild only the top levels.
    renderer->Step(scene);
    renderer->Context().WaitIdle();
    const double host = Fastest([] {}, [&] { renderer->Step(scene); });
    renderer->Context().WaitIdle();
    const double ms = Fastest([] {},
                              [&] {
                                  renderer->Step(scene);
                                  renderer->Context().WaitIdle();
                              });
    const double pixels = static_cast<double>(worlds) * size.x * size.y;
    std::printf("  %d views of %u x %u: %.0f views, %.2f gigapixels a second, %.2f ms submitting, "
                "on %s\n",
                worlds, size.x, size.y, 1000.0 * worlds / ms, pixels / ms / 1e6, host,
                renderer->Context().Info().name.c_str());
    if (std::getenv("ERNIE_BENCH_PROFILE") != nullptr) {
        profiler::SetEnabled(true);
        profiler::SetGpuEnabled(false);
        for (int step = 0; step < 8; ++step) {
            profiler::BeginFrame();
            renderer->Step(scene);
            renderer->Context().WaitIdle();
        }
        profiler::BeginFrame();
        const std::vector<profiler::Frame> frames = profiler::Frames();
        std::printf("%s", profiler::ReportText(profiler::Summarize(frames, profiler::ThreadNames()),
                                               {.depth = 4, .minMs = 0.1, .byTime = true})
                              .c_str());
        profiler::SetEnabled(false);
    }
    return {ms, targetMs};
}

/// Registers a step of `worlds` views of `side` x `side` pixels.
bool RegisterBatch(int worlds, std::uint32_t side, double targetMs) {
    return Register("render.batch_step." + std::to_string(worlds) + "_worlds_" +
                        std::to_string(side) + "x" + std::to_string(side),
                    [=] { return StepWorlds(worlds, glm::uvec2(side), targetMs); });
}

// Madrona's benchmark resolutions, then beyond, at 1024 worlds; and many worlds at 64 x 64.
const bool kBatches = RegisterBatch(1024, 64, 5.0) && RegisterBatch(1024, 128, 15.0) &&
                      RegisterBatch(1024, 256, 50.0) && RegisterBatch(1024, 512, 200.0) &&
                      RegisterBatch(4096, 64, 20.0);

}  // namespace
}  // namespace ernie::bench
