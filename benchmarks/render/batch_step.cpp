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
constexpr glm::uvec2 kSize{64, 64};

/// @brief A scene of `worlds` worlds side by side, each with shapes scattered in front of its
/// camera, and the products rendering every camera's colour and depth.
void BuildWorlds(Scene& scene, int worlds) {
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
            .SetResolution(glm::dvec2(kSize))
            .SetOrderedVars(vars);
    }
    RenderSettings::Define(scene, "/Render/Settings").SetProducts(products);
}

Timing StepWorlds(int worlds, double targetMs) {
    std::unique_ptr<Renderer> renderer;
    try {
        renderer = std::make_unique<Renderer>();
    } catch (const std::exception& error) {
        std::printf("skipped: %s\n", error.what());
        return {0.0, targetMs};
    }
    Scene scene;
    BuildWorlds(scene, worlds);
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
    std::printf("  %d views of %u x %u: %.0f views a second, %.2f ms submitting, on %s\n", worlds,
                kSize.x, kSize.y, 1000.0 * worlds / ms, host,
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

const bool kBatch1k =
    Register("render.batch_step.1024_worlds_64x64", [] { return StepWorlds(1024, 5.0); });
const bool kBatch4k =
    Register("render.batch_step.4096_worlds_64x64", [] { return StepWorlds(4096, 20.0); });

}  // namespace
}  // namespace ernie::bench
