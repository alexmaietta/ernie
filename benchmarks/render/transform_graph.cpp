// SPDX-License-Identifier: BSD-3-Clause
/// @file transform_graph.cpp
/// @brief The transform graph at 10,000 environments of a body each: every body's transform
/// authored on the host, and the whole world removed in one update.
#include "ernie/render/transform_graph.h"
#include "benchmarks/bench.h"
#include "ernie/scene/xform.h"

#include <cstdio>
#include <memory>
#include <string>

namespace ernie::bench {
namespace {

constexpr int kEnvironments = 10'000;

/// A world of environments, each a body, and its transform graph.
struct World {
    std::unique_ptr<DeviceContext> context;
    std::unique_ptr<ShaderCompiler> compiler;
    std::unique_ptr<Arenas> arenas;
    std::unique_ptr<Scene> scene;
    std::unique_ptr<TransformGraph> graph;

    /// @return Whether there is a device to run on.
    bool Open() {
        try {
            context = std::make_unique<DeviceContext>();
        } catch (const DeviceError& error) {
            std::printf("skipped: %s\n", error.what());
            return false;
        }
        compiler = std::make_unique<ShaderCompiler>();
        arenas = std::make_unique<Arenas>(*context, MemoryUse::Device, "transforms", 1 << 22);
        return true;
    }

    void Build() {
        graph.reset();
        scene = std::make_unique<Scene>();
        scene->Define("/World", "Xform");
        for (int i = 0; i < kEnvironments; ++i) {
            const std::string env = "/World/Env_" + std::to_string(i);
            scene->Define(env, "Xform");
            SetXform(*scene, env, {.translate = glm::dvec3(i, 0, 0)});
            scene->Define(env + "/Body", "Xform");
            SetXform(*scene, env + "/Body", {.translate = glm::dvec3(0), .scale = glm::dvec3(1)});
        }
        graph = std::make_unique<TransformGraph>(*context, *arenas, *scene, *compiler);
        graph->Update().Wait();
    }
};

const bool kHostEdits = Register("render.transform_graph.edit_10k_bodies", [] {
    World world;
    if (!world.Open()) {
        return Timing{0.0, 5.0};
    }
    world.Build();
    double height = 0.0;
    const auto edit = [&] {
        height += 1.0;
        for (int i = 0; i < kEnvironments; ++i) {
            world.scene->Set(*world.scene->Find("/World/Env_" + std::to_string(i) + "/Body"),
                             "xformOp:translate", glm::dvec3(0, 0, height));
        }
    };
    return Timing{Fastest(edit, [&] { world.graph->Update().Wait(); }), 5.0};
});

const bool kRemoveWorld = Register("render.transform_graph.remove_20k_prims", [] {
    World world;
    if (!world.Open()) {
        return Timing{0.0, 10.0};
    }
    const auto remove = [&] {
        world.scene->Remove("/World");
        world.graph->Update().Wait();
    };
    return Timing{Fastest([&] { world.Build(); }, remove, 3), 10.0};
});

}  // namespace
}  // namespace ernie::bench
