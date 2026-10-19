// SPDX-License-Identifier: BSD-3-Clause
/// @file shared_prims.cpp
/// @brief How batch worlds best trace the prims they share (RT-044), as Isaac Lab's scenes have
/// them: a shared ground and room, with furniture, and a few small objects a world whose poses
/// device columns hold, so every step rebuilds the top levels as a learning step does. Each world
/// renders colour and depth at 64 x 64. Reports a step's time with the shared prims duplicated
/// into every world's top level and with them in a top level of their own.
#include "benchmarks/bench.h"
#include "ernie/device/context.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/schema/camera.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/planet.h"
#include "ernie/scene/schema/render.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"

#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace ernie::bench {
namespace {

constexpr int kObjectsPerWorld = 8;
constexpr std::uint32_t kSide = 64;

/// @brief A room 10 m square around the origin: the ground, four walls, and furniture making up
/// `shared` prims in all; and `worlds` worlds, each with small objects and a camera inside it.
void BuildRoom(Scene& scene, int worlds, int shared) {
    GroundPlane::Define(scene, "/Room/Ground");
    for (int wall = 0; wall < 4; ++wall) {
        const std::string path = "/Room/Wall" + std::to_string(wall);
        Cube::Define(scene, path).SetSize(1.0);
        const double angle = 1.5707963267948966 * wall;
        SetXform(
            scene, path,
            {.translate = glm::dvec3(5.0 * std::cos(angle), 5.0 * std::sin(angle), 1.5),
             .scale = wall % 2 == 0 ? glm::dvec3(0.2, 10.0, 3.0) : glm::dvec3(10.0, 0.2, 3.0)});
    }
    std::mt19937 random(11);
    std::uniform_real_distribution<double> across(-4.0, 4.0);
    for (int item = 5; item < shared; ++item) {
        const std::string path = "/Room/Furniture" + std::to_string(item);
        Cube::Define(scene, path).SetSize(0.5);
        SetXform(scene, path, {.translate = glm::dvec3(across(random), across(random), 0.25)});
    }
    const std::vector<Path> vars{Path("/Render/Vars/Colour"), Path("/Render/Vars/Depth")};
    RenderVar::Define(scene, vars[0]).SetSourceName(std::string(render_vars::kLdrColor));
    RenderVar::Define(scene, vars[1]).SetSourceName(std::string(render_vars::kDistanceToCamera));
    std::vector<Path> products;
    for (int world = 0; world < worlds; ++world) {
        const std::string root = "/World" + std::to_string(world);
        Xform::Define(scene, root).SetWorld(world);
        for (int object = 0; object < kObjectsPerWorld; ++object) {
            const std::string path = root + "/Object" + std::to_string(object);
            if (object % 2 == 0) {
                Sphere::Define(scene, path).SetRadius(0.15);
            } else {
                Cube::Define(scene, path).SetSize(0.25);
            }
            SetXform(scene, path, {.translate = glm::dvec3(across(random), across(random), 0.2)});
        }
        const std::string camera = root + "/Camera";
        Camera::Define(scene, camera).SetHorizontalAperture(0.036).SetVerticalAperture(0.036);
        scene.Set(camera, tokens::kTransform,
                  LookAtTransform({across(random), -4.5, 1.5}, {0.0, 0.0, 0.5}, {0.0, 0.0, 1.0}));
        products.emplace_back("/Render/View" + std::to_string(world));
        RenderProduct::Define(scene, products.back())
            .SetCamera(camera)
            .SetResolution(glm::dvec2(kSide))
            .SetOrderedVars(vars);
    }
    RenderSettings::Define(scene, "/Render/Settings").SetProducts(products);
}

Timing StepRoom(SharedPrims policy, int worlds, int shared) {
    const double targetMs = 0.005 * worlds;
    std::unique_ptr<Renderer> renderer;
    try {
        renderer = std::make_unique<Renderer>(RendererOptions{.sharedPrims = policy});
    } catch (const std::exception& error) {
        std::printf("skipped: %s\n", error.what());
        return {0.0, targetMs};
    }
    Scene scene;
    BuildRoom(scene, worlds, shared);
    // The objects' poses on the device, as a simulation writes them: every step rebuilds the top
    // levels.
    renderer->BindColumn(scene, "/World0/Object0", "xformOp:translate");
    renderer->BindColumn(scene, "/World0/Object1", "xformOp:translate");
    renderer->Step(scene);
    renderer->Context().WaitIdle();
    const double ms = Fastest([] {},
                              [&] {
                                  renderer->Step(scene);
                                  renderer->Context().WaitIdle();
                              });
    std::printf("  %s, %d worlds, %d shared prims: %.0f views a second\n",
                policy == SharedPrims::Duplicated ? "duplicated" : "own top level", worlds, shared,
                1000.0 * worlds / ms);
    return {ms, targetMs};
}

bool RegisterRoom(SharedPrims policy, int worlds, int shared) {
    const std::string name = policy == SharedPrims::Duplicated ? "duplicated" : "own_top_level";
    return Register("render.shared_prims." + name + "." + std::to_string(worlds) + "_worlds_" +
                        std::to_string(shared) + "_shared",
                    [=] { return StepRoom(policy, worlds, shared); });
}

bool RegisterRooms() {
    for (const int shared : {8, 64}) {
        for (const int worlds : {64, 1024, 4096, 16384}) {
            for (const SharedPrims policy :
                 {SharedPrims::Duplicated, SharedPrims::SharedTopLevel}) {
                RegisterRoom(policy, worlds, shared);
            }
        }
    }
    return true;
}

const bool kRooms = RegisterRooms();

}  // namespace
}  // namespace ernie::bench
