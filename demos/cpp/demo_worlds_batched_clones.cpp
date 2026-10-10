// SPDX-License-Identifier: BSD-3-Clause
// Renders 16 environments ("worlds") headless in one batch, the way a reinforcement learning
// loop renders one camera per environment, and writes the views as a 4 x 4 contact sheet.
//
//     ./build/bin/demo_worlds_batched_clones [out.png]
//
// Each environment is a clone of the first, tagged with its world: the environments overlap at
// the origin, yet every view sees only its own world plus what no world claims (the ground and
// the sun). One camera carries a pose and a world per view.

#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "ernie/session/renderer.h"

#include <stb_image_write.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace {

constexpr int kWorlds = 16;
constexpr int kSize = 128;
constexpr int kColumns = 4;

ernie::Scene BuildScene() {
    using namespace ernie;
    Scene scene;
    GroundPlane::Define(scene, "/World/Ground");
    DistantLight::Define(scene, "/World/Sun");
    scene.Set("/World/Sun", tokens::kTransform, DistantLightTransform({0.4, -0.6, 0.7}));

    // One environment, then clones of it; each world moves and recolours its box.
    Xform::Define(scene, "/Envs/Env0");
    Cube::Define(scene, "/Envs/Env0/Box");
    for (int world = 0; world < kWorlds; ++world) {
        const Path env("/Envs/Env" + std::to_string(world));
        if (world > 0) {
            scene.Clone("/Envs/Env0", env);
        }
        Xform(scene, env).SetWorld(world);
        const double t = world / static_cast<double>(kWorlds);
        Cube(scene, env.Child("Box")).SetDisplayColor({t, 0.4, 1.0 - t});
        SetXform(
            scene, env.Child("Box"),
            {.translate = glm::dvec3(3.0 * std::cos(6.28 * t), 3.0 * std::sin(6.28 * t), 1.0)});
    }

    // One camera, one pose and one world per view (poses in the math convention, row-major).
    std::vector<double> poses;
    std::vector<std::int32_t> worlds;
    for (int view = 0; view < kWorlds; ++view) {
        const glm::dmat4 pose =
            LookAtTransform({0.0, -12.0, 8.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0});
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                poses.push_back(pose[col][row]);
            }
        }
        worlds.push_back(view);
    }
    Camera::Define(scene, "/World/Camera")
        .SetFocalLength(18.0)
        .SetPoses(Tensor::Copy<double>(poses, {kWorlds, 4, 4}))
        .SetWorlds(Tensor::Copy<std::int32_t>(worlds, {kWorlds}));
    RenderProduct::Define(scene, "/Render/Batch")
        .SetCamera("/World/Camera")
        .SetResolution({kSize, kSize});
    return scene;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string out = argc > 1 ? argv[1] : "demo_worlds_batched_clones.png";
    try {
        const ernie::Scene scene = BuildScene();
        ernie::Renderer renderer;
        const ernie::Path products[] = {"/Render/Batch"};
        ernie::RenderProductSetOutputs outputs = renderer.Step(scene, products);
        // (views, rows, cols, 4) uint8; map to MapDevice::Cuda to keep it on the GPU instead.
        const ernie::Tensor views =
            outputs.at("/Render/Batch").vars.at("LdrColor").Map(ernie::MapDevice::Cpu).tensor;
        const auto* texels = views.As<std::uint8_t>();
        const int rows = kWorlds / kColumns;
        std::vector<std::uint8_t> sheet(
            static_cast<std::size_t>(rows * kSize * kColumns * kSize * 4));
        for (int view = 0; view < kWorlds; ++view) {
            for (int y = 0; y < kSize; ++y) {
                const std::uint8_t* from = texels + ((view * kSize + y) * kSize) * 4;
                std::uint8_t* to =
                    sheet.data() + (((view / kColumns) * kSize + y) * kColumns * kSize +
                                    (view % kColumns) * kSize) *
                                       4;
                std::copy(from, from + kSize * 4, to);
            }
        }
        if (stbi_write_png(out.c_str(), kColumns * kSize, rows * kSize, 4, sheet.data(),
                           kColumns * kSize * 4) == 0) {
            std::fprintf(stderr, "Cannot write %s\n", out.c_str());
            return 1;
        }
        std::printf("Rendered %d worlds in one step into %s\n", kWorlds, out.c_str());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
