// SPDX-License-Identifier: BSD-3-Clause
// Renders 16 environments ("worlds") in one batch, the way a reinforcement learning loop renders
// one camera per environment, and writes the views as a 4 x 4 contact sheet (--output): a C++
// demo on ernie::app::Demo.
//
//     ./build/bin/demo_worlds_batched_clones --viewer null --output clones.png
//
// Each environment is a clone of the first, tagged with its world: the environments overlap at
// the origin, yet every view sees only its own world plus what no world claims (the ground and
// the sun). One camera carries a pose and a world per view. The viewer, and the frames without
// one, show every world at once through an overview camera.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"

#include <stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ernie;

constexpr int kWorlds = 16;
constexpr int kSize = 128;
constexpr int kColumns = 4;

class BatchedClonesDemo : public app::Demo {
  public:
    BatchedClonesDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        description = "16 cloned worlds rendered in one batch, one view per world.";
        Scene& scene = GetScene();
        GroundPlane::Define(scene, "/World/Ground");
        DistantLight::Define(scene, "/World/Sun")
            .SetTransform(DistantLightTransform({0.4, -0.6, 0.7}));

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

        // An overview of every world, for the frames without a viewer and the tests.
        const glm::dmat4 pose =
            LookAtTransform({0.0, -12.0, 8.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0});
        Camera::Define(scene, "/World/Overview").SetFocalLength(18.0).SetTransform(pose);
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera("/World/Overview")
            .SetResolution({640, 480});

        // One camera, one pose and one world per view (poses in the math convention, row-major).
        std::vector<double> poses;
        std::vector<std::int32_t> worlds;
        for (int view = 0; view < kWorlds; ++view) {
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
    }

    // The scene is still: render the batch once, all 16 views in one step.
    void Step() override {
        if (!views_.empty()) {
            return;
        }
        const Path products[] = {"/Render/Batch"};
        RenderProductSetOutputs outputs = GetRenderer().Step(GetScene(), products);
        // (views, rows, cols, 4) uint8; map to MapDevice::Cuda to keep it on the GPU instead.
        const Tensor views =
            outputs.at("/Render/Batch").vars.at("LdrColor").Map(MapDevice::Cpu).tensor;
        const auto* texels = static_cast<const std::uint8_t*>(views.Data());
        views_.assign(texels, texels + static_cast<std::size_t>(kWorlds * kSize * kSize * 4));
        std::printf("Rendered %d worlds in one step\n", kWorlds);
    }

    void SaveOutput(const std::filesystem::path& path) override {
        const int rows = kWorlds / kColumns;
        std::vector<std::uint8_t> sheet(
            static_cast<std::size_t>(rows * kSize * kColumns * kSize * 4));
        for (int view = 0; view < kWorlds; ++view) {
            for (int y = 0; y < kSize; ++y) {
                const std::uint8_t* from = View(view) + static_cast<std::ptrdiff_t>(y * kSize * 4);
                std::uint8_t* to =
                    sheet.data() + (((view / kColumns) * kSize + y) * kColumns * kSize +
                                    (view % kColumns) * kSize) *
                                       4;
                std::copy(from, from + kSize * 4, to);
            }
        }
        if (stbi_write_png(path.string().c_str(), kColumns * kSize, rows * kSize, 4, sheet.data(),
                           kColumns * kSize * 4) == 0) {
            throw std::runtime_error("Cannot write " + path.string());
        }
    }

    // Each view sees its own box: the views share a pose, so where one differs from all of them
    // (their median, the ground) it sees its box, which is redder the later its world.
    void TestFinal() override {
        if (views_.empty()) {
            throw std::runtime_error("The batch was not rendered");
        }
        std::vector<std::uint8_t> median(static_cast<std::size_t>(kSize * kSize * 4));
        for (int i = 0; i < kSize * kSize * 4; ++i) {
            std::uint8_t values[kWorlds];
            for (int view = 0; view < kWorlds; ++view) {
                values[view] = View(view)[i];
            }
            std::nth_element(values, values + kWorlds / 2, values + kWorlds);
            median[static_cast<std::size_t>(i)] = values[kWorlds / 2];
        }
        double lastRedness = -1e9;
        int rising = 0;
        for (int view = 0; view < kWorlds; ++view) {
            const std::uint8_t* texels = View(view);
            double redness = 0.0;
            int box = 0;
            for (int i = 0; i < kSize * kSize; ++i) {
                int difference = 0;
                for (int c = 0; c < 3; ++c) {
                    difference =
                        std::max(difference, std::abs(texels[i * 4 + c] - median[i * 4 + c]));
                }
                if (difference > 24) {
                    redness += texels[i * 4] - texels[i * 4 + 2];
                    ++box;
                }
            }
            if (box < 20) {
                throw std::runtime_error("View " + std::to_string(view) + " misses its box");
            }
            redness /= box;
            rising += redness > lastRedness ? 1 : 0;
            lastRedness = redness;
        }
        if (rising < kWorlds - 3) {
            throw std::runtime_error("The views do not see their own worlds' boxes");
        }
    }

  private:
    const std::uint8_t* View(int view) const {
        return views_.data() + static_cast<std::ptrdiff_t>(view * kSize * kSize * 4);
    }

    std::vector<std::uint8_t> views_;
};

}  // namespace

int main(int argc, char** argv) {
    return ernie::app::DemoMain(
        argc, argv, {ernie::app::MakeDemoInfo<BatchedClonesDemo>("worlds", "batched clones")});
}
