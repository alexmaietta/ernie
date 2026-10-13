// SPDX-License-Identifier: BSD-3-Clause
// The cost of keeping the renderer in step with a large scene: a grid
// of cubes in nested groups, one kind of edit each step, so a step's
// time is mostly the CPU's sync of what changed (or of everything).

#include "../bench.h"

#include "ernie/device/compute_interop.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "ernie/session/renderer.h"
#include "ernie/util/tensor.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie::bench {

namespace {

constexpr double kSpacing = 1.5;

class SceneSync : public Scenario {
  public:
    std::vector<Param> Params() const override {
        return {
            {"edit", "colour",
             "Each step: none, colour (one cube's), xform (one "
             "cube's), "
             "parent (one group's), batch (every cube's translate as "
             "rows of one tensor), gpubatch (the same from CUDA), "
             "mapped (the column mapped on CUDA, the renderer's own "
             "array, and written there), cpumapped (the column "
             "mapped "
             "on the host and written there), add (a cube added, the "
             "last "
             "removed), walk (every cube's world transform and "
             "visibility read)"},
            {"prims", "10000", "Cubes"},
            {"fanout", "100", "Cubes in each group"},
            {"depth", "3", "Nested transforms above each cube"},
            {"resolution", "64",
             "Pixels along each side of the view"},
        };
    }

    void Build(Scene& scene, const Case& c) override {
        const int prims = GetInt(c, "prims");
        const int fanout = std::max(GetInt(c, "fanout"), 1);
        const int depth = std::max(GetInt(c, "depth"), 1);
        const int resolution = GetInt(c, "resolution");
        const std::string edit = Get(c, "edit");
        if (edit != "none" && edit != "colour" && edit != "xform" &&
            edit != "parent" && edit != "batch" &&
            edit != "gpubatch" && edit != "mapped" &&
            edit != "cpumapped" && edit != "add" && edit != "walk") {
            throw std::invalid_argument("Unknown edit " + edit);
        }

        // Seen from above, the whole grid in view.
        const int side = static_cast<int>(
            std::ceil(std::sqrt(static_cast<double>(prims))));
        const double extent = side * kSpacing;
        Camera::Define(scene, kPaths.camera)
            .SetFocalLength(18.0)
            .SetSensorNoise(false)
            .SetTransform(LookAt({0.0, -0.6 * extent, 0.9 * extent},
                                 {0.0, 0.0, 0.0}));
        DistantLight::Define(scene, kPaths.sun)
            .SetTransform(DistantLightTransform({0.4, -0.6, 0.7}));
        const Path var =
            kPaths.product.Child("Vars").Child("LdrColor");
        RenderVar::Define(scene, var).SetSourceName("LdrColor");
        RenderProduct::Define(scene, kPaths.product)
            .SetCamera(kPaths.camera)
            .SetResolution(glm::dvec2(resolution, resolution))
            .SetOrderedVars({var});
        RenderSettings::Define(scene, kPaths.settings)
            .SetProducts({kPaths.product})
            .SetAmbient(0.3);
        FindPreset("minimal").apply(scene);

        // Groups of `fanout` cubes, each under `depth` transforms.
        cubes_.clear();
        groups_.clear();
        const Path root("/World/Grid");
        Xform::Define(scene, root);
        for (int i = 0; i < prims; ++i) {
            const int g = i / fanout;
            Path parent = root.Child("G" + std::to_string(g));
            if (i % fanout == 0) {
                Xform::Define(scene, parent);
                SetXform(scene, parent,
                         {.translate = glm::dvec3(0.0)});
                groups_.push_back(parent);
                for (int level = 1; level < depth; ++level) {
                    parent =
                        parent.Child("L" + std::to_string(level));
                    Xform::Define(scene, parent);
                }
            } else {
                for (int level = 1; level < depth; ++level) {
                    parent =
                        parent.Child("L" + std::to_string(level));
                }
            }
            const Path cube = parent.Child("C" + std::to_string(i));
            Cube::Define(scene, cube)
                .SetSize(1.0)
                .SetDisplayColor(
                    glm::dvec3(0.3 + 0.5 * (i % 7) / 6.0, 0.5, 0.6));
            SetXform(scene, cube,
                     {.translate = Position(i, side, 0.0)});
            cubes_.push_back(cube);
        }
        AddStat("cubes", prims);
        AddStat("groups", static_cast<double>(groups_.size()));
        AddStat("prims", static_cast<double>(scene.Prims().size()));

        // Two layouts of every cube's translate, for batched writes.
        batch_.clear();
        if (edit == "batch" || edit == "gpubatch" ||
            edit == "mapped" || edit == "cpumapped") {
            set_ = PrimSet(cubes_);
            for (int layout = 0; layout < 2; ++layout) {
                std::vector<float> rows;
                rows.reserve(static_cast<std::size_t>(prims) * 3);
                for (int i = 0; i < prims; ++i) {
                    const glm::dvec3 p =
                        Position(i, side, layout == 0 ? 0.0 : 0.25);
                    rows.push_back(static_cast<float>(p.x));
                    rows.push_back(static_cast<float>(p.y));
                    rows.push_back(static_cast<float>(p.z));
                }
                batch_.push_back(
                    Tensor::Copy<float>(rows, {prims, 3}));
            }
        }
        side_ = side;
    }

    void Setup(Renderer& renderer, Scene& scene,
               const Case& c) override {
        const std::string edit = Get(c, "edit");
        poses_ = scene.GetColumn(set_, tokens::kTranslate);
        if (edit != "gpubatch" && edit != "mapped") {
            return;
        }
        // The layouts on CUDA, as a simulation holds its state.
        compute_ = renderer.Delegate().Compute();
        if (compute_ == nullptr) {
            throw std::invalid_argument(edit + " needs CUDA interop");
        }
        for (Tensor& layout : batch_) {
            layout = compute_->Upload(layout);
        }
        // Posed on the GPU from the first, so mapping it there gives
        // the renderer's own array.
        if (edit == "mapped") {
            poses_.Write(batch_[0]);
        }
    }

    void Step(Renderer&, Scene& scene, const Case& c,
              int step) override {
        const std::string edit = Get(c, "edit");
        const auto n = static_cast<int>(cubes_.size());
        // A different cube or group each step, spread over the grid.
        const int i = static_cast<int>((step * 7919LL) % n);
        const double lift = 0.05 * (step % 8);
        if (edit == "colour") {
            Cube(scene, cubes_[static_cast<std::size_t>(i)])
                .SetDisplayColor(
                    glm::dvec3(0.2 + 0.1 * (step % 8), 0.4, 0.3));
        } else if (edit == "xform") {
            SetXform(scene, cubes_[static_cast<std::size_t>(i)],
                     {.translate = Position(i, side_, lift)});
        } else if (edit == "parent") {
            const std::size_t g =
                static_cast<std::size_t>(step) % groups_.size();
            SetXform(scene, groups_[g],
                     {.translate = glm::dvec3(0.0, 0.0, lift)});
        } else if (edit == "batch" || edit == "gpubatch") {
            scene.Write(set_, tokens::kTranslate,
                        batch_[static_cast<std::size_t>(step % 2)]);
        } else if (edit == "mapped") {
            const Tensor rows = poses_.Map(compute_->Location());
            compute_->Copy(batch_[static_cast<std::size_t>(step % 2)],
                           rows);
            poses_.Unmap();
        } else if (edit == "cpumapped") {
            const Tensor rows = poses_.Map(Device{});
            const Tensor& layout =
                batch_[static_cast<std::size_t>(step % 2)];
            std::memcpy(rows.Data(), layout.Data(),
                        static_cast<std::size_t>(layout.Elements()) *
                            sizeof(float));
            poses_.Unmap();
        } else if (edit == "add") {
            const Path extra("/World/Extra");
            if (step > 0) {
                scene.Remove(
                    extra.Child("E" + std::to_string(step - 1)));
            }
            const Path cube = extra.Child("E" + std::to_string(step));
            Cube::Define(scene, cube).SetSize(1.0);
            SetXform(scene, cube,
                     {.translate = Position(i, side_, 1.0)});
        } else if (edit == "walk") {
            // What a full sync's draw items ask of each cube.
            double sum = 0.0;
            for (const Path& cube : cubes_) {
                sum += scene.WorldTransform(cube)[3][2];
                sum += scene.IsVisible(cube) ? 1.0 : 0.0;
            }
            KeepAlive(sum);
        }
    }

  private:
    /// Cube `i`'s place in a grid `side` cubes across, lifted by
    /// `lift`.
    static glm::dvec3 Position(int i, int side, double lift) {
        const double half = 0.5 * (side - 1) * kSpacing;
        return {(i % side) * kSpacing - half,
                (i / side) * kSpacing - half, 0.5 + lift};
    }

    std::vector<Path> cubes_;
    std::vector<Path> groups_;
    PrimSet set_;
    ComputeInterop* compute_ = nullptr;
    Column poses_;
    std::vector<Tensor> batch_;
    int side_ = 1;
};

}  // namespace

ERNIE_BENCH_SCENARIO(SceneSync, "scene_sync",
                     "A grid of cubes in nested groups, one kind of "
                     "edit a step: the cost of syncing the renderer");

}  // namespace ernie::bench
