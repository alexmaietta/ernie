// SPDX-License-Identifier: BSD-3-Clause
#include "batch.h"

#include "ernie/device/compute_interop.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/renderer.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace ernie::bench {

namespace {

/// Whether a var renders colour, which minimal "none" leaves out.
bool IsColour(const std::string& var) {
    return var == "LdrColor" || var == "HdrColor" ||
           var == "SpectralRadiance";
}

}  // namespace

std::vector<Param> BatchScenario::Params() const {
    std::vector<Param> params = {
        {"preset", "minimal", "Render settings, a registered preset"},
        {"outputs", "rgb", "Render vars, a registered output set"},
        {"resolution", "128", "Pixels along each side of a view"},
        {"worlds", "1024", "Worlds in the batch"},
        {"views", "1", "Views of each world"},
    };
    for (Param& param : WorldParams()) {
        params.push_back(std::move(param));
    }
    return params;
}

Definition BatchScenario::Describe(const Case&) const {
    return {.scope = Scope::Workflow,
            .completion = Completion::Pipelined};
}

std::string
BatchScenario::Validate(const Case&,
                        const RenderProductSetOutputs& outputs) {
    const auto product = outputs.find(kPaths.product);
    if (product == outputs.end()) {
        return "the batch rendered no outputs";
    }
    for (const auto& [name, var] : product->second.vars) {
        if (name != "LdrColor" && name != "DistanceToCamera") {
            continue;
        }
        RenderVarOutput output = var;
        const Tensor tensor = output.Map(MapDevice::Cpu).tensor;
        // Colour channels, or finite distances, and their spread.
        double sum = 0.0;
        double squares = 0.0;
        std::size_t count = 0;
        const auto add = [&](double value) {
            sum += value;
            squares += value * value;
            ++count;
        };
        if (name == "LdrColor") {
            const auto* texels =
                static_cast<const std::uint8_t*>(tensor.Data());
            for (std::int64_t i = 0; i < tensor.Elements(); ++i) {
                if (i % 4 != 3) {
                    add(texels[i]);
                }
            }
        } else {
            const auto* metres =
                static_cast<const float*>(tensor.Data());
            for (std::int64_t i = 0; i < tensor.Elements(); ++i) {
                if (std::isfinite(metres[i])) {
                    add(static_cast<double>(metres[i]));
                }
            }
        }
        if (count == 0) {
            return name + " has no finite values";
        }
        const double mean = sum / static_cast<double>(count);
        if (squares / static_cast<double>(count) - mean * mean <
            1e-6) {
            return name + " renders flat";
        }
    }
    return {};
}

void BatchScenario::Build(Scene& scene, const Case& c) {
    const int resolution = GetInt(c, "resolution");
    const int views = Worlds(c) * ViewsPerWorld(c);
    Camera::Define(scene, kPaths.camera)
        .SetFocalLength(FocalLength())
        .SetSensorNoise(false);
    DistantLight::Define(scene, kPaths.sun)
        .SetTransform(DistantLightTransform(SunDirection()));

    const OutputsInfo& outputs = FindOutputs(Get(c, "outputs"));
    std::vector<Path> vars;
    for (const std::string& name : outputs.vars) {
        const Path var = kPaths.product.Child("Vars").Child(name);
        RenderVar::Define(scene, var).SetSourceName(name);
        vars.push_back(var);
    }
    RenderProduct::Define(scene, kPaths.product)
        .SetCamera(kPaths.camera)
        .SetResolution(glm::dvec2(resolution, resolution))
        .SetOrderedVars(vars);
    // Ambient light besides the sun, so shadows and rooms are not
    // black.
    RenderSettings::Define(scene, kPaths.settings)
        .SetProducts({kPaths.product})
        .SetAmbient(Ambient());
    FindPreset(Get(c, "preset")).apply(scene);
    // Minimal shading without colour vars renders none.
    RenderProduct product(scene, kPaths.product);
    if (product.GetMinimalMode() != tokens::kOff &&
        std::none_of(outputs.vars.begin(), outputs.vars.end(),
                     IsColour)) {
        product.SetMinimalMode(std::string(tokens::kNone));
    }

    // The worlds first: their layouts may place the views.
    BuildWorlds(scene, c);
    std::vector<std::int32_t> worlds(static_cast<std::size_t>(views));
    for (int v = 0; v < views; ++v) {
        worlds[static_cast<std::size_t>(v)] = v / ViewsPerWorld(c);
    }
    Camera(scene, kPaths.camera)
        .SetPoses(PoseTensor(c, 0))
        .SetWorlds(Tensor::Copy<std::int32_t>(worlds, {views}));
}

void BatchScenario::Setup(Renderer& renderer, Scene&, const Case& c) {
    ComputeInterop* compute = renderer.Delegate().Compute();
    posesOnGpu_ = compute != nullptr;
    poses_.clear();
    for (int s = 0; s < kPoseSets; ++s) {
        Tensor poses = PoseTensor(c, s);
        poses_.push_back(compute != nullptr ? compute->Upload(poses)
                                            : std::move(poses));
    }
}

void BatchScenario::Step(Renderer&, Scene& scene, const Case&,
                         int step) {
    Camera(scene, kPaths.camera)
        .SetPoses(poses_[static_cast<std::size_t>(step % kPoseSets)]);
}

std::vector<Unit> BatchScenario::Units(const Case& c) const {
    const double views = Worlds(c) * ViewsPerWorld(c);
    const double resolution = GetDouble(c, "resolution");
    return {{"views", views},
            {"worlds", static_cast<double>(Worlds(c))},
            {"Mpixels", views * resolution * resolution, 1e6}};
}

Tensor BatchScenario::PoseTensor(const Case& c, int step) const {
    const int views = Worlds(c) * ViewsPerWorld(c);
    std::vector<glm::dmat4> poses(static_cast<std::size_t>(views));
    Poses(c, step, poses);
    // float32 (views, 4, 4) in the math convention, rows of four.
    std::vector<float> rows;
    rows.reserve(poses.size() * 16);
    for (const glm::dmat4& pose : poses) {
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                rows.push_back(static_cast<float>(pose[col][row]));
            }
        }
    }
    return Tensor::Copy<float>(rows, {views, 4, 4});
}

}  // namespace ernie::bench
