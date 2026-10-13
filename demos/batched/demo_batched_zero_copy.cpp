// SPDX-License-Identifier: BSD-3-Clause
// Times a batched camera's observations mapped to CUDA in place
// against copied at each map (ernie:outputs:inPlace), for a few sets
// of render vars, and checks both give the same observations.

#include "ernie/app/demo.h"
#include "ernie/device/compute_interop.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;
using Clock = std::chrono::steady_clock;

constexpr int kSize = 64;
constexpr int kSteps = 30;  // timed steps
constexpr int kWarmup = 5;  // steps before timing
constexpr int kBands = 3;

// The render vars timed, from one small one to the spectrum.
const std::vector<std::vector<std::string>> kVarSets = {
    {"LdrColor"},
    {"LdrColor", "DistanceToCamera", "Normal", "PrimId"},
    {"LdrColor", "SpectralRadiance"},
};

const Path kCamera("/World/Camera");
const Path kProduct("/Render/BatchCamera");

/// A cube and a sphere in each world, under the sun.
void BuildScene(Scene& scene, int worlds) {
    GroundPlane::Define(scene, "/World/Ground");
    DistantLight::Define(scene, "/World/Sun")
        .SetTransform(DistantLightTransform({0.4, -0.6, 0.7}));
    Xform::Define(scene, "/Envs/Env0");
    Cube::Define(scene, "/Envs/Env0/Cube")
        .SetDisplayColor({0.8, 0.3, 0.2});
    SetXform(scene, "/Envs/Env0/Cube",
             {.translate = glm::dvec3(-1.0, 0.0, 0.5)});
    Sphere::Define(scene, "/Envs/Env0/Sphere")
        .SetDisplayColor({0.2, 0.5, 0.8});
    SetXform(scene, "/Envs/Env0/Sphere",
             {.translate = glm::dvec3(1.2, 0.5, 0.5)});
    for (int world = 0; world < worlds; ++world) {
        const Path env("/Envs/Env" + std::to_string(world));
        if (world > 0) {
            scene.Clone("/Envs/Env0", env);
        }
        Xform(scene, env).SetWorld(world);
    }
    Camera::Define(scene, kCamera)
        .SetFocalLength(18.0)
        .SetSpectralBands(kBands)
        .SetSensorNoise(false);
}

/// One pose per view, circling its world from its own heading:
/// float32 (views, 4, 4) in the math convention, rows of four.
Tensor Orbit(int views, double angle) {
    std::vector<float> poses;
    poses.reserve(static_cast<std::size_t>(views) * 16);
    for (int v = 0; v < views; ++v) {
        const double heading =
            angle + 2.0 * glm::pi<double>() * v / views;
        const glm::dmat4 pose = LookAtTransform(
            {6.0 * std::cos(heading), 6.0 * std::sin(heading), 4.0},
            {0.0, 0.0, 0.5}, {0.0, 0.0, 1.0});
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                poses.push_back(static_cast<float>(pose[col][row]));
            }
        }
    }
    return Tensor::Copy<float>(poses, {views, 4, 4});
}

ComputeInterop& RequireCuda(Renderer& renderer) {
    ComputeInterop* compute = renderer.Delegate().Compute();
    if (compute == nullptr) {
        throw std::runtime_error("This demo needs CUDA interop");
    }
    return *compute;
}

/// A batch camera of `views` views over a scene of their worlds, its
/// outputs mapped in place or copied, as Python's BatchCamera
/// defines it.
class Batch {
  public:
    Batch(Renderer& renderer, Scene& scene, int views,
          const std::vector<std::string>& vars, bool inPlace)
        : renderer_(renderer), scene_(scene) {
        BuildScene(scene, views);
        std::vector<Path> paths;
        for (const std::string& name : vars) {
            const Path var = kProduct.Child("Vars").Child(name);
            RenderVar::Define(scene, var).SetSourceName(name);
            paths.push_back(var);
        }
        RenderProduct::Define(scene, kProduct)
            .SetCamera(kCamera)
            .SetResolution({kSize, kSize})
            .SetOrderedVars(paths)
            .SetOutputsInPlace(inPlace);
        std::vector<std::int32_t> worlds(
            static_cast<std::size_t>(views));
        for (int v = 0; v < views; ++v) {
            worlds[static_cast<std::size_t>(v)] = v;
        }
        Camera(scene, kCamera)
            .SetPoses(Orbit(views, 0.0))
            .SetWorlds(Tensor::Copy<std::int32_t>(worlds, {views}));
        ComputeInterop& compute = RequireCuda(renderer);
        for (int s = 0; s < kWarmup + kSteps; ++s) {
            poses_.push_back(compute.Upload(Orbit(views, 0.05 * s)));
        }
    }

    /// Renders step `s` and returns each var's observations on the
    /// GPU, without waiting for the render.
    std::map<std::string, Tensor> Step(int s) {
        Camera(scene_, kCamera)
            .SetPoses(
                poses_[static_cast<std::size_t>(s) % poses_.size()]);
        const Path products[] = {kProduct};
        StepOperation operation =
            renderer_.StepAsync(scene_, products);
        RenderProductSetOutputs outputs = operation.Outputs();
        std::map<std::string, Tensor> observations;
        for (auto& [name, var] : outputs.at(kProduct).vars) {
            observations.emplace(name,
                                 var.Map(MapDevice::Gpu).tensor);
        }
        return observations;
    }

    /// Milliseconds per step until its observations are mapped, and
    /// per step with the render finished.
    std::pair<double, double> Time() {
        for (int s = 0; s < kWarmup; ++s) {
            Step(s);
        }
        double mapped = 0.0;
        std::map<std::string, Tensor> observations;
        const Clock::time_point start = Clock::now();
        for (int s = 0; s < kSteps; ++s) {
            const Clock::time_point begun = Clock::now();
            observations = Step(kWarmup + s);
            mapped +=
                std::chrono::duration<double>(Clock::now() - begun)
                    .count();
        }
        // The last step has finished.
        RequireCuda(renderer_).Download(observations.at("LdrColor"));
        const double total =
            std::chrono::duration<double>(Clock::now() - start)
                .count();
        return {1e3 * mapped / kSteps, 1e3 * total / kSteps};
    }

  private:
    Renderer& renderer_;
    Scene& scene_;
    std::vector<Tensor> poses_;
};

class ZeroCopyDemo : public app::Demo {
  public:
    ZeroCopyDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        ParseOptions(options.extra);
        Renderer& renderer = GetRenderer();
        RequireCuda(renderer);
        std::printf(
            "%s, %d views of %d x %d, %d bands, ms per step\n",
            renderer.DeviceName().c_str(), views_, kSize, kSize,
            kBands);
        std::printf("%-52s %15s %15s  (until mapped / rendered)\n",
                    "render vars", "copied", "in place");
        for (const std::vector<std::string>& vars : kVarSets) {
            std::string names;
            for (const std::string& name : vars) {
                names += (names.empty() ? "" : ", ") + name;
            }
            std::printf("%-52s", names.c_str());
            for (const bool inPlace : {false, true}) {
                Scene scene;
                Batch batch(renderer, scene, views_, vars, inPlace);
                const auto [until, whole] = batch.Time();
                std::printf(" %6.2f / %6.2f", until, whole);
            }
            std::printf("\n");
        }
        // The views of the last set, for the output and the test: the
        // same step mapped in place and copied.
        inPlace_ = std::make_unique<Batch>(
            renderer, GetScene(), views_, kVarSets.back(), true);
        copied_ = std::make_unique<Batch>(
            renderer, copiedScene_, views_, kVarSets.back(), false);
    }

    void Step() override {}

    // Without a viewer the timed steps were the renders.
    void Render() override {
        if (GetViewer() != nullptr) {
            GetViewer()->Step();
        }
    }

    void SaveOutput(const std::filesystem::path& path) override {
        const Tensor images = Observations(*inPlace_).at("LdrColor");
        const auto* texels =
            static_cast<const std::uint8_t*>(images.Data());
        // 16 views spread round the orbit.
        const int stride = std::max(views_ / 16, 1);
        Tensor sheet =
            Tensor::Zeros(DType::UInt8, {4 * kSize, 4 * kSize, 4});
        auto* to = static_cast<std::uint8_t*>(sheet.Data());
        for (int i = 0; i < 16 && i * stride < views_; ++i) {
            const int r = i / 4;
            const int c = i % 4;
            for (int y = 0; y < kSize; ++y) {
                std::memcpy(
                    to +
                        ((r * kSize + y) * 4 * kSize + c * kSize) * 4,
                    texels + ((static_cast<std::size_t>(i * stride) *
                                   kSize +
                               static_cast<std::size_t>(y)) *
                              kSize) *
                                 4,
                    kSize * 4);
            }
        }
        app::WritePng(path, sheet);
    }

    void TestFinal() override {
        const std::map<std::string, Tensor> inPlace =
            Observations(*inPlace_);
        const std::map<std::string, Tensor> copied =
            Observations(*copied_);
        for (const auto& [name, tensor] : copied) {
            const Tensor& other = inPlace.at(name);
            const std::size_t bytes =
                static_cast<std::size_t>(tensor.Elements()) *
                SizeOf(tensor.Type());
            Check(other.Shape() == tensor.Shape() &&
                      std::memcmp(other.Data(), tensor.Data(),
                                  bytes) == 0,
                  name + " in place differs from copied");
        }
        const Tensor& images = inPlace.at("LdrColor");
        const auto* texels =
            static_cast<const std::uint8_t*>(images.Data());
        double sum = 0.0;
        double squares = 0.0;
        std::size_t count = 0;
        for (std::int64_t i = 0; i < images.Elements(); ++i) {
            if (i % 4 != 3) {
                sum += texels[i];
                squares += static_cast<double>(texels[i]) * texels[i];
                ++count;
            }
        }
        const double mean = sum / static_cast<double>(count);
        Check(std::sqrt(squares / static_cast<double>(count) -
                        mean * mean) > 1.0,
              "the views should not render flat");
    }

  private:
    /// --views among the demo's own arguments.
    void ParseOptions(const std::vector<std::string>& extra) {
        for (std::size_t i = 0; i < extra.size(); ++i) {
            if (extra[i] == "--views" && i + 1 < extra.size()) {
                views_ = std::stoi(extra[++i]);
            } else {
                throw std::invalid_argument("Unknown option " +
                                            extra[i]);
            }
        }
        if (views_ < 1) {
            throw std::invalid_argument(
                "--views needs at least one view");
        }
    }

    /// Its first pose's observations on the host, after a step that
    /// leaves its outputs in place.
    std::map<std::string, Tensor> Observations(Batch& batch) {
        batch.Step(0);
        std::map<std::string, Tensor> host;
        for (auto& [name, tensor] : batch.Step(0)) {
            host.emplace(name, RequireCuda(GetRenderer())
                                   .Download(tensor)
                                   .Contiguous());
        }
        return host;
    }

    int views_ = 1024;
    Scene copiedScene_;
    std::unique_ptr<Batch> inPlace_;
    std::unique_ptr<Batch> copied_;
};

}  // namespace

ERNIE_REGISTER_DEMO(ZeroCopyDemo, "batched", "zero_copy",
                    "Batched observations mapped to CUDA in place "
                    "against copied, timed per step.");
