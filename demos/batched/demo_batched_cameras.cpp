// SPDX-License-Identifier: BSD-3-Clause
// Benchmarks a batched camera, one view per cloned environment, with
// poses and images kept on the GPU.

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
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;
using Clock = std::chrono::steady_clock;

constexpr int kSize = 64;
constexpr int kSteps = 60;  // distinct poses, cycled
constexpr int kWarmup =
    5;  // steps before timing: uploads, pipelines, allocations

// An environment's things, cubes and spheres in turn: their colours
// and where they stand, the same in demo_batched_cameras.py.
struct Thing {
    glm::dvec3 colour;
    double x;
    double y;
};
constexpr std::array<Thing, 6> kThings = {{
    {{0.700, 0.918, 0.821}, -1.649, -1.199},
    {{0.899, 0.204, 0.857}, 1.782, -0.192},
    {{0.442, 0.423, 0.404}, -0.330, 0.027},
    {{0.643, 0.996, 0.834}, 0.733, 2.934},
    {{0.372, 0.328, 0.690}, -2.736, -2.786},
    {{0.612, 0.573, 0.934}, 0.775, 0.085},
}};

constexpr std::array<const char*, 4> kMinimalModes = {
    "off", "none", "constantDiffuse", "texturedDiffuse"};

const Path kCamera("/World/Camera");
const Path kProduct("/Render/BatchCamera");

void BuildScene(Scene& scene, int worlds, int bands) {
    GroundPlane::Define(scene, "/World/Ground");
    DistantLight::Define(scene, "/World/Sun")
        .SetTransform(DistantLightTransform({0.4, -0.6, 0.7}));
    Xform::Define(scene, "/Envs/Env0");
    for (std::size_t i = 0; i < kThings.size(); ++i) {
        const Path thing("/Envs/Env0/Thing" + std::to_string(i));
        if (i % 2 == 0) {
            Cube::Define(scene, thing)
                .SetDisplayColor(kThings[i].colour);
        } else {
            Sphere::Define(scene, thing)
                .SetDisplayColor(kThings[i].colour);
        }
        SetXform(scene, thing,
                 {.translate =
                      glm::dvec3(kThings[i].x, kThings[i].y, 0.5)});
    }
    for (int world = 0; world < worlds; ++world) {
        const Path env("/Envs/Env" + std::to_string(world));
        if (world > 0) {
            scene.Clone("/Envs/Env0", env);
        }
        Xform(scene, env).SetWorld(world);
    }
    Camera::Define(scene, kCamera)
        .SetFocalLength(18.0)
        .SetSpectralBands(bands);
}

/// One pose per view, circling its environment from its own heading:
/// float32 (views, 4, 4) in the math convention, rows of four.
Tensor Orbit(int views, double angle) {
    std::vector<float> poses;
    poses.reserve(static_cast<std::size_t>(views) * 16);
    for (int v = 0; v < views; ++v) {
        const double heading =
            angle + 2.0 * glm::pi<double>() * v / views;
        const glm::dmat4 pose = LookAtTransform(
            {8.0 * std::cos(heading), 8.0 * std::sin(heading), 5.0},
            {0.0, 0.0, 0.5}, {0.0, 0.0, 1.0});
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                poses.push_back(static_cast<float>(pose[col][row]));
            }
        }
    }
    return Tensor::Copy<float>(poses, {views, 4, 4});
}

/// The render vars a batch renders: distances besides the black
/// colour of minimal "none".
std::vector<std::string> RenderVars(const std::string& minimal) {
    if (minimal == "none") {
        return {"LdrColor", "DistanceToCamera"};
    }
    return {"LdrColor"};
}

/// A product rendering `views` views through the camera, one world
/// each, as Python's ernie.batch.BatchCamera defines it: minimal
/// shading unless `minimal` is "off".
void DefineBatch(Scene& scene, int views,
                 const std::string& minimal) {
    std::vector<Path> vars;
    for (const std::string& name : RenderVars(minimal)) {
        const Path var = kProduct.Child("Vars").Child(name);
        RenderVar::Define(scene, var).SetSourceName(name);
        vars.push_back(var);
    }
    RenderProduct product = RenderProduct::Define(scene, kProduct)
                                .SetCamera(kCamera)
                                .SetResolution({kSize, kSize})
                                .SetOrderedVars(vars);
    if (minimal != "off") {
        product.SetMinimalMode(minimal);
    }
    std::vector<std::int32_t> worlds(static_cast<std::size_t>(views));
    for (int v = 0; v < views; ++v) {
        worlds[static_cast<std::size_t>(v)] = v;
    }
    Camera(scene, kCamera)
        .SetPoses(Orbit(views, 0.0))
        .SetWorlds(Tensor::Copy<std::int32_t>(worlds, {views}));
}

/// Every step's poses, in CUDA memory, as a simulator would hand them
/// over.
std::vector<Tensor> UploadPoses(ComputeInterop& compute, int views) {
    std::vector<Tensor> poses;
    for (int s = 0; s < kSteps; ++s) {
        poses.push_back(compute.Upload(Orbit(views, 0.03 * s)));
    }
    return poses;
}

/// Renders a step with `poses` and returns each render var's
/// observations on the GPU, without waiting for the render.
std::map<std::string, Tensor>
StepBatch(Renderer& renderer, Scene& scene, const Tensor& poses) {
    Camera(scene, kCamera).SetPoses(poses);
    const Path products[] = {kProduct};
    StepOperation operation = renderer.StepAsync(scene, products);
    RenderProductSetOutputs outputs = operation.Outputs();
    std::map<std::string, Tensor> observations;
    for (auto& [name, var] : outputs.at(kProduct).vars) {
        observations.emplace(name, var.Map(MapDevice::Gpu).tensor);
    }
    return observations;
}

ComputeInterop& RequireCuda(Renderer& renderer) {
    ComputeInterop* compute = renderer.Delegate().Compute();
    if (compute == nullptr) {
        throw std::runtime_error("This demo needs CUDA interop");
    }
    return *compute;
}

/// Seconds per step of `views` views at `bands` bands, in a scene of
/// their own.
double Benchmark(Renderer& renderer, int views, int bands,
                 const std::string& minimal) {
    Scene scene;
    BuildScene(scene, views, bands);
    DefineBatch(scene, views, minimal);
    ComputeInterop& compute = RequireCuda(renderer);
    const std::vector<Tensor> poses = UploadPoses(compute, views);
    for (int s = 0; s < kWarmup; ++s) {
        StepBatch(renderer, scene,
                  poses[static_cast<std::size_t>(s)]);
    }
    const Clock::time_point start = Clock::now();
    std::map<std::string, Tensor> observations;
    for (const Tensor& pose : poses) {
        observations = StepBatch(renderer, scene, pose);
    }
    compute.Download(
        observations.at("LdrColor"));  // the last step has finished
    return std::chrono::duration<double>(Clock::now() - start)
               .count() /
           kSteps;
}

void PrintRate(int views, int bands, double perStep,
               const std::string& note = {}) {
    std::printf("%6d views, %2d bands: %7.2f ms per step, %10.0f "
                "views per second%s\n",
                views, bands, perStep * 1e3, views / perStep,
                note.c_str());
}

class BatchCameraDemo : public app::Demo {
  public:
    BatchCameraDemo(app::Viewer* viewer,
                    const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = kWarmup + kSteps;
        bool sweep = false;
        ParseOptions(options.extra, sweep);
        Renderer& renderer = GetRenderer();
        ComputeInterop& compute = RequireCuda(renderer);
        std::printf("%s, %d x %d LdrColor observations\n",
                    renderer.DeviceName().c_str(), kSize, kSize);
        if (sweep) {
            for (const int bands : {3, 16}) {
                for (const int views : {256, 1024, 4096}) {
                    PrintRate(
                        views, bands,
                        Benchmark(renderer, views, bands, "off"));
                }
            }
            // Minimal shading has no spectral bands to speak of: one
            // row per mode.
            for (std::size_t m = 1; m < kMinimalModes.size(); ++m) {
                for (const int views : {256, 1024, 4096}) {
                    PrintRate(views, 3,
                              Benchmark(renderer, views, 3,
                                        kMinimalModes[m]),
                              std::string(", minimal ") +
                                  kMinimalModes[m]);
                }
            }
        }
        BuildScene(GetScene(), views_, bands_);
        DefineBatch(GetScene(), views_, minimal_);
        poses_ = UploadPoses(compute, views_);
    }

    void Step() override {
        if (frame == kWarmup) {
            Images();  // the warm-up has finished
            timerStart_ = Clock::now();
            timedSteps_ = 0;
        }
        observations_ = StepBatch(
            GetRenderer(), GetScene(),
            poses_[static_cast<std::size_t>(frame % kSteps)]);
        if (timerStart_) {
            ++timedSteps_;
        }
    }

    // Without a viewer the batch step is the render.
    void Render() override {
        if (GetViewer() != nullptr) {
            GetViewer()->Step();
        }
    }

    void SaveOutput(const std::filesystem::path& path) override {
        Report();
        // 16 views spread round the orbit; neighbouring views differ
        // by only a few degrees.
        const Tensor images = Images();
        const auto* texels =
            static_cast<const std::uint8_t*>(images.Data());
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
                               y) *
                              kSize) *
                                 4,
                    kSize * 4);
            }
        }
        app::WritePng(path, sheet);
    }

    void Close() override {
        Report();
    }

    void TestFinal() override {
        Report();
        const Tensor images = Images();
        Check(images.Shape() ==
                  std::vector<std::int64_t>{views_, kSize, kSize, 4},
              "the observations should hold a view per environment");
        const std::size_t pixels =
            static_cast<std::size_t>(views_) * kSize * kSize;
        // The values the checks look at: colours, or for minimal
        // "none" (black by design) the distances in tenths of a
        // metre, as the colours' scale.
        std::vector<double> values(pixels * 3);
        const auto* texels =
            static_cast<const std::uint8_t*>(images.Data());
        if (minimal_ == "none") {
            for (std::size_t i = 0; i < pixels; ++i) {
                Check(texels[i * 4] == 0 && texels[i * 4 + 1] == 0 &&
                          texels[i * 4 + 2] == 0,
                      "minimal none renders no colour");
            }
            const Tensor distances =
                RequireCuda(GetRenderer())
                    .Download(observations_.at("DistanceToCamera"));
            const auto* metres =
                static_cast<const float*>(distances.Data());
            for (std::size_t i = 0; i < pixels; ++i) {
                const double tenths =
                    std::isfinite(metres[i])
                        ? std::min(10.0 * metres[i], 255.0)
                        : 0.0;
                values[i * 3] = values[i * 3 + 1] =
                    values[i * 3 + 2] = tenths;
            }
        } else {
            for (std::size_t i = 0; i < pixels; ++i) {
                for (int c = 0; c < 3; ++c) {
                    values[i * 3 + c] = texels[i * 4 + c];
                }
            }
        }
        double sum = 0.0;
        double squares = 0.0;
        for (const double value : values) {
            sum += value;
            squares += value * value;
        }
        const double mean = sum / static_cast<double>(values.size());
        Check(std::sqrt(squares / static_cast<double>(values.size()) -
                        mean * mean) > 1.0,
              "the views should not render flat");
        // Views from opposite headings see their environment from
        // opposite sides.
        if (views_ > 1) {
            const std::size_t view =
                static_cast<std::size_t>(kSize) * kSize * 3;
            const std::size_t opposite =
                static_cast<std::size_t>(views_ / 2) * view;
            double largest = 0.0;
            for (std::size_t i = 0; i < view; ++i) {
                largest =
                    std::max(largest, std::abs(values[i] -
                                               values[opposite + i]));
            }
            Check(largest > 16.0,
                  "views from opposite headings should differ");
        }
    }

  private:
    /// --views, --bands, --minimal and --sweep among the demo's own
    /// arguments.
    void ParseOptions(const std::vector<std::string>& extra,
                      bool& sweep) {
        for (std::size_t i = 0; i < extra.size(); ++i) {
            const auto value = [&]() -> const std::string& {
                if (i + 1 >= extra.size()) {
                    throw std::invalid_argument(extra[i] +
                                                " needs a value");
                }
                return extra[++i];
            };
            if (extra[i] == "--views") {
                views_ = std::stoi(value());
            } else if (extra[i] == "--bands") {
                bands_ = std::stoi(value());
            } else if (extra[i] == "--minimal") {
                minimal_ = value();
                if (std::find(kMinimalModes.begin(),
                              kMinimalModes.end(),
                              minimal_) == kMinimalModes.end()) {
                    throw std::invalid_argument(
                        "--minimal is off, none, constantDiffuse or "
                        "texturedDiffuse, not " +
                        minimal_);
                }
            } else if (extra[i] == "--sweep") {
                sweep = true;
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

    /// The last step's views, uint8 (views, kSize, kSize, 4) on the
    /// host, waiting for the render.
    Tensor Images() {
        return RequireCuda(GetRenderer())
            .Download(observations_.at("LdrColor"));
    }

    /// Prints the time per step since the warm-up, once the last step
    /// has finished; before the output and the test, so that their
    /// downloads are not counted.
    void Report() {
        if (observations_.empty() || timedSteps_ == 0) {
            return;
        }
        Images();  // the last step has finished
        const double perStep =
            std::chrono::duration<double>(Clock::now() - *timerStart_)
                .count() /
            timedSteps_;
        std::string note = minimal_ == "off"
                               ? std::string()
                               : ", minimal " + minimal_;
        note += GetViewer() == nullptr ? ""
                                       : " (with the viewer drawing)";
        PrintRate(views_, bands_, perStep, note);
        timedSteps_ = 0;
    }

    int views_ = 1024;
    int bands_ = 3;
    std::string minimal_ = "off";
    std::vector<Tensor> poses_;
    std::map<std::string, Tensor> observations_;
    std::optional<Clock::time_point> timerStart_;
    int timedSteps_ = 0;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    BatchCameraDemo, "batched", "cameras",
    "A view per environment rendered in batches, timed per step.");
