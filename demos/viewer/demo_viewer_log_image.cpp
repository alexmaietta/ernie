// SPDX-License-Identifier: BSD-3-Clause
// An onboard camera circling a few objects, its colour (kept on the
// GPU) and depth logged each step as live images (LogImage).

#include "ernie/app/demo.h"
#include "ernie/device/compute_interop.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr int kCols = 160;  // the onboard camera's columns and rows
constexpr int kRows = 120;
constexpr double kNear =
    2.0;  // metres the depth image spans, bright to dark
constexpr double kFar = 14.0;
const Path kProduct("/Render/Onboard");

// Things around the origin: cube or sphere, colour, x, y.
struct Thing {
    bool cube;
    glm::dvec3 colour;
    double x;
    double y;
};
const std::array<Thing, 5> kThings = {{
    {true, {0.85, 0.3, 0.25}, 2.0, 0.0},
    {false, {0.25, 0.6, 0.9}, -1.5, 1.5},
    {true, {0.95, 0.8, 0.2}, -1.0, -2.0},
    {false, {0.4, 0.85, 0.4}, 0.5, 2.5},
    {true, {0.7, 0.4, 0.9}, -3.0, -0.5},
}};

class LogImageDemo : public app::Demo {
  public:
    LogImageDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 60;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(20.0)
            .SetTransform(LookAtTransform(
                {0.0, -11.0, 7.0}, {0.0, 0.0, 0.5}, {0.0, 0.0, 1.0}));
        GroundPlane::Define(scene, "/World/Ground")
            .SetChecker(true)
            .SetCheckerSize(1.0);
        DistantLight::Define(scene, "/World/Sun")
            .SetTransform(DistantLightTransform({0.4, -0.6, 0.7}));
        for (std::size_t i = 0; i < kThings.size(); ++i) {
            const Path thing("/World/Thing" + std::to_string(i));
            if (kThings[i].cube) {
                Cube::Define(scene, thing)
                    .SetDisplayColor(kThings[i].colour);
            } else {
                Sphere::Define(scene, thing)
                    .SetDisplayColor(kThings[i].colour);
            }
            SetXform(scene, thing,
                     {.translate = glm::dvec3(kThings[i].x,
                                              kThings[i].y, 0.5)});
        }
        Camera::Define(scene, "/World/Onboard").SetFocalLength(14.0);
        // The onboard camera's product: its colour and its distances.
        RenderProduct product = RenderProduct::Define(scene, kProduct)
                                    .SetCamera(Path("/World/Onboard"))
                                    .SetResolution({kCols, kRows});
        const Path colour = kProduct.Child("Vars").Child("LdrColor");
        const Path depth =
            kProduct.Child("Vars").Child("DistanceToCamera");
        RenderVar::Define(scene, colour).SetSourceName("LdrColor");
        RenderVar::Define(scene, depth)
            .SetSourceName("DistanceToCamera");
        product.SetOrderedVars({colour, depth});
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({640, 360});
        Widgets().Slider("Orbit speed", 0.0, 2.0, speed_,
                         [this](double value) { speed_ = value; });
    }

    void Step() override {
        // The onboard camera circles the things, looking in.
        const double angle = speed_ * time;
        Camera(GetScene(), "/World/Onboard")
            .SetTransform(LookAtTransform(
                {6.0 * std::cos(angle), 6.0 * std::sin(angle), 1.5},
                {0.0, 0.0, 0.5}, {0.0, 0.0, 1.0}));
        const Path products[] = {kProduct};
        RenderProductSetOutputs outputs =
            GetRenderer().Step(GetScene(), products);
        RenderProductOutput& output = outputs.at(kProduct);
        // Colour: on the GPU with CUDA interop, never reaching the
        // host.
        const bool gpu =
            GetRenderer().Delegate().Compute() != nullptr;
        LogImage("onboard/colour",
                 output.vars.at("LdrColor")
                     .Map(gpu ? MapDevice::Gpu : MapDevice::Cpu)
                     .tensor);
        // Depth: on the host, scaled to grey.
        const Tensor metres = output.vars.at("DistanceToCamera")
                                  .Map(MapDevice::Cpu)
                                  .tensor;
        const auto* distance =
            static_cast<const float*>(metres.Data());
        std::vector<float> grey(static_cast<std::size_t>(kRows) *
                                kCols);
        for (std::size_t i = 0; i < grey.size(); ++i) {
            const double d = std::isfinite(distance[i])
                                 ? static_cast<double>(distance[i])
                                 : kFar;
            grey[i] = static_cast<float>(
                1.0 -
                std::clamp((d - kNear) / (kFar - kNear), 0.0, 1.0));
        }
        LogImage("onboard/depth",
                 Tensor::Copy<float>(grey, {kRows, kCols}));
    }

    void TestFinal() override {
        const Tensor depth = LoggedImage("onboard/depth");
        Check(depth.Shape() ==
                  std::vector<std::int64_t>{kRows, kCols},
              "a depth image");
        const auto* grey = static_cast<const float*>(depth.Data());
        const auto [darkest, brightest] =
            std::minmax_element(grey, grey + kRows * kCols);
        Check(*brightest > 0.3f && *darkest < 0.1f,
              "near things bright, the sky dark");
        Tensor colour = LoggedImage("onboard/colour");
        if (colour.Location().kind != DeviceKind::Cpu) {
            colour =
                GetRenderer().Delegate().Compute()->Download(colour);
        }
        Check(colour.Shape() ==
                  std::vector<std::int64_t>{kRows, kCols, 4},
              "a colour image");
        Check(demos::ColourDeviation(colour) > 5.0,
              "the onboard colour image should not be flat");
    }

  private:
    double speed_ = 0.4;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    LogImageDemo, "viewer", "log image",
    "An onboard camera's colour and depth logged as live images.");
