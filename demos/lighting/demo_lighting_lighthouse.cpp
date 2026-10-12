// SPDX-License-Identifier: BSD-3-Clause
// A lighthouse at night: its two beams sweep round through the hazy
// air, seen from the shore by the light the air scatters out of them,
// the lantern's iron bars cutting dark shafts through each; the
// lantern glass glows all round, the keeper's cottage windows throw
// their light out into the haze and the pier's lamps pour cones down
// onto it.
//
// References:
// - C. Kulla and M. Fajardo, "Importance Sampling Techniques for Path
//   Tracing in Participating Media", Computer Graphics Forum
//   31(4):1519-1528, 2012 (the beams' light along each pixel's ray,
//   sampled equiangularly from the lamp).

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/stage_schemas.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr glm::ivec2 kResolution{320, 200};
constexpr glm::dvec3 kLamp{0.0, 0.0, 24.0};  // m, the lantern
constexpr double kHaze = 0.5;  // the haze's optical depth at 550 nm
constexpr double kPeriod = 10.0;  // s, a turn of the beacon
// A lighthouse's beam: about 10^6 candela in a cone a few degrees
// across, from a halogen lamp behind a Fresnel lens.
constexpr double kLumens = 2e4;
constexpr double kCone = 3.0;  // degrees, the beam's half angle
constexpr glm::dvec2 kCottage{22.0,
                              -28.0};  // m, the keeper's cottage
constexpr int kPierLamps =
    4;  // along the shore in front of the tower

/// The beacon's transform turned `angle` radians about the tower.
glm::dmat4 Beam(double angle) {
    return LookAtTransform(
        kLamp,
        kLamp + glm::dvec3(std::cos(angle), std::sin(angle), 0.02),
        {0.0, 0.0, 1.0});
}

class LighthouseDemo : public app::Demo {
  public:
    LighthouseDemo(app::Viewer* viewer,
                   const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 120;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Material::Define(scene, "/World/Looks/Shore")
            .SetAlbedo(glm::dvec3(0.08));
        Material::Define(scene, "/World/Looks/Tower")
            .SetAlbedo(glm::dvec3(0.8));
        Material::Define(scene, "/World/Looks/Iron")
            .SetAlbedo(glm::dvec3(0.05));
        GroundPlane::Define(scene, "/World/Shore")
            .SetMaterialBinding("/World/Looks/Shore");
        Atmosphere::Define(scene, "/World/Air")
            .SetHazeOpticalDepth(kHaze);
        Cylinder::Define(scene, "/World/Tower")
            .SetRadius(2.5)
            .SetHeight(22.0)
            .SetMaterialBinding("/World/Looks/Tower");
        SetXform(scene, "/World/Tower",
                 {.translate = glm::dvec3(0.0, 0.0, 11.0)});
        // The lantern's bars, which cut shafts through the beams.
        for (int i = 0; i < 8; ++i) {
            const double a = 2.0 * std::numbers::pi * (i + 0.5) / 8.0;
            const Path path("/World/Bar" + std::to_string(i));
            Cylinder::Define(scene, path)
                .SetRadius(0.1)
                .SetHeight(3.0)
                .SetMaterialBinding("/World/Looks/Iron");
            SetXform(
                scene, path,
                {.translate = glm::dvec3(1.4 * std::cos(a),
                                         1.4 * std::sin(a), 24.0)});
        }
        for (const char* path : {"/World/Beam", "/World/Back"}) {
            SphereLight::Define(scene, path)
                .SetRadius(0.3)
                .SetShapingConeAngle(kCone)
                .SetShapingConeSoftness(0.3)
                .SetPhotometricPower(kLumens)
                .SetEnableColorTemperature(true)
                .SetColorTemperature(3200.0);
        }
        lamps_ = {"/World/Beam", "/World/Back", "/World/Glow"};
        // The lantern glass's own glow, all round (a cylinder light,
        // its axis turned upright), which lights the tower's top and
        // haloes the lamp.
        CylinderLight::Define(scene, "/World/Glow")
            .SetRadius(1.2)
            .SetLength(2.5)
            .SetPhotometricPower(3000.0)
            .SetEnableColorTemperature(true)
            .SetColorTemperature(3200.0);
        SetXform(scene, "/World/Glow",
                 {.translate = kLamp,
                  .rotate = glm::dvec3(0.0, -90.0, 0.0)});
        // The keeper's cottage, its two lit windows (rect lights)
        // facing the shore.
        Material::Define(scene, "/World/Looks/Cottage")
            .SetAlbedo(glm::dvec3(0.5));
        Cube::Define(scene, "/World/Cottage")
            .SetSize(1.0)
            .SetMaterialBinding("/World/Looks/Cottage");
        SetXform(scene, "/World/Cottage",
                 {.translate = glm::dvec3(kCottage, 2.5),
                  .scale = glm::dvec3(8.0, 5.0, 5.0)});
        for (int i = 0; i < 2; ++i) {
            const glm::dvec3 window(kCottage.x +
                                        (i == 0 ? -2.0 : 2.0),
                                    kCottage.y - 2.55, 2.5);
            const Path path("/World/Window" + std::to_string(i));
            RectLight::Define(scene, path)
                .SetWidth(1.0)
                .SetHeight(1.2)
                .SetPhotometricPower(1200.0)
                .SetEnableColorTemperature(true)
                .SetColorTemperature(2700.0)
                .SetTransform(LookAtTransform(
                    window, window - glm::dvec3(0.0, 1.0, 0.0),
                    {0.0, 0.0, 1.0}));
            lamps_.push_back(path);
        }
        // The pier's lamps (disk lights) on their posts, facing down.
        for (int i = 0; i < kPierLamps; ++i) {
            const glm::dvec2 at(-48.0 + 12.0 * i, -48.0);
            const Path post("/World/Post" + std::to_string(i));
            Cylinder::Define(scene, post)
                .SetRadius(0.08)
                .SetHeight(4.0)
                .SetMaterialBinding("/World/Looks/Iron");
            SetXform(scene, post, {.translate = glm::dvec3(at, 2.0)});
            const Path lamp("/World/PierLamp" + std::to_string(i));
            DiskLight::Define(scene, lamp)
                .SetRadius(0.2)
                .SetPhotometricPower(1500.0)
                .SetEnableColorTemperature(true)
                .SetColorTemperature(4000.0)
                .SetTransform(LookAtTransform(glm::dvec3(at, 4.1),
                                              glm::dvec3(at, 0.0),
                                              {0.0, 1.0, 0.0}));
            lamps_.push_back(lamp);
        }
        Turn(0.0);
        // A long exposure, as for the night sky.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(18.0)
            .SetSensorNoise(false)
            .SetExposureTime(0.5)
            .SetExposureFStop(2.0)
            .SetExposureIso(3200.0)
            .SetClippingRange({0.05, 1e5})
            .SetTransform(LookAtTransform({-70.0, -100.0, 2.0},
                                          {0.0, 0.0, 26.0},
                                          {0.0, 0.0, 1.0}));
        const Path var("/Render/Camera/SpectralRadiance");
        RenderVar::Define(scene, var)
            .SetSourceName("SpectralRadiance");
        RenderProduct::Define(scene, "/Render/Camera")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(kResolution)
            .SetOrderedVars({var});
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(kResolution)
            .SetAmbient(0.0);
        Widgets().Slider("Haze optical depth", 0.0, 3.0, kHaze,
                         [this](double depth) { Thicken(depth); });
        Widgets().Slider("Turns per minute", 0.0, 30.0,
                         60.0 / kPeriod, [this](double turns) {
                             // From where the beam points now, not a
                             // jump.
                             speed_ = 2.0 * std::numbers::pi * turns /
                                      60.0;
                             phase_ = angle_ - speed_ * Time();
                         });
        // A turn of the beacon is the timeline's range, which loops.
        Stage(scene)
            .SetStartTimeCode(0.0)
            .SetEndTimeCode(kPeriod *
                            Stage(scene).GetTimeCodesPerSecond())
            .SetTimeCode(0.0);
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            viewer->SetLighting(app::Lighting::Stage);
            app::ViewportDisplay display =
                viewer->GetViewportDisplay();
            display.lights = false;
            display.grid = false;
            viewer->SetViewportDisplay(display);
            // The timeline turns the beacon: play, scrub or step it.
            viewer->OnFrame(
                [this](app::Viewer&, const app::FrameInfo&) {
                    TurnAt(Time());
                });
            viewer->GetTimeline().Play();
        }
    }

    void Step() override {
        if (GetViewer() != nullptr) {
            return;
        }
        // Without a viewer, the demo plays the timeline itself.
        Stage stage(GetScene());
        const double length =
            stage.GetEndTimeCode() - stage.GetStartTimeCode();
        stage.SetTimeCode(
            std::fmod(stage.GetTimeCode() +
                          frameDt * stage.GetTimeCodesPerSecond(),
                      length));
        TurnAt(Time());
    }

    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, GetRenderer().Render(GetScene()));
    }

    void TestFinal() override {
        constexpr double kDegree = std::numbers::pi / 180.0;
        // The beam swinging past the camera's side, a quarter turn
        // on.
        Turn(200.0 * kDegree);
        const std::vector<double> side = Capture(true);
        Turn(290.0 * kDegree);
        const std::vector<double> turned = Capture(true);
        Thicken(2.0 * kHaze);
        const std::vector<double> hazier = Capture(true);
        Thicken(kHaze);
        const std::vector<double> dark = Capture(false);
        // The beams in the sky above the horizon, the frame's top
        // half.
        const std::size_t sky = static_cast<std::size_t>(
            kResolution.x * kResolution.y / 2);
        const double brightest =
            *std::max_element(side.begin(), side.begin() + sky);
        std::size_t lit = 0;
        double sideSum = 0.0;
        double turnedSky = 0.0;
        double hazierSky = 0.0;
        double moved = 0.0;
        for (std::size_t i = 0; i < sky; ++i) {
            sideSum += side[i];
            moved += std::abs(turned[i] - side[i]);
            lit += side[i] > 0.02 * brightest ? 1 : 0;
            turnedSky += turned[i];
            hazierSky += hazier[i];
        }
        const double share =
            static_cast<double>(lit) / static_cast<double>(sky);
        const double ratio = hazierSky / turnedSky;
        const double darkest =
            *std::max_element(dark.begin(), dark.end());
        std::printf("the beams light %.1f%% of the sky; a quarter "
                    "turn changes "
                    "%.0f%% of its light; twice the haze scatters "
                    "%.2f times "
                    "as much; dark, the brightest pixel is %.3g\n",
                    100.0 * share, 100.0 * moved / sideSum, ratio,
                    darkest);
        Check(share > 0.02, "the beams show in the air");
        Check(moved > 0.5 * sideSum, "they sweep round");
        Check(ratio > 1.2, "haze shows them more");
        Check(darkest == 0.0, "nothing else lights the night");
    }

  private:
    std::vector<Path>
        lamps_;  ///< Every light, which `Capture` turns off
    double angle_ = 0.0;
    double speed_ = 2.0 * std::numbers::pi / kPeriod;
    double phase_ = 0.0;  ///< The beam's angle at time 0.

    /// Seconds into the timeline.
    double Time() {
        return Stage(GetScene()).GetTime();
    }

    /// The beacon as it is `t` seconds into the timeline.
    void TurnAt(double t) {
        Turn(phase_ + speed_ * t);
    }

    void Turn(double angle) {
        angle_ = angle;
        SphereLight(GetScene(), "/World/Beam")
            .SetTransform(Beam(angle));
        SphereLight(GetScene(), "/World/Back")
            .SetTransform(Beam(angle + std::numbers::pi));
    }

    void Thicken(double depth) {
        Atmosphere(GetScene(), "/World/Air")
            .SetHazeOpticalDepth(depth);
    }

    /// The radiance summed over the bands, with the beacon lit or
    /// dark.
    std::vector<double> Capture(bool lit) {
        Scene& scene = GetScene();
        for (const Path& path : lamps_) {
            scene.Set(path, tokens::kVisibility,
                      std::string(lit ? tokens::kInherited
                                      : tokens::kInvisible));
        }
        const std::vector<Path> products{Path("/Render/Camera")};
        RenderProductSetOutputs outputs =
            GetRenderer().Step(scene, products);
        const Tensor bands = outputs.at(Path("/Render/Camera"))
                                 .vars.at("SpectralRadiance")
                                 .Map(MapDevice::Cpu)
                                 .tensor.Contiguous();
        const auto count = static_cast<std::size_t>(bands.Shape()[0]);
        const auto pixels =
            static_cast<std::size_t>(kResolution.x * kResolution.y);
        std::vector<double> radiance(pixels, 0.0);
        const auto* values = bands.As<float>();
        for (std::size_t b = 0; b < count; ++b) {
            for (std::size_t i = 0; i < pixels; ++i) {
                radiance[i] +=
                    static_cast<double>(values[b * pixels + i]);
            }
        }
        return radiance;
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(LighthouseDemo, "lighting", "lighthouse",
                    "A lighthouse's beams sweeping the night air.");
