// SPDX-License-Identifier: BSD-3-Clause
// A rover driving at night on Mars: its LED headlights and mast
// floodlight light the rocks ahead and show as beams in the dusty
// air, the dust scattering their light forwards, towards the chase
// camera behind; a dust storm thickens them.
//
// References:
// - M. T. Lemmon et al., "Dust aerosol, clouds, and the atmospheric
//   optical depth record over 5 Mars years of the Mars Exploration
//   Rover mission", Icarus 251:96-111, 2015 (the dust's optical
//   depth, about 0.3 to 0.9 in clear seasons and several in storms).

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr glm::ivec2 kResolution{320, 200};
constexpr double kDust = 0.6;   // the dust's optical depth at 880 nm
constexpr double kSpeed = 0.1;  // m/s, about a rover's top speed

/// A rock ahead of the rover: its place (m) and radius.
struct Rock {
    glm::dvec2 at;
    double radius;
};

constexpr std::array<Rock, 6> kRocks{{{{-2.0, 8.0}, 0.5},
                                      {{1.5, 12.0}, 0.8},
                                      {{-0.5, 17.0}, 0.4},
                                      {{3.0, 20.0}, 1.1},
                                      {{-3.5, 24.0}, 0.9},
                                      {{0.8, 28.0}, 0.6}}};

constexpr const char* kLamps[] = {"/World/Rover/HeadLeft",
                                  "/World/Rover/HeadRight",
                                  "/World/Rover/Flood"};

class MarsRoverDemo : public app::Demo {
  public:
    MarsRoverDemo(app::Viewer* viewer,
                  const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 120;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // Mars's soil and rocks, reddish, about its mean albedo 0.25.
        Material::Define(scene, "/World/Looks/Soil")
            .SetAlbedo(glm::dvec3(0.33, 0.22, 0.14));
        Material::Define(scene, "/World/Looks/Rock")
            .SetAlbedo(glm::dvec3(0.3, 0.2, 0.14));
        Material::Define(scene, "/World/Looks/Body")
            .SetAlbedo(glm::dvec3(0.7));
        Material::Define(scene, "/World/Looks/Wheel")
            .SetAlbedo(glm::dvec3(0.15));
        GroundPlane::Define(scene, "/World/Ground")
            .SetMaterialBinding("/World/Looks/Soil");
        Atmosphere::Define(scene, "/World/Air")
            .SetPreset(std::string(tokens::kMars))
            .SetDustOpticalDepth(kDust);
        for (std::size_t i = 0; i < kRocks.size(); ++i) {
            const Path path("/World/Rock" + std::to_string(i));
            Sphere::Define(scene, path)
                .SetRadius(kRocks[i].radius)
                .SetMaterialBinding("/World/Looks/Rock");
            SetXform(scene, path,
                     {.translate = glm::dvec3(
                          kRocks[i].at, 0.4 * kRocks[i].radius)});
        }
        // The rover, which drives along +y: its body, six wheels and
        // mast, and its lights, all carried by its transform.
        Xform::Define(scene, "/World/Rover");
        Cube::Define(scene, "/World/Rover/Body")
            .SetSize(1.0)
            .SetMaterialBinding("/World/Looks/Body");
        SetXform(scene, "/World/Rover/Body",
                 {.translate = glm::dvec3(0.0, 0.0, 0.9),
                  .scale = glm::dvec3(1.6, 2.4, 0.5)});
        for (int i = 0; i < 6; ++i) {
            const Path path("/World/Rover/Wheel" + std::to_string(i));
            Cylinder::Define(scene, path)
                .SetRadius(0.26)
                .SetHeight(0.25)
                .SetAxis("X")
                .SetMaterialBinding("/World/Looks/Wheel");
            SetXform(scene, path,
                     {.translate =
                          glm::dvec3((i % 2 == 0 ? -1.0 : 1.0) * 0.95,
                                     -0.9 + 0.9 * (i / 2), 0.26)});
        }
        Cylinder::Define(scene, "/World/Rover/Mast")
            .SetRadius(0.05)
            .SetHeight(1.0)
            .SetMaterialBinding("/World/Looks/Wheel");
        SetXform(scene, "/World/Rover/Mast",
                 {.translate = glm::dvec3(0.5, 0.9, 1.6)});
        // Two LED headlights, cool white, a little apart and dipped.
        for (int i = 0; i < 2; ++i) {
            const double x = i == 0 ? -0.55 : 0.55;
            SphereLight::Define(scene, kLamps[i])
                .SetRadius(0.04)
                .SetShapingConeAngle(22.0)
                .SetShapingConeSoftness(0.4)
                .SetPhotometricPower(2500.0)
                .SetEnableColorTemperature(true)
                .SetColorTemperature(6000.0)
                .SetTransform(LookAtTransform({x, 1.22, 0.9},
                                              {2.0 * x, 12.0, 0.0},
                                              {0.0, 0.0, 1.0}));
        }
        // A floodlight on the mast, wider and looking down ahead.
        SphereLight::Define(scene, kLamps[2])
            .SetRadius(0.06)
            .SetShapingConeAngle(45.0)
            .SetShapingConeSoftness(0.5)
            .SetPhotometricPower(3000.0)
            .SetEnableColorTemperature(true)
            .SetColorTemperature(5000.0)
            .SetTransform(LookAtTransform(
                {0.5, 0.9, 2.15}, {0.5, 6.0, 0.0}, {0.0, 0.0, 1.0}));
        Drive(0.0);
        // A chase camera behind and above, a long exposure for the
        // night.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(16.0)
            .SetSensorNoise(false)
            .SetExposureTime(0.25)
            .SetExposureFStop(2.8)
            .SetExposureIso(400.0)
            .SetClippingRange({0.05, 1e4});
        Follow();
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
        Widgets().Slider("Dust optical depth", 0.0, 5.0, kDust,
                         [this](double depth) { Dust(depth); });
        Widgets().Slider("Speed (m/s)", 0.0, 1.0, kSpeed,
                         [this](double speed) { speed_ = speed; });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            viewer->SetLighting(app::Lighting::Stage);
            app::ViewportDisplay display =
                viewer->GetViewportDisplay();
            display.lights = false;
            display.grid = false;
            viewer->SetViewportDisplay(display);
        }
    }

    void Step() override {
        Drive(y_ + speed_ * frameDt);
        Follow();
    }

    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, GetRenderer().Render(GetScene()));
    }

    void TestFinal() override {
        const std::vector<double> lit = Capture(true);
        Dust(4.0 * kDust);
        const std::vector<double> storm = Capture(true);
        Scene& scene = GetScene();
        scene.Set("/World/Air", tokens::kVisibility,
                  std::string(tokens::kInvisible));
        const std::vector<double> airless = Capture(true);
        scene.Set("/World/Air", tokens::kVisibility,
                  std::string(tokens::kInherited));
        Dust(kDust);
        const std::vector<double> dark = Capture(false);
        // The beams: what the dusty air adds over the ground and
        // rocks.
        std::vector<double> beams(lit.size());
        double beamSum = 0.0;
        double stormSum = 0.0;
        for (std::size_t i = 0; i < lit.size(); ++i) {
            beams[i] = lit[i] - airless[i];
            beamSum += beams[i];
            stormSum += storm[i] - airless[i];
        }
        std::vector<double> sorted = beams;
        const auto at = static_cast<std::ptrdiff_t>(
            0.999 * static_cast<double>(sorted.size() - 1));
        std::nth_element(sorted.begin(), sorted.begin() + at,
                         sorted.end());
        const double threshold =
            0.05 * sorted[static_cast<std::size_t>(at)];
        std::size_t shown = 0;
        for (const double value : beams) {
            shown += value > threshold ? 1 : 0;
        }
        const double share = static_cast<double>(shown) /
                             static_cast<double>(beams.size());
        const double ratio = stormSum / beamSum;
        const double darkest =
            *std::max_element(dark.begin(), dark.end());
        std::printf("the beams light %.1f%% of the frame; a dust "
                    "storm four times "
                    "as thick scatters %.2f times as much; dark, the "
                    "brightest "
                    "pixel is %.3g\n",
                    100.0 * share, ratio, darkest);
        Check(share > 0.05, "the beams show in the dust");
        Check(ratio > 1.5, "the storm thickens them");
        Check(darkest == 0.0, "nothing else lights the night");
    }

  private:
    double y_ = 0.0;
    double speed_ = kSpeed;

    void Drive(double y) {
        y_ = y;
        SetXform(GetScene(), "/World/Rover",
                 {.translate = glm::dvec3(0.0, y, 0.0)});
    }

    /// The chase camera behind and above the rover.
    void Follow() {
        Camera(GetScene(), "/World/Camera")
            .SetTransform(LookAtTransform({-4.0, y_ - 9.0, 3.5},
                                          {0.0, y_ + 9.0, 0.5},
                                          {0.0, 0.0, 1.0}));
    }

    void Dust(double depth) {
        Atmosphere(GetScene(), "/World/Air")
            .SetDustOpticalDepth(depth);
    }

    /// The radiance summed over the bands, with the lights on or off.
    std::vector<double> Capture(bool lit) {
        Scene& scene = GetScene();
        for (const char* path : kLamps) {
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

ERNIE_REGISTER_DEMO(
    MarsRoverDemo, "lighting", "mars_rover",
    "A rover's headlights in the dusty Martian night.");
