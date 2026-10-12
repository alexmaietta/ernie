// SPDX-License-Identifier: BSD-3-Clause
// Laser beams in the night air: a green and a blue beam and a red fan
// rise from the ground, shown only by the light the air scatters, the
// haze lighting them more and a balloon in the green beam stopping
// it.
//
// References:
// - C. F. Bohren and D. R. Huffman, "Absorption and Scattering of
//   Light by Small Particles", Wiley, 1983 (a beam seen side-on by
//   the light its path scatters: the gas's Rayleigh scattering, the
//   haze's or dust's forward peak).

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
constexpr glm::dvec3 kRig{0.0, 0.0, 1.5};  // m, the lasers' aperture
// The haze's (Earth's) or dust's (Mars's) optical depth.
constexpr double kHaze = 0.1;

/// The lasers' rig turned to `elevation` above the horizon and
/// `azimuth` from +y towards +x, in degrees.
glm::dmat4 Aim(double elevation, double azimuth) {
    const double e = elevation * std::numbers::pi / 180.0;
    const double a = azimuth * std::numbers::pi / 180.0;
    return LookAtTransform(
        kRig,
        kRig + glm::dvec3(std::cos(e) * std::sin(a),
                          std::cos(e) * std::cos(a), std::sin(e)),
        {0.0, 0.0, 1.0});
}

class NightAirDemo : public app::Demo {
  public:
    NightAirDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Material::Define(scene, "/World/Looks/Ground")
            .SetAlbedo(glm::dvec3(0.1));
        Material::Define(scene, "/World/Looks/Balloon")
            .SetAlbedo(glm::dvec3(0.7));
        GroundPlane::Define(scene, "/World/Ground")
            .SetMaterialBinding("/World/Looks/Ground");
        Atmosphere::Define(scene, "/World/Air")
            .SetHazeOpticalDepth(kHaze);
        LaserLight::Define(scene, "/World/Green")
            .SetWavelength(532.0)
            .SetPower(5.0)
            .SetWaistRadius(glm::dvec2(0.003))
            .SetTransform(Aim(50.0, 0.0));
        LaserLight::Define(scene, "/World/Blue")
            .SetWavelength(450.0)
            .SetPower(3.0)
            .SetWaistRadius(glm::dvec2(0.003))
            .SetTransform(Aim(70.0, -30.0));
        LaserLight::Define(scene, "/World/Red")
            .SetWavelength(638.0)
            .SetPower(3.0)
            .SetWaistRadius(glm::dvec2(0.003))
            .SetPattern("line")
            .SetFanAngle(40.0)
            .SetTransform(Aim(30.0, 25.0));
        // A balloon in the green beam, 15 m along it.
        const double e = 50.0 * std::numbers::pi / 180.0;
        Sphere::Define(scene, "/World/Balloon")
            .SetRadius(1.0)
            .SetMaterialBinding("/World/Looks/Balloon");
        SetXform(
            scene, "/World/Balloon",
            {.translate = kRig + glm::dvec3(0.0, 15.0 * std::cos(e),
                                            15.0 * std::sin(e))});
        // A long, wide-open exposure, as for the night sky.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(14.0)
            .SetSensorNoise(false)
            .SetExposureTime(1.0)
            .SetExposureFStop(1.4)
            .SetExposureIso(6400.0)
            .SetClippingRange({0.05, 1e5})
            .SetTransform(LookAtTransform({-20.0, -12.0, 1.7},
                                          {0.0, 12.0, 14.0},
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
        Widgets().Combo(
            "Air", {"Earth", "Mars"}, 0, [this](std::size_t index) {
                Atmosphere(GetScene(), "/World/Air")
                    .SetPreset(std::string(
                        index == 1 ? tokens::kMars : tokens::kEarth));
            });
        Widgets().Slider("Haze or dust optical depth", 0.0, 2.0,
                         kHaze,
                         [this](double depth) { Thicken(depth); });
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

    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, GetRenderer().Render(GetScene()));
    }

    void TestFinal() override {
        const std::vector<double> clear = Capture(true);
        Thicken(1.0);
        const std::vector<double> hazy = Capture(true);
        Thicken(kHaze);
        const std::vector<double> dark = Capture(false);
        // The beams by the brightest of their pixels, past the few
        // the green spot lights on the balloon.
        const double threshold = 0.05 * Percentile(clear, 99.9);
        std::size_t visible = 0;
        for (const double value : clear) {
            visible += value > threshold ? 1 : 0;
        }
        const double share = static_cast<double>(visible) /
                             static_cast<double>(clear.size());
        const double ratio =
            Percentile(hazy, 99.0) / Percentile(clear, 99.0);
        const double darkest =
            *std::max_element(dark.begin(), dark.end());
        std::printf(
            "the beams light %.1f%% of the pixels in clear air; haze "
            "of optical depth 1 scatters %.1f times their light; "
            "with "
            "the lasers off, the brightest pixel is %.3g\n",
            100.0 * share, ratio, darkest);
        Check(share > 0.003, "the beams show in the air");
        Check(ratio > 1.5, "haze shows them more");
        Check(darkest == 0.0, "nothing else lights the night");
    }

  private:
    /// The value `percent` of the way up `values` sorted.
    static double Percentile(std::vector<double> values,
                             double percent) {
        const auto at = static_cast<std::ptrdiff_t>(
            percent / 100.0 * static_cast<double>(values.size() - 1));
        std::nth_element(values.begin(), values.begin() + at,
                         values.end());
        return values[static_cast<std::size_t>(at)];
    }

    void Thicken(double depth) {
        Atmosphere(GetScene(), "/World/Air")
            .SetHazeOpticalDepth(depth)
            .SetDustOpticalDepth(depth);
    }

    /// The radiance summed over the bands, with the lasers on or off.
    std::vector<double> Capture(bool lasers) {
        Scene& scene = GetScene();
        for (const char* path :
             {"/World/Green", "/World/Blue", "/World/Red"}) {
            scene.Set(path, tokens::kVisibility,
                      std::string(lasers ? tokens::kInherited
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

ERNIE_REGISTER_DEMO(NightAirDemo, "lasers", "night_air",
                    "Laser beams in the night air.");
