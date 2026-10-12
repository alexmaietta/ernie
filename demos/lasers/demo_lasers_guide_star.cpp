// SPDX-License-Identifier: BSD-3-Clause
// A sodium laser guide star: a 20 W beam tuned to sodium's 589 nm D2
// line rises to the mesosphere, where the sodium layer 92 km up glows
// back as an artificial star; seen from behind the laser, and
// through a telescope 30 m away that sees it drawn out along the
// beam.
//
// References:
// - R. Foy and A. Labeyrie, "Feasibility of adaptive telescope with
//   laser probe", Astronomy and Astrophysics 152:L29-L31, 1985 (the
//   sodium laser guide star).
// - D. A. Steck, "Sodium D Line Data" (the D2 line).
//   https://steck.us/alkalidata/

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
#include <numbers>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr glm::ivec2 kResolution{320, 240};
constexpr double kD2 = 589.158;  // nm, sodium's D2 line in vacuum
constexpr glm::dvec3 kLaunch{0.0, 0.0, 2.0};      // m, the laser
constexpr glm::dvec3 kTelescope{30.0, 0.0, 2.0};  // m, beside it
constexpr double kZenith = 10.0;  // degrees, the beam from the zenith
constexpr double kLayer = 92e3;   // m, the sodium layer's height

/// Where the beam meets the sodium layer.
glm::dvec3 Spot() {
    return kLaunch +
           glm::dvec3(
               0.0,
               kLayer * std::tan(kZenith * std::numbers::pi / 180.0),
               kLayer);
}

/// A camera's place, target, focal length (mm) and exposure.
struct View {
    const char* name;
    glm::dvec3 eye;
    glm::dvec3 target;
    double focal;
    double time;
    double fStop;
    double iso;
};

std::array<View, 2> Views() {
    return {
        {{"Behind the laser",
          {8.0, -30.0, 1.7},
          {0.0, 400.0, 1200.0},
          12.0,
          4.0,
          1.4,
          6400.0},
         {"Telescope", kTelescope, Spot(), 50e3, 1.0, 12.0, 1600.0}}};
}

class GuideStarDemo : public app::Demo {
  public:
    GuideStarDemo(app::Viewer* viewer,
                  const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Material::Define(scene, "/World/Looks/Ground")
            .SetAlbedo(glm::dvec3(0.1));
        GroundPlane::Define(scene, "/World/Ground")
            .SetMaterialBinding("/World/Looks/Ground");
        Atmosphere::Define(scene, "/World/Air");
        // A launch telescope's beam, 0.3 m across, its line a few
        // megahertz wide.
        LaserLight::Define(scene, "/World/Laser")
            .SetWavelength(kD2)
            .SetLinewidth(1e-5)
            .SetPower(20.0)
            .SetWaistRadius(glm::dvec2(0.15))
            .SetTransform(
                LookAtTransform(kLaunch, Spot(), {0.0, 1.0, 0.0}));
        Camera::Define(scene, "/World/Camera")
            .SetSensorNoise(false)
            .SetClippingRange({0.05, 1e6});
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
        Look(0);
        std::vector<std::string> names;
        for (const View& view : Views()) {
            names.emplace_back(view.name);
        }
        Widgets().Combo("View", std::move(names), 0,
                        [this](std::size_t index) { Look(index); });
        Widgets().Checkbox("Tuned to the D2 line", true,
                           [this](bool tuned) { Tune(tuned); });
        // About 2e13 to 8e13 atoms per square metre over the seasons.
        Widgets().Slider("Sodium column (1e13 / m^2)", 0.0, 10.0, 4.0,
                         [this](double column) {
                             Atmosphere(GetScene(), "/World/Air")
                                 .SetSodiumColumn(column * 1e13);
                         });
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
        Look(1);
        const std::vector<double> tuned = Capture();
        Tune(false);
        const std::vector<double> detuned = Capture();
        Tune(true);
        Look(0);
        // The star's pixels, and how far they reach along and across
        // the image.
        const double peak =
            *std::max_element(tuned.begin(), tuned.end());
        const double detunedPeak =
            *std::max_element(detuned.begin(), detuned.end());
        int top = kResolution.y;
        int bottom = -1;
        int left = kResolution.x;
        int right = -1;
        for (int y = 0; y < kResolution.y; ++y) {
            for (int x = 0; x < kResolution.x; ++x) {
                if (tuned[static_cast<std::size_t>(y * kResolution.x +
                                                   x)] > 0.1 * peak) {
                    top = std::min(top, y);
                    bottom = std::max(bottom, y);
                    left = std::min(left, x);
                    right = std::max(right, x);
                }
            }
        }
        const int rows = bottom - top + 1;
        const int cols = right - left + 1;
        std::printf(
            "the guide star peaks at %.3g W m^-2 sr^-1 nm^-1, "
            "%.3g times the detuned beam's; %d x %d pixels, "
            "drawn out along the beam\n",
            peak, peak / std::max(detunedPeak, 1e-30), rows, cols);
        Check(peak > 100.0 * detunedPeak, "sodium glows");
        Check(std::max(rows, cols) > 2 * std::min(rows, cols),
              "elongated");
    }

  private:
    void Look(std::size_t index) {
        const View view = Views()[index];
        Camera(GetScene(), "/World/Camera")
            .SetFocalLength(view.focal)
            .SetExposureTime(view.time)
            .SetExposureFStop(view.fStop)
            .SetExposureIso(view.iso)
            .SetTransform(LookAtTransform(view.eye, view.target,
                                          {0.0, 0.0, 1.0}));
    }

    void Tune(bool tuned) {
        // Two hundredths of a nanometre off, far past the line's
        // width.
        LaserLight(GetScene(), "/World/Laser")
            .SetWavelength(tuned ? kD2 : kD2 - 0.02);
    }

    /// The radiance summed over the bands.
    std::vector<double> Capture() {
        const std::vector<Path> products{Path("/Render/Camera")};
        RenderProductSetOutputs outputs =
            GetRenderer().Step(GetScene(), products);
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

ERNIE_REGISTER_DEMO(GuideStarDemo, "lasers", "guide_star",
                    "A sodium laser guide star.");
