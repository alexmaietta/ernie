// SPDX-License-Identifier: BSD-3-Clause
// Procedural cloud layers over the Earth, after three-geospatial's
// clouds: low and mid cumulus and cirrus carved from a weather map
// and Perlin-Worley noise at each sample (a CloudLayersField),
// drifting on the wind, lit through the atmosphere by a two-lobed
// phase function, the powder effect and multiple-scattering octaves.
// Seen from above, as over Fuji, or from below, as over Tokyo.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/volume_schemas.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kCoverage = 0.4;
constexpr double kSunElevation = 20.0;
constexpr double kSunAzimuth = 30.0;
constexpr double kWind = 15.0;

struct View {
    const char* name;
    glm::dvec3 eye;
    double pitch;  ///< Degrees, looking along -x.
};

constexpr std::array<View, 2> kViews{{
    {"Above the clouds", {0.0, 0.0, 3800.0}, -27.0},
    {"Below the clouds", {0.0, 0.0, 160.0}, 9.0},
}};

class CloudLayersDemo : public app::Demo {
  public:
    CloudLayersDemo(app::Viewer* viewer,
                    const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        demos::DefinePlanetSurface(scene, "/World/Earth", 6371.0088e3,
                                   glm::dvec3(0.08, 0.12, 0.06),
                                   "/World/Looks/Ground");
        Atmosphere::Define(scene, "/World/Air")
            .SetPreset(std::string(tokens::kEarth));
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(0.53)
            .SetShadowEnable(true);
        PlaceSun();

        // three-geospatial's default layers (low and mid cumulus,
        // cirrus) over 60 km, their density scales in 1/m.
        Volume::Define(scene, "/World/Clouds")
            .SetField("density", "/World/Clouds/Layers")
            .SetMaterialBinding("/World/Looks/Cloud");
        CloudLayersField::Define(scene, "/World/Clouds/Layers")
            .SetCoverage(kCoverage)
            .SetExtent(30000.0);
        Material::Define(scene, "/World/Looks/Cloud")
            .SetDensity(1.0)
            .SetVolumeAlbedo(glm::dvec3(1.0))
            .SetPhaseFunction(
                std::string(tokens::kDualHenyeyGreenstein))
            .SetAnisotropy(0.7)
            .SetSecondAnisotropy(-0.2)
            .SetLobeMix(0.5)
            .SetPowder(0.8)
            .SetPowderLength(150.0)
            .SetScatteringOctaves(8)
            .SetOctaveAttenuation(0.5)
            .SetOctaveExtinction(0.5)
            .SetOctavePhase(0.5);

        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(12.0)
            .SetSensorNoise(false)
            .SetExposureIso(100.0)
            .SetExposureFStop(11.0)
            .SetExposureTime(1.0 / 250.0);
        LookFrom(0);
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({640, 360})
            .SetAmbient(0.0);

        Widgets().Combo(
            "View", {kViews[0].name, kViews[1].name}, 0,
            [this](std::size_t index) { LookFrom(index); });
        Widgets().Slider(
            "Coverage", 0.0, 1.0, kCoverage,
            [this](double coverage) { Cover(coverage); });
        Widgets().Slider("Sun elevation", 1.0, 90.0, kSunElevation,
                         [this](double degrees) {
                             elevation_ = degrees;
                             PlaceSun();
                         });
        Widgets().Slider("Sun azimuth", 0.0, 360.0, kSunAzimuth,
                         [this](double degrees) {
                             azimuth_ = degrees;
                             PlaceSun();
                         });
        Widgets().Slider("Wind (m/s)", 0.0, 60.0, kWind,
                         [this](double speed) { wind_ = speed; });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    void Step() override {
        // The weather drifts downwind, the clouds with it.
        CloudLayersField(GetScene(), "/World/Clouds/Layers")
            .SetWeatherOffset(glm::dvec2(-wind_ * time, 0.0));
    }

    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, GetRenderer().Render(GetScene()));
    }

    void TestFinal() override {
        Scene& scene = GetScene();
        const glm::dvec3 cloudy =
            demos::MeanLinearRgb(GetRenderer().Render(scene));
        Cover(0.0);
        const glm::dvec3 clear =
            demos::MeanLinearRgb(GetRenderer().Render(scene));
        Cover(kCoverage);
        std::printf(
            "Mean linear RGB with clouds %.3f %.3f %.3f, clear "
            "%.3f %.3f %.3f\n",
            cloudy.r, cloudy.g, cloudy.b, clear.r, clear.g, clear.b);
        // Above the clouds, they whiten the view of the ground below.
        Check(glm::length(cloudy - clear) > 0.01, "the clouds show");
        Check(cloudy.r / cloudy.b > clear.r / clear.b,
              "the clouds are whiter than the ground and air");
    }

  private:
    void LookFrom(std::size_t index) {
        const View& view = kViews[index];
        const double p = view.pitch * std::numbers::pi / 180.0;
        const glm::dvec3 target =
            view.eye +
            1000.0 * glm::dvec3(-std::cos(p), 0.0, std::sin(p));
        Camera(GetScene(), "/World/Camera")
            .SetTransform(
                LookAtTransform(view.eye, target, {0.0, 0.0, 1.0}));
    }

    void Cover(double coverage) {
        CloudLayersField(GetScene(), "/World/Clouds/Layers")
            .SetCoverage(coverage);
    }

    void PlaceSun() {
        const double a = azimuth_ * std::numbers::pi / 180.0;
        const double e = elevation_ * std::numbers::pi / 180.0;
        DistantLight(GetScene(), "/World/Sun")
            .SetTransform(DistantLightTransform(
                {std::cos(a) * std::cos(e), std::sin(a) * std::cos(e),
                 std::sin(e)}));
    }

    double elevation_ = kSunElevation;
    double azimuth_ = kSunAzimuth;
    double wind_ = kWind;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    CloudLayersDemo, "volumes", "cloud_layers",
    "Procedural cloud layers over the Earth, seen from "
    "above or below: three-geospatial's clouds.");
