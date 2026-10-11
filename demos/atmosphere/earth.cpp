// SPDX-License-Identifier: BSD-3-Clause
// Earth's atmosphere from the ground and from orbit: a blue noon sky,
// a red sunset, and the planet's limb from the International Space
// Station's 400 km, under the Sun's 1361 W/m^2 as a 5772 K black
// body.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kPi = std::numbers::pi;

/// A camera and the sun's elevation, with an exposure for its light:
/// the "sunny 16" rule (f/16 at 1/ISO s) in daylight, opened up at
/// sunset.
struct View {
    const char* name;
    glm::dvec3 eye;
    glm::dvec3 target;
    glm::dvec3 up;
    double
        sunElevation;  ///< Degrees above the horizon at the origin.
    double fStop;
};

const std::array<View, 4> kViews = {{
    {"Ground, noon",
     {0.0, 0.0, 2.0},
     {-1000.0, 0.0, 250.0},
     {0.0, 0.0, 1.0},
     60.0,
     16.0},
    {"Ground, sunset",
     {0.0, 0.0, 2.0},
     {1000.0, 0.0, 60.0},
     {0.0, 0.0, 1.0},
     2.0,
     5.6},
    {"ISS, 400 km",
     {0.0, 0.0, 400e3},
     {0.0, 1e6, 0.0},
     {0.0, 0.0, 1.0},
     30.0,
     16.0},
    // Over the night side, looking towards the sun at the limb, 19.8
    // degrees below the horizontal from 400 km, where the sun sets.
    {"ISS, limb at sunset",
     {0.0, 0.0, 400e3},
     {1e6, 0.0, 40e3},
     {0.0, 0.0, 1.0},
     -19.0,
     4.0},
}};

class EarthDemo : public app::Demo {
  public:
    EarthDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // Earth's air, ozone and haze around a planet whose surface
        // is the world's z = 0, the ground beyond any geometry its
        // own.
        Atmosphere::Define(scene, "/World/Air")
            .SetPreset(std::string(tokens::kEarth));
        // The Sun above the atmosphere.
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kBlackbody))
            .SetColorTemperature(5772.0)
            .SetAngle(0.53);
        // Grey spheres on the ground, lit by the sun through the air.
        for (int i = 0; i < 3; ++i) {
            const Path path("/World/Rock" + std::to_string(i));
            Sphere::Define(scene, path)
                .SetRadius(1.0)
                .SetDisplayColor(glm::dvec3(0.5));
            SetXform(scene, path,
                     {.translate = glm::dvec3(i == 0 ? -25.0 : 30.0,
                                              -6.0 + 6.0 * i, 1.0)});
        }
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(14.0)
            .SetSensorNoise(false)
            .SetExposureTime(1.0 / 125.0)
            .SetExposureIso(125.0);
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({480, 320})
            .SetAmbient(0.0);
        Show(0);

        std::vector<std::string> names;
        for (const View& view : kViews) {
            names.emplace_back(view.name);
        }
        Widgets().Combo("View", std::move(names), 0,
                        [this](std::size_t index) {
                            Show(static_cast<int>(index));
                        });
        Widgets().Slider("Sun elevation", -20.0, 90.0,
                         kViews[0].sunElevation,
                         [this](double degrees) { SunAt(degrees); });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    /// The four views side by side.
    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, demos::SideBySide(RenderAll()));
    }

    void TestFinal() override {
        const std::vector<Tensor> images = RenderAll();
        std::array<glm::dvec3, kViews.size()> mean{};
        for (std::size_t i = 0; i < images.size(); ++i) {
            mean[i] = demos::MeanLinearRgb(images[i]);
            std::printf("%s: mean linear RGB %.3f %.3f %.3f\n",
                        kViews[i].name, mean[i].r, mean[i].g,
                        mean[i].b);
        }
        // Rayleigh scattering: a blue sky at noon, and a red sun low
        // through the long path at sunset.
        Check(mean[0].b > 1.2 * mean[0].r,
              "the noon sky should be blue");
        Check(mean[1].r / mean[1].b > 2.0 * mean[0].r / mean[0].b,
              "the sunset should be redder than noon");
        Check(mean[2].r + mean[2].g + mean[2].b > 0.05,
              "the planet should be lit from orbit");
    }

  private:
    void Show(int index) {
        view_ = index;
        const View& view = kViews[static_cast<std::size_t>(index)];
        Scene& scene = GetScene();
        Camera(scene, "/World/Camera")
            .SetExposureFStop(view.fStop)
            .SetTransform(
                LookAtTransform(view.eye, view.target, view.up));
        SunAt(view.sunElevation);
    }

    void SunAt(double degrees) {
        const double e = degrees * kPi / 180.0;
        DistantLight(GetScene(), "/World/Sun")
            .SetTransform(DistantLightTransform(
                {std::cos(e), 0.0, std::sin(e)}));
    }

    /// The scene from each view in turn, RGBA uint8; leaves the view
    /// as it was.
    std::vector<Tensor> RenderAll() {
        const int current = view_;
        std::vector<Tensor> images;
        for (int i = 0; i < static_cast<int>(kViews.size()); ++i) {
            Show(i);
            images.push_back(GetRenderer().Render(GetScene()));
        }
        Show(current);
        return images;
    }

    int view_ = 0;
};

}  // namespace

ERNIE_REGISTER_DEMO(EarthDemo, "atmosphere", "earth",
                    "Earth's sky from the ground at noon and sunset, "
                    "and from the ISS.");
