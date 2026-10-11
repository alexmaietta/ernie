// SPDX-License-Identifier: BSD-3-Clause
// Mars's atmosphere from a rover and from orbit: the butterscotch
// sky of suspended dust at noon, the blue glow around the setting sun
// that the dust scatters forward, and the thin limb from 300 km,
// under the Sun's measured spectrum at Mars's mean distance.

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
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kPi = std::numbers::pi;

/// The Sun's irradiance at Mars's mean distance, 1.5237 au (its
/// orbit's semi-major axis): the total solar irradiance at 1 au over
/// the distance squared.
constexpr double kMarsIrradiance = 1361.0 / (1.5237 * 1.5237);

/// A camera and the sun's elevation, with an exposure for its light:
/// a stop more than on Earth in daylight (the sun is 43% as bright),
/// opened up at sunset.
struct View {
    const char* name;
    glm::dvec3 eye;
    glm::dvec3 target;
    double
        sunElevation;  ///< Degrees above the horizon at the origin.
    double fStop;
};

const std::array<View, 4> kViews = {{
    {"Rover, noon",
     {0.0, 0.0, 2.0},
     {-1000.0, 0.0, 350.0},
     60.0,
     11.0},
    {"Rover, sunset",
     {0.0, 0.0, 2.0},
     {1000.0, 0.0, 60.0},
     3.0,
     16.0},
    {"Orbiter, 300 km",
     {0.0, 0.0, 300e3},
     {0.0, 1e6, 0.0},
     30.0,
     11.0},
    // Looking towards the sun at the limb, 23.3 degrees below the
    // horizontal from 300 km, where the sun sets.
    {"Limb at sunset",
     {0.0, 0.0, 300e3},
     {1e6, 0.0, -131e3},
     -22.0,
     2.8},
}};

/// The mean linear colour of the rows [`top`, `bottom`) and columns
/// [`left`, `right`) of an RGBA uint8 image, as fractions of its
/// size.
glm::dvec3 MeanLinearRgbIn(const Tensor& image, double top,
                           double bottom, double left, double right) {
    const std::int64_t rows = image.Shape()[0];
    const std::int64_t cols = image.Shape()[1];
    const std::int64_t channels = image.Shape()[2];
    const auto* texels =
        static_cast<const std::uint8_t*>(image.Data());
    glm::dvec3 sum(0.0);
    std::int64_t count = 0;
    for (auto y = static_cast<std::int64_t>(top * rows);
         y < static_cast<std::int64_t>(bottom * rows); ++y) {
        for (auto x = static_cast<std::int64_t>(left * cols);
             x < static_cast<std::int64_t>(right * cols); ++x) {
            const std::int64_t i = (y * cols + x) * channels;
            sum +=
                glm::dvec3(texels[i], texels[i + 1], texels[i + 2]);
            ++count;
        }
    }
    return demos::SrgbToLinear(sum /
                               (255.0 * static_cast<double>(count)));
}

class MarsDemo : public app::Demo {
  public:
    MarsDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // Carbon dioxide and dust (feldspar with hematite) around a
        // planet whose surface is the world's z = 0.
        Atmosphere::Define(scene, "/World/Air")
            .SetPreset(std::string(tokens::kMars))
            .SetDustOpticalDepth(0.5)
            .SetGroundAlbedo(0.25);
        // The Sun's measured spectrum at Mars's distance.
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(kMarsIrradiance)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(0.35);
        // Reddish rocks on the ground (albedo about Mars's mean
        // 0.25).
        for (int i = 0; i < 3; ++i) {
            const Path path("/World/Rock" + std::to_string(i));
            Sphere::Define(scene, path)
                .SetRadius(0.6 + 0.3 * i)
                .SetDisplayColor(glm::dvec3(0.35, 0.2, 0.12));
            SetXform(scene, path,
                     {.translate = glm::dvec3(i == 0 ? -12.0 : 15.0,
                                              -5.0 + 5.0 * i, 0.5)});
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
        Widgets().Slider("Sun elevation", -25.0, 90.0,
                         kViews[0].sunElevation,
                         [this](double degrees) { SunAt(degrees); });
        // About 0.3 to 0.9 in clear seasons; dust storms above 1.
        Widgets().Slider("Dust optical depth", 0.0, 5.0, 0.5,
                         [this](double depth) {
                             Atmosphere(GetScene(), "/World/Air")
                                 .SetDustOpticalDepth(depth);
                         });
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
        // The noon sky away from the sun, the top of the frame.
        const glm::dvec3 noon =
            MeanLinearRgbIn(images[0], 0.0, 0.3, 0.0, 1.0);
        // At sunset, the sky around the sun and far to its side.
        const glm::dvec3 aureole =
            MeanLinearRgbIn(images[1], 0.25, 0.45, 0.4, 0.6);
        const glm::dvec3 aside =
            MeanLinearRgbIn(images[1], 0.0, 0.2, 0.0, 0.15);
        const glm::dvec3 orbit = demos::MeanLinearRgb(images[2]);
        std::printf("noon sky: %.3f %.3f %.3f\n", noon.r, noon.g,
                    noon.b);
        std::printf("around the setting sun: %.3f %.3f %.3f\n",
                    aureole.r, aureole.g, aureole.b);
        std::printf("aside at sunset: %.3f %.3f %.3f\n", aside.r,
                    aside.g, aside.b);
        // Lemmon et al. 2004: dust makes the daytime sky
        // butterscotch, and scatters blue light forward around the
        // setting sun.
        Check(noon.r > 1.1 * noon.b,
              "the noon sky should be redder than it is blue");
        Check(aureole.b / aureole.r > 1.2 * aside.b / aside.r,
              "the sky around the setting sun should be bluer");
        Check(orbit.r + orbit.g + orbit.b > 0.03,
              "the planet should be lit from orbit");
    }

  private:
    void Show(int index) {
        view_ = index;
        const View& view = kViews[static_cast<std::size_t>(index)];
        Camera(GetScene(), "/World/Camera")
            .SetExposureFStop(view.fStop)
            .SetTransform(LookAtTransform(view.eye, view.target,
                                          {0.0, 0.0, 1.0}));
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

ERNIE_REGISTER_DEMO(
    MarsDemo, "atmosphere", "mars",
    "Mars's dusty sky from a rover at noon and sunset, "
    "and from orbit.");
