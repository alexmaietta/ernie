// SPDX-License-Identifier: BSD-3-Clause
// Solar eclipses by a CelestialBody Moon: partial and annular through
// a solar filter, totality from the ground, and its shadow from
// orbit.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/light_schemas.h"
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

constexpr double kPi = std::numbers::pi;
constexpr double kSunElevation = 40.0;  // degrees

/// A camera, an exposure, and the Moon's distance and offset from the
/// sun (towards the zenith).
struct View {
    const char* name;
    glm::dvec3 eye;
    double elevation;  ///< Of the camera's aim, degrees, towards +x.
    glm::dvec3 up;
    double focalLength;   ///< Millimetres.
    double moonDistance;  ///< Metres.
    double moonOffset;    ///< Degrees.
    double fStop;
    double exposureTime;  ///< Seconds, at ISO 100.
    double filter;        ///< Stops; -13.3 is an ND 4.0 solar filter.
};

// At 360,000 km the Moon covers the sun; at 405,500 km it leaves a
// ring.
const std::array<View, 4> kViews = {{
    {"Partial, through a solar filter",
     {0.0, 0.0, 2.0},
     kSunElevation,
     {0.0, 0.0, 1.0},
     1200.0,
     384400e3,
     0.3,
     16.0,
     1.0 / 1000.0,
     -13.3},
    {"Annular, through a solar filter",
     {0.0, 0.0, 2.0},
     kSunElevation,
     {0.0, 0.0, 1.0},
     1200.0,
     405500e3,
     0.0,
     16.0,
     1.0 / 1000.0,
     -13.3},
    {"Totality, from the ground",
     {0.0, 0.0, 2.0},
     22.0,
     {0.0, 0.0, 1.0},
     14.0,
     360000e3,
     0.0,
     2.8,
     1.0 / 2.0,
     0.0},
    {"The Moon's shadow from geostationary orbit",
     {0.0, 0.0, 35786e3},
     -90.0,
     {1.0, 0.0, 0.0},
     100.0,
     360000e3,
     0.0,
     16.0,
     1.0 / 125.0,
     0.0},
}};

/// The mean linear colour of rows [top, bottom) and columns [left,
/// right) of a uint8 (rows, cols, 4) image.
glm::dvec3 RegionLinearRgb(const Tensor& image, std::int64_t top,
                           std::int64_t bottom, std::int64_t left,
                           std::int64_t right) {
    const std::int64_t cols = image.Shape()[1];
    const auto* pixels =
        static_cast<const std::uint8_t*>(image.Data());
    glm::dvec3 sum(0.0);
    for (std::int64_t row = top; row < bottom; ++row) {
        for (std::int64_t col = left; col < right; ++col) {
            const std::uint8_t* p = pixels + (row * cols + col) * 4;
            sum += demos::SrgbToLinear(glm::dvec3(p[0], p[1], p[2]) /
                                       255.0);
        }
    }
    return sum / static_cast<double>((bottom - top) * (right - left));
}

double Luminance(glm::dvec3 rgb) {
    return 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
}

/// The pixels of a uint8 (rows, cols, 4) image brighter than half
/// white.
int BrightPixels(const Tensor& image) {
    const auto* pixels =
        static_cast<const std::uint8_t*>(image.Data());
    const std::int64_t count = image.Shape()[0] * image.Shape()[1];
    int bright = 0;
    for (std::int64_t i = 0; i < count; ++i) {
        bright += pixels[i * 4 + 1] > 128 ? 1 : 0;
    }
    return bright;
}

class EclipseDemo : public app::Demo {
  public:
    EclipseDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        demos::DefinePlanetSurface(scene, "/World/Earth", 6371.0088e3,
                                   glm::dvec3(0.3),
                                   "/World/Looks/Ground");
        Atmosphere::Define(scene, "/World/Air")
            .SetPreset(std::string(tokens::kEarth));
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(0.53);
        const double e = kSunElevation * kPi / 180.0;
        DistantLight(scene, "/World/Sun")
            .SetTransform(DistantLightTransform(
                {std::cos(e), 0.0, std::sin(e)}));
        // The Moon: its mean radius and normal albedo.
        CelestialBody::Define(scene, "/World/Moon")
            .SetRadius(1737.4e3)
            .SetAlbedo(glm::dvec3(0.12));
        Camera::Define(scene, "/World/Camera")
            .SetSensorNoise(false)
            .SetExposureIso(100.0);
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
        Widgets().Slider(
            "Moon offset", -1.0, 1.0, kViews[0].moonOffset,
            [this](double degrees) {
                MoonAt(kViews[static_cast<std::size_t>(view_)]
                           .moonDistance,
                       degrees);
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
        for (std::size_t i = 0; i < images.size(); ++i) {
            const glm::dvec3 mean = demos::MeanLinearRgb(images[i]);
            std::printf("%s: mean linear RGB %.4f %.4f %.4f\n",
                        kViews[i].name, mean.r, mean.g, mean.b);
        }
        // A crescent and a ring: less of the sun than its whole disc.
        const int whole = BrightPixels(RenderWithoutMoon(0));
        const int crescent = BrightPixels(images[0]);
        const int ring = BrightPixels(images[1]);
        std::printf("bright pixels: whole %d, crescent %d, ring %d\n",
                    whole, crescent, ring);
        Check(crescent > 0 && crescent < 0.8 * whole,
              "the partial eclipse should leave a crescent");
        Check(ring > 0 && ring < 0.3 * whole,
              "the annular eclipse should leave a thin ring");
        // The ring: dark at the Moon's centre.
        const std::int64_t rows = images[1].Shape()[0];
        const std::int64_t cols = images[1].Shape()[1];
        Check(Luminance(RegionLinearRgb(images[1], rows / 2 - 2,
                                        rows / 2 + 2, cols / 2 - 2,
                                        cols / 2 + 2)) < 0.01,
              "the Moon should be dark against the sun");
        // Totality: a dark sky, brightest along the horizon.
        const Tensor day = RenderWithoutMoon(2);
        const double total =
            Luminance(demos::MeanLinearRgb(images[2]));
        const double daylight = Luminance(demos::MeanLinearRgb(day));
        std::printf("totality %.5f of daylight\n", total / daylight);
        Check(total < 0.05 * daylight,
              "totality should darken the sky");
        const double horizon = Luminance(RegionLinearRgb(
            images[2], rows - rows / 5, rows, 0, cols));
        const double overhead = Luminance(
            RegionLinearRgb(images[2], 0, rows / 5, 0, cols));
        std::printf("totality: horizon %.5f, top %.5f\n", horizon,
                    overhead);
        Check(horizon > 2.0 * overhead,
              "the horizon should glow in totality");
        // From orbit: a dark umbra under the camera.
        const double umbra = Luminance(
            RegionLinearRgb(images[3], rows / 2 - 2, rows / 2 + 2,
                            cols / 2 - 2, cols / 2 + 2));
        const double around = Luminance(
            RegionLinearRgb(images[3], 0, rows / 8, 0, cols));
        std::printf("from orbit: umbra %.5f, around %.5f\n", umbra,
                    around);
        Check(umbra < 0.1 * around,
              "the umbra should be dark from orbit");
    }

  private:
    void Show(int index) {
        view_ = index;
        const View& view = kViews[static_cast<std::size_t>(index)];
        const double a = view.elevation * kPi / 180.0;
        Camera(GetScene(), "/World/Camera")
            .SetFocalLength(view.focalLength)
            .SetExposureFStop(view.fStop)
            .SetExposureTime(view.exposureTime)
            .SetExposure(view.filter)
            .SetTransform(LookAtTransform(
                view.eye,
                view.eye + glm::dvec3(std::cos(a), 0.0, std::sin(a)),
                view.up));
        MoonAt(view.moonDistance, view.moonOffset);
    }

    /// The Moon `distance` from the world's origin, `degrees` from
    /// the sun's direction towards the zenith.
    void MoonAt(double distance, double degrees) {
        const double e = (kSunElevation + degrees) * kPi / 180.0;
        SetXform(GetScene(), "/World/Moon",
                 {.translate = distance * glm::dvec3(std::cos(e), 0.0,
                                                     std::sin(e))});
    }

    /// View `index` with the Moon far from the sun; leaves the view
    /// as it was.
    Tensor RenderWithoutMoon(int index) {
        const int current = view_;
        Show(index);
        MoonAt(kViews[static_cast<std::size_t>(index)].moonDistance,
               -30.0);
        Tensor image = GetRenderer().Render(GetScene());
        Show(current);
        return image;
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

ERNIE_REGISTER_DEMO(EclipseDemo, "atmosphere", "eclipse",
                    "Solar eclipses: partial, annular and total from "
                    "the ground, and the Moon's shadow from orbit.");
