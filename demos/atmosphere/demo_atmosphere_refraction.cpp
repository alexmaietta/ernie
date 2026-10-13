// SPDX-License-Identifier: BSD-3-Clause
// Refraction lifts and flattens the low sun and Moon: a telephoto
// sunset, the sun seen after it sets, moonrise, and sunset from the
// ISS.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
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
constexpr double kMoonDistance = 384400e3;

/// A camera aimed at `azimuth` and `elevation` (degrees), the sun's
/// true elevation towards +x, and an exposure.
struct View {
    const char* name;
    double altitude;
    double azimuth;
    double elevation;
    double focalLength;  ///< Millimetres.
    double sunElevation;
    double fStop;
    double exposureTime;  ///< Seconds, at ISO 100.
    double filter;        ///< Stops, dimming the bright sun.
};

const std::array<View, 4> kViews = {{
    {"Sunset, the sun flattened", 2.0, 0.0, 0.5, 1200.0, 0.0, 16.0,
     1.0 / 1000.0, -3.0},
    {"Set, and still seen", 2.0, 0.0, 0.2, 1200.0, -0.47, 16.0,
     1.0 / 1000.0, -1.0},
    {"Full Moon rising", 2.0, 183.0, 1.9, 600.0, -0.8, 11.0,
     1.0 / 30.0, 0.0},
    // Bent going into the air and again coming out.
    {"Sunset from the ISS", 400e3, 0.0, -19.6, 600.0, -19.95, 16.0,
     1.0 / 1000.0, -11.0},
}};

/// The rows and columns of the pixels of a uint8 (rows, cols, 4)
/// image whose green is above half its brightest: their extent.
struct Extent {
    std::int64_t top = 0;
    std::int64_t bottom = -1;
    std::int64_t left = 0;
    std::int64_t right = -1;
};

Extent BrightExtent(const Tensor& image) {
    const std::int64_t rows = image.Shape()[0];
    const std::int64_t cols = image.Shape()[1];
    const auto* pixels =
        static_cast<const std::uint8_t*>(image.Data());
    int brightest = 0;
    for (std::int64_t i = 0; i < rows * cols; ++i) {
        brightest = std::max<int>(brightest, pixels[i * 4 + 1]);
    }
    Extent extent{rows, -1, cols, -1};
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t col = 0; col < cols; ++col) {
            if (2 * pixels[(row * cols + col) * 4 + 1] > brightest) {
                extent.top = std::min(extent.top, row);
                extent.bottom = std::max(extent.bottom, row);
                extent.left = std::min(extent.left, col);
                extent.right = std::max(extent.right, col);
            }
        }
    }
    return extent;
}

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

glm::dvec3 Towards(double azimuth, double elevation) {
    const double a = azimuth * kPi / 180.0;
    const double e = elevation * kPi / 180.0;
    return {std::cos(e) * std::cos(a), std::cos(e) * std::sin(a),
            std::sin(e)};
}

class RefractionDemo : public app::Demo {
  public:
    RefractionDemo(app::Viewer* viewer,
                   const app::DemoOptions& options)
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
        // The full Moon opposite the sun, aside from the Earth's
        // shadow.
        CelestialBody::Define(scene, "/World/Moon")
            .SetRadius(1737.4e3)
            .SetAlbedo(glm::dvec3(0.12));
        SetXform(scene, "/World/Moon",
                 {.translate = kMoonDistance * Towards(183.0, 1.5)});
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
        Widgets().Slider("Sun elevation", -25.0, 3.0,
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
        for (std::size_t i = 0; i < images.size(); ++i) {
            const glm::dvec3 mean = demos::MeanLinearRgb(images[i]);
            std::printf("%s: mean linear RGB %.4f %.4f %.4f\n",
                        kViews[i].name, mean.r, mean.g, mean.b);
        }
        // The low sun: flattened and lifted.
        const double pixelsPerDegree =
            480.0 /
            (2.0 * std::atan(20.955 / 2.0 / 1200.0) * 180.0 / kPi);
        for (const std::size_t i : {std::size_t{0}, std::size_t{1}}) {
            const Extent disc = BrightExtent(images[i]);
            const double wide =
                static_cast<double>(disc.right - disc.left + 1);
            const double high =
                static_cast<double>(disc.bottom - disc.top + 1);
            const double centre =
                kViews[i].elevation +
                (static_cast<double>(images[i].Shape()[0]) / 2.0 -
                 0.5 * static_cast<double>(disc.top + disc.bottom)) /
                    pixelsPerDegree;
            std::printf("%s: %.0f x %.0f pixels, centre %.2f degrees "
                        "up (true %.2f)\n",
                        kViews[i].name, wide, high, centre,
                        kViews[i].sunElevation);
            Check(disc.bottom >= disc.top && high < 0.92 * wide,
                  "the low sun should show flattened");
            Check(centre - kViews[i].sunElevation > 0.4,
                  "the low sun should show lifted");
        }
        // The rising Moon: redder than the sky beside it.
        const std::int64_t middle = images[2].Shape()[0] / 2;
        const glm::dvec3 moon = RegionLinearRgb(images[2], middle - 4,
                                                middle + 4, 236, 244);
        const glm::dvec3 sky =
            RegionLinearRgb(images[2], middle - 4, middle + 4, 0, 40);
        std::printf(
            "rising Moon %.4f %.4f %.4f, sky %.4f %.4f %.4f\n",
            moon.r, moon.g, moon.b, sky.r, sky.g, sky.b);
        Check(moon.r > sky.r && moon.r / moon.b > 1.3 * sky.r / sky.b,
              "the rising Moon should show reddened");
        // From orbit: squashed.
        const Extent squashed = BrightExtent(images[3]);
        const double wide =
            static_cast<double>(squashed.right - squashed.left + 1);
        const double high =
            static_cast<double>(squashed.bottom - squashed.top + 1);
        std::printf("%s: %.0f x %.0f pixels\n", kViews[3].name, wide,
                    high);
        Check(squashed.bottom >= squashed.top && high < 0.8 * wide,
              "the sun should show squashed from orbit");
    }

  private:
    void Show(int index) {
        view_ = index;
        const View& view = kViews[static_cast<std::size_t>(index)];
        Camera(GetScene(), "/World/Camera")
            .SetFocalLength(view.focalLength)
            .SetExposureFStop(view.fStop)
            .SetExposureTime(view.exposureTime)
            .SetExposure(view.filter)
            .SetTransform(LookAtTransform(
                {0.0, 0.0, view.altitude},
                glm::dvec3(0.0, 0.0, view.altitude) +
                    Towards(view.azimuth, view.elevation),
                {0.0, 0.0, 1.0}));
        SunAt(view.sunElevation);
    }

    void SunAt(double degrees) {
        DistantLight(GetScene(), "/World/Sun")
            .SetTransform(
                DistantLightTransform(Towards(0.0, degrees)));
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
    RefractionDemo, "atmosphere", "refraction",
    "Refraction at the horizon: the flattened setting "
    "sun, seen after it sets, a rising full Moon, "
    "and the sun squashed from orbit.");
