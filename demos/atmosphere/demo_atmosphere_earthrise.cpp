// SPDX-License-Identifier: BSD-3-Clause
// The Earth and its atmosphere rising over the Moon's horizon:
// gibbous over the sunlit surface, and a crescent ringed by its lit
// air.

#include "ernie/app/demo.h"
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

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kPi = std::numbers::pi;
constexpr double kMoonRadius = 1737.4e3;
constexpr double kEarthRadius = 6371.0088e3;
constexpr double kEarthDistance = 384400e3;

/// The Sun's direction (degrees, azimuth from the Earth's) and an
/// exposure.
struct View {
    const char* name;
    double sunAzimuth;
    double sunElevation;
    double fStop;
    double exposureTime;  ///< Seconds, at ISO 100.
};

const std::array<View, 2> kViews = {{
    {"Earthrise", 110.0, 25.0, 11.0, 1.0 / 250.0},
    {"Crescent Earth", 20.0, -6.0, 4.0, 1.0 / 15.0},
}};

glm::dvec3 Towards(double azimuth, double elevation) {
    const double a = azimuth * kPi / 180.0;
    const double e = elevation * kPi / 180.0;
    return {std::cos(e) * std::cos(a), std::cos(e) * std::sin(a),
            std::sin(e)};
}

/// The mean linear colour of rows [top, bottom) of a uint8 (rows,
/// cols, 4) image.
glm::dvec3 RowsLinearRgb(const Tensor& image, std::int64_t top,
                         std::int64_t bottom) {
    const std::int64_t cols = image.Shape()[1];
    const auto* pixels =
        static_cast<const std::uint8_t*>(image.Data());
    glm::dvec3 sum(0.0);
    for (std::int64_t row = top; row < bottom; ++row) {
        for (std::int64_t col = 0; col < cols; ++col) {
            const std::uint8_t* p = pixels + (row * cols + col) * 4;
            sum += demos::SrgbToLinear(glm::dvec3(p[0], p[1], p[2]) /
                                       255.0);
        }
    }
    return sum / static_cast<double>((bottom - top) * cols);
}

class EarthriseDemo : public app::Demo {
  public:
    EarthriseDemo(app::Viewer* viewer,
                  const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // The Moon: a smooth planet Terrain about the world's origin,
        // its regolith's normal albedo.
        constexpr std::int64_t kSide = 65;
        Terrain::Define(scene, "/World/Moon")
            .SetHeights(Tensor::Copy<float>(
                std::vector<float>(6 * kSide * kSide, 0.0f),
                {6, kSide, kSide}))
            .SetPlanetRadius(kMoonRadius)
            .SetMaterialBinding("/World/Looks/Regolith");
        Material::Define(scene, "/World/Looks/Regolith")
            .SetAlbedo(glm::dvec3(0.12));
        // The Earth 2 degrees over the horizon to +x: an icosphere
        // with its air centred on it.
        const glm::dvec3 eye(0.0, 0.0, kMoonRadius + 2.0);
        Xform::Define(scene, "/World/Earth");
        SetXform(
            scene, "/World/Earth",
            {.translate = eye + kEarthDistance * Towards(0.0, 2.0)});
        Sphere::Define(scene, "/World/Earth/Globe")
            .SetRadius(kEarthRadius)
            .SetTessellation(std::string(tokens::kIcosphere))
            .SetTessellationLevel(6)
            .SetMaterialBinding("/World/Looks/Earth");
        Material::Define(scene, "/World/Looks/Earth")
            .SetAlbedo(glm::dvec3(0.25, 0.28, 0.35));
        Atmosphere::Define(scene, "/World/Earth/Air")
            .SetPreset(std::string(tokens::kEarth))
            .SetAnchor(std::string(tokens::kAnchorCentre));
        // No shadows at these distances.
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(0.53)
            .SetShadowEnable(false);
        // Two metres up through a telephoto lens, its far plane
        // infinite.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(150.0)
            .SetClippingRange(glm::dvec2(
                0.5, std::numeric_limits<double>::infinity()))
            .SetSensorNoise(false)
            .SetExposureIso(100.0)
            .SetTransform(LookAtTransform(
                eye, eye + Towards(0.0, 1.4), {0.0, 0.0, 1.0}));
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
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    /// The views side by side.
    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, demos::SideBySide(RenderAll()));
    }

    void TestFinal() override {
        const std::vector<Tensor> images = RenderAll();
        const std::int64_t rows = images[0].Shape()[0];
        for (std::size_t i = 0; i < images.size(); ++i) {
            const glm::dvec3 sky =
                RowsLinearRgb(images[i], 0, rows / 8);
            std::printf("%s: sky %.4f %.4f %.4f\n", kViews[i].name,
                        sky.r, sky.g, sky.b);
            Check(sky.r + sky.g + sky.b < 0.01,
                  "the Moon's sky is black");
        }
        const glm::dvec3 ground =
            RowsLinearRgb(images[0], rows - rows / 8, rows);
        const glm::dvec3 earth =
            RowsLinearRgb(images[0], rows / 4, rows / 2);
        std::printf(
            "Earthrise: ground %.4f %.4f %.4f, Earth %.4f %.4f "
            "%.4f\n",
            ground.r, ground.g, ground.b, earth.r, earth.g, earth.b);
        Check(ground.g > 0.05, "the sunlit Moon shows");
        Check(earth.b > earth.r, "the Earth shows blue");
        // The crescent: the Earth's lit air beside its night side.
        const glm::dvec3 crescent =
            RowsLinearRgb(images[1], rows / 4, rows / 2);
        Check(crescent.r + crescent.g + crescent.b > 0.002,
              "the crescent shows");
    }

  private:
    void Show(int index) {
        view_ = index;
        const View& view = kViews[static_cast<std::size_t>(index)];
        Scene& scene = GetScene();
        Camera(scene, "/World/Camera")
            .SetExposureFStop(view.fStop)
            .SetExposureTime(view.exposureTime);
        DistantLight(scene, "/World/Sun")
            .SetTransform(DistantLightTransform(
                Towards(view.sunAzimuth, view.sunElevation)));
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

ERNIE_REGISTER_DEMO(EarthriseDemo, "atmosphere", "earthrise",
                    "The Earth and its air rising over the Moon's "
                    "horizon, gibbous and as a crescent.");
