// SPDX-License-Identifier: BSD-3-Clause
// A ridge 6 km high on a Moon-sized planet Terrain under a sun 10
// degrees up, seen from 300 km: its shadow through the cascades
// alone, through the terrain's horizon maps alone, and through both.
// The cascades show acne and ragged edges on the curved ground far
// from the ridge.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
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
#include <numbers>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kMoonRadius = 1737.4e3;
constexpr int kCells = 1024;
constexpr float kRidgeHeight = 6000.0f;
constexpr double kAltitude = 300e3;
constexpr double kEyeY = -30e3;
constexpr std::int64_t kWidth = 640;
constexpr std::int64_t kHeight = 360;

/// What shadows the ground: horizon maps on or off, and how far the
/// cascades reach (0 for as far as the view sees).
struct Mode {
    const char* name;
    bool horizons;
    double distance;
};

const std::array<Mode, 3> kModes = {{
    {"Cascades alone", false, 0.0},
    {"Horizon maps alone", true, 10.0},
    {"Both", true, 0.0},
}};

/// Six cube faces of heights, a ridge along face 0's meridian.
std::vector<float> RidgeHeights() {
    constexpr int points = kCells + 1;
    std::vector<float> heights(6 * points * points, 0.0f);
    for (int row = 0; row < points; ++row) {
        for (int col = kCells / 2 - 2; col <= kCells / 2 + 2; ++col) {
            heights[row * points + col] = kRidgeHeight;
        }
    }
    return heights;
}

double LinearGreen(std::uint8_t value) {
    return demos::SrgbToLinear(glm::dvec3(value / 255.0)).g;
}

/// The linear green along the middle rows at ground y (m).
double GroundGreen(const Tensor& image, double y) {
    const double u = (y - kEyeY) / (kAltitude * 18.0 / 24.0);
    const auto col =
        static_cast<std::int64_t>((u + 1.0) * 0.5 * kWidth);
    const auto* pixels =
        static_cast<const std::uint8_t*>(image.Data());
    double sum = 0.0;
    for (std::int64_t row = kHeight / 2 - 8; row < kHeight / 2 + 8;
         ++row) {
        sum += LinearGreen(pixels[(row * kWidth + col) * 4 + 1]);
    }
    return sum / 16.0;
}

/// How much the lit ground far west of the ridge varies from column
/// to column, mean over rows: rings of acne raise it.
double Acne(const Tensor& image) {
    const auto* pixels =
        static_cast<const std::uint8_t*>(image.Data());
    double sum = 0.0;
    for (std::int64_t row = 0; row < kHeight; ++row) {
        for (std::int64_t col = 0; col + 1 < 64; ++col) {
            const std::uint8_t* p = pixels + (row * kWidth + col) * 4;
            sum += std::abs(LinearGreen(p[5]) - LinearGreen(p[1]));
        }
    }
    return sum / static_cast<double>(kHeight * 63);
}

class LowSunDemo : public app::Demo {
  public:
    LowSunDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 3;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Material::Define(scene, "/World/Looks/Regolith")
            .SetAlbedo(glm::dvec3(0.3));
        constexpr std::int64_t points = kCells + 1;
        Terrain::Define(scene, "/World/Moon")
            .SetHeights(Tensor::Copy<float>(RidgeHeights(),
                                            {6, points, points}))
            .SetPlanetRadius(kMoonRadius)
            .SetMaterialBinding("/World/Looks/Regolith");
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(0.53);
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetHorizontalAperture(36.0)
            .SetVerticalAperture(36.0 * kHeight / kWidth)
            .SetSensorNoise(false)
            .SetExposure(-14.0)
            .SetClippingRange({1e3, 1e7})
            .SetTransform(LookAtTransform(
                {kMoonRadius + kAltitude, kEyeY, 0.0},
                {kMoonRadius, kEyeY, 0.0}, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(glm::dvec2(kWidth, kHeight))
            .SetAmbient(0.0);
        SunAt(10.0);
        Show(2);

        std::vector<std::string> names;
        for (const Mode& mode : kModes) {
            names.emplace_back(mode.name);
        }
        Widgets().Combo("Shadows", std::move(names), 2,
                        [this](std::size_t index) {
                            Show(static_cast<int>(index));
                        });
        Widgets().Slider("Sun elevation", 2.0, 30.0, 10.0,
                         [this](double degrees) { SunAt(degrees); });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    /// The three modes side by side.
    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, demos::SideBySide(RenderAll()));
    }

    void TestFinal() override {
        const std::vector<Tensor> images = RenderAll();
        for (std::size_t i = 0; i < images.size(); ++i) {
            std::printf("%s: acne %.4f\n", kModes[i].name,
                        Acne(images[i]));
        }
        // The ridge's foot at y = -5.4 km; the ground falling away as
        // d^2 / 2R meets the ray from its top 36 km on, at -41.6 km.
        for (std::size_t i = 0; i < images.size(); ++i) {
            const double lit = GroundGreen(images[i], -60e3);
            const double near = GroundGreen(images[i], -15e3);
            const double far = GroundGreen(images[i], -35e3);
            std::printf("%s: lit %.4f, shadow %.4f %.4f\n",
                        kModes[i].name, lit, near, far);
            Check(lit > 0.03 && lit < 0.9,
                  "the open ground should show lit");
            Check(GroundGreen(images[i], 40e3) > lit,
                  "the sunward side, under a higher sun, should be "
                  "brighter");
            Check(near < 0.1 * lit && far < 0.1 * lit,
                  std::string(kModes[i].name) +
                      ": the ridge should shadow 36 km of ground");
        }
    }

  private:
    /// The sun `degrees` up over the ridge, from the east (+y).
    void SunAt(double degrees) {
        const double e = degrees * std::numbers::pi / 180.0;
        DistantLight(GetScene(), "/World/Sun")
            .SetTransform(DistantLightTransform(
                {std::sin(e), std::cos(e), 0.0}));
    }

    void Show(int index) {
        mode_ = index;
        const Mode& mode = kModes[static_cast<std::size_t>(index)];
        RenderSettings(GetScene(), "/Render/Settings")
            .SetShadowHorizons(mode.horizons)
            .SetShadowDistance(mode.distance);
    }

    /// Each mode's image, RGBA uint8; leaves the mode as it was.
    std::vector<Tensor> RenderAll() {
        const int current = mode_;
        std::vector<Tensor> images;
        for (int i = 0; i < static_cast<int>(kModes.size()); ++i) {
            Show(i);
            Tensor image;
            for (int frame = 0; frame < numFrames; ++frame) {
                image = GetRenderer().Render(GetScene());
            }
            images.push_back(image);
        }
        Show(current);
        return images;
    }

    int mode_ = 2;
};

}  // namespace

ERNIE_REGISTER_DEMO(LowSunDemo, "shadows", "low_sun",
                    "A ridge's shadow on a Moon-sized planet under a "
                    "low sun: cascades, horizon maps and both.");
