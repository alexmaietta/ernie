// SPDX-License-Identifier: BSD-3-Clause
// A polarization camera's view of the sky and of glare on glass: its
// mosaic of 0, 45, 90 and 135 degree polarizers gives the degree and
// angle of linear polarization per 2 x 2 cell.
//
// References:
// - Sony Semiconductor Solutions, "Polarization Image Sensor
//   IMX250MZR/MYR", 2018 (the mosaic, and Stokes from its four
//   angles).
// - S. Chandrasekhar, "Radiative Transfer", Oxford, 1950 (the sky's
//   Rayleigh pattern).

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr glm::ivec2 kResolution{320, 256};
constexpr double kFocal = 10.0;          // mm
constexpr double kVerticalField = 80.0;  // degrees
constexpr double kHeight = 2.0;  // m, the camera above the ground
// The sun's elevation: its glint on glass at Brewster's angle.
const double kSun = std::atan(1.0 / 1.5);
constexpr double kSky =
    0.75;  // the sky's greatest degree of polarization

/// The mosaic cell (row, col) seeing along `azimuth` right of ahead
/// and `elevation` up, in degrees.
std::pair<int, int> CellOf(double azimuth, double elevation) {
    const double a = glm::radians(azimuth);
    const double e = glm::radians(elevation);
    const double x = std::sin(a) * std::cos(e);
    const double y = std::sin(e);
    const double z = std::cos(a) * std::cos(e);
    const double focal = 0.5 * kResolution.y /
                         std::tan(glm::radians(0.5 * kVerticalField));
    const double col = 0.5 * kResolution.x + focal * x / z;
    const double row = 0.5 * kResolution.y - focal * y / z;
    return {static_cast<int>(row) / 2, static_cast<int>(col) / 2};
}

/// The sky's degree of polarization along a direction: Rayleigh's
/// sin^2 g / (1 + cos^2 g) at its angle g from the sun, times kSky.
double Rayleigh(double azimuth, double elevation) {
    const double a = glm::radians(azimuth);
    const double e = glm::radians(elevation);
    const double cosine = std::cos(a) * std::cos(e) * std::cos(kSun) +
                          std::sin(e) * std::sin(kSun);
    return kSky * (1.0 - cosine * cosine) / (1.0 + cosine * cosine);
}

/// Per mosaic cell: its intensity S0, and degree and angle (degrees,
/// 0 to 180) of linear polarization.
struct Measured {
    int rows = 0;
    int cols = 0;
    std::vector<double> intensity;
    std::vector<double> degree;
    std::vector<double> angle;
};

class PolarizationDemo : public app::Demo {
  public:
    PolarizationDemo(app::Viewer* viewer,
                     const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Atmosphere::Define(scene, "/World/Air")
            .SetPreset(std::string(tokens::kEarth));
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(0.53)
            .SetTransform(DistantLightTransform(
                {0.0, std::cos(kSun), std::sin(kSun)}));
        GroundPlane::Define(scene, "/World/Ground")
            .SetChecker(true)
            .SetCheckerColor0(glm::dvec3(0.3))
            .SetCheckerColor1(glm::dvec3(0.25));
        // Black glass (index 1.5) where the camera sees the sun's
        // glint.
        Material::Define(scene, "/World/Glass")
            .SetBrdf("preview_surface")
            .SetAlbedo(glm::dvec3(0.02));
        scene.Set("/World/Glass", "inputs:roughness", 0.2);
        scene.Set("/World/Glass", "inputs:ior", 1.5);
        Cube::Define(scene, "/World/Sheet")
            .SetSize(1.0)
            .SetMaterialBinding(Path("/World/Glass"));
        SetXform(scene, "/World/Sheet",
                 {.translate =
                      glm::dvec3(0.0, kHeight / std::tan(kSun), 0.01),
                  .scale = glm::dvec3(4.0, 4.0, 0.02)});
        Cube::Define(scene, "/World/Block")
            .SetSize(1.0)
            .SetDisplayColor(glm::dvec3(0.5));
        SetXform(scene, "/World/Block",
                 {.translate = glm::dvec3(2.5, 6.0, 0.5)});
        const double aperture =
            2.0 * kFocal *
            std::tan(glm::radians(0.5 * kVerticalField));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(kFocal)
            .SetHorizontalAperture(aperture * kResolution.x /
                                   kResolution.y)
            .SetVerticalAperture(aperture)
            .SetSpectralRange({400.0, 700.0})
            .SetSpectralBands(6)
            .SetSensorNoise(false)
            .SetPolarizerType(std::string(tokens::kMosaic))
            .SetClippingRange({0.05, 1e5})
            .SetTransform(LookAtTransform({0.0, 0.0, kHeight},
                                          {0.0, 1.0, kHeight},
                                          {0.0, 0.0, 1.0}));
        RenderVar::Define(scene, "/Render/Camera/SpectralRadiance")
            .SetSourceName("SpectralRadiance");
        RenderProduct::Define(scene, "/Render/Camera")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(kResolution)
            .SetOrderedVars(
                {Path("/Render/Camera/SpectralRadiance")});
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(kResolution);
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    /// Its intensity (a square root of S0 over the median's 4 times),
    /// degree (0 to 1) and angle (0 to 180 degrees).
    void SaveOutput(const std::filesystem::path& path) override {
        const Measured& m = Measure();
        std::vector<double> sorted = m.intensity;
        const auto middle =
            sorted.begin() +
            static_cast<std::ptrdiff_t>(sorted.size() / 2);
        std::nth_element(sorted.begin(), middle, sorted.end());
        const double scale = 4.0 * *middle;
        Tensor grey =
            Tensor::Zeros(DType::UInt8, {m.rows, m.cols, 3});
        auto* data = static_cast<std::uint8_t*>(grey.Data());
        for (std::size_t i = 0; i < m.intensity.size(); ++i) {
            const auto level = static_cast<std::uint8_t>(
                255.0 * std::sqrt(std::clamp(m.intensity[i] / scale,
                                             0.0, 1.0)));
            for (std::size_t c = 0; c < 3; ++c) {
                data[i * 3 + c] = level;
            }
        }
        const std::vector<Tensor> images = {
            grey, demos::Heatmap(m.degree, m.rows, m.cols, 0.0, 1.0),
            demos::Heatmap(m.angle, m.rows, m.cols, 0.0, 180.0)};
        app::WritePng(path, demos::SideBySide(images));
    }

    void TestFinal() override {
        const Measured& m = Measure();
        // The median over 5 x 5 cells about a direction: edges, whose
        // four pixels see different things, read as polarized.
        const auto at = [&](double azimuth, double elevation) {
            const auto [row, col] = CellOf(azimuth, elevation);
            std::vector<double> patch;
            for (int r = row - 2; r <= row + 2; ++r) {
                for (int c = col - 2; c <= col + 2; ++c) {
                    patch.push_back(m.degree[static_cast<std::size_t>(
                        r * m.cols + c)]);
                }
            }
            std::nth_element(patch.begin(), patch.begin() + 12,
                             patch.end());
            return patch[12];
        };
        const double sun = glm::degrees(kSun);
        const double glare = at(0.0, -sun);
        const double near = at(0.0, sun - 10.0);
        const double away = at(40.0, 10.0);
        const double ground = at(-30.0, -10.0);
        std::printf(
            "degree of polarization: glare %.2f, sky 10 deg from "
            "the sun %.3f, 44 deg from it %.3f (Rayleigh %.3f), "
            "ground %.3f\n",
            glare, near, away, Rayleigh(40.0, 10.0), ground);
        Check(glare > 0.6, "glare at Brewster's angle is polarized");
        Check(near < 0.05,
              "the sky by the sun is nearly unpolarized");
        Check(std::abs(away - Rayleigh(40.0, 10.0)) < 0.05,
              "the sky away from the sun follows Rayleigh's pattern");
        Check(ground < 0.05, "the grey ground scatters unpolarized");
    }

  private:
    /// Per mosaic cell (90 and 45 degrees over 135 and 0): S0, S1 and
    /// S2 over the camera's bands, then their degree and angle.
    const Measured& Measure() {
        if (measured_) {
            return *measured_;
        }
        RenderProductSetOutputs outputs = GetRenderer().Step(
            GetScene(), std::vector<Path>{Path("/Render/Camera")});
        const Tensor bands = outputs.at(Path("/Render/Camera"))
                                 .vars.at("SpectralRadiance")
                                 .Map(MapDevice::Cpu)
                                 .tensor.Contiguous();
        const auto count = static_cast<std::size_t>(bands.Shape()[0]);
        const auto rows = static_cast<int>(bands.Shape()[1]);
        const auto cols = static_cast<int>(bands.Shape()[2]);
        const auto* values = bands.As<float>();
        const auto pixel = [&](int r, int c) {
            double sum = 0.0;
            for (std::size_t b = 0; b < count; ++b) {
                sum += static_cast<double>(
                    values[(b * static_cast<std::size_t>(rows) +
                            static_cast<std::size_t>(r)) *
                               static_cast<std::size_t>(cols) +
                           static_cast<std::size_t>(c)]);
            }
            return sum;
        };
        Measured m;
        m.rows = rows / 2;
        m.cols = cols / 2;
        for (int r = 0; r < m.rows; ++r) {
            for (int c = 0; c < m.cols; ++c) {
                const double i90 = pixel(2 * r, 2 * c);
                const double i45 = pixel(2 * r, 2 * c + 1);
                const double i135 = pixel(2 * r + 1, 2 * c);
                const double i0 = pixel(2 * r + 1, 2 * c + 1);
                const double s0 = (i0 + i45 + i90 + i135) / 2.0;
                const double s1 = i0 - i90;
                const double s2 = i45 - i135;
                m.intensity.push_back(s0);
                m.degree.push_back(std::sqrt(s1 * s1 + s2 * s2) /
                                   std::max(s0, 1e-30));
                const double angle =
                    glm::degrees(0.5 * std::atan2(s2, s1));
                m.angle.push_back(angle < 0.0 ? angle + 180.0
                                              : angle);
            }
        }
        measured_ = std::move(m);
        return *measured_;
    }

    std::optional<Measured> measured_;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    PolarizationDemo, "cameras", "polarization",
    "A polarization camera's view of the sky and of glare on glass.");
