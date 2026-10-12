// SPDX-License-Identifier: BSD-3-Clause
// Laser speckle on a matte wall through a machine-vision camera: the
// grain grows as the lens stops down, and a surface that depolarizes
// the light shows two patterns at once, lowering the contrast.
//
// References:
// - J. W. Goodman, "Speckle Phenomena in Optics", Roberts and
//   Company, 2007 (grain size lambda z / D, contrast 1 / sqrt(M)).

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kWavelength = 532.0;  // nm
constexpr double kPixel = 5e-3;        // mm
constexpr double kFocal = 8.0;         // mm
constexpr glm::ivec2 kResolution{256, 192};
constexpr double kDistance = 1.0;  // m, camera to wall
constexpr std::array<double, 3> kFNumbers{2.0, 8.0, 16.0};

/// Goodman's 1 / sqrt(M): `patterns` independent patterns, each pixel
/// summing 1 + p^2 / A_c of them for the grain's correlation area
/// A_c = 1.08 (1.086 lambda N)^2 on the sensor.
double ExpectedContrast(double fNumber, double patterns) {
    const double cell = 1.086 * kWavelength * 1e-6 * fNumber;  // mm
    return 1.0 /
           std::sqrt(patterns *
                     (1.0 + kPixel * kPixel / (1.08 * cell * cell)));
}

class SpeckleDemo : public app::Demo {
  public:
    SpeckleDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // The wall's left half keeps the laser's polarization (a
        // surface scattering once), its right half scrambles it.
        for (const auto& [name, retention, x] :
             {std::tuple{"Keeps", 1.0, -0.25},
              std::tuple{"Scrambles", 0.0, 0.25}}) {
            const Path look =
                Path("/World").Child(std::string(name) + "Look");
            Material::Define(scene, look)
                .SetAlbedo(glm::dvec3(0.8))
                .SetPolarizationRetention(retention);
            const Path wall = Path("/World").Child(name);
            Cube::Define(scene, wall)
                .SetSize(1.0)
                .SetMaterialBinding(look);
            SetXform(scene, wall,
                     {.translate = glm::dvec3(x, kDistance, 0.0),
                      .scale = glm::dvec3(0.5, 0.01, 0.5)});
        }
        LaserLight::Define(scene, "/World/Laser")
            .SetWavelength(kWavelength)
            .SetLinewidth(0.0)
            .SetPower(0.01)
            .SetWaistRadius(glm::dvec2(0.04))
            .SetPolarizationDegree(1.0)
            .SetSpeckle(true)
            .SetTransform(LookAtTransform({0.05, 0.0, 0.0},
                                          {0.0, kDistance, 0.0},
                                          {0.0, 0.0, 1.0}));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(kFocal)
            .SetHorizontalAperture(kPixel * kResolution.x)
            .SetVerticalAperture(kPixel * kResolution.y)
            .SetSensorType("mono")
            .SetSpectralRange({520.0, 545.0})
            .SetSpectralBands(1)
            .SetSensorNoise(false)
            .SetClippingRange({0.05, 1e4})
            .SetTransform(LookAtTransform({0.0, 0.0, 0.0},
                                          {0.0, kDistance, 0.0},
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
            .SetResolution(kResolution)
            .SetAmbient(0.0);
        StopDown(1);
        std::vector<std::string> names;
        for (const double n : kFNumbers) {
            char name[16];
            std::snprintf(name, sizeof(name), "f/%g", n);
            names.emplace_back(name);
        }
        Widgets().Combo(
            "f-number", std::move(names), 1,
            [this](std::size_t index) { StopDown(index); });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            viewer->SetLighting(app::Lighting::Stage);
        }
    }

    /// Each f-number's image, the light without speckle shown at half
    /// white.
    void SaveOutput(const std::filesystem::path& path) override {
        const std::vector<double>& plain = Radiance(-1.0);
        const double peak =
            *std::max_element(plain.begin(), plain.end());
        std::vector<Tensor> images;
        for (const double n : kFNumbers) {
            const std::vector<double>& image = Radiance(n);
            Tensor grey = Tensor::Zeros(
                DType::UInt8, {kResolution.y, kResolution.x, 3});
            auto* data = static_cast<std::uint8_t*>(grey.Data());
            for (std::size_t i = 0; i < image.size(); ++i) {
                const auto level = static_cast<std::uint8_t>(
                    255.0 *
                    std::clamp(image[i] / (2.0 * peak), 0.0, 1.0));
                for (std::size_t k = 0; k < 3; ++k) {
                    data[i * 3 + k] = level;
                }
            }
            images.push_back(std::move(grey));
        }
        app::WritePng(path, demos::SideBySide(images));
    }

    void TestFinal() override {
        for (const double n : kFNumbers) {
            const auto [keeps, scrambles] = Contrasts(n);
            const double one = ExpectedContrast(n, 1.0);
            const double two = ExpectedContrast(n, 2.0);
            std::printf(
                "f/%g: contrast %.2f (Goodman %.2f) where the wall "
                "keeps the polarization, %.2f (%.2f) where it "
                "scrambles it\n",
                n, keeps, one, scrambles, two);
            Check(std::abs(keeps - one) < 0.1, "one pattern");
            Check(std::abs(scrambles - two) < 0.1, "two patterns");
        }
    }

  private:
    /// Sets the lens to kFNumbers[index], the exposure time keeping
    /// the image as bright.
    void StopDown(std::size_t index) {
        const double n = kFNumbers[index];
        Camera(GetScene(), "/World/Camera")
            .SetExposureFStop(n)
            .SetExposureTime(1e-3 * n * n);
    }

    /// The wall's radiance at f-number `n`, or without speckle for a
    /// negative one. Each renders once.
    const std::vector<double>& Radiance(double n) {
        if (const auto found = images_.find(n);
            found != images_.end()) {
            return found->second;
        }
        LaserLight laser(GetScene(), "/World/Laser");
        laser.SetSpeckle(n > 0.0);
        if (n > 0.0) {
            StopDown(static_cast<std::size_t>(
                std::find(kFNumbers.begin(), kFNumbers.end(), n) -
                kFNumbers.begin()));
        }
        const std::vector<Path> products{Path("/Render/Camera")};
        RenderProductSetOutputs outputs =
            GetRenderer().Step(GetScene(), products);
        laser.SetSpeckle(true);
        const Tensor bands = outputs.at(Path("/Render/Camera"))
                                 .vars.at("SpectralRadiance")
                                 .Map(MapDevice::Cpu)
                                 .tensor.Contiguous();
        const auto count = static_cast<std::size_t>(bands.Shape()[0]);
        const std::size_t pixels =
            static_cast<std::size_t>(kResolution.x * kResolution.y);
        std::vector<double> image(pixels, 0.0);
        const auto* values = bands.As<float>();
        for (std::size_t b = 0; b < count; ++b) {
            for (std::size_t i = 0; i < pixels; ++i) {
                image[i] +=
                    static_cast<double>(values[b * pixels + i]);
            }
        }
        return images_.emplace(n, std::move(image)).first->second;
    }

    /// The speckle's contrast (deviation over mean) on each half of
    /// the wall, over the beam's core, against the light without it.
    std::pair<double, double> Contrasts(double n) {
        const std::vector<double>& plain = Radiance(-1.0);
        const std::vector<double>& speckled = Radiance(n);
        const double peak =
            *std::max_element(plain.begin(), plain.end());
        const auto width = static_cast<std::size_t>(kResolution.x);
        const std::size_t half = width / 2;
        std::array<double, 2> result{};
        for (std::size_t side = 0; side < 2; ++side) {
            const std::size_t first = side == 0 ? 8 : half + 8;
            const std::size_t last = side == 0 ? half - 8 : width - 8;
            double sum = 0.0;
            double squares = 0.0;
            std::size_t count = 0;
            for (std::size_t i = 0; i < plain.size(); ++i) {
                const std::size_t column = i % width;
                if (column < first || column >= last ||
                    plain[i] <= 0.5 * peak) {
                    continue;
                }
                const double ratio = speckled[i] / plain[i];
                sum += ratio;
                squares += ratio * ratio;
                ++count;
            }
            const double mean = sum / static_cast<double>(count);
            result[side] =
                std::sqrt(squares / static_cast<double>(count) -
                          mean * mean) /
                mean;
        }
        return {result[0], result[1]};
    }

    std::map<double, std::vector<double>>
        images_;  ///< Per f-number (negative: speckle off).
};

}  // namespace

ERNIE_REGISTER_DEMO(SpeckleDemo, "lasers", "speckle",
                    "Laser speckle's grain and contrast against the "
                    "lens's f-number.");
