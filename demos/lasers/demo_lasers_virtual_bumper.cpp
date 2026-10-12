// SPDX-License-Identifier: BSD-3-Clause
// NASA Ames' structured-light virtual bumper: 11 dots at 660 nm and a
// filtered mono camera on a rover, at night, in Earth daylight and in
// lunar daylight.
//
// References:
// - A. Nefian, U. Wong, M. Dille et al., "Structured Light-based
//   Hazard Detection For Planetary Surface Navigation", NASA NTRS
//   20170009200, 2017.
// - C. M. Pieters, L. A. Taylor, S. K. Noble et al., "Spectral
//   characterization of lunar mare soils", LPSC XXXI, 2000 (the
//   soils' spectra, read off the paper's Fig. 1).
// - A. S. McEwen, "Photometric functions for photoclinometry and
//   other applications", Icarus 92:298-311, 1991 (Lunar-Lambert).
// - Osela Inc., "SL Streamline Laser" datasheet, 2020 (660 nm at 50
//   or 100 mW).

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

constexpr std::array<const char*, 3> kCases = {"night", "Earth day",
                                               "lunar day"};
constexpr int kDots = 11;
constexpr double kGamma = 1.52;     // degrees between beams: 20 px
constexpr double kBaseline = 0.26;  // m, camera to projector
constexpr double kPitch = 28.0;     // degrees down
constexpr double kHeight = 0.6;     // m, the rig above the ground
constexpr double kFocal = 3.5;      // mm, at f/5.6
constexpr double kPixel = 4.65e-3;  // mm, the ICX267's
constexpr glm::ivec2 kResolution{1392, 1040};
constexpr double kDisparity = 15.0;  // pixels before an obstacle
constexpr double kPower = 0.05;      // W, in all the beams
// Their binary thresholds (Fig. 11), of 255, and the shutters (s)
// putting the dots at 90% of white on clear ground, as `Expose` finds
// them.
constexpr std::array<double, 3> kThreshold = {0.2, 0.675, 0.675};
constexpr std::array<double, 3> kShutter = {8.2e-4, 6.1e-4, 6.0e-4};

/// The ICX267's quantum efficiency (Fig. 4) behind a 100 nm wide
/// Gaussian band-pass at 660 nm: their C(lambda) F(lambda), as (nm,
/// value) pairs.
std::vector<float> FilteredQe() {
    constexpr std::array<std::pair<double, double>, 14> kQe = {{
        {400, 0.23},
        {450, 0.33},
        {500, 0.39},
        {520, 0.40},
        {550, 0.38},
        {600, 0.31},
        {650, 0.27},
        {700, 0.23},
        {750, 0.18},
        {800, 0.13},
        {850, 0.09},
        {900, 0.06},
        {950, 0.035},
        {1000, 0.02},
    }};
    const double sigma =
        100.0 / (2.0 * std::sqrt(2.0 * std::log(2.0)));
    std::vector<float> pairs;
    for (double nm = 560.0; nm <= 760.0; nm += 10.0) {
        std::size_t i = 1;
        while (kQe[i].first < nm) {
            ++i;
        }
        const auto [x0, y0] = kQe[i - 1];
        const auto [x1, y1] = kQe[i];
        const double qe = y0 + (y1 - y0) * (nm - x0) / (x1 - x0);
        const double band =
            std::exp(-0.5 * std::pow((nm - 660.0) / sigma, 2.0));
        pairs.push_back(static_cast<float>(nm));
        pairs.push_back(static_cast<float>(qe * band));
    }
    return pairs;
}

// Apollo soils' reflectance (Fig. 1): 12001 mare and 67641 highland.
const std::vector<float> kMare = {400,  0.05,  500,  0.06, 600,
                                  0.07, 700,   0.08, 800,  0.09,
                                  900,  0.092, 1000, 0.095};
const std::vector<float> kHighland = {
    400,   0.245, 500,   0.285, 600,  0.315, 700,
    0.345, 800,   0.365, 900,   0.36, 1000,  0.38};

/// A transform `x` m to the right on the rover, `kPitch` down.
glm::dmat4 Rig(double x) {
    const double pitch = glm::radians(kPitch);
    return LookAtTransform(
        {x, 0.0, kHeight},
        {x, std::cos(pitch), kHeight - std::sin(pitch)},
        {0.0, 0.0, 1.0});
}

/// Centroids of the runs of set columns in `mask`.
std::vector<double> Clusters(const std::vector<bool>& mask) {
    std::vector<double> centroids;
    std::size_t start = 0;
    while (start < mask.size()) {
        if (!mask[start]) {
            ++start;
            continue;
        }
        std::size_t end = start;
        while (end < mask.size() && mask[end]) {
            ++end;
        }
        if (end - start >= 2) {
            centroids.push_back(0.5 *
                                static_cast<double>(start + end - 1));
        }
        start = end;
    }
    return centroids;
}

/// One path's frame: its image, the dots' sunlight ratio and the
/// dots its detector finds.
struct Result {
    Tensor image;
    double ratio = 0.0;
    std::vector<double> found;
};

class VirtualBumperDemo : public app::Demo {
  public:
    VirtualBumperDemo(app::Viewer* viewer,
                      const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Material::Define(scene, "/World/Looks/Mare")
            .SetBrdf("lunar_lambert")
            .SetReflectance(kMare);
        Material::Define(scene, "/World/Looks/Highland")
            .SetBrdf("lunar_lambert")
            .SetReflectance(kHighland);
        Cube::Define(scene, "/World/Ground")
            .SetSize(1.0)
            .SetMaterialBinding(Path("/World/Looks/Mare"));
        SetXform(scene, "/World/Ground",
                 {.translate = glm::dvec3(0.0, 20.0, -0.05),
                  .scale = glm::dvec3(60.0, 60.0, 0.1)});
        // A 10 cm rock, their smallest hazard, under two of the dots.
        Cube::Define(scene, "/World/Rock")
            .SetSize(1.0)
            .SetMaterialBinding(Path("/World/Looks/Highland"));
        SetXform(scene, "/World/Rock",
                 {.translate = glm::dvec3(0.18, 1.13, 0.05),
                  .scale = glm::dvec3(0.08, 0.12, 0.1)});
        DistantLight::Define(scene, "/World/Sun")
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetTransform(DistantLightTransform(
                glm::normalize(glm::dvec3(0.5, -0.5, 0.707))));
        Atmosphere::Define(scene, "/World/Air")
            .SetPreset(std::string(tokens::kEarth));
        // 660 nm in 11 beams 1.52 degrees apart: 50 mW, the
        // Streamline's lowest 660 nm option (the paper gives none),
        // 80% into the dots.
        LaserLight::Define(scene, "/World/Projector")
            .SetWavelength(660.0)
            .SetPower(kPower)
            .SetWaistRadius(glm::dvec2(0.001))
            .SetPattern("grid")
            .SetGridColumns(kDots)
            .SetGridRows(1)
            .SetGridPitch({kGamma, kGamma})
            .SetEfficiency(0.8)
            .SetZeroOrder(0.0)
            .SetTransform(Rig(kBaseline));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(kFocal)
            .SetHorizontalAperture(kPixel * kResolution.x)
            .SetVerticalAperture(kPixel * kResolution.y)
            .SetSensorType("mono")
            .SetSensorQeMono(FilteredQe())
            .SetSpectralRange({560.0, 760.0})
            .SetSpectralBands(20)
            .SetSensorNoise(false)
            .SetExposureFStop(5.6)
            .SetExposureIso(400.0)
            .SetExposureTime(kShutter[0])
            .SetClippingRange({0.05, 1e4})
            .SetTransform(Rig(0.0));
        for (const char* name : {"LdrColor", "Electrons"}) {
            RenderVar::Define(scene,
                              Path("/Render/Camera").Child(name))
                .SetSourceName(name);
        }
        RenderProduct::Define(scene, "/Render/Camera")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(kResolution)
            .SetOrderedVars({Path("/Render/Camera/LdrColor"),
                             Path("/Render/Camera/Electrons")});
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(kResolution)
            .SetAmbient(0.0);
        Show(0);
        Widgets().Combo(
            "Case",
            std::vector<std::string>(kCases.begin(), kCases.end()), 0,
            [this](std::size_t index) { Show(index); });
        // Through the camera, lit by the scene alone.
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            viewer->SetLighting(app::Lighting::Stage);
        }
    }

    /// The three cases with the rock, side by side.
    void SaveOutput(const std::filesystem::path& path) override {
        std::vector<Tensor> images;
        for (std::size_t c = 0; c < kCases.size(); ++c) {
            images.push_back(RunCase(c)[1].image);
        }
        app::WritePng(path, demos::SideBySide(images));
    }

    void TestFinal() override {
        const std::vector<double> expected = RunCase(0)[0].found;
        Check(expected.size() == kDots,
              "flat ground at night should show 11 dots");
        std::array<double, kCases.size()> ratios{};
        for (std::size_t c = 0; c < kCases.size(); ++c) {
            const std::array<Result, 2> results = RunCase(c);
            ratios[c] = results[0].ratio;
            std::printf(
                "%s: (L_laser + L_sun) / L_sun %.3g, clear path %zu "
                "dots (%s), rock %zu dots (%s)\n",
                kCases[c], ratios[c], results[0].found.size(),
                Obstacle(results[0].found, expected) ? "obstacle"
                                                     : "clear",
                results[1].found.size(),
                Obstacle(results[1].found, expected) ? "obstacle"
                                                     : "clear");
            Check(Obstacle(results[1].found, expected),
                  std::string(kCases[c]) +
                      ": the rock should be found");
        }
        Check(ratios[0] > ratios[1], "daylight should fade the dots");
        Check(ratios[1] > ratios[2],
              "the bare Sun should fade them more");
    }

  private:
    /// Lights and exposes the scene for case `c`.
    void Show(std::size_t c) {
        Scene& scene = GetScene();
        Camera(scene, "/World/Camera").SetExposureTime(kShutter[c]);
        DistantLight(scene, "/World/Sun")
            .SetIrradiance(c == 0 ? 0.0 : 1361.0);
        scene.Set("/World/Air", tokens::kVisibility,
                  std::string(c == 1 ? tokens::kInherited
                                     : tokens::kInvisible));
    }

    /// The camera's LdrColor (uint8, rows x cols x 4) and its
    /// electrons' first channel.
    std::pair<Tensor, std::vector<float>> Capture() {
        RenderProductSetOutputs outputs = GetRenderer().Step(
            GetScene(), std::vector<Path>{Path("/Render/Camera")});
        auto& vars = outputs.at(Path("/Render/Camera")).vars;
        Tensor ldr = vars.at("LdrColor").Map(MapDevice::Cpu).tensor;
        const Tensor electrons = vars.at("Electrons")
                                     .Map(MapDevice::Cpu)
                                     .tensor.Contiguous();
        const auto pixels =
            static_cast<std::size_t>(ldr.Shape()[0] * ldr.Shape()[1]);
        const auto channels =
            static_cast<std::size_t>(electrons.Elements()) / pixels;
        const auto* values = electrons.As<float>();
        std::vector<float> first(pixels);
        for (std::size_t i = 0; i < pixels; ++i) {
            first[i] = values[i * channels];
        }
        return {std::move(ldr), std::move(first)};
    }

    static double Ldr(const Tensor& image, std::size_t row,
                      std::size_t col) {
        const auto cols = static_cast<std::size_t>(image.Shape()[1]);
        const auto channels =
            static_cast<std::size_t>(image.Shape()[2]);
        return static_cast<const std::uint8_t*>(
            image.Data())[(row * cols + col) * channels];
    }

    /// Sets the shutter so the brightest dot is 90% of white, as
    /// theirs, and returns the frame.
    std::pair<Tensor, std::vector<float>> Expose() {
        Camera camera(GetScene(), "/World/Camera");
        camera.SetExposureTime(0.001);
        for (int i = 0; i < 8; ++i) {
            const Tensor image = Capture().first;
            double peak = 0.0;
            const auto rows =
                static_cast<std::size_t>(image.Shape()[0]);
            const auto cols =
                static_cast<std::size_t>(image.Shape()[1]);
            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t x = 0; x < cols; ++x) {
                    peak = std::max(peak, Ldr(image, r, x));
                }
            }
            if (std::abs(peak - 230.0) <= 4.0) {
                break;
            }
            // LdrColor is gamma encoded: about the 2.2th root of
            // light.
            const double scale =
                peak < 254.0
                    ? std::pow(230.0 / std::max(peak, 1.0), 2.2)
                    : 0.3;
            camera.SetExposureTime(camera.GetExposureTime() * scale);
        }
        return Capture();
    }

    /// Case `c`'s clear path and rock under one shutter, set on the
    /// clear path as theirs was per condition.
    std::array<Result, 2> RunCase(std::size_t c) {
        Show(c);
        Scene& scene = GetScene();
        LaserLight laser(scene, "/World/Projector");
        std::array<Result, 2> results;
        for (std::size_t rock = 0; rock < 2; ++rock) {
            scene.Set("/World/Rock", tokens::kVisibility,
                      std::string(rock != 0 ? tokens::kInherited
                                            : tokens::kInvisible));
            auto [ldr, on] = rock != 0 ? Capture() : Expose();
            laser.SetPower(0.0);
            const std::vector<float> off = Capture().second;
            laser.SetPower(kPower);
            const auto rows = static_cast<std::size_t>(kResolution.y);
            const auto cols = static_cast<std::size_t>(kResolution.x);
            // The dots' row, and the band about it the detector
            // reads.
            std::size_t row = 0;
            float brightest = 0.0f;
            for (std::size_t i = 0; i < rows * cols; ++i) {
                if (on[i] - off[i] > brightest) {
                    brightest = on[i] - off[i];
                    row = i / cols;
                }
            }
            const std::size_t top = row > 40 ? row - 40 : 0;
            const std::size_t bottom = std::min(row + 40, rows - 1);
            std::vector<float> peaks(cols, 0.0f);
            std::vector<float> ambient;
            std::vector<bool> lit(cols, false);
            for (std::size_t r = top; r <= bottom; ++r) {
                for (std::size_t x = 0; x < cols; ++x) {
                    const std::size_t i = r * cols + x;
                    peaks[x] = std::max(peaks[x], on[i] - off[i]);
                    ambient.push_back(off[i]);
                    lit[x] = lit[x] ||
                             Ldr(ldr, r, x) > kThreshold[c] * 255.0;
                }
            }
            std::vector<bool> dotMask(cols);
            const float topPeak =
                *std::max_element(peaks.begin(), peaks.end());
            for (std::size_t x = 0; x < cols; ++x) {
                dotMask[x] = peaks[x] > 0.2f * topPeak;
            }
            const auto middle =
                ambient.begin() +
                static_cast<std::ptrdiff_t>(ambient.size() / 2);
            std::nth_element(ambient.begin(), middle, ambient.end());
            const double sun =
                std::max(static_cast<double>(*middle), 1e-9);
            double ratio = 0.0;
            bool first = true;
            for (const double centroid : Clusters(dotMask)) {
                double signal = 0.0;
                const auto x = static_cast<std::size_t>(centroid);
                for (std::size_t r = top; r <= bottom; ++r) {
                    signal = std::max(signal, static_cast<double>(
                                                  on[r * cols + x]));
                }
                ratio = first ? signal / sun
                              : std::min(ratio, signal / sun);
                first = false;
            }
            results[rock] = {std::move(ldr), ratio, Clusters(lit)};
        }
        return results;
    }

    /// Their decision: fewer dots than beams, or one moved more than
    /// `kDisparity` from where flat ground puts it.
    static bool Obstacle(const std::vector<double>& found,
                         const std::vector<double>& expected) {
        if (found.size() != expected.size()) {
            return true;
        }
        for (std::size_t i = 0; i < found.size(); ++i) {
            if (std::abs(found[i] - expected[i]) > kDisparity) {
                return true;
            }
        }
        return false;
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    VirtualBumperDemo, "lasers", "virtual_bumper",
    "NASA Ames' structured-light virtual bumper: 11 dots at "
    "660 nm against night, Earth and lunar daylight.");
