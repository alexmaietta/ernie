// SPDX-License-Identifier: BSD-3-Clause
// A rover's structured-light dots smearing over a long night
// exposure: still on flat ground, which moves with the rig, but drawn
// out along their row on a rock face the rover drives towards.
//
// References:
// - A. Nefian, U. Wong, M. Dille et al., "Structured Light-based
//   Hazard Detection For Planetary Surface Navigation", NASA NTRS
//   20170009200, 2017 (motion blur's night false negatives, Sec. VI).

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
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

constexpr int kDots = 11;
constexpr double kGamma = 1.52;     // degrees between beams
constexpr double kBaseline = 0.26;  // m, camera to projector
constexpr double kPitch = 28.0;     // degrees down
constexpr double kHeight = 0.6;     // m, the rig above the ground
constexpr double kFocal = 3.5;      // mm
constexpr double kPixel = 4.65e-3;  // mm, the ICX267's
constexpr glm::ivec2 kResolution{1392, 1040};
constexpr double kSpeed = 0.1;  // m/s, their most common
constexpr double kStep = 0.5;   // s between frames
// Exposures (s): a short one, and a long one at night; each frame's
// shutter is open for its exposure before the frame's time.
constexpr std::array<std::pair<const char*, double>, 2> kExposures = {
    {{"1 ms", 0.001}, {"500 ms", 0.5}}};
constexpr double kStart = -0.3;  // m, the rover's first position
constexpr double kEnd = 0.0;     // m, where it turns back

/// A transform `x` m to the right on the rover, `kPitch` down.
glm::dmat4 Rig(double x) {
    const double pitch = glm::radians(kPitch);
    return LookAtTransform(
        {x, 0.0, kHeight},
        {x, std::cos(pitch), kHeight - std::sin(pitch)},
        {0.0, 0.0, 1.0});
}

/// (first, last) columns of each dot along `row` above half its own
/// peak: the dots on the rock face outshine those on the ground.
std::vector<std::pair<std::size_t, std::size_t>>
Dots(const std::vector<double>& row) {
    const double brightest =
        *std::max_element(row.begin(), row.end());
    std::vector<std::pair<std::size_t, std::size_t>> spans;
    std::size_t start = 0;
    while (start < row.size()) {
        if (row[start] <= 0.01 * brightest) {
            ++start;
            continue;
        }
        std::size_t end = start;
        double peak = 0.0;
        while (end < row.size() && row[end] > 0.01 * brightest) {
            peak = std::max(peak, row[end]);
            ++end;
        }
        std::size_t first = end;
        std::size_t last = start;
        for (std::size_t x = start; x < end; ++x) {
            if (row[x] > 0.5 * peak) {
                first = std::min(first, x);
                last = std::max(last, x);
            }
        }
        spans.emplace_back(first, last);
        start = end;
    }
    return spans;
}

/// An image of rows x cols values.
struct RadianceImage {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<double> values;

    double At(std::size_t row, std::size_t col) const {
        return values[row * cols + col];
    }

    /// The row with the brightest value.
    std::size_t BrightestRow() const {
        const auto at =
            std::max_element(values.begin(), values.end());
        return static_cast<std::size_t>(at - values.begin()) / cols;
    }
};

class MotionSmearDemo : public app::Demo {
  public:
    MotionSmearDemo(app::Viewer* viewer,
                    const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 4;
        fixedDt = kStep;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Cube::Define(scene, "/World/Ground")
            .SetSize(1.0)
            .SetDisplayColor(glm::dvec3(0.1));
        SetXform(scene, "/World/Ground",
                 {.translate = glm::dvec3(0.0, 20.0, -0.05),
                  .scale = glm::dvec3(60.0, 60.0, 0.1)});
        // A rock face 1 m ahead, under the left dots (the projector
        // is 26 cm right of the camera).
        Cube::Define(scene, "/World/Rock")
            .SetSize(1.0)
            .SetDisplayColor(glm::dvec3(0.3));
        SetXform(scene, "/World/Rock",
                 {.translate = glm::dvec3(0.12, 1.2, 0.15),
                  .scale = glm::dvec3(0.24, 0.4, 0.3)});
        Xform::Define(scene, "/World/Rover");
        LaserLight::Define(scene, "/World/Rover/Projector")
            .SetWavelength(660.0)
            .SetPower(0.05)
            .SetWaistRadius(glm::dvec2(0.001))
            .SetPattern("grid")
            .SetGridColumns(kDots)
            .SetGridRows(1)
            .SetGridPitch({kGamma, kGamma})
            .SetEfficiency(0.8)
            .SetZeroOrder(0.0)
            .SetTransform(Rig(kBaseline));
        Camera::Define(scene, "/World/Rover/Camera")
            .SetFocalLength(kFocal)
            .SetHorizontalAperture(kPixel * kResolution.x)
            .SetVerticalAperture(kPixel * kResolution.y)
            .SetSensorType("mono")
            .SetSensorQeMono(std::vector<float>{600, 0.3f, 720, 0.3f})
            .SetSpectralRange({600.0, 720.0})
            .SetSpectralBands(3)
            .SetSensorNoise(false)
            .SetExposureFStop(5.6)
            .SetClippingRange({0.05, 1e4})
            .SetShutterSamples(8)
            .SetTransform(Rig(0.0));
        // Radiance, which the shutter averages and nothing saturates.
        RenderVar::Define(scene, "/Render/Camera/SpectralRadiance")
            .SetSourceName("SpectralRadiance");
        RenderProduct::Define(scene, "/Render/Camera")
            .SetCamera(Path("/World/Rover/Camera"))
            .SetResolution(kResolution)
            .SetOrderedVars(
                {Path("/Render/Camera/SpectralRadiance")});
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Rover/Camera"))
            .SetResolution(kResolution)
            .SetAmbient(0.0);
        Place(kStart);
        Expose(1);
        std::vector<std::string> names;
        for (const auto& [name, exposure] : kExposures) {
            names.emplace_back(name);
        }
        Widgets().Combo("Exposure", std::move(names), 1,
                        [this](std::size_t index) { Expose(index); });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Rover/Camera");
            viewer->SetLighting(app::Lighting::Stage);
        }
    }

    void Step() override {
        position_ += kSpeed * kStep;
        if (position_ > kEnd) {
            position_ = kStart;
        }
        Place(position_);
    }

    /// Each exposure's image about the dots: a square root, so the
    /// dim dots on the ground show, three times larger.
    void SaveOutput(const std::filesystem::path& path) override {
        std::vector<Tensor> images;
        for (std::size_t e = 0; e < kExposures.size(); ++e) {
            const RadianceImage& image = Radiance(e);
            const std::size_t row = image.BrightestRow();
            constexpr std::size_t kRows = 41;
            constexpr std::size_t kCols = 250;
            double brightest = 0.0;
            for (std::size_t r = 0; r < kRows; ++r) {
                for (std::size_t c = 0; c < kCols; ++c) {
                    brightest = std::max(
                        brightest, image.At(row - 20 + r, 730 + c));
                }
            }
            Tensor grey = Tensor::Zeros(DType::UInt8,
                                        {3 * kRows, 3 * kCols, 3});
            auto* data = static_cast<std::uint8_t*>(grey.Data());
            for (std::size_t r = 0; r < 3 * kRows; ++r) {
                for (std::size_t c = 0; c < 3 * kCols; ++c) {
                    const double value =
                        image.At(row - 20 + r / 3, 730 + c / 3);
                    const auto level = static_cast<std::uint8_t>(
                        255.0 * std::sqrt(std::clamp(
                                    value / brightest, 0.0, 1.0)));
                    for (std::size_t k = 0; k < 3; ++k) {
                        data[(r * 3 * kCols + c) * 3 + k] = level;
                    }
                }
            }
            images.push_back(std::move(grey));
        }
        app::WritePng(path, demos::SideBySide(images));
    }

    void TestFinal() override {
        std::array<std::vector<std::size_t>, kExposures.size()>
            widths;
        for (std::size_t e = 0; e < kExposures.size(); ++e) {
            const RadianceImage& image = Radiance(e);
            const std::size_t row = image.BrightestRow();
            std::vector<double> band(image.cols, 0.0);
            for (std::size_t r = row - 3; r <= row + 3; ++r) {
                for (std::size_t c = 0; c < image.cols; ++c) {
                    band[c] = std::max(band[c], image.At(r, c));
                }
            }
            std::printf("%s: dot widths [", kExposures[e].first);
            for (const auto& [first, last] : Dots(band)) {
                widths[e].push_back(last - first + 1);
                std::printf("%s%zu", widths[e].size() > 1 ? ", " : "",
                            widths[e].back());
            }
            std::printf("]\n");
        }
        const auto& shortOnes = widths[0];
        const auto& longOnes = widths[1];
        // The rock's edge hides a dot.
        Check(shortOnes.size() == kDots - 1 &&
                  longOnes.size() == kDots - 1,
              "10 dots each");
        if (shortOnes.size() != longOnes.size() ||
            shortOnes.size() < 10) {
            return;
        }
        // The left five on the rock face, the right five on the
        // ground.
        const std::size_t n = shortOnes.size();
        for (std::size_t i = 0; i < 5; ++i) {
            Check(longOnes[i] >= shortOnes[i] + 5,
                  "dots on the rock face should smear");
            const auto ground =
                static_cast<long>(longOnes[n - 1 - i]) -
                static_cast<long>(shortOnes[n - 1 - i]);
            Check(std::abs(ground) <= 1,
                  "dots on flat ground should not");
        }
    }

  private:
    /// Puts the rover `y` m along its track.
    void Place(double y) {
        SetXform(GetScene(), "/World/Rover",
                 {.translate = glm::dvec3(0.0, y, 0.0)});
    }

    /// Opens the shutter for exposure `e` before each frame's time.
    void Expose(std::size_t e) {
        const double exposure = kExposures[e].second;
        Camera(GetScene(), "/World/Rover/Camera")
            .SetExposureTime(exposure)
            .SetShutterOpen(-exposure / kStep)
            .SetShutterClose(0.0);
    }

    /// The camera's radiance over its bands for exposure `e`, the
    /// rover arriving at `kEnd` from where it was a frame before
    /// (drawn with the shutter shut, for its poses alone). Each
    /// exposure renders once.
    const RadianceImage& Radiance(std::size_t e) {
        if (const auto found = frames_.find(e);
            found != frames_.end()) {
            return found->second;
        }
        const std::vector<Path> products{Path("/Render/Camera")};
        Place(kEnd - kSpeed * kStep);
        Camera(GetScene(), "/World/Rover/Camera")
            .SetShutterOpen(0.0)
            .SetShutterClose(0.0);
        GetRenderer().Step(GetScene(), products);
        Place(kEnd);
        Expose(e);
        RenderProductSetOutputs outputs =
            GetRenderer().Step(GetScene(), products);
        const Tensor bands = outputs.at(Path("/Render/Camera"))
                                 .vars.at("SpectralRadiance")
                                 .Map(MapDevice::Cpu)
                                 .tensor.Contiguous();
        const auto count = static_cast<std::size_t>(bands.Shape()[0]);
        RadianceImage image;
        image.rows = static_cast<std::size_t>(bands.Shape()[1]);
        image.cols = static_cast<std::size_t>(bands.Shape()[2]);
        const std::size_t pixels = image.rows * image.cols;
        image.values.assign(pixels, 0.0);
        const auto* values = bands.As<float>();
        for (std::size_t b = 0; b < count; ++b) {
            for (std::size_t i = 0; i < pixels; ++i) {
                image.values[i] += values[b * pixels + i];
            }
        }
        return frames_.emplace(e, std::move(image)).first->second;
    }

    double position_ = kStart;
    std::map<std::size_t, RadianceImage>
        frames_;  ///< Per exposure, as `Radiance` renders it.
};

}  // namespace

ERNIE_REGISTER_DEMO(
    MotionSmearDemo, "lasers", "motion_smear",
    "A rover's dots smearing on a rock face over a long exposure.");
