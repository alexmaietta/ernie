// SPDX-License-Identifier: BSD-3-Clause
// A flash fired by the camera's exposure at night: in sync it lights
// the frame, a late trigger loses part of it, a later one misses it;
// a rolling shutter catches it in a band of rows, and a xenon flash's
// jittery trigger makes frames flicker.
//
// References:
// - ON Semiconductor, "Rolling Shutter vs. Global Shutter" (AND9858),
//   2018: a flash during a rolling readout lights only the rows
//   exposing then.

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
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr glm::ivec2 kResolution{240, 180};
constexpr double kFrame = 1.0 / 30.0;  // s
constexpr double kExposure = 2e-3;     // s, each row's
constexpr double kFlash = 1e-3;        // s, the flash's width
constexpr double kReadout = 15e-3;     // s, top row to bottom row

struct Case {
    const char* name;
    double delay;    // the trigger's, s
    double readout;  // a rolling shutter's, s
    double decay;    // s
    double jitter;   // s
};

constexpr std::array<Case, 5> kCases{{
    {"In sync", 0.5e-3, 0.0, 0.0, 0.0},
    {"Late trigger", 1.5e-3, 0.0, 0.0, 0.0},
    {"Missed", 2.5e-3, 0.0, 0.0, 0.0},
    {"Rolling shutter", 8e-3, kReadout, 0.0, 0.0},
    {"Xenon, jittery trigger", 0.5e-3, 0.0, 0.3e-3, 1e-3},
}};

class FlashSyncDemo : public app::Demo {
  public:
    FlashSyncDemo(app::Viewer* viewer,
                  const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        fixedDt = kFrame;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        const auto box = [&](const char* name, glm::dvec3 colour,
                             glm::dvec3 at, glm::dvec3 scale) {
            const Path path = Path("/World").Child(name);
            Cube::Define(scene, path)
                .SetSize(1.0)
                .SetDisplayColor(colour);
            SetXform(scene, path, {.translate = at, .scale = scale});
        };
        box("Floor", glm::dvec3(0.5), {0.0, 0.0, -0.05},
            {8.0, 8.0, 0.1});
        box("Wall", glm::dvec3(0.6), {0.0, 2.0, 1.5},
            {8.0, 0.1, 3.0});
        box("Red", {0.7, 0.1, 0.1}, {-0.8, 0.5, 0.25},
            glm::dvec3(0.5));
        box("Green", {0.1, 0.6, 0.1}, {0.0, 0.5, 0.25},
            glm::dvec3(0.5));
        box("Blue", {0.1, 0.2, 0.7}, {0.8, 0.5, 0.25},
            glm::dvec3(0.5));
        // An LED flash beside the lens: its intensity is its level
        // while it fires.
        SphereLight::Define(scene, "/World/Flash")
            .SetRadius(0.02)
            .SetIntensity(2e7)
            .SetPulseWidth(kFlash)
            .SetPulseTrigger(std::string(tokens::kExposureTrigger));
        SetXform(scene, "/World/Flash",
                 {.translate = glm::dvec3(0.1, -3.0, 1.3)});
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetSensorNoise(false)
            .SetExposureTime(kExposure)
            .SetExposureFStop(2.8)
            .SetExposureIso(400.0)
            .SetShutterSamples(64)
            .SetTransform(LookAtTransform(
                {0.0, -3.0, 1.2}, {0.0, 1.0, 0.8}, {0.0, 0.0, 1.0}));
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
        Show(0);
        std::vector<std::string> names;
        for (const Case& c : kCases) {
            names.emplace_back(c.name);
        }
        Widgets().Combo("Case", std::move(names), 0,
                        [this](std::size_t index) { Show(index); });
        Widgets().Slider("Trigger delay (ms)", -2.0, 20.0,
                         kCases[0].delay * 1e3, [this](double ms) {
                             SphereLight(GetScene(), "/World/Flash")
                                 .SetPulseDelay(ms * 1e-3);
                         });
        Widgets().Slider("Trigger jitter (ms)", 0.0, 3.0, 0.0,
                         [this](double ms) {
                             SphereLight(GetScene(), "/World/Flash")
                                 .SetPulseJitter(ms * 1e-3);
                         });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            app::ViewportDisplay display =
                viewer->GetViewportDisplay();
            display.lights = false;
            display.grid = false;
            viewer->SetViewportDisplay(display);
        }
    }

    /// Each case as grey, as bright as the frame in sync.
    void SaveOutput(const std::filesystem::path& path) override {
        std::vector<double> sorted = Radiance(0);
        std::sort(sorted.begin(), sorted.end());
        const double peak = sorted[sorted.size() * 99 / 100];
        std::vector<Tensor> images;
        for (std::size_t c = 0; c < kCases.size(); ++c) {
            const std::vector<double>& image = Radiance(c);
            Tensor grey = Tensor::Zeros(
                DType::UInt8, {kResolution.y, kResolution.x, 3});
            auto* data = static_cast<std::uint8_t*>(grey.Data());
            for (std::size_t i = 0; i < image.size(); ++i) {
                const auto level = static_cast<std::uint8_t>(
                    255.0 * std::clamp(image[i] / peak, 0.0, 1.0));
                for (std::size_t k = 0; k < 3; ++k) {
                    data[i * 3 + k] = level;
                }
            }
            images.push_back(std::move(grey));
        }
        app::WritePng(path, demos::SideBySide(images));
    }

    void TestFinal() override {
        const auto mean = [](const std::vector<double>& values) {
            double sum = 0.0;
            for (const double value : values) {
                sum += value;
            }
            return sum / static_cast<double>(values.size());
        };
        const double sync = mean(Radiance(0));
        const double late = mean(Radiance(1)) / sync;
        const double missed = mean(Radiance(2)) / sync;
        std::printf("late trigger %.2f of the light in sync (0.50), "
                    "missed %.2f\n",
                    late, missed);
        Check(std::abs(late - 0.5) < 0.01,
              "half the flash after the close");
        Check(missed == 0.0, "the flash after the exposure");
        // The rows exposing while it fires, 8 to 9 ms after the top
        // row opens, against the frame in sync (all of the flash): at
        // least half of it in those opening 6.5 to 8.5 ms in.
        const std::vector<double>& rolling = Radiance(3);
        const std::vector<double>& inSync = Radiance(0);
        const auto width = static_cast<std::size_t>(kResolution.x);
        const auto height = static_cast<std::size_t>(kResolution.y);
        double first = 1.0;
        double last = 0.0;
        double top = 0.0;
        for (std::size_t r = 0; r < height; ++r) {
            double lit = 0.0;
            double full = 0.0;
            for (std::size_t c = 0; c < width; ++c) {
                lit += rolling[r * width + c];
                full += inSync[r * width + c];
            }
            if (r < height / 4) {
                top = std::max(top, lit);
            }
            if (lit > 0.5 * full) {
                first = std::min(first, double(r) / double(height));
                last = std::max(last, double(r) / double(height));
            }
        }
        std::printf("rolling shutter: rows %.2f to %.2f of the way "
                    "down lit (0.43 to 0.57)\n",
                    first, last);
        Check(std::abs(first - 0.43) < 0.03 &&
                  std::abs(last - 0.57) < 0.03,
              "a band of rows");
        Check(top == 0.0, "the top dark");
    }

  private:
    /// Sets up case `index`: the trigger's delay and jitter, the
    /// flash's decay, and a global or rolling shutter, each row
    /// exposed for kExposure.
    void Show(std::size_t index) {
        const Case& c = kCases[index];
        SphereLight(GetScene(), "/World/Flash")
            .SetPulseDelay(c.delay)
            .SetPulseDecay(c.decay)
            .SetPulseJitter(c.jitter);
        Camera camera(GetScene(), "/World/Camera");
        if (c.readout > 0.0) {
            // Rows read out over kReadout, each open for kExposure
            // before the frame's time.
            camera.SetShutterOpen(-kExposure / kFrame)
                .SetShutterClose(0.0)
                .SetShutterRolling(c.readout / kFrame);
        } else {
            camera.SetShutterOpen(0.0)
                .SetShutterClose(0.0)
                .SetShutterRolling(0.0);
        }
    }

    /// Case `index`'s radiance over the bands, rows by columns; each
    /// renders once, a frame of the sensor clock long.
    const std::vector<double>& Radiance(std::size_t index) {
        if (const auto found = images_.find(index);
            found != images_.end()) {
            return found->second;
        }
        Show(index);
        const std::vector<Path> products{Path("/Render/Camera")};
        RenderProductSetOutputs outputs =
            GetRenderer().Step(GetScene(), products, Path(), kFrame);
        const Tensor bands = outputs.at(Path("/Render/Camera"))
                                 .vars.at("SpectralRadiance")
                                 .Map(MapDevice::Cpu)
                                 .tensor.Contiguous();
        const auto count = static_cast<std::size_t>(bands.Shape()[0]);
        const auto pixels =
            static_cast<std::size_t>(kResolution.x * kResolution.y);
        std::vector<double> image(pixels, 0.0);
        const auto* values = bands.As<float>();
        for (std::size_t b = 0; b < count; ++b) {
            for (std::size_t i = 0; i < pixels; ++i) {
                image[i] +=
                    static_cast<double>(values[b * pixels + i]);
            }
        }
        return images_.emplace(index, std::move(image)).first->second;
    }

    std::map<std::size_t, std::vector<double>> images_;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    FlashSyncDemo, "lighting", "flash_sync",
    "A flash fired by the exposure: in sync, late, missed, rolling.");
