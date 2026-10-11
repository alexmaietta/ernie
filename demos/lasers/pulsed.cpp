// SPDX-License-Identifier: BSD-3-Clause
// A pulsed 650 nm line laser, 20 us pulses at 1 kHz (2% of the time
// on), across a wall, seen by three cameras: a 1/60 s exposure sees
// its average, a 50 us exposure in step with the pulses catches one
// at twenty times the average, and the same 50 us between pulses sees
// none of it. The renderer's sensor clock (`RenderDelegate::Time`)
// times each frame.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kRate = 1000.0;       // pulses per second
constexpr double kWidth = 20e-6;       // s
constexpr double kShort = 50e-6;       // the short exposures, s
constexpr double kFrame = 1.0 / 60.0;  // s
// The pulses start 40 us before each whole millisecond, so a short
// exposure ending on one holds a pulse.
constexpr double kDelay = -40e-6;

// The cameras: exposure, f-number and ISO (the short ones opened up
// and turned up to make up for their exposure), and where their
// exposure ends in frames from the frame's time: in step with the
// pulses, or 0.01 frame (167 us) early, between them.
struct View {
    const char* name;
    const char* product;
    double exposure;
    double fStop;
    double iso;
    double end;
};

constexpr std::array<View, 3> kViews = {{
    {"1/60 s: the average", "/Render/Long", kFrame, 4.0, 400.0, 0.0},
    {"50 us in step: a pulse", "/Render/InStep", kShort, 1.4, 16000.0,
     0.0},
    {"50 us between pulses", "/Render/Between", kShort, 1.4, 16000.0,
     -0.01},
}};

constexpr glm::dvec2 kResolution(320.0, 200.0);

class PulsedDemo : public app::Demo {
  public:
    PulsedDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 60;
        fixedDt = kFrame;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Floor")
            .SetChecker(true)
            .SetCheckerColor0(glm::dvec3(0.4))
            .SetCheckerColor1(glm::dvec3(0.3));
        Cube::Define(scene, "/World/Wall")
            .SetSize(1.0)
            .SetDisplayColor(glm::dvec3(0.7));
        SetXform(scene, "/World/Wall",
                 {.translate = glm::dvec3(0.0, 0.05, 1.0),
                  .scale = glm::dvec3(4.0, 0.1, 2.0)});
        SphereLight::Define(scene, "/World/Lamp")
            .SetRadius(0.1)
            .SetPhotometricPower(2000.0)
            .SetTransform(LookAtTransform(
                {0.0, -1.5, 2.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}));
        // 50 mW average, fanned into a flat 40 degree line along the
        // wall.
        LaserLight::Define(scene, "/World/Laser")
            .SetWavelength(650.0)
            .SetPower(0.05)
            .SetWaistRadius(glm::dvec2(0.001))
            .SetPattern("line")
            .SetFanAngle(40.0)
            .SetPulseRate(kRate)
            .SetPulseWidth(kWidth)
            .SetPulseDelay(kDelay)
            .SetTransform(LookAtTransform(
                {0.0, -1.5, 1.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}));
        for (std::size_t c = 0; c < kViews.size(); ++c) {
            const View& view = kViews[c];
            const Path camera = CameraPath(c);
            Camera::Define(scene, camera)
                .SetFocalLength(24.0)
                .SetSensorNoise(false)
                .SetExposureTime(view.exposure)
                .SetExposureFStop(view.fStop)
                .SetExposureIso(view.iso)
                .SetShutterOpen(view.end)
                .SetShutterClose(view.end)
                .SetTransform(LookAtTransform({0.6, -3.0, 1.2},
                                              {0.0, 0.0, 1.0},
                                              {0.0, 0.0, 1.0}));
            const Path ldr = Path(view.product).Child("LdrColor");
            RenderVar::Define(scene, ldr).SetSourceName("LdrColor");
            RenderProduct::Define(scene, view.product)
                .SetCamera(camera)
                .SetResolution(kResolution)
                .SetOrderedVars({ldr});
        }
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(CameraPath(1))
            .SetResolution(kResolution)
            .SetAmbient(0.0);

        std::vector<std::string> names;
        for (const View& view : kViews) {
            names.emplace_back(view.name);
        }
        Widgets().Combo(
            "Camera", std::move(names), 1, [this](std::size_t index) {
                if (GetViewer() != nullptr) {
                    GetViewer()->SetActiveCamera(CameraPath(index));
                }
            });
        if (viewer != nullptr) {
            viewer->SetActiveCamera(CameraPath(1));
        }
    }

    /// The three cameras' frames ending at 0.1 s, side by side.
    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, demos::SideBySide(RenderAt(0.1)));
    }

    void TestFinal() override {
        const std::vector<Tensor> images = RenderAt(0.1);
        // The cameras share a pose: the line lies where the long
        // exposure sees it.
        const double row = LineRow(images[0]);
        std::array<double, 3> line{};
        std::array<double, 3> wall{};
        for (std::size_t c = 0; c < images.size(); ++c) {
            line[c] = Red(images[c], 0.5, row);
            wall[c] = Red(images[c], 0.5, 0.3);
            std::printf("%s: line %.3f, wall above %.3f\n",
                        kViews[c].name, line[c], wall[c]);
        }
        Check(line[0] > wall[0] + 0.2,
              "the long exposure should see the line's average");
        Check(line[1] > wall[1] + 0.2,
              "the exposure in step should catch a pulse");
        Check(line[2] < wall[2] + 0.05,
              "the exposure between pulses should see no line");
    }

  private:
    static Path CameraPath(std::size_t c) {
        return Path("/World/Camera" + std::to_string(c));
    }

    /// Every camera's LdrColor for the frame from `time - kFrame` to
    /// `time` on the sensor clock, uint8 (rows, cols, 4).
    std::vector<Tensor> RenderAt(double time) {
        RenderDelegate& delegate = GetRenderer().Delegate();
        delegate.ResetTime(time - kFrame);
        std::vector<Path> products;
        for (const View& view : kViews) {
            products.emplace_back(view.product);
        }
        RenderProductSetOutputs outputs =
            GetRenderer().Step(GetScene(), products, Path(), kFrame);
        std::vector<Tensor> images;
        for (const View& view : kViews) {
            images.push_back(outputs.at(Path(view.product))
                                 .vars.at("LdrColor")
                                 .Map(MapDevice::Cpu)
                                 .tensor);
        }
        return images;
    }

    /// The image row, as a fraction from the top, where the long
    /// exposure's line is reddest down the middle column.
    static double LineRow(const Tensor& image) {
        const auto rows = image.Shape()[0];
        double best = 0.0;
        double where = 0.5;
        for (std::int64_t r = 0; r < rows; ++r) {
            const double at = (static_cast<double>(r) + 0.5) /
                              static_cast<double>(rows);
            const double red = Red(image, 0.5, at);
            if (red > best) {
                best = red;
                where = at;
            }
        }
        return where;
    }

    /// The red channel (0 to 1) at the fractions `x` across and `y`
    /// down the image.
    static double Red(const Tensor& image, double x, double y) {
        const auto rows = image.Shape()[0];
        const auto cols = image.Shape()[1];
        const auto channels = image.Shape()[2];
        const auto r = std::min<std::int64_t>(
            static_cast<std::int64_t>(y * static_cast<double>(rows)),
            rows - 1);
        const auto c = std::min<std::int64_t>(
            static_cast<std::int64_t>(x * static_cast<double>(cols)),
            cols - 1);
        return static_cast<const std::uint8_t*>(
                   image.Data())[(r * cols + c) * channels] /
               255.0;
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(PulsedDemo, "lasers", "pulsed",
                    "A pulsed line laser seen by a long exposure, a "
                    "short one in step "
                    "with its pulses and one between them.");
