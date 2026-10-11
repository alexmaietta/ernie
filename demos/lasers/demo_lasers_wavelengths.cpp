// SPDX-License-Identifier: BSD-3-Clause
// Six lasers from violet to near infrared seen by four cameras, each
// seeing only the lines its bands reach.

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
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

// Typical diode and DPSS lasers.
//
// References:
// - A. E. Siegman, Lasers, University Science Books, 1986.
// - Thorlabs, laser diode selection guide (405 to 940 nm diodes).
struct Laser {
    const char* name;
    double nm;
    double milliwatts;
};

constexpr std::array<Laser, 6> kLasers = {{
    {"405 nm violet", 405.0, 5.0},
    {"450 nm blue", 450.0, 10.0},
    {"532 nm green", 532.0, 1.0},
    {"650 nm red", 650.0, 5.0},
    {"850 nm NIR", 850.0, 10.0},
    {"940 nm NIR", 940.0, 20.0},
}};

// The cameras: their spectral range, and for a mono one its quantum
// efficiency as (nm, value) pairs.
struct Band {
    const char* name;
    const char* product;
    glm::dvec2 range;
    int bands;
    std::vector<float> qe;  // empty: the RGB camera's own
};

const std::array<Band, 4> kBands = {{
    {"visible RGB", "/Render/Rgb", {380.0, 720.0}, 17, {}},
    // A 10 nm interference filter at 532 nm.
    {"532 nm narrowband",
     "/Render/Green",
     {517.0, 547.0},
     6,
     {517.0f, 0.0f, 526.0f, 0.0f, 527.0f, 0.6f, 537.0f, 0.6f, 538.0f,
      0.0f, 547.0f, 0.0f}},
    // A visible-blocking filter on silicon: 780 to 1000 nm.
    {"NIR 780-1000 nm",
     "/Render/Nir",
     {760.0, 1000.0},
     12,
     {760.0f, 0.0f, 779.0f, 0.0f, 781.0f, 0.35f, 900.0f, 0.25f,
      1000.0f, 0.05f}},
    // Bare silicon, 400 to 1000 nm.
    {"wideband mono",
     "/Render/Mono",
     {380.0, 1000.0},
     31,
     {380.0f, 0.1f, 500.0f, 0.5f, 700.0f, 0.6f, 850.0f, 0.4f, 1000.0f,
      0.05f}},
}};

constexpr glm::dvec2 kResolution(400.0, 240.0);
constexpr double kDistance = 2.0;  // camera to wall, m
constexpr double kHeight = 1.2;    // the cameras' and spots' height
constexpr double kFocalLength = 50.0;  // mm
constexpr double kAperture = 20.955;   // mm, USD's horizontal default
constexpr double kSpacing = 0.15;      // between spots on the wall

/// Where on the wall laser `i`'s spot lies along x.
double SpotX(std::size_t i) {
    return kSpacing * (static_cast<double>(i) -
                       0.5 * static_cast<double>(kLasers.size() - 1));
}

class WavelengthsDemo : public app::Demo {
  public:
    WavelengthsDemo(app::Viewer* viewer,
                    const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Floor")
            .SetChecker(true)
            .SetCheckerColor0(glm::dvec3(0.3))
            .SetCheckerColor1(glm::dvec3(0.25));
        // A matte white wall at y = 0.
        Cube::Define(scene, "/World/Wall")
            .SetSize(1.0)
            .SetDisplayColor(glm::dvec3(0.8));
        SetXform(scene, "/World/Wall",
                 {.translate = glm::dvec3(0.0, 0.05, 2.0),
                  .scale = glm::dvec3(6.0, 0.1, 4.0)});
        // A dim incandescent ceiling lamp, whose blackbody reaches
        // the near infrared too.
        SphereLight::Define(scene, "/World/Lamp")
            .SetRadius(0.1)
            .SetPhotometricPower(200.0)
            .SetPhysicalIlluminant(std::string(tokens::kBlackbody))
            .SetColorTemperature(2800.0)
            .SetTransform(LookAtTransform(
                {0.0, -1.5, 3.5}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}));
        // The lasers on a bar 1.5 m from the wall, beams level and
        // square to it: collimated, 4 mm in radius, M^2 1.2.
        Xform::Define(scene, "/World/Lasers");
        for (std::size_t i = 0; i < kLasers.size(); ++i) {
            const glm::dvec3 from(SpotX(i), -1.5, kHeight);
            LaserLight::Define(scene, LaserPath(i))
                .SetWavelength(kLasers[i].nm)
                .SetPower(kLasers[i].milliwatts * 1e-3)
                .SetWaistRadius(glm::dvec2(0.004))
                .SetBeamQuality(glm::dvec2(1.2))
                .SetTransform(LookAtTransform(
                    from, from + glm::dvec3(0.0, 1.0, 0.0),
                    {0.0, 0.0, 1.0}));
        }
        // The four cameras side by side, square to the wall, exposed
        // automatically and without noise.
        for (std::size_t c = 0; c < kBands.size(); ++c) {
            const Band& band = kBands[c];
            const Path camera = CameraPath(c);
            Camera view = Camera::Define(scene, camera);
            view.SetFocalLength(kFocalLength)
                .SetHorizontalAperture(kAperture)
                .SetVerticalAperture(kAperture * kResolution.y /
                                     kResolution.x)
                .SetSensorNoise(false)
                .SetSpectralRange(band.range)
                .SetSpectralBands(band.bands)
                .SetExposureMode(std::string(tokens::kAuto))
                .SetTransform(LookAtTransform(
                    {0.0, -kDistance, kHeight}, {0.0, 0.0, kHeight},
                    {0.0, 0.0, 1.0}));
            if (!band.qe.empty()) {
                view.SetSensorType("mono").SetSensorQeMono(band.qe);
            }
            const Path ldr = Path(band.product).Child("LdrColor");
            RenderVar::Define(scene, ldr).SetSourceName("LdrColor");
            RenderProduct::Define(scene, band.product)
                .SetCamera(camera)
                .SetResolution(kResolution)
                .SetOrderedVars({ldr});
        }
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(CameraPath(0))
            .SetResolution(kResolution)
            .SetAmbient(0.0);

        // The demo's option: which camera the viewport looks through,
        // the viewer's own perspective camera first.
        std::vector<std::string> names = {"perspective"};
        for (const Band& band : kBands) {
            names.emplace_back(band.name);
        }
        Widgets().Combo(
            "Camera", std::move(names), 0, [this](std::size_t index) {
                if (GetViewer() != nullptr) {
                    GetViewer()->SetActiveCamera(
                        index == 0 ? app::Viewer::PerspectiveCamera()
                                   : CameraPath(index - 1));
                }
            });
    }

    /// The four cameras' images side by side.
    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, demos::SideBySide(RenderAll()));
    }

    void TestFinal() override {
        const std::vector<Tensor> images = RenderAll();
        // Per camera, which lasers' spots stand out of the wall
        // around them.
        const std::array<std::array<bool, 6>, 4> expected = {{
            {true, true, true, true, false, false},
            {false, false, true, false, false, false},
            {false, false, false, false, true, true},
            {true, true, true, true, true, true},
        }};
        for (std::size_t c = 0; c < images.size(); ++c) {
            const double around = Brightness(images[c], 0.0, 0.12);
            for (std::size_t i = 0; i < kLasers.size(); ++i) {
                const double spot =
                    Brightness(images[c], SpotX(i), 0.0);
                std::printf("%s sees %s: %.3f (wall %.3f)\n",
                            kBands[c].name, kLasers[i].name, spot,
                            around);
                const bool seen = spot > around + 0.1;
                Check(seen == expected[c][i],
                      std::string(kBands[c].name) +
                          (expected[c][i] ? " should see the "
                                          : " should not see the ") +
                          kLasers[i].name + " laser");
            }
        }
    }

  private:
    static Path LaserPath(std::size_t i) {
        return Path("/World/Lasers/Laser" + std::to_string(i));
    }

    static Path CameraPath(std::size_t c) {
        return Path("/World/Camera" + std::to_string(c));
    }

    /// Every camera's LdrColor, uint8 (rows, cols, 4).
    std::vector<Tensor> RenderAll() {
        std::vector<Path> products;
        for (const Band& band : kBands) {
            products.emplace_back(band.product);
        }
        RenderProductSetOutputs outputs =
            GetRenderer().Step(GetScene(), products);
        std::vector<Tensor> images;
        for (const Band& band : kBands) {
            images.push_back(outputs.at(Path(band.product))
                                 .vars.at("LdrColor")
                                 .Map(MapDevice::Cpu)
                                 .tensor);
        }
        return images;
    }

    /// The brightest value (0 to 1, the brightest channel) within 3
    /// pixels of the wall's point `x` along it and `z` above the
    /// spots.
    static double Brightness(const Tensor& image, double x,
                             double z) {
        const auto rows = static_cast<int>(image.Shape()[0]);
        const auto cols = static_cast<int>(image.Shape()[1]);
        const auto channels = static_cast<int>(image.Shape()[2]);
        const double pixels =
            kFocalLength / kAperture * cols / kDistance;
        const int col = static_cast<int>(0.5 * cols + x * pixels);
        const int row = static_cast<int>(0.5 * rows - z * pixels);
        const auto* data =
            static_cast<const std::uint8_t*>(image.Data());
        int brightest = 0;
        for (int r = std::max(row - 3, 0);
             r <= std::min(row + 3, rows - 1); ++r) {
            for (int c = std::max(col - 3, 0);
                 c <= std::min(col + 3, cols - 1); ++c) {
                for (int k = 0; k < std::min(channels, 3); ++k) {
                    brightest = std::max(
                        brightest,
                        static_cast<int>(
                            data[(r * cols + c) * channels + k]));
                }
            }
        }
        return brightest / 255.0;
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    WavelengthsDemo, "lasers", "wavelengths",
    "Six lasers from 405 to 940 nm seen by RGB, narrowband, NIR and "
    "wideband cameras.");
