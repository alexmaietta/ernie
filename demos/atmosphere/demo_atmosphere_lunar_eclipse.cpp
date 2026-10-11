// SPDX-License-Identifier: BSD-3-Clause
// A lunar eclipse from the ground: the Moon in the Earth's shadow,
// lit red by sunlight the atmosphere refracts into it, darker with
// clouds.

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
constexpr double kEarthRadius = 6371.0088e3;
constexpr double kMoonDistance = 384400e3;
constexpr double kMoonElevation = 40.0;  // degrees

/// The Moon's offset from the shadow's axis (degrees, from the
/// Earth's centre), the clouds' tops, and an exposure.
struct View {
    const char* name;
    double offset;
    double cloudTop;  ///< Metres.
    double fStop;
    double exposureTime;  ///< Seconds, at ISO 100.
};

// The umbra is some 0.7 degrees in radius at the Moon, the penumbra
// 1.2.
const std::array<View, 4> kViews = {{
    {"Full Moon", 3.0, 0.0, 11.0, 1.0 / 125.0},
    {"Partial: the umbra's edge", 0.75, 0.0, 11.0, 1.0 / 30.0},
    {"Totality, clear air", 0.2, 0.0, 2.8, 1.0 / 2.0},
    {"Totality, clouds to 10 km", 0.2, 10e3, 2.8, 1.0 / 2.0},
}};

/// The mean linear colour of a square of `half` pixels either side of
/// the middle of a uint8 (rows, cols, 4) image.
glm::dvec3 MiddleLinearRgb(const Tensor& image, std::int64_t half) {
    const std::int64_t rows = image.Shape()[0];
    const std::int64_t cols = image.Shape()[1];
    const auto* pixels =
        static_cast<const std::uint8_t*>(image.Data());
    glm::dvec3 sum(0.0);
    for (std::int64_t row = rows / 2 - half; row < rows / 2 + half;
         ++row) {
        for (std::int64_t col = cols / 2 - half;
             col < cols / 2 + half; ++col) {
            const std::uint8_t* p = pixels + (row * cols + col) * 4;
            sum += demos::SrgbToLinear(glm::dvec3(p[0], p[1], p[2]) /
                                       255.0);
        }
    }
    return sum / static_cast<double>(4 * half * half);
}

class LunarEclipseDemo : public app::Demo {
  public:
    LunarEclipseDemo(app::Viewer* viewer,
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
            .SetAngle(0.53);
        CelestialBody::Define(scene, "/World/Moon")
            .SetRadius(1737.4e3)
            .SetAlbedo(glm::dvec3(0.12));
        // 1200 mm: the Moon fills half the frame's height.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(1200.0)
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
        Widgets().Slider("Offset from the axis", 0.0, 2.0,
                         kViews[0].offset,
                         [this](double degrees) { SunAt(degrees); });
        Widgets().Slider("Cloud tops (km)", 0.0, 15.0, 0.0,
                         [this](double km) {
                             Atmosphere(GetScene(), "/World/Air")
                                 .SetCloudTop(km * 1e3);
                         });
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
        std::array<glm::dvec3, kViews.size()> middle{};
        for (std::size_t i = 0; i < images.size(); ++i) {
            middle[i] = MiddleLinearRgb(images[i], 10);
            std::printf("%s: the Moon's middle %.4f %.4f %.4f\n",
                        kViews[i].name, middle[i].r, middle[i].g,
                        middle[i].b);
        }
        Check(middle[0].r > 0.05 && middle[0].b > 0.6 * middle[0].r,
              "the full Moon should be bright and grey");
        Check(middle[2].r > 3.0 * middle[2].b,
              "the eclipsed Moon should be red");
        // The same exposure: the clouds darken the umbra.
        Check(middle[3].r < 0.75 * middle[2].r,
              "clouds should darken the eclipse");
    }

  private:
    void Show(int index) {
        view_ = index;
        const View& view = kViews[static_cast<std::size_t>(index)];
        Scene& scene = GetScene();
        Atmosphere(scene, "/World/Air").SetCloudTop(view.cloudTop);
        // The Moon 40 degrees up from the Earth's centre; the camera
        // on the ground aims at it.
        const double e = kMoonElevation * kPi / 180.0;
        const glm::dvec3 centre(0.0, 0.0, -kEarthRadius);
        const glm::dvec3 moon =
            centre +
            kMoonDistance * glm::dvec3(std::cos(e), 0.0, std::sin(e));
        SetXform(scene, "/World/Moon", {.translate = moon});
        const glm::dvec3 eye(0.0, 0.0, 2.0);
        Camera(scene, "/World/Camera")
            .SetExposureFStop(view.fStop)
            .SetExposureTime(view.exposureTime)
            .SetTransform(
                LookAtTransform(eye, eye + glm::normalize(moon - eye),
                                {0.0, 0.0, 1.0}));
        SunAt(view.offset);
    }

    /// The sun opposite the Moon from the Earth's centre, `degrees`
    /// off the line, so the Moon lies that far from the shadow's
    /// axis.
    void SunAt(double degrees) {
        const double e = (kMoonElevation + degrees) * kPi / 180.0;
        DistantLight(GetScene(), "/World/Sun")
            .SetTransform(DistantLightTransform(
                -glm::dvec3(std::cos(e), 0.0, std::sin(e))));
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

ERNIE_REGISTER_DEMO(LunarEclipseDemo, "atmosphere", "lunar_eclipse",
                    "A lunar eclipse: the Moon red in the Earth's "
                    "shadow, darker with clouds.");
