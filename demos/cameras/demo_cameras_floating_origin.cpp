// SPDX-License-Identifier: BSD-3-Clause
// Metre-sized probes from the origin out to Neptune's distance, each
// as sharp as the last: the renderer draws relative to the eye.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

// Metres in an astronomical unit.
constexpr double kAu = 1.495978707e11;

struct Place {
    const char* key;
    const char* label;
    double distance;  // metres from the origin
};

// The places, each a probe at its distance from the origin along one
// direction: a red cube, a green sphere and a blue cube a metre
// apart, with a 10 cm cube on the sphere. Doubles hold such detail to
// about a millimetre out to the solar system's edge (1e13 m); beyond,
// they too run short.
const std::array<Place, 5> kPlaces = {{
    {"Home", "Home", 0.0},
    {"Moon", "The Moon's distance", 3.844e8},
    {"Mars", "Mars (1.52 AU)", 1.524 * kAu},
    {"Saturn", "Saturn (9.58 AU)", 9.583 * kAu},
    {"Neptune", "Neptune (30.07 AU)", 30.07 * kAu},
}};
// Where the camera starts, and the test looks: the farthest.
constexpr std::size_t kStart = kPlaces.size() - 1;

glm::dvec3 PositionAt(double distance) {
    return glm::normalize(glm::dvec3(1.0, 0.35, 0.12)) * distance;
}

/// The spacing of floating-point numbers near `distance` with a
/// `mantissa`-bit significand (23 for float32, 52 for float64).
double Step(double distance, int mantissa) {
    if (distance <= 0.0) {
        return 0.0;
    }
    return std::exp2(std::floor(std::log2(distance)) - mantissa);
}

std::string Length(double metres) {
    char text[32];
    const std::array<std::pair<const char*, double>, 3> units = {
        {{"km", 1e3}, {"m", 1.0}, {"mm", 1e-3}}};
    for (const auto& [unit, size] : units) {
        if (metres >= size) {
            std::snprintf(text, sizeof(text), "%.3g %s",
                          metres / size, unit);
            return text;
        }
    }
    std::snprintf(text, sizeof(text), "%.3g um", metres * 1e6);
    return text;
}

std::string Describe(std::size_t index) {
    const Place& place = kPlaces[index];
    if (place.distance == 0.0) {
        return std::string(place.label) + ": at the origin.";
    }
    return std::string(place.label) + ": " + Length(place.distance) +
           " out, where a float32 coordinate steps by " +
           Length(Step(place.distance, 23)) + " and a float64 by " +
           Length(Step(place.distance, 52)) + ".";
}

void DefineShape(Scene& scene, const Path& path, bool sphere,
                 double size, glm::dvec3 at, glm::dvec3 colour) {
    if (sphere) {
        Sphere::Define(scene, path)
            .SetRadius(size)
            .SetDisplayColor(colour);
    } else {
        Cube::Define(scene, path)
            .SetSize(size)
            .SetDisplayColor(colour);
    }
    SetXform(scene, path, {.translate = at});
}

class FloatingOriginDemo : public app::Demo {
  public:
    FloatingOriginDemo(app::Viewer* viewer,
                       const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        for (const Place& place : kPlaces) {
            const Path probe = Path("/World").Child(place.key);
            Xform::Define(scene, probe);
            SetXform(scene, probe,
                     {.translate = PositionAt(place.distance)});
            DefineShape(scene, probe.Child("Red"), false, 0.6,
                        {-1.0, 0.0, 0.0}, {0.8, 0.1, 0.1});
            DefineShape(scene, probe.Child("Green"), true, 0.35,
                        {0.0, 0.0, 0.0}, {0.1, 0.7, 0.2});
            DefineShape(scene, probe.Child("Blue"), false, 0.6,
                        {1.0, 0.0, 0.0}, {0.1, 0.2, 0.8});
            DefineShape(scene, probe.Child("Tip"), false, 0.1,
                        {0.0, 0.0, 0.4}, {0.9, 0.9, 0.9});
            probes_.push_back(probe);
        }
        DistantLight::Define(scene, "/World/Sun")
            .SetTransform(DistantLightTransform({-0.4, -0.8, 0.45}));
        // Looking at the farthest probe from 3.5 m; the viewer's
        // navigation drives it, fitting its clipping to the distance.
        const glm::dvec3 target =
            PositionAt(kPlaces[kStart].distance);
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetClippingRange({0.05, 1000.0})
            .SetExposureMode(std::string(tokens::kAuto))
            .SetTransform(
                LookAtTransform(target + glm::dvec3(0.0, -3.5, 0.6),
                                target, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({480, 270});
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }

        // The places, a button each, and where the view is.
        app::WidgetPanel& w = Widgets();
        where_ = w.Text(Describe(kStart));
        w.Separator();
        for (std::size_t i = 0; i < kPlaces.size(); ++i) {
            w.Button(kPlaces[i].label, [this, i] { Go(i); });
        }
    }

    void TestFinal() override {
        const Tensor image = RenderImage();
        const auto rows = static_cast<std::size_t>(image.Shape()[0]);
        const auto cols = static_cast<std::size_t>(image.Shape()[1]);
        const auto* data =
            static_cast<const std::uint8_t*>(image.Data());
        // The probe's three shapes, left to right, 30 AU out.
        const std::array<const char*, 3> names{"red", "green",
                                               "blue"};
        for (std::size_t third = 0; third < 3; ++third) {
            glm::dvec3 sum(0.0);
            double count = 0.0;
            for (std::size_t r = rows / 3; r < 2 * rows / 3; ++r) {
                for (std::size_t c = third * cols / 3;
                     c < (third + 1) * cols / 3; ++c) {
                    const std::uint8_t* p = data + (r * cols + c) * 4;
                    const glm::dvec3 rgb(p[0], p[1], p[2]);
                    if (rgb.r + rgb.g + rgb.b > 30.0) {
                        sum += rgb;
                        count += 1.0;
                    }
                }
            }
            Check(count > 0.0, "a shape in every third");
            const glm::dvec3 colour = sum / count;
            double others = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                if (k != third) {
                    others =
                        std::max(others, colour[static_cast<int>(k)]);
                }
            }
            Check(colour[static_cast<int>(third)] > 1.3 * others,
                  std::string("the ") + names[third] +
                      " shape should stand in its place");
        }
    }

  private:
    /// Frames place `index`'s probe with the viewer's camera.
    void Go(std::size_t index) {
        Widgets().Set(where_, Describe(index));
        if (app::Viewer* viewer = GetViewer()) {
            const Path probe = probes_[index];
            viewer->Frame(std::span<const Path>(&probe, 1));
        }
    }

    std::vector<Path> probes_;
    app::WidgetPanel::Id where_ = 0;
};

}  // namespace

ERNIE_REGISTER_DEMO(FloatingOriginDemo, "cameras", "floating origin",
                    "Probes out to Neptune's distance, each as sharp "
                    "as the last.");
