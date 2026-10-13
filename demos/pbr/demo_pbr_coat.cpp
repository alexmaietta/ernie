// SPDX-License-Identifier: BSD-3-Clause
// OpenPBR's coat: a clear (or tinted) layer over four rough bases, red
// and white plastic, black and a blue metal. A smooth coat lays a sharp
// reflection over the base's blurred one; its colour tints what passes
// through it and its darkening deepens the base beneath.
//
// References:
// - J. Andersson, A. Portsmouth et al., "OpenPBR Surface
//   Specification", v1.1, 2024 (the coat layer).
// - C. Kulla and A. Conty, "Revisiting Physically Based Shading at
//   Imageworks", SIGGRAPH 2017 course (layering by albedo scaling).

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <algorithm>
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

constexpr int kWidth = 800;
constexpr int kHeight = 320;
constexpr double kFocal = 50.0;     // mm
constexpr double kAperture = 36.0;  // mm, horizontal
constexpr glm::dvec3 kEye(0.0, -6.0, 1.0);
constexpr glm::dvec3 kTarget(0.0, 0.0, 0.45);
constexpr double kRadius = 0.4;

struct Base {
    const char* name;
    glm::dvec3 colour;
    double metalness;
    double roughness;  // the base's specular roughness
};

const std::array<Base, 4> kBases = {{
    {"Red", {0.6, 0.04, 0.03}, 0.0, 0.5},
    {"White", {0.8, 0.8, 0.8}, 0.0, 0.5},
    {"Black", {0.02, 0.02, 0.02}, 0.0, 0.6},
    {"BlueMetal", {0.15, 0.3, 0.75}, 1.0, 0.45},
}};

struct CoatColour {
    const char* name;
    glm::dvec3 colour;
};

const std::array<CoatColour, 3> kColours = {{
    {"Clear", {1.0, 1.0, 1.0}},
    {"Amber varnish", {1.0, 0.7, 0.35}},
    {"Blue tint", {0.6, 0.8, 1.0}},
}};

struct CoatInput {
    const char* label;
    const char* name;
    double low;
    double high;
    double value;
};

const std::array<CoatInput, 4> kCoat = {{
    {"Coat weight", "coat_weight", 0.0, 1.0, 1.0},
    {"Coat roughness", "coat_roughness", 0.0, 1.0, 0.0},
    {"Coat IOR", "coat_ior", 1.0, 2.5, 1.5},
    {"Coat darkening", "coat_darkening", 0.0, 1.0, 1.0},
}};

/// A latitude-longitude studio: a sky brightening upwards, a dim
/// floor and two soft boxes, linear RGB.
Tensor Studio(int rows = 64, int cols = 128) {
    std::vector<float> texels(static_cast<std::size_t>(rows * cols * 3));
    for (int i = 0; i < rows; ++i) {
        const double e = 90.0 - 180.0 * (i + 0.5) / rows;
        for (int j = 0; j < cols; ++j) {
            const double a = 360.0 * (j + 0.5) / cols;
            double sky =
                e > 0.0 ? 0.2 + 0.6 * std::sin(e * std::numbers::pi /
                                                180.0)
                        : 0.03;
            for (const double centre : {240.0, 330.0}) {
                if (std::abs(e - 35.0) < 15.0 &&
                    std::abs(a - centre) < 18.0) {
                    sky += 6.0;
                }
            }
            for (int c = 0; c < 3; ++c) {
                texels[static_cast<std::size_t>((i * cols + j) * 3 +
                                                c)] =
                    static_cast<float>(sky);
            }
        }
    }
    return Tensor::Copy<float>(texels, {rows, cols, 3});
}

/// The pixel (col, row) of a world point, and pixels per metre at its
/// depth.
glm::dvec3 Project(const glm::dvec3& point) {
    const glm::dvec3 forward = glm::normalize(kTarget - kEye);
    const glm::dvec3 right =
        glm::normalize(glm::cross(forward, glm::dvec3(0.0, 0.0, 1.0)));
    const glm::dvec3 up = glm::cross(right, forward);
    const glm::dvec3 d = point - kEye;
    const double scale =
        kFocal / kAperture * kWidth / glm::dot(d, forward);
    return {0.5 * kWidth + scale * glm::dot(d, right),
            0.5 * kHeight - scale * glm::dot(d, up), scale};
}

/// Calls `visit(texel)` for each pixel within `outer` of the ball's
/// radius about `point`, and at least `inner`.
template <typename Visit>
void ForDisc(const Tensor& image, const glm::dvec3& point, double inner,
             double outer, Visit visit) {
    const glm::dvec3 at = Project(point);
    const auto* texels = static_cast<const std::uint8_t*>(image.Data());
    const std::int64_t rows = image.Shape()[0];
    const std::int64_t cols = image.Shape()[1];
    const std::int64_t channels = image.Shape()[2];
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t col = 0; col < cols; ++col) {
            const double r = std::hypot(col + 0.5 - at.x,
                                        row + 0.5 - at.y) /
                             (at.z * kRadius);
            if (r >= inner && r < outer) {
                visit(texels + (row * cols + col) * channels);
            }
        }
    }
}

/// The mean linear RGB of the pixels between `inner` and `outer`
/// radii (fractions of the ball's) about the ball at `point`.
glm::dvec3 DiscMean(const Tensor& image, const glm::dvec3& point,
                    double inner, double outer) {
    glm::dvec3 sum(0.0);
    double count = 0.0;
    ForDisc(image, point, inner, outer, [&](const std::uint8_t* texel) {
        sum += glm::dvec3(texel[0], texel[1], texel[2]);
        count += 1.0;
    });
    return demos::SrgbToLinear(sum / (255.0 * count));
}

/// The brightest pixel's mean of R, G and B (0 to 255) on the ball at
/// `point`.
double DiscPeak(const Tensor& image, const glm::dvec3& point) {
    double peak = 0.0;
    ForDisc(image, point, 0.0, 0.95, [&](const std::uint8_t* texel) {
        peak = std::max(peak, (texel[0] + texel[1] + texel[2]) / 3.0);
    });
    return peak;
}

glm::dvec3 BallCentre(int i) {
    return {1.0 * i - 1.5, 0.0, kRadius};
}

class CoatDemo : public app::Demo {
  public:
    CoatDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Floor")
            .SetChecker(true)
            .SetCheckerSize(0.5)
            .SetCheckerColor0(glm::dvec3(0.06))
            .SetCheckerColor1(glm::dvec3(0.03));
        DomeLight::Define(scene, "/World/Sky")
            .SetIntensity(1.0)
            .SetTexture(Studio());
        DistantLight::Define(scene, "/World/Sun")
            .SetIntensity(2.0)
            .SetAngle(2.0)
            .SetTransform(DistantLightTransform({-0.4, -0.5, 0.77}));
        for (int i = 0; i < static_cast<int>(kBases.size()); ++i) {
            const Base& base = kBases[static_cast<std::size_t>(i)];
            const std::string look =
                "/World/Looks/" + std::string(base.name);
            Material::Define(scene, look)
                .SetBrdf("open_pbr")
                .SetAlbedo(base.colour);
            scene.Set(look, "inputs:base_metalness", base.metalness);
            scene.Set(look, "inputs:specular_roughness", base.roughness);
            for (const CoatInput& input : kCoat) {
                scene.Set(look, "inputs:" + std::string(input.name),
                          input.value);
            }
            scene.Set(look, "inputs:coat_color", kColours[0].colour);
            const std::string ball = "/World/" + std::string(base.name);
            Sphere::Define(scene, ball)
                .SetRadius(kRadius)
                .SetMaterialBinding(look);
            SetXform(scene, ball, {.translate = BallCentre(i)});
        }
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(kFocal)
            .SetHorizontalAperture(kAperture)
            .SetVerticalAperture(kAperture * kHeight / kWidth)
            .SetSensorNoise(false)
            .SetTransform(
                LookAtTransform(kEye, kTarget, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({kWidth, kHeight})
            .SetAmbient(0.0);

        for (const CoatInput& input : kCoat) {
            const std::string name = input.name;
            Widgets().Slider(input.label, input.low, input.high,
                             input.value, [this, name](double value) {
                                 Coat(name, value);
                             });
        }
        std::vector<std::string> names;
        for (const CoatColour& colour : kColours) {
            names.emplace_back(colour.name);
        }
        Widgets().Combo("Coat colour", std::move(names), 0,
                        [this](std::size_t index) {
                            Coat("coat_color", kColours[index].colour);
                        });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            viewer->LookAt(kEye, kTarget);
        }
    }

    void TestFinal() override {
        const std::vector<glm::dvec3> coated = Balls();
        const double sharp = BlackPeak();
        Coat("coat_weight", 0.0);
        const std::vector<glm::dvec3> bare = Balls();
        const double blurred = BlackPeak();
        Coat("coat_weight", 1.0);
        Coat("coat_color", kColours[1].colour);
        const std::vector<glm::dvec3> amber = Balls();
        Coat("coat_color", kColours[0].colour);
        Coat("coat_darkening", 0.0);
        const std::vector<glm::dvec3> undarkened = Balls();
        Coat("coat_darkening", 1.0);
        for (std::size_t i = 0; i < kBases.size(); ++i) {
            std::printf(
                "%s: coated %.3f %.3f %.3f, bare %.3f %.3f %.3f, amber "
                "%.3f %.3f %.3f, undarkened %.3f %.3f %.3f\n",
                kBases[i].name, coated[i].r, coated[i].g, coated[i].b,
                bare[i].r, bare[i].g, bare[i].b, amber[i].r, amber[i].g,
                amber[i].b, undarkened[i].r, undarkened[i].g,
                undarkened[i].b);
        }
        std::printf(
            "Black ball's brightest pixel coated %.0f, bare %.0f\n",
            sharp, blurred);
        const std::size_t white = 1;
        Check(sharp > 200.0 && blurred < 0.7 * sharp,
              "the smooth coat reflects the soft boxes sharply off the "
              "black ball");
        Check(amber[white].r / amber[white].b >
                  1.5 * coated[white].r / coated[white].b,
              "an amber coat warms the white ball");
        Check(undarkened[0].r > coated[0].r * 1.02,
              "darkening deepens the red ball under its coat");
    }

  private:
    /// Sets coat input `name` on every ball.
    template <typename T>
    void Coat(const std::string& name, const T& value) {
        for (const Base& base : kBases) {
            GetScene().Set("/World/Looks/" + std::string(base.name),
                           "inputs:" + name, value);
        }
    }

    std::vector<glm::dvec3> Balls() {
        const Tensor image = GetRenderer().Render(GetScene());
        std::vector<glm::dvec3> means;
        for (int i = 0; i < static_cast<int>(kBases.size()); ++i) {
            means.push_back(DiscMean(image, BallCentre(i), 0.0, 0.95));
        }
        return means;
    }

    double BlackPeak() {
        return DiscPeak(GetRenderer().Render(GetScene()), BallCentre(2));
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    CoatDemo, "pbr", "coat",
    "OpenPBR's coat over four bases: its weight, roughness, index, "
    "colour and darkening.");
