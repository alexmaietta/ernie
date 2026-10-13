// SPDX-License-Identifier: BSD-3-Clause
// Metals in OpenPBR: base_metalness 1, base_color their reflectance
// head on and specular_color their F82 tint towards grazing (Hoffman
// 2023), in a studio of two soft boxes over a dim floor.
//
// References:
// - J. Andersson, A. Portsmouth et al., "OpenPBR Surface
//   Specification", v1.1, 2024 (the metal's F82-tint Fresnel).
// - N. Hoffman, "Generalization of Adobe's Fresnel Model", 2023.

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
constexpr glm::dvec3 kEye(0.0, -7.0, 1.2);
constexpr glm::dvec3 kTarget(0.0, 0.0, 0.45);
constexpr double kRadius = 0.4;
constexpr double kRoughness = 0.2;

struct Metal {
    const char* name;
    glm::dvec3 colour;
    glm::dvec3 tint;
};

// Linear sRGB reflectance head on and F82 tint, approximated from
// measured indices of refraction.
const std::array<Metal, 5> kMetals = {{
    {"Gold", {1.0, 0.78, 0.34}, {0.98, 0.93, 0.74}},
    {"Copper", {0.96, 0.64, 0.54}, {0.99, 0.89, 0.83}},
    {"Aluminium", {0.92, 0.92, 0.93}, {0.97, 0.98, 0.99}},
    {"Silver", {0.97, 0.96, 0.92}, {1.0, 1.0, 0.99}},
    {"Titanium", {0.54, 0.50, 0.45}, {0.85, 0.83, 0.80}},
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

/// The mean linear RGB of the pixels between `inner` and `outer`
/// radii (fractions of the ball's) about the ball at `point`; with
/// `upper`, its upper half only.
glm::dvec3 DiscMean(const Tensor& image, const glm::dvec3& point,
                    double inner, double outer, bool upper = false) {
    const glm::dvec3 at = Project(point);
    const double radius = at.z * kRadius;
    const auto* texels = static_cast<const std::uint8_t*>(image.Data());
    const std::int64_t rows = image.Shape()[0];
    const std::int64_t cols = image.Shape()[1];
    const std::int64_t channels = image.Shape()[2];
    glm::dvec3 sum(0.0);
    double count = 0.0;
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t col = 0; col < cols; ++col) {
            const double r =
                std::hypot(col + 0.5 - at.x, row + 0.5 - at.y) / radius;
            if (r < inner || r >= outer || (upper && row + 0.5 >= at.y)) {
                continue;
            }
            for (int c = 0; c < 3; ++c) {
                sum[c] += texels[(row * cols + col) * channels + c];
            }
            count += 1.0;
        }
    }
    return demos::SrgbToLinear(sum / (255.0 * count));
}

glm::dvec3 BallCentre(int i) {
    return {1.0 * (i - 2), 0.0, kRadius};
}

class MetalsDemo : public app::Demo {
  public:
    MetalsDemo(app::Viewer* viewer, const app::DemoOptions& options)
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
        for (int i = 0; i < static_cast<int>(kMetals.size()); ++i) {
            const Metal& metal = kMetals[static_cast<std::size_t>(i)];
            const std::string look =
                "/World/Looks/" + std::string(metal.name);
            Material::Define(scene, look)
                .SetBrdf("open_pbr")
                .SetAlbedo(metal.colour);
            scene.Set(look, "inputs:base_metalness", 1.0);
            scene.Set(look, "inputs:specular_roughness", kRoughness);
            scene.Set(look, "inputs:specular_color", metal.tint);
            const std::string ball = "/World/" + std::string(metal.name);
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

        Widgets().Slider("Roughness", 0.0, 1.0, kRoughness,
                         [this](double roughness) { Roughen(roughness); });
        Widgets().Combo("Edge tint", {"F82 tint", "None (white)"}, 0,
                        [this](std::size_t index) { Tint(index); });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            viewer->LookAt(kEye, kTarget);
        }
    }

    void TestFinal() override {
        const Tensor image = RenderNow();
        std::vector<glm::dvec3> balls;
        for (int i = 0; i < static_cast<int>(kMetals.size()); ++i) {
            balls.push_back(DiscMean(image, BallCentre(i), 0.0, 0.9));
        }
        // Gold's upper rim, which reflects the sky near grazing.
        const glm::dvec3 tinted =
            DiscMean(image, BallCentre(0), 0.93, 0.99, true);
        Tint(1);
        const glm::dvec3 plain =
            DiscMean(RenderNow(), BallCentre(0), 0.93, 0.99, true);
        Tint(0);
        Roughen(0.7);
        const Tensor rough = RenderNow();
        Roughen(kRoughness);
        for (std::size_t i = 0; i < kMetals.size(); ++i) {
            std::printf("%s: %.3f %.3f %.3f\n", kMetals[i].name,
                        balls[i].r, balls[i].g, balls[i].b);
        }
        std::printf(
            "Gold's rim tinted %.3f %.3f %.3f, untinted %.3f %.3f "
            "%.3f\n",
            tinted.r, tinted.g, tinted.b, plain.r, plain.g, plain.b);
        const glm::dvec3& gold = balls[0];
        const glm::dvec3& copper = balls[1];
        const glm::dvec3& aluminium = balls[2];
        Check(gold.r > 1.5 * gold.b, "gold is yellow");
        Check(copper.r > 1.3 * copper.g, "copper is red");
        Check(std::max({aluminium.r, aluminium.g, aluminium.b}) <
                  1.1 * std::min({aluminium.r, aluminium.g, aluminium.b}),
              "aluminium is neutral");
        Check(tinted.b / tinted.r < 0.96 * plain.b / plain.r,
              "the F82 tint colours gold's grazing reflection");
        Check(MeanDifference(rough, image) > 1.0,
              "roughness blurs the reflections");
    }

  private:
    void Roughen(double roughness) {
        for (const Metal& metal : kMetals) {
            GetScene().Set("/World/Looks/" + std::string(metal.name),
                           "inputs:specular_roughness", roughness);
        }
    }

    void Tint(std::size_t index) {
        for (const Metal& metal : kMetals) {
            GetScene().Set("/World/Looks/" + std::string(metal.name),
                           "inputs:specular_color",
                           index == 0 ? metal.tint : glm::dvec3(1.0));
        }
    }

    Tensor RenderNow() {
        return GetRenderer().Render(GetScene());
    }

    /// The mean absolute difference of two uint8 images' bytes.
    static double MeanDifference(const Tensor& a, const Tensor& b) {
        const auto* x = static_cast<const std::uint8_t*>(a.Data());
        const auto* y = static_cast<const std::uint8_t*>(b.Data());
        const std::int64_t count = a.Elements();
        double sum = 0.0;
        for (std::int64_t i = 0; i < count; ++i) {
            sum += std::abs(int(x[i]) - int(y[i]));
        }
        return sum / static_cast<double>(count);
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    MetalsDemo, "pbr", "metals",
    "OpenPBR metals: base colour head on, F82 edge tint towards "
    "grazing, with a roughness slider.");
