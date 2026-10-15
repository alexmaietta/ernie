// SPDX-License-Identifier: BSD-3-Clause
// OpenPBR's fuzz: velvet balls of dark cloth under a sheen of fibres
// (fuzz_weight, fuzz_color, fuzz_roughness), lit by a low sun from
// behind. The fibres catch the light at grazing, so the balls glow
// about their rims.
//
// References:
// - J. Andersson, A. Portsmouth et al., "OpenPBR Surface
//   Specification", v1.1, 2024 (the fuzz layer).
// - T. Zeltner, B. Burley and M. J.-Y. Chiang, "Practical
//   Multiple-Scattering Sheen Using Linearly Transformed Cosines",
//   SIGGRAPH 2022 Talks.

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

constexpr int kWidth = 720;
constexpr int kHeight = 320;
constexpr double kFocal = 50.0;     // mm
constexpr double kAperture = 36.0;  // mm, horizontal
constexpr glm::dvec3 kEye(0.0, -6.0, 1.0);
constexpr glm::dvec3 kTarget(0.0, 0.0, 0.5);
constexpr double kRadius = 0.5;
constexpr double kFuzzWeight = 1.0;
constexpr double kFuzzRoughness = 0.5;
constexpr double kSunAzimuth = 30.0;  // degrees from behind the balls
constexpr double kSunElevation = 15.0;

struct Cloth {
    const char* name;
    glm::dvec3 colour;
    glm::dvec3 fuzz;
};

const std::array<Cloth, 3> kCloths = {{
    {"Crimson", {0.2, 0.005, 0.02}, {1.0, 0.45, 0.55}},
    {"Navy", {0.01, 0.02, 0.12}, {0.45, 0.6, 1.0}},
    {"Black", {0.01, 0.01, 0.01}, {0.8, 0.8, 0.8}},
}};

/// A latitude-longitude studio: a sky brightening upwards, a dim
/// floor and two soft boxes, linear RGB.
Tensor Studio(int rows = 64, int cols = 128) {
    std::vector<float> texels(
        static_cast<std::size_t>(rows * cols * 3));
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
    const glm::dvec3 right = glm::normalize(
        glm::cross(forward, glm::dvec3(0.0, 0.0, 1.0)));
    const glm::dvec3 up = glm::cross(right, forward);
    const glm::dvec3 d = point - kEye;
    const double scale =
        kFocal / kAperture * kWidth / glm::dot(d, forward);
    return {0.5 * kWidth + scale * glm::dot(d, right),
            0.5 * kHeight - scale * glm::dot(d, up), scale};
}

/// The mean linear RGB of the pixels between `inner` and `outer`
/// radii (fractions of the ball's) about the ball at `point`.
glm::dvec3 DiscMean(const Tensor& image, const glm::dvec3& point,
                    double inner, double outer) {
    const glm::dvec3 at = Project(point);
    const auto* texels =
        static_cast<const std::uint8_t*>(image.Data());
    const std::int64_t rows = image.Shape()[0];
    const std::int64_t cols = image.Shape()[1];
    const std::int64_t channels = image.Shape()[2];
    glm::dvec3 sum(0.0);
    double count = 0.0;
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t col = 0; col < cols; ++col) {
            const double r =
                std::hypot(col + 0.5 - at.x, row + 0.5 - at.y) /
                (at.z * kRadius);
            if (r < inner || r >= outer) {
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
    return {1.3 * (i - 1), 0.0, kRadius};
}

class FuzzDemo : public app::Demo {
  public:
    FuzzDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Floor")
            .SetChecker(true)
            .SetCheckerSize(0.5)
            .SetCheckerColor0(glm::dvec3(0.06))
            .SetCheckerColor1(glm::dvec3(0.03));
        // A dim studio, so the sun's grazing light shows.
        DomeLight::Define(scene, "/World/Sky")
            .SetIntensity(0.15)
            .SetTexture(Studio());
        DistantLight::Define(scene, "/World/Sun")
            .SetIntensity(3.0)
            .SetAngle(2.0);
        LightFrom(kSunAzimuth);
        for (int i = 0; i < static_cast<int>(kCloths.size()); ++i) {
            const Cloth& cloth = kCloths[static_cast<std::size_t>(i)];
            const std::string look =
                "/World/Looks/" + std::string(cloth.name);
            Material::Define(scene, look)
                .SetBrdf("open_pbr")
                .SetAlbedo(cloth.colour);
            // Matte cloth: rough diffuse, no specular.
            scene.Set(look, "inputs:base_diffuse_roughness", 1.0);
            scene.Set(look, "inputs:specular_weight", 0.0);
            scene.Set(look, "inputs:fuzz_weight", kFuzzWeight);
            scene.Set(look, "inputs:fuzz_roughness", kFuzzRoughness);
            scene.Set(look, "inputs:fuzz_color", cloth.fuzz);
            const std::string ball =
                "/World/" + std::string(cloth.name);
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

        Widgets().Slider(
            "Fuzz weight", 0.0, 1.0, kFuzzWeight,
            [this](double value) { Fuzz("fuzz_weight", value); });
        Widgets().Slider(
            "Fuzz roughness", 0.0, 1.0, kFuzzRoughness,
            [this](double value) { Fuzz("fuzz_roughness", value); });
        Widgets().Slider("Sun azimuth (0 behind the balls)", 0.0,
                         180.0, kSunAzimuth, [this](double degrees) {
                             LightFrom(degrees);
                         });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            viewer->LookAt(kEye, kTarget);
        }
    }

    void TestFinal() override {
        const std::vector<double> fuzzy = Rims();
        Fuzz("fuzz_weight", 0.0);
        const std::vector<double> bare = Rims();
        Fuzz("fuzz_weight", kFuzzWeight);
        Fuzz("fuzz_roughness", 1.0);
        const std::vector<double> rough = Rims();
        Fuzz("fuzz_roughness", kFuzzRoughness);
        for (std::size_t i = 0; i < kCloths.size(); ++i) {
            std::printf("%s's rim: fuzz %.4f, none %.4f, roughest "
                        "fuzz %.4f\n",
                        kCloths[i].name, fuzzy[i], bare[i], rough[i]);
        }
        for (std::size_t i = 0; i < kCloths.size(); ++i) {
            Check(fuzzy[i] > 2.0 * bare[i],
                  "the fuzz lights the rims");
        }
        Check(
            rough[2] > 1.5 * fuzzy[2],
            "rougher fibres spread the sheen further over the balls");
    }

  private:
    /// Sets fuzz input `name` on every ball.
    void Fuzz(const std::string& name, double value) {
        for (const Cloth& cloth : kCloths) {
            GetScene().Set("/World/Looks/" + std::string(cloth.name),
                           "inputs:" + name, value);
        }
    }

    void LightFrom(double degrees) {
        const double a = degrees * std::numbers::pi / 180.0;
        const double e = kSunElevation * std::numbers::pi / 180.0;
        DistantLight(GetScene(), "/World/Sun")
            .SetTransform(DistantLightTransform(
                {std::sin(a) * std::cos(e), std::cos(a) * std::cos(e),
                 std::sin(e)}));
    }

    /// Each ball's rim, in linear luminance.
    std::vector<double> Rims() {
        const Tensor image = GetRenderer().Render(GetScene());
        const glm::dvec3 weights(0.2126, 0.7152, 0.0722);
        std::vector<double> rims;
        for (int i = 0; i < static_cast<int>(kCloths.size()); ++i) {
            rims.push_back(glm::dot(
                DiscMean(image, BallCentre(i), 0.8, 0.97), weights));
        }
        return rims;
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    FuzzDemo, "pbr", "fuzz",
    "OpenPBR's fuzz: velvet balls whose fibres glow about their rims "
    "in a low back light.");
