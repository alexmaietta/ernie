// SPDX-License-Identifier: BSD-3-Clause
// OpenPBR's rough diffuse: a Lambert ball beside one of
// base_diffuse_roughness (EON, an energy-preserving Oren-Nayar), lit
// by a sun near the camera. Rough, the ball flattens towards a full
// moon's look, its limb as bright as its middle.
//
// References:
// - J. Portsmouth, P. Kutz and S. Hill, "EON: A practical
//   energy-preserving rough diffuse BRDF", JCGT 14(1), 2025.
// - M. Oren and S. K. Nayar, "Generalization of Lambert's reflectance
//   model", SIGGRAPH 1994.

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

namespace {

using namespace ernie;
using demos::Check;

constexpr int kWidth = 720;
constexpr int kHeight = 360;
constexpr double kFocal = 50.0;     // mm
constexpr double kAperture = 36.0;  // mm, horizontal
constexpr glm::dvec3 kEye(0.0, -7.0, 0.0);
constexpr glm::dvec3 kTarget(0.0, 0.0, 0.0);
constexpr double kRadius = 0.8;
constexpr double kRoughness = 1.0;
// Degrees between the sun and the camera, seen from the balls.
constexpr double kPhase = 20.0;
constexpr glm::dvec3 kAlbedo(0.6, 0.58, 0.55);
constexpr std::array<glm::dvec3, 2> kCentres = {
    glm::dvec3(-1.0, 0.0, 0.0), glm::dvec3(1.0, 0.0, 0.0)};

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

class DiffuseRoughnessDemo : public app::Demo {
  public:
    DiffuseRoughnessDemo(app::Viewer* viewer,
                         const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // A dim sky, so the sun's shading shows.
        DomeLight::Define(scene, "/World/Sky").SetIntensity(0.02);
        DistantLight::Define(scene, "/World/Sun")
            .SetIntensity(0.8)
            .SetAngle(0.53);
        LightAt(kPhase);
        const std::array<const char*, 2> names = {"Lambert", "Rough"};
        for (std::size_t i = 0; i < names.size(); ++i) {
            const std::string look =
                "/World/Looks/" + std::string(names[i]);
            Material::Define(scene, look)
                .SetBrdf("open_pbr")
                .SetAlbedo(kAlbedo);
            // No specular, the diffuse alone.
            scene.Set(look, "inputs:specular_weight", 0.0);
            scene.Set(look, "inputs:base_diffuse_roughness",
                      i == 0 ? 0.0 : kRoughness);
            const std::string ball =
                "/World/" + std::string(names[i]);
            Sphere::Define(scene, ball)
                .SetRadius(kRadius)
                .SetMaterialBinding(look);
            SetXform(scene, ball, {.translate = kCentres[i]});
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
            "Diffuse roughness (right)", 0.0, 1.0, kRoughness,
            [this](double roughness) { Roughen(roughness); });
        Widgets().Slider(
            "Sun phase angle", 0.0, 150.0, kPhase,
            [this](double degrees) { LightAt(degrees); });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            viewer->LookAt(kEye, kTarget);
        }
    }

    void TestFinal() override {
        LightAt(0.0);
        const Tensor image = GetRenderer().Render(GetScene());
        const double lambert = LimbDarkening(image, 0);
        const double rough = LimbDarkening(image, 1);
        Roughen(0.0);
        const Tensor smooth = GetRenderer().Render(GetScene());
        Roughen(kRoughness);
        LightAt(kPhase);
        std::printf("Limb over middle with the sun behind the "
                    "camera: Lambert "
                    "%.3f, rough %.3f\n",
                    lambert, rough);
        Check(0.35 < lambert && lambert < 0.75,
              "Lambert's limb darkens as cos");
        Check(rough > lambert + 0.1,
              "the rough ball's limb is flatter");
        Check(std::abs(LimbDarkening(smooth, 1) - lambert) < 0.02,
              "roughness 0 is Lambert");
    }

  private:
    void Roughen(double roughness) {
        GetScene().Set("/World/Looks/Rough",
                       "inputs:base_diffuse_roughness", roughness);
    }

    /// The sun `degrees` round from behind the camera to the right, a
    /// little above.
    void LightAt(double degrees) {
        const double a = degrees * std::numbers::pi / 180.0;
        DistantLight(GetScene(), "/World/Sun")
            .SetTransform(DistantLightTransform(
                {std::sin(a), -std::cos(a), 0.1}));
    }

    /// The ball's limb over its middle, in linear luminance.
    static double LimbDarkening(const Tensor& image, std::size_t i) {
        const glm::dvec3 weights(0.2126, 0.7152, 0.0722);
        const double middle =
            glm::dot(DiscMean(image, kCentres[i], 0.0, 0.3), weights);
        const double limb = glm::dot(
            DiscMean(image, kCentres[i], 0.8, 0.95), weights);
        return limb / middle;
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    DiffuseRoughnessDemo, "pbr", "diffuse_roughness",
    "OpenPBR's EON rough diffuse beside Lambert: rough, a ball lit "
    "from the camera flattens like the full moon.");
