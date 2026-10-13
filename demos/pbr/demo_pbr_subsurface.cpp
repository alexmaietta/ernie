// SPDX-License-Identifier: BSD-3-Clause
// OpenPBR's subsurface scattering: balls of skin, marble and jade lit
// from the side (subsurface_weight, subsurface_color,
// subsurface_radius and its scale per channel). Light entering each
// ball leaves it further on, softening the terminator and the shadow
// one ball casts on the next; in skin red travels furthest, so their
// edges glow red.
//
// References:
// - J. Andersson, A. Portsmouth et al., "OpenPBR Surface
//   Specification", v1.1, 2024 (the subsurface lobe).
// - P. H. Christensen and B. Burley, "Approximate Reflectance
//   Profiles for Efficient Subsurface Scattering", Pixar, 2015.
// - J. Jimenez et al., "Separable Subsurface Scattering", Computer
//   Graphics Forum 34(6), 2015.

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
constexpr double kSunAzimuth =
    90.0;  // degrees from behind: the right
constexpr double kSunElevation = 20.0;

struct Translucent {
    const char* name;
    glm::dvec3 base;
    glm::dvec3 colour;
    double radius;  // m
    glm::dvec3 scale;
};

const std::array<Translucent, 3> kMaterials = {{
    {"Skin",
     {0.8, 0.55, 0.45},
     {0.9, 0.55, 0.45},
     0.04,
     {1.0, 0.35, 0.2}},
    {"Marble",
     {0.85, 0.85, 0.83},
     {0.92, 0.92, 0.9},
     0.03,
     {1.0, 1.0, 1.0}},
    {"Jade",
     {0.3, 0.6, 0.4},
     {0.35, 0.8, 0.5},
     0.05,
     {0.6, 1.0, 0.6}},
}};

/// A dim latitude-longitude studio: a sky brightening upwards and a
/// dark floor, linear RGB.
Tensor Studio(int rows = 64, int cols = 128) {
    std::vector<float> texels(
        static_cast<std::size_t>(rows * cols * 3));
    for (int i = 0; i < rows; ++i) {
        const double e = 90.0 - 180.0 * (i + 0.5) / rows;
        const double sky =
            e > 0.0
                ? 0.1 + 0.3 * std::sin(e * std::numbers::pi / 180.0)
                : 0.02;
        for (int j = 0; j < cols * 3; ++j) {
            texels[static_cast<std::size_t>(i * cols * 3 + j)] =
                static_cast<float>(sky);
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

/// The mean linear RGB of a strip just past a ball's terminator, on
/// its side away from the sun.
glm::dvec3 ShadowSide(const Tensor& image, const glm::dvec3& point) {
    const glm::dvec3 at = Project(point);
    const double size = at.z * kRadius;
    const auto* texels =
        static_cast<const std::uint8_t*>(image.Data());
    const std::int64_t rows = image.Shape()[0];
    const std::int64_t cols = image.Shape()[1];
    const std::int64_t channels = image.Shape()[2];
    glm::dvec3 sum(0.0);
    double count = 0.0;
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t col = 0; col < cols; ++col) {
            const double x = (col + 0.5 - at.x) / size;
            const double y = (row + 0.5 - at.y) / size;
            if (x <= -0.3 || x >= -0.05 || std::abs(y) >= 0.3) {
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

std::string Look(const Translucent& material) {
    return "/World/Looks/" + std::string(material.name);
}

class SubsurfaceDemo : public app::Demo {
  public:
    SubsurfaceDemo(app::Viewer* viewer,
                   const app::DemoOptions& options)
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
            .SetIntensity(0.1)
            .SetTexture(Studio());
        const double a = kSunAzimuth * std::numbers::pi / 180.0;
        const double e = kSunElevation * std::numbers::pi / 180.0;
        DistantLight::Define(scene, "/World/Sun")
            .SetIntensity(3.0)
            .SetAngle(1.0)
            .SetTransform(DistantLightTransform(
                {std::sin(a) * std::cos(e), std::cos(a) * std::cos(e),
                 std::sin(e)}));
        for (int i = 0; i < static_cast<int>(kMaterials.size());
             ++i) {
            const Translucent& material =
                kMaterials[static_cast<std::size_t>(i)];
            const std::string look = Look(material);
            Material::Define(scene, look)
                .SetBrdf("open_pbr")
                .SetAlbedo(material.base);
            scene.Set(look, "inputs:specular_roughness", 0.35);
            scene.Set(look, "inputs:subsurface_weight", 1.0);
            scene.Set(look, "inputs:subsurface_color",
                      material.colour);
            scene.Set(look, "inputs:subsurface_radius",
                      material.radius);
            scene.Set(look, "inputs:subsurface_radius_scale_r",
                      material.scale.r);
            scene.Set(look, "inputs:subsurface_radius_scale_g",
                      material.scale.g);
            scene.Set(look, "inputs:subsurface_radius_scale_b",
                      material.scale.b);
            const std::string ball =
                "/World/" + std::string(material.name);
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

        Widgets().Slider("Subsurface weight", 0.0, 1.0, 1.0,
                         [this](double value) { Weight(value); });
        Widgets().Slider("Radius (times each material's)", 0.1, 4.0,
                         1.0,
                         [this](double times) { Radius(times); });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            viewer->LookAt(kEye, kTarget);
        }
    }

    void TestFinal() override {
        const glm::dvec3 weights(0.2126, 0.7152, 0.0722);
        const std::vector<glm::dvec3> scattered = ShadowSides();
        Weight(0.0);
        const std::vector<glm::dvec3> opaque = ShadowSides();
        Weight(1.0);
        for (std::size_t i = 0; i < kMaterials.size(); ++i) {
            const glm::dvec3& s = scattered[i];
            const glm::dvec3& o = opaque[i];
            std::printf("%s past its terminator: subsurface %.4f "
                        "(red/blue %.2f), "
                        "none %.4f (red/blue %.2f)\n",
                        kMaterials[i].name, glm::dot(s, weights),
                        s.r / s.b, glm::dot(o, weights), o.r / o.b);
        }
        for (std::size_t i = 0; i < kMaterials.size(); ++i) {
            Check(glm::dot(scattered[i], weights) >
                      1.2 * glm::dot(opaque[i], weights),
                  "light scattered beneath the surface lights its "
                  "shadowed side");
        }
        Check(scattered[0].r / scattered[0].b >
                  1.1 * opaque[0].r / opaque[0].b,
              "red light travels furthest in skin");
    }

  private:
    void Weight(double value) {
        for (const Translucent& material : kMaterials) {
            GetScene().Set(Look(material), "inputs:subsurface_weight",
                           value);
        }
    }

    void Radius(double times) {
        for (const Translucent& material : kMaterials) {
            GetScene().Set(Look(material), "inputs:subsurface_radius",
                           material.radius * times);
        }
    }

    std::vector<glm::dvec3> ShadowSides() {
        const Tensor image = GetRenderer().Render(GetScene());
        std::vector<glm::dvec3> sides;
        for (int i = 0; i < static_cast<int>(kMaterials.size());
             ++i) {
            sides.push_back(ShadowSide(image, BallCentre(i)));
        }
        return sides;
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    SubsurfaceDemo, "pbr", "subsurface",
    "OpenPBR's subsurface scattering: skin, marble and jade balls "
    "whose "
    "light bleeds past their terminators and shadows.");
