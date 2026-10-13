// SPDX-License-Identifier: BSD-3-Clause
// OpenPBR's transmission before a wall of coloured tiles: a solid
// glass ball refracting the wall by its specular_ior
// (geometry_thin_walled 0), a tinted thin-walled pane letting it
// through unbent (transmission_color), and beside them plain alpha
// blending, a UsdPreviewSurface panel of opacity below 1. Transparent
// surfaces draw over the opaque scene, the nearest one per pixel.
//
// References:
// - J. Andersson, A. Portsmouth et al., "OpenPBR Surface
//   Specification", v1.1, 2024 (transmission and thin walls).
// - Pixar, "UsdPreviewSurface Specification" (opacity).

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
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
#include <utility>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr int kWidth = 800;
constexpr int kHeight = 360;
constexpr double kFocal = 40.0;     // mm
constexpr double kAperture = 36.0;  // mm, horizontal
constexpr glm::dvec3 kEye(0.0, -6.0, 1.1);
constexpr glm::dvec3 kTarget(0.0, 0.0, 0.8);
constexpr double kIor = 1.5;
constexpr double kOpacity = 0.5;
constexpr double kRadius = 0.6;
constexpr glm::dvec3 kBall(-1.6, 0.0, 0.8);
constexpr glm::dvec3 kPane(0.0, 0.0, 0.8);
constexpr glm::dvec3 kPanel(1.6, 0.0, 0.8);

struct Tint {
    const char* name;
    glm::dvec3 colour;
};

const std::array<Tint, 3> kTints = {{
    {"Green", {0.2, 0.9, 0.3}},
    {"Amber", {1.0, 0.6, 0.1}},
    {"Clear", {1.0, 1.0, 1.0}},
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

/// Hue, saturation and value (0 to 1) to RGB, as Python's colorsys.
glm::dvec3 HsvToRgb(double h, double s, double v) {
    const int i = static_cast<int>(h * 6.0) % 6;
    const double f = h * 6.0 - std::floor(h * 6.0);
    const double p = v * (1.0 - s);
    const double q = v * (1.0 - s * f);
    const double t = v * (1.0 - s * (1.0 - f));
    switch (i) {
    case 0:
        return {v, t, p};
    case 1:
        return {q, v, p};
    case 2:
        return {p, v, t};
    case 3:
        return {p, q, v};
    case 4:
        return {t, p, v};
    default:
        return {v, p, q};
    }
}

/// The wall's tile in column `i` and row `j`: hues round the wheel
/// along it, light and dark rows alternating, linear RGB.
glm::dvec3 TileColour(int i, int j) {
    const double value = (i + j) % 2 == 0 ? 0.8 : 0.35;
    return HsvToRgb(i / 32.0, 0.8, value);
}

/// A vertical quad facing the camera.
void DefineQuad(Scene& scene, const Path& path, const glm::dvec3& centre,
                const Path& material, double width = 1.0,
                double height = 1.2) {
    const double w = 0.5 * width;
    const double h = 0.5 * height;
    const std::vector<double> points = {
        centre.x - w, 0.0, centre.z - h, centre.x + w, 0.0, centre.z - h,
        centre.x + w, 0.0, centre.z + h, centre.x - w, 0.0, centre.z + h};
    Mesh::Define(scene, path)
        .SetPoints(Tensor::Copy<double>(points, {4, 3}))
        .SetFaceVertexCounts(Tensor::Copy<std::int64_t>(
            std::vector<std::int64_t>{4}, {1}))
        .SetFaceVertexIndices(Tensor::Copy<std::int64_t>(
            std::vector<std::int64_t>{0, 1, 2, 3}, {4}))
        .SetMaterialBinding(material);
}

/// The pixel (col, row) of a world point.
glm::dvec2 Project(const glm::dvec3& point) {
    const glm::dvec3 forward = glm::normalize(kTarget - kEye);
    const glm::dvec3 right =
        glm::normalize(glm::cross(forward, glm::dvec3(0.0, 0.0, 1.0)));
    const glm::dvec3 up = glm::cross(right, forward);
    const glm::dvec3 d = point - kEye;
    const double scale =
        kFocal / kAperture * kWidth / glm::dot(d, forward);
    return {0.5 * kWidth + scale * glm::dot(d, right),
            0.5 * kHeight - scale * glm::dot(d, up)};
}

/// The RGB (0 to 255) of the pixels within `half` of `point`.
std::vector<glm::dvec3> Window(const Tensor& image,
                               const glm::dvec3& point, int half = 20) {
    const glm::dvec2 at = Project(point);
    const int col0 = static_cast<int>(at.x) - half;
    const int row0 = static_cast<int>(at.y) - half;
    const auto* texels = static_cast<const std::uint8_t*>(image.Data());
    const std::int64_t cols = image.Shape()[1];
    const std::int64_t channels = image.Shape()[2];
    std::vector<glm::dvec3> pixels;
    for (int r = 0; r < 2 * half; ++r) {
        for (int c = 0; c < 2 * half; ++c) {
            const std::uint8_t* texel =
                texels + ((row0 + r) * cols + col0 + c) * channels;
            pixels.emplace_back(texel[0], texel[1], texel[2]);
        }
    }
    return pixels;
}

glm::dvec3 Mean(const std::vector<glm::dvec3>& pixels) {
    glm::dvec3 sum(0.0);
    for (const glm::dvec3& p : pixels) {
        sum += p;
    }
    return sum / static_cast<double>(pixels.size());
}

/// The mean absolute difference of two windows' channels.
double Difference(const std::vector<glm::dvec3>& a,
                  const std::vector<glm::dvec3>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const glm::dvec3 d = glm::abs(a[i] - b[i]);
        sum += d.r + d.g + d.b;
    }
    return sum / (3.0 * static_cast<double>(a.size()));
}

double Grey(const glm::dvec3& rgb) {
    return (rgb.r + rgb.g + rgb.b) / 3.0;
}

class TransmissionDemo : public app::Demo {
  public:
    TransmissionDemo(app::Viewer* viewer,
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
            .SetIntensity(1.0)
            .SetTexture(Studio());
        // High, away from the panes' mirror direction to the camera.
        DistantLight::Define(scene, "/World/Sun")
            .SetIntensity(2.0)
            .SetAngle(2.0)
            .SetTransform(DistantLightTransform({-0.3, -0.3, 0.9}));
        // A wall of 32 x 12 tiles of 0.25 m, 2.5 m behind.
        Xform::Define(scene, "/World/Wall");
        for (int i = 0; i < 32; ++i) {
            for (int j = 0; j < 12; ++j) {
                const std::string tile = "/World/Wall/T" +
                                         std::to_string(i) + "_" +
                                         std::to_string(j);
                Cube::Define(scene, tile)
                    .SetSize(1.0)
                    .SetDisplayColor(TileColour(i, j));
                SetXform(scene, tile,
                         {.translate = glm::dvec3(-4.0 + 0.25 * (i + 0.5),
                                                  2.5, 0.25 * (j + 0.5)),
                          .scale = glm::dvec3(0.25, 0.1, 0.25)});
            }
        }

        const Path glass("/World/Looks/Glass");
        Material::Define(scene, glass).SetBrdf("open_pbr");
        scene.Set(glass, "inputs:specular_roughness", 0.0);
        scene.Set(glass, "inputs:specular_ior", kIor);
        scene.Set(glass, "inputs:transmission_weight", 1.0);
        Sphere::Define(scene, "/World/Ball")
            .SetRadius(kRadius)
            .SetMaterialBinding(glass);
        SetXform(scene, "/World/Ball", {.translate = kBall});
        const Path tinted("/World/Looks/Tinted");
        Material::Define(scene, tinted).SetBrdf("open_pbr");
        scene.Set(tinted, "inputs:specular_roughness", 0.0);
        scene.Set(tinted, "inputs:transmission_weight", 1.0);
        scene.Set(tinted, "inputs:transmission_color", kTints[0].colour);
        scene.Set(tinted, "inputs:geometry_thin_walled", 1.0);
        DefineQuad(scene, "/World/Pane", kPane, tinted);
        // Alpha blending: a white UsdPreviewSurface, partly opaque.
        const Path look("/World/Looks/Frosted");
        Material::Define(scene, look);
        Shader::Define(scene, look.Child("Surface"))
            .SetInfoId(tokens::kUsdPreviewSurface)
            .SetDiffuseColor(glm::dvec3(0.6))
            .SetOpacity(kOpacity);
        Connect(scene, look, tokens::kSurfaceOutput, look.Child("Surface"),
                tokens::kSurfaceOutput);
        DefineQuad(scene, "/World/Panel", kPanel, look);
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

        Widgets().Slider("Ball's IOR", 1.0, 2.5, kIor, [this](double ior) {
            GetScene().Set("/World/Looks/Glass", "inputs:specular_ior",
                           ior);
        });
        Widgets().Checkbox("Ball thin-walled", false, [this](bool thin) {
            GetScene().Set("/World/Looks/Glass",
                           "inputs:geometry_thin_walled",
                           thin ? 1.0 : 0.0);
        });
        Widgets().Slider("Transmission", 0.0, 1.0, 1.0,
                         [this](double weight) { Transmit(weight); });
        std::vector<std::string> names;
        for (const Tint& tint : kTints) {
            names.emplace_back(tint.name);
        }
        Widgets().Combo("Pane's colour", std::move(names), 0,
                        [this](std::size_t index) {
                            GetScene().Set("/World/Looks/Tinted",
                                           "inputs:transmission_color",
                                           kTints[index].colour);
                        });
        Widgets().Slider("Panel's opacity", 0.0, 1.0, kOpacity,
                         [this](double opacity) { SetOpacity(opacity); });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            viewer->LookAt(kEye, kTarget);
        }
    }

    void TestFinal() override {
        Scene& scene = GetScene();
        const Tensor image = RenderNow();
        const Tensor wall = Hidden();
        scene.Set("/World/Looks/Glass", "inputs:specular_ior", 1.0);
        const Tensor unbent = RenderNow();
        scene.Set("/World/Looks/Glass", "inputs:specular_ior", kIor);
        SetOpacity(1.0);
        const Tensor opaque = RenderNow();
        SetOpacity(kOpacity);

        // Most of the ball, which turns the wall over about its edge.
        const std::vector<glm::dvec3> behindBall = Window(wall, kBall, 55);
        const double bent = Difference(Window(image, kBall, 55), behindBall);
        const double straight =
            Difference(Window(unbent, kBall, 55), behindBall);
        const glm::dvec3 pane = Mean(Window(image, kPane));
        const glm::dvec3 behind = Mean(Window(wall, kPane));
        const double half = Grey(Mean(Window(image, kPanel)));
        const double full = Grey(Mean(Window(opaque, kPanel)));
        const double through = Grey(Mean(Window(wall, kPanel)));
        std::printf(
            "Ball against the wall: refracted %.1f, at IOR 1 %.1f; pane "
            "%.1f %.1f %.1f over the wall %.1f %.1f %.1f; panel %.1f "
            "between opaque %.1f and the wall %.1f\n",
            bent, straight, pane.r, pane.g, pane.b, behind.r, behind.g,
            behind.b, half, full, through);
        Check(bent > 2.0 * straight + 10.0, "the glass ball refracts");
        Check(pane.g / pane.r > 1.3 * behind.g / behind.r &&
                  pane.g / pane.b > 1.3 * behind.g / behind.b,
              "the pane tints the wall green");
        const double low = std::min(full, through);
        const double high = std::max(full, through);
        Check(low + 0.2 * (high - low) < half &&
                  half < high - 0.2 * (high - low),
              "the half-opaque panel blends with the wall");
    }

  private:
    void Transmit(double weight) {
        for (const char* look : {"/World/Looks/Glass", "/World/Looks/Tinted"}) {
            GetScene().Set(look, "inputs:transmission_weight", weight);
        }
    }

    void SetOpacity(double opacity) {
        Shader(GetScene(), "/World/Looks/Frosted/Surface")
            .SetOpacity(opacity);
    }

    Tensor RenderNow() {
        return GetRenderer().Render(GetScene());
    }

    /// The view without the ball, pane and panel.
    Tensor Hidden() {
        Scene& scene = GetScene();
        const std::array<const char*, 3> names = {
            "/World/Ball", "/World/Pane", "/World/Panel"};
        for (const char* name : names) {
            scene.Set(name, tokens::kVisibility,
                      std::string(tokens::kInvisible));
        }
        Tensor image = RenderNow();
        for (const char* name : names) {
            scene.Set(name, tokens::kVisibility,
                      std::string(tokens::kInherited));
        }
        return image;
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    TransmissionDemo, "pbr", "transmission",
    "OpenPBR's transmission: a refracting glass ball, a tinted "
    "thin-walled pane and an alpha-blended panel before coloured "
    "tiles.");
