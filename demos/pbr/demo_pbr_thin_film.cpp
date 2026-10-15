// SPDX-License-Identifier: BSD-3-Clause
// OpenPBR's thin film: iridescence from a film a fraction of a micron
// thick on the base's specular (thin_film_weight,
// thin_film_thickness, thin_film_ior), like the colours of
// heat-tinted steel, anodized titanium or oil on black glass. The
// balls' films thicken from left to right through the interference
// colours, which shift with the angle of view.
//
// References:
// - J. Andersson, A. Portsmouth et al., "OpenPBR Surface
//   Specification", v1.1, 2024 (the thin film).
// - L. Belcour and P. Barla, "A Practical Extension to Microfacet
//   Theory for the Modeling of Varying Iridescence", ACM Transactions
//   on Graphics 36(4), 2017.

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
constexpr double kThickness = 0.25;  // um, the middle ball's film
// Each ball's film over the middle one's.
constexpr std::array<double, 5> kScales = {0.4, 0.7, 1.0, 1.3, 1.6};
constexpr double kFilmIor = 2.0;

struct Base {
    const char* name;
    glm::dvec3 colour;
    double metalness;
    double roughness;  // specular roughness
};

const std::array<Base, 3> kBases = {{
    {"Steel", {0.56, 0.57, 0.58}, 1.0, 0.15},
    {"Titanium", {0.54, 0.50, 0.45}, 1.0, 0.15},
    {"Black glass", {0.01, 0.01, 0.01}, 0.0, 0.05},
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

/// The RGB (0 to 255) of the pixels on the ball at `point`.
std::vector<glm::dvec3> DiscPixels(const Tensor& image,
                                   const glm::dvec3& point) {
    const glm::dvec3 at = Project(point);
    const auto* texels =
        static_cast<const std::uint8_t*>(image.Data());
    const std::int64_t rows = image.Shape()[0];
    const std::int64_t cols = image.Shape()[1];
    const std::int64_t channels = image.Shape()[2];
    std::vector<glm::dvec3> pixels;
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t col = 0; col < cols; ++col) {
            const double r =
                std::hypot(col + 0.5 - at.x, row + 0.5 - at.y) /
                (at.z * kRadius);
            if (r < 0.95) {
                const std::uint8_t* texel =
                    texels + (row * cols + col) * channels;
                pixels.emplace_back(texel[0], texel[1], texel[2]);
            }
        }
    }
    return pixels;
}

/// The mean of each pixel's largest channel less its smallest: how
/// coloured they look.
double Chroma(const std::vector<glm::dvec3>& pixels) {
    double sum = 0.0;
    for (const glm::dvec3& p : pixels) {
        sum += std::max({p.r, p.g, p.b}) - std::min({p.r, p.g, p.b});
    }
    return sum / static_cast<double>(pixels.size());
}

glm::dvec3 Mean(const std::vector<glm::dvec3>& pixels) {
    glm::dvec3 sum(0.0);
    for (const glm::dvec3& p : pixels) {
        sum += p;
    }
    return sum / static_cast<double>(pixels.size());
}

glm::dvec3 BallCentre(int i) {
    return {1.0 * (i - 2), 0.0, kRadius};
}

std::string LookOf(std::size_t i) {
    return "/World/Looks/Film" + std::to_string(i);
}

class ThinFilmDemo : public app::Demo {
  public:
    ThinFilmDemo(app::Viewer* viewer, const app::DemoOptions& options)
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
        for (std::size_t i = 0; i < kScales.size(); ++i) {
            Material::Define(scene, LookOf(i)).SetBrdf("open_pbr");
            scene.Set(LookOf(i), "inputs:thin_film_weight", 1.0);
            scene.Set(LookOf(i), "inputs:thin_film_ior", kFilmIor);
            const std::string ball =
                "/World/Ball" + std::to_string(i);
            Sphere::Define(scene, ball)
                .SetRadius(kRadius)
                .SetMaterialBinding(LookOf(i));
            SetXform(scene, ball,
                     {.translate = BallCentre(static_cast<int>(i))});
        }
        SetBase(0);
        Thicken(kThickness);
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

        std::vector<std::string> names;
        for (const Base& base : kBases) {
            names.emplace_back(base.name);
        }
        Widgets().Combo(
            "Base", std::move(names), 0,
            [this](std::size_t index) { SetBase(index); });
        Widgets().Slider(
            "Middle ball's film (um)", 0.0, 1.0, kThickness,
            [this](double thickness) { Thicken(thickness); });
        Widgets().Slider(
            "Film IOR", 1.0, 2.5, kFilmIor,
            [this](double ior) { Film("thin_film_ior", ior); });
        Widgets().Slider("Film weight", 0.0, 1.0, 1.0,
                         [this](double weight) {
                             Film("thin_film_weight", weight);
                         });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            viewer->LookAt(kEye, kTarget);
        }
    }

    void TestFinal() override {
        const std::vector<std::vector<glm::dvec3>> filmed = Balls();
        Film("thin_film_weight", 0.0);
        const std::vector<std::vector<glm::dvec3>> bare = Balls();
        Film("thin_film_weight", 1.0);
        std::vector<glm::dvec3> hues;
        for (std::size_t i = 0; i < kScales.size(); ++i) {
            const glm::dvec3 mean = Mean(filmed[i]);
            std::printf("Film %.2f um: chroma %.1f (bare %.1f), mean "
                        "RGB %.1f "
                        "%.1f %.1f\n",
                        kScales[i] * kThickness, Chroma(filmed[i]),
                        Chroma(bare[i]), mean.r, mean.g, mean.b);
            hues.push_back(mean / (mean.r + mean.g + mean.b));
        }
        for (std::size_t i = 0; i < kScales.size(); ++i) {
            Check(Chroma(filmed[i]) > Chroma(bare[i]) + 10.0,
                  "the film colours the steel");
        }
        double nearest = 1e30;
        for (std::size_t i = 1; i < hues.size(); ++i) {
            const glm::dvec3 d = glm::abs(hues[i] - hues[i - 1]);
            nearest = std::min(nearest, d.r + d.g + d.b);
        }
        Check(nearest > 0.03,
              "each thickness has a colour of its own");
    }

  private:
    /// Sets input `name` on every ball.
    void Film(const std::string& name, double value) {
        for (std::size_t i = 0; i < kScales.size(); ++i) {
            GetScene().Set(LookOf(i), "inputs:" + name, value);
        }
    }

    void Thicken(double thickness) {
        for (std::size_t i = 0; i < kScales.size(); ++i) {
            GetScene().Set(LookOf(i), "inputs:thin_film_thickness",
                           kScales[i] * thickness);
        }
    }

    void SetBase(std::size_t index) {
        const Base& base = kBases[index];
        for (std::size_t i = 0; i < kScales.size(); ++i) {
            Material(GetScene(), LookOf(i)).SetAlbedo(base.colour);
            GetScene().Set(LookOf(i), "inputs:base_metalness",
                           base.metalness);
            GetScene().Set(LookOf(i), "inputs:specular_roughness",
                           base.roughness);
        }
    }

    std::vector<std::vector<glm::dvec3>> Balls() {
        const Tensor image = GetRenderer().Render(GetScene());
        std::vector<std::vector<glm::dvec3>> balls;
        for (int i = 0; i < static_cast<int>(kScales.size()); ++i) {
            balls.push_back(DiscPixels(image, BallCentre(i)));
        }
        return balls;
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    ThinFilmDemo, "pbr", "thin_film",
    "OpenPBR's thin film: interference colours on steel, titanium or "
    "black glass as the film thickens.");
