// SPDX-License-Identifier: BSD-3-Clause
// Adaptive spheres from a marble to a moon: an Earth-sized sphere as
// the ground, a spinning globe, a ball, a tilted egg (a scaled
// sphere) and the Moon at its distance, each refined towards the eye.
// The panel picks the view and tints the LOD levels.

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

#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kEarth = 6371000.0;
constexpr double kMoon = 1737400.0;
constexpr double kMoonDistance = 384400000.0;
constexpr double kMarble = 0.02;

// Views: a name, the eye and what it looks at.
struct View {
    const char* name;
    glm::dvec3 eye;
    glm::dvec3 target;
};

const glm::dvec3 kMoonCentre =
    kMoonDistance * glm::normalize(glm::dvec3(1.0, 0.4, 0.12));

const std::array<View, 4> kViews = {{
    {"The scene", {-2.0, -9.0, 2.5}, {0.0, 0.0, 0.8}},
    {"The marble", {0.05, -0.05, 0.04}, {0.0, 0.0, kMarble}},
    {"The horizon and the Moon", {0.0, 0.0, 1.7}, kMoonCentre},
    {"From orbit", {0.0, -1.6e7, 1.2e7}, {0.0, 0.0, -kEarth}},
}};

/// A globe's map, latitude by longitude (rows from the north): blue
/// and white tiles of 22.5 degrees, the equator and prime meridian
/// red.
Tensor GlobeMap() {
    constexpr std::int64_t kRows = 64;
    constexpr std::int64_t kCols = 128;
    std::vector<std::uint8_t> texels;
    for (std::int64_t row = 0; row < kRows; ++row) {
        for (std::int64_t col = 0; col < kCols; ++col) {
            const bool line =
                row == kRows / 2 || row == kRows / 2 - 1 ||
                col == kCols / 2 || col == kCols / 2 - 1;
            const bool white = ((row / 8) + (col / 8)) % 2 == 0;
            const std::uint8_t r = line ? 220 : white ? 230 : 40;
            const std::uint8_t g = line ? 30 : white ? 230 : 90;
            const std::uint8_t b = line ? 30 : white ? 230 : 200;
            texels.insert(texels.end(), {r, g, b, 255});
        }
    }
    return Tensor::Copy<std::uint8_t>(texels, {kRows, kCols, 4});
}

/// A material of `colour`, or of `map` read by latitude and
/// longitude.
void DefineLook(Scene& scene, const std::string& look,
                const glm::dvec3& colour,
                const Tensor* map = nullptr) {
    Material::Define(scene, look);
    Shader::Define(scene, look + "/Surface")
        .SetInfoId(tokens::kUsdPreviewSurface)
        .SetDiffuseColor(colour)
        .SetRoughness(0.6);
    Connect(scene, look, tokens::kSurfaceOutput, look + "/Surface",
            tokens::kSurfaceOutput);
    if (map != nullptr) {
        Shader::Define(scene, look + "/Map")
            .SetInfoId(tokens::kUsdUVTexture)
            .SetSourceColorSpace("sRGB")
            .SetTexture(*map);
        Connect(scene, look + "/Surface", tokens::kDiffuseColor,
                look + "/Map", "outputs:rgb");
    }
}

/// An adaptive sphere of `radius` at `transform`, wearing `look`.
void DefineSphere(Scene& scene, const std::string& path,
                  double radius, const glm::dmat4& transform,
                  const std::string& look) {
    Sphere::Define(scene, path)
        .SetRadius(radius)
        .SetTessellation(std::string(tokens::kAdaptiveSphere))
        .SetMaterialBinding(look)
        .SetTransform(transform);
}

class AdaptiveSpheresDemo : public app::Demo {
  public:
    AdaptiveSpheresDemo(app::Viewer* viewer,
                        const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 3;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        const Tensor map = GlobeMap();
        DefineLook(scene, "/World/Looks/Ground", {0.35, 0.42, 0.3});
        DefineLook(scene, "/World/Looks/Globe", {1.0, 1.0, 1.0},
                   &map);
        DefineLook(scene, "/World/Looks/Ball", {0.8, 0.35, 0.2});
        DefineLook(scene, "/World/Looks/Egg", {0.9, 0.85, 0.7});
        DefineLook(scene, "/World/Looks/Moon", {0.5, 0.5, 0.5});
        const auto at = [](const glm::dvec3& p) {
            return glm::translate(glm::dmat4(1.0), p);
        };
        // The ground is the top of an Earth-sized sphere.
        DefineSphere(scene, "/World/Earth", kEarth,
                     at({0.0, 0.0, -kEarth}), "/World/Looks/Ground");
        DefineSphere(scene, "/World/Globe", kMarble,
                     at({0.0, 0.0, kMarble}), "/World/Looks/Globe");
        DefineSphere(scene, "/World/Ball", 1.0, at({3.0, 1.0, 1.0}),
                     "/World/Looks/Ball");
        // An egg: a unit sphere squashed along x and y, tipped over.
        DefineSphere(
            scene, "/World/Egg", 1.0,
            glm::scale(glm::rotate(at({-3.0, 1.0, 1.0}),
                                   glm::radians(35.0),
                                   glm::dvec3(1.0, 0.0, 0.0)),
                       glm::dvec3(0.6, 0.6, 1.0)),
            "/World/Looks/Egg");
        DefineSphere(scene, "/World/Moon", kMoon, at(kMoonCentre),
                     "/World/Looks/Moon");
        DistantLight::Define(scene, "/World/Sun")
            .SetIntensity(1.0)
            .SetTransform(DistantLightTransform({0.4, -0.5, 0.75}));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(35.0)
            .SetClippingRange(glm::dvec2(
                1e-3, std::numeric_limits<double>::infinity()));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera("/World/Camera")
            .SetResolution(glm::dvec2(1280.0, 720.0));

        std::vector<std::string> names;
        for (const View& view : kViews) {
            names.emplace_back(view.name);
        }
        Widgets().Combo("View", names, 0,
                        [this](std::size_t i) { Look(i); });
        Widgets().Checkbox("Tint LOD levels", false, [this](bool on) {
            RenderSettings(GetScene(), "/Render/Settings")
                .SetLodShowLevels(on);
        });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }
        Look(0);
    }

    // The globe turns a degree a frame.
    void Step() override {
        spin_ += 1.0;
        Sphere(GetScene(), "/World/Globe")
            .SetTransform(glm::rotate(
                glm::translate(glm::dmat4(1.0),
                               glm::dvec3(0.0, 0.0, kMarble)),
                glm::radians(spin_), glm::dvec3(0.0, 0.0, 1.0)));
    }

    void TestFinal() override {
        // Each view's centre pixel shows its sphere, and the marble's
        // close-up fills most of the frame with its map.
        std::array<double, kViews.size()> centres{};
        double marbleCover = 0.0;
        for (std::size_t i = 0; i < kViews.size(); ++i) {
            if (i == 2) {
                continue;  // The Moon is small and may be unlit.
            }
            Look(i);
            const Tensor image = GetRenderer().Render(GetScene());
            const auto rows = image.Shape()[0];
            const auto cols = image.Shape()[1];
            const auto* p =
                static_cast<const std::uint8_t*>(image.Data());
            const auto* centre =
                p + ((rows / 2) * cols + cols / 2) * 4;
            centres[i] = centre[0] + centre[1] + centre[2];
            if (i == 1) {
                std::int64_t lit = 0;
                for (std::int64_t k = 0; k < rows * cols; ++k) {
                    lit +=
                        p[k * 4] + p[k * 4 + 1] + p[k * 4 + 2] > 60;
                }
                marbleCover = static_cast<double>(lit) /
                              static_cast<double>(rows * cols);
            }
        }
        std::printf("centres %.0f %.0f %.0f; marble covers %.2f\n",
                    centres[0], centres[1], centres[3], marbleCover);
        Check(centres[0] > 60.0, "the scene's ground is lit");
        Check(centres[1] > 60.0, "the marble is lit");
        Check(centres[3] > 60.0, "the Earth from orbit is lit");
        Check(marbleCover > 0.3, "the marble fills its close-up");
    }

  private:
    void Look(std::size_t i) {
        const View& view = kViews[i];
        if (app::Viewer* viewer = GetViewer()) {
            viewer->LookAt(view.eye, view.target);
        } else {
            Camera(GetScene(), "/World/Camera")
                .SetTransform(LookAtTransform(view.eye, view.target,
                                              {0.0, 0.0, 1.0}));
        }
    }

    double spin_ = 0.0;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    AdaptiveSpheresDemo, "planets", "adaptive_spheres",
    "Adaptive spheres from a marble to a moon, each refined towards "
    "the eye.");
