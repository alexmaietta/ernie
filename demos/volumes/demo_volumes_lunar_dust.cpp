// SPDX-License-Identifier: BSD-3-Clause
// A lander hovering over the Moon, its exhaust blowing a sheet of
// dust outward: a low sun lights the dust, which shades the ground
// under it.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/volume_schemas.h"
#include "ernie/scene/xform.h"
#include "ernie/util/nanovdb.h"
#include "ernie/util/noise.h"
#include "util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

/// The image's mean linear RGB over rows [top, bottom).
glm::dvec3 MeanRows(const Tensor& image, std::int64_t top,
                    std::int64_t bottom) {
    const std::int64_t cols = image.Shape()[1];
    const std::uint8_t* pixels = image.As<std::uint8_t>();
    glm::dvec3 sum(0.0);
    for (std::int64_t r = top; r < bottom; ++r) {
        for (std::int64_t c = 0; c < cols; ++c) {
            const std::uint8_t* p = pixels + 4 * (r * cols + c);
            sum += demos::SrgbToLinear(glm::dvec3(p[0], p[1], p[2]) /
                                       255.0);
        }
    }
    return sum / static_cast<double>((bottom - top) * cols);
}

/// The dust's density over a 48 m square, 6 m deep, in 0.25 m voxels:
/// a sheet streaming out from under the engine, streaked along its
/// rays and billowing, thinning with height and distance.
std::vector<std::uint8_t> MakeDust() {
    constexpr double kVoxel = 0.25;
    const glm::ivec3 shape(192, 192, 24);
    const glm::dvec3 low(-24.0, -24.0, 0.0);
    const GradientNoise noise(11);
    std::vector<float> density(
        static_cast<std::size_t>(shape.x) * shape.y * shape.z, 0.0f);
    for (int i = 0; i < shape.x; ++i) {
        for (int j = 0; j < shape.y; ++j) {
            for (int k = 0; k < shape.z; ++k) {
                const glm::dvec3 p =
                    low + kVoxel * (glm::dvec3(i, j, k) + 0.5);
                const double r = std::hypot(p.x, p.y);
                const double theta = std::atan2(p.y, p.x);
                // Out from 1.5 m, fading over 10 m; thicker further
                // out.
                const double radial =
                    std::clamp((r - 1.5) / 2.0, 0.0, 1.0) *
                    std::exp(-r / 10.0);
                const double vertical =
                    std::exp(-p.z / (0.2 + 0.05 * r));
                // Streaks along the rays, and billows.
                const double streaks =
                    0.5 +
                    0.5 * noise.Fbm({6.0 * std::cos(theta),
                                     6.0 * std::sin(theta), 0.15 * r},
                                    3);
                const double billows =
                    0.5 + 0.5 * noise.Fbm(0.6 * p + 17.0, 4);
                const double value =
                    radial * vertical *
                    std::clamp(2.5 * streaks - 1.0, 0.0, 1.0) *
                    (0.5 + billows);
                density[(static_cast<std::size_t>(i) * shape.y + j) *
                            shape.z +
                        k] =
                    value > 1e-3 ? static_cast<float>(value) : 0.0f;
            }
        }
    }
    return DenseNanoVdbGrid(density, shape, kVoxel,
                            low + 0.5 * kVoxel, "density");
}

class LunarDustDemo : public app::Demo {
  public:
    LunarDustDemo(app::Viewer* viewer,
                  const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Ground")
            .SetMaterialBinding("/World/Looks/Regolith");
        Material::Define(scene, "/World/Looks/Regolith")
            .SetAlbedo(glm::dvec3(0.12));
        // The lander 4 m up: its body, legs and engine bell.
        Cube::Define(scene, "/World/Lander/Body")
            .SetSize(1.0)
            .SetMaterialBinding("/World/Looks/Foil");
        SetXform(scene, "/World/Lander/Body",
                 {.translate = glm::dvec3(0.0, 0.0, 5.5),
                  .scale = glm::dvec3(3.0, 3.0, 2.0)});
        for (int leg = 0; leg < 4; ++leg) {
            const double angle =
                std::numbers::pi * (0.25 + 0.5 * leg);
            const std::string path =
                "/World/Lander/Leg" + std::to_string(leg);
            Cylinder::Define(scene, path)
                .SetRadius(0.08)
                .SetHeight(2.0)
                .SetMaterialBinding("/World/Looks/Metal");
            SetXform(scene, path,
                     {.translate =
                          glm::dvec3(2.0 * std::cos(angle),
                                     2.0 * std::sin(angle), 4.0)});
        }
        Cone::Define(scene, "/World/Lander/Bell")
            .SetRadius(0.5)
            .SetHeight(0.8)
            .SetMaterialBinding("/World/Looks/Metal");
        SetXform(scene, "/World/Lander/Bell",
                 {.translate = glm::dvec3(0.0, 0.0, 4.1)});
        Material::Define(scene, "/World/Looks/Foil")
            .SetAlbedo(glm::dvec3(0.35, 0.28, 0.12));
        Material::Define(scene, "/World/Looks/Metal")
            .SetAlbedo(glm::dvec3(0.3));

        // The dust, built here: grains that scatter about half the
        // light, mostly back towards the sun, as regolith does
        // (Hapke's double Henyey-Greenstein lobes).
        const std::vector<std::uint8_t> dust = MakeDust();
        Volume::Define(scene, "/World/Dust")
            .SetField("density", "/World/Dust/Density")
            .SetMaterialBinding("/World/Looks/Dust");
        NanoVDBField::Define(scene, "/World/Dust/Density")
            .SetBuffer(Tensor::Copy<std::uint8_t>(
                dust, {static_cast<std::int64_t>(dust.size())}));
        Material::Define(scene, "/World/Looks/Dust")
            .SetDensity(kDensity)
            .SetVolumeAlbedo(glm::dvec3(0.55))
            .SetPhaseFunction(
                std::string(tokens::kDoubleHenyeyGreenstein));
        scene.Set("/World/Looks/Dust", tokens::kPhaseB, 0.3);
        scene.Set("/World/Looks/Dust", tokens::kPhaseC, 0.6);

        // A low sun from the left.
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetShadowEnable(true)
            .SetTransform(DistantLightTransform({-1.0, -0.7, 0.3}));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetSensorNoise(false)
            .SetExposureIso(100.0)
            .SetExposureFStop(8.0)
            .SetExposureTime(1.0 / 250.0)
            .SetTransform(LookAtTransform(
                {8.0, -22.0, 2.5}, {0.0, 0.0, 2.5}, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({480, 320})
            .SetAmbient(0.0);

        Widgets().Slider("Dust density (1/m)", 0.0, 2.0, kDensity,
                         [this](double density) {
                             Material(GetScene(), "/World/Looks/Dust")
                                 .SetDensity(density);
                         });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, GetRenderer().Render(GetScene()));
    }

    void TestFinal() override {
        Scene& scene = GetScene();
        const Tensor dusty = GetRenderer().Render(scene);
        scene.Set("/World/Dust", tokens::kVisibility,
                  std::string(tokens::kInvisible));
        const Tensor clear = GetRenderer().Render(scene);
        scene.Set("/World/Dust", tokens::kVisibility,
                  std::string(tokens::kInherited));
        const std::int64_t rows = dusty.Shape()[0];
        // The sky above, and the ground beyond the lander, seen
        // through the dust or without it.
        const glm::dvec3 sky = MeanRows(dusty, 0, rows / 8);
        const auto band = [&](const Tensor& image) {
            return MeanRows(image, rows / 2 + rows / 16,
                            rows / 2 + rows / 8);
        };
        const glm::dvec3 with = band(dusty);
        const glm::dvec3 without = band(clear);
        std::printf("Sky %.4f, ground band with dust %.4f %.4f %.4f, "
                    "without %.4f %.4f %.4f\n",
                    sky.r + sky.g + sky.b, with.r, with.g, with.b,
                    without.r, without.g, without.b);
        Check(sky.r + sky.g + sky.b < 0.01,
              "the Moon's sky is black");
        Check(std::abs((with.r + with.g + with.b) -
                       (without.r + without.g + without.b)) > 0.01,
              "the dust shows over the ground");
    }

  private:
    static constexpr double kDensity = 0.3;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    LunarDustDemo, "volumes", "lunar_dust",
    "A lander's exhaust blowing a sheet of lunar dust in a low sun.");
