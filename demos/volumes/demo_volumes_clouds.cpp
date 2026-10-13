// SPDX-License-Identifier: BSD-3-Clause
// Cumulus clouds over the Earth under its atmosphere, after
// three-geospatial's clouds: a Volume whose grid is built here from a
// coverage map, a height profile and eroding noise, scattering many
// times and lit by the sun, the sky and the ground.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/terrain_schemas.h"
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

constexpr double kCoverage = 0.55;
constexpr double kSunElevation = 25.0;

/// A field of cumulus over a 20 km square, from 600 m up to 2100 m,
/// in 50 m voxels, its density over 0 to 1: where a low-frequency
/// coverage map lets clouds form, a profile rounding their bases and
/// tops, and billowing noise eroded by finer noise at their edges.
///
/// References:
/// - A. Schneider and N. Vos, "The Real-Time Volumetric Cloudscapes
/// of
///   Horizon: Zero Dawn", SIGGRAPH 2015 Advances in Real-Time
///   Rendering.
/// - takram, three-geospatial's clouds (packages/clouds), 2024.
std::vector<std::uint8_t> MakeClouds(double coverage) {
    constexpr double kVoxel = 50.0;
    constexpr double kBase = 600.0;
    constexpr double kThickness = 1500.0;
    const glm::ivec3 shape(400, 400, 30);
    const glm::dvec3 low(-10000.0, -10000.0, kBase);
    const GradientNoise noise(21);
    std::vector<float> density(
        static_cast<std::size_t>(shape.x * shape.y * shape.z), 0.0f);
    for (int i = 0; i < shape.x; ++i) {
        for (int j = 0; j < shape.y; ++j) {
            const glm::dvec2 xy =
                glm::dvec2(low) + kVoxel * (glm::dvec2(i, j) + 0.5);
            // Where clouds form: coverage's share of the map, its
            // edges soft.
            const double map =
                0.5 +
                0.5 *
                    noise.Fbm({xy.x / 4000.0, xy.y / 4000.0, 0.5}, 3);
            const double cover =
                std::clamp((map - (1.0 - coverage)) / 0.25, 0.0, 1.0);
            if (cover <= 0.0) {
                continue;
            }
            for (int k = 0; k < shape.z; ++k) {
                const glm::dvec3 p =
                    low + kVoxel * (glm::dvec3(i, j, k) + 0.5);
                // Flat bases and rounded tops, taller where cover is
                // full.
                const double h = (p.z - kBase) / kThickness;
                const double profile =
                    std::clamp(h / 0.08, 0.0, 1.0) *
                    std::clamp((0.35 + 0.65 * cover - h) / 0.25, 0.0,
                               1.0);
                if (profile <= 0.0) {
                    continue;
                }
                const double billows =
                    0.5 + 0.5 * noise.Fbm(p / 900.0, 4);
                const double base = std::clamp(
                    (billows * profile * cover - 0.35) / 0.4, 0.0,
                    1.0);
                if (base <= 0.0) {
                    continue;
                }
                // Finer noise wears the edges away, wispier below.
                const double detail =
                    0.5 + 0.5 * noise.Fbm(p / 180.0 + 31.0, 3);
                const double value =
                    base - (1.0 - base) * detail * (0.6 - 0.3 * h);
                density[static_cast<std::size_t>(
                    (i * shape.y + j) * shape.z + k)] =
                    value > 1e-3 ? static_cast<float>(value) : 0.0f;
            }
        }
    }
    return DenseNanoVdbGrid(density, shape, kVoxel,
                            low + 0.5 * kVoxel, "density");
}

class CloudsDemo : public app::Demo {
  public:
    CloudsDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // The Earth's surface at z = 0, grassland's albedo, and its
        // air.
        demos::DefinePlanetSurface(scene, "/World/Earth", 6371.0088e3,
                                   glm::dvec3(0.08, 0.12, 0.06),
                                   "/World/Looks/Ground");
        Atmosphere::Define(scene, "/World/Air")
            .SetPreset(std::string(tokens::kEarth));
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(0.53)
            .SetShadowEnable(true);
        SunAt(kSunElevation);

        // The clouds: water droplets scatter nearly all the light,
        // mostly forwards, many times over.
        Volume::Define(scene, "/World/Clouds")
            .SetField("density", "/World/Clouds/Density")
            .SetMaterialBinding("/World/Looks/Cloud");
        NanoVDBField::Define(scene, "/World/Clouds/Density");
        Material::Define(scene, "/World/Looks/Cloud")
            .SetDensity(0.04)
            .SetVolumeAlbedo(glm::dvec3(0.99))
            .SetAnisotropy(0.6)
            .SetScatteringOctaves(6)
            .SetOctaveAttenuation(0.5)
            .SetOctaveExtinction(0.5)
            .SetOctavePhase(0.5);
        Cover(kCoverage);

        // From 300 m up, looking across the field, the sun behind.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(18.0)
            .SetSensorNoise(false)
            .SetExposureIso(100.0)
            .SetExposureFStop(11.0)
            .SetExposureTime(1.0 / 250.0)
            .SetTransform(LookAtTransform({0.0, -9000.0, 300.0},
                                          {2000.0, 0.0, 1500.0},
                                          {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({480, 320})
            .SetAmbient(0.0);

        Widgets().Slider("Sun elevation", 2.0, 90.0, kSunElevation,
                         [this](double degrees) { SunAt(degrees); });
        Widgets().Slider(
            "Coverage", 0.1, 0.9, kCoverage,
            [this](double coverage) { Cover(coverage); });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, GetRenderer().Render(GetScene()));
    }

    void TestFinal() override {
        Scene& scene = GetScene();
        const Tensor cloudy = GetRenderer().Render(scene);
        scene.Set("/World/Clouds", tokens::kVisibility,
                  std::string(tokens::kInvisible));
        const Tensor clear = GetRenderer().Render(scene);
        scene.Set("/World/Clouds", tokens::kVisibility,
                  std::string(tokens::kInherited));
        const glm::dvec3 withClouds = demos::MeanLinearRgb(cloudy);
        const glm::dvec3 sky = demos::MeanLinearRgb(clear);
        std::printf("Mean linear RGB with clouds %.3f %.3f %.3f, "
                    "clear %.3f %.3f %.3f\n",
                    withClouds.r, withClouds.g, withClouds.b, sky.r,
                    sky.g, sky.b);
        Check(sky.b > sky.r, "the clear sky is blue");
        // The clouds change the view, white where the sky was blue.
        Check(glm::length(withClouds - sky) > 0.01,
              "the clouds show");
        Check(withClouds.r / withClouds.b > sky.r / sky.b,
              "the clouds are whiter than the sky");
    }

  private:
    void SunAt(double degrees) {
        const double e = degrees * std::numbers::pi / 180.0;
        DistantLight(GetScene(), "/World/Sun")
            .SetTransform(DistantLightTransform({-0.8 * std::cos(e),
                                                 -0.6 * std::cos(e),
                                                 std::sin(e)}));
    }

    void Cover(double coverage) {
        const std::vector<std::uint8_t> clouds = MakeClouds(coverage);
        NanoVDBField(GetScene(), "/World/Clouds/Density")
            .SetBuffer(Tensor::Copy<std::uint8_t>(
                clouds, {static_cast<std::int64_t>(clouds.size())}));
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(CloudsDemo, "volumes", "clouds",
                    "Cumulus over the Earth: many-times-scattered "
                    "sunlight, sky light "
                    "and air.");
