// SPDX-License-Identifier: BSD-3-Clause
// A cumulus lit from behind by a low sun: its water droplets scatter
// by Mie theory (the silver lining about its edges, a fogbow and
// glory opposite the sun), their light scattered many times found by
// collectors on its lit surface (Bouthors et al. 2008) rather than
// octaves.

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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kSunAzimuth = 20.0;
constexpr double kSunElevation = 8.0;
constexpr double kDropletRadius = 7.0;

struct Blob {
    glm::dvec3 centre;
    double radius;
};

// Blobs of the cloud: centre (m) and radius (m).
constexpr Blob kBlobs[] = {
    {{0.0, 0.0, 1900.0}, 1100.0},
    {{-900.0, 300.0, 1500.0}, 800.0},
    {{850.0, -200.0, 1450.0}, 750.0},
    {{300.0, 700.0, 2500.0}, 650.0},
    {{-300.0, -500.0, 1300.0}, 700.0},
};

/// A cumulus congestus of a few rounded blobs on a flat base at 900
/// m, in 40 m voxels, its density over 0 to 1: billowing noise on the
/// blobs, finer noise wearing their edges away.
std::vector<std::uint8_t> MakeCumulus() {
    constexpr double kVoxel = 40.0;
    const glm::ivec3 shape(120, 120, 80);
    const glm::dvec3 low(-2400.0, -2400.0, 800.0);
    const GradientNoise noise(7);
    std::vector<float> density(
        static_cast<std::size_t>(shape.x * shape.y * shape.z), 0.0f);
    for (int i = 0; i < shape.x; ++i) {
        for (int j = 0; j < shape.y; ++j) {
            for (int k = 0; k < shape.z; ++k) {
                const glm::dvec3 p =
                    low + kVoxel * (glm::dvec3(i, j, k) + 0.5);
                double inside = 0.0;
                for (const Blob& blob : kBlobs) {
                    inside = std::max(
                        inside, 1.0 - glm::length(p - blob.centre) /
                                          blob.radius);
                }
                inside *= std::clamp((p.z - 900.0) / 150.0, 0.0, 1.0);
                if (inside <= 0.0) {
                    continue;
                }
                const double billows =
                    0.5 + 0.5 * noise.Fbm(p / 450.0, 5);
                const double base = std::clamp(
                    (inside - 0.55 * (1.0 - billows)) * 3.0, 0.0,
                    1.0);
                const double detail =
                    0.5 + 0.5 * noise.Fbm(p / 120.0 + 17.0, 3);
                const double value =
                    base - (1.0 - base) * detail * 0.8;
                density[static_cast<std::size_t>(
                    (i * shape.y + j) * shape.z + k)] =
                    value > 1e-3 ? static_cast<float>(value) : 0.0f;
            }
        }
    }
    return DenseNanoVdbGrid(density, shape, kVoxel,
                            low + 0.5 * kVoxel, "density");
}

class CloudDropletsDemo : public app::Demo {
  public:
    CloudDropletsDemo(app::Viewer* viewer,
                      const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
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
        PlaceSun();

        Volume::Define(scene, "/World/Cloud")
            .SetField("density", "/World/Cloud/Density")
            .SetMaterialBinding("/World/Looks/Cloud");
        const std::vector<std::uint8_t> cumulus = MakeCumulus();
        NanoVDBField::Define(scene, "/World/Cloud/Density")
            .SetBuffer(Tensor::Copy<std::uint8_t>(
                cumulus,
                {static_cast<std::int64_t>(cumulus.size())}));
        // Water droplets of 7 um, their surface where the density
        // reaches 0.1; octaves for the other setting.
        Material::Define(scene, "/World/Looks/Cloud")
            .SetDensity(0.05)
            .SetVolumeAlbedo(glm::dvec3(1.0))
            .SetPhaseFunction(std::string(tokens::kCloudDroplets))
            .SetDropletRadius(kDropletRadius)
            .SetSurfaceDensity(0.1)
            .SetAnisotropy(0.6)
            .SetScatteringOctaves(6)
            .SetOctaveAttenuation(0.5)
            .SetOctaveExtinction(0.5)
            .SetOctavePhase(0.5);

        // From 6.5 km off at the cloud's middle, the sun behind it.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetSensorNoise(false)
            .SetExposureIso(100.0)
            .SetExposureFStop(11.0)
            .SetExposureTime(1.0 / 250.0)
            .SetTransform(LookAtTransform({0.0, -6500.0, 1300.0},
                                          {0.0, 0.0, 1700.0},
                                          {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({640, 400})
            .SetAmbient(0.0)
            .SetVolumeMultipleScattering(
                std::string(tokens::kCollectors));

        Widgets().Slider("Sun azimuth (0 behind the cloud)", 0.0,
                         360.0, kSunAzimuth, [this](double degrees) {
                             azimuth_ = degrees;
                             PlaceSun();
                         });
        Widgets().Slider("Sun elevation", 1.0, 60.0, kSunElevation,
                         [this](double degrees) {
                             elevation_ = degrees;
                             PlaceSun();
                         });
        Widgets().Combo("Multiple scattering",
                        {"Collectors", "Octaves"}, 0,
                        [this](std::size_t index) {
                            ScatterBy(index == 0 ? tokens::kCollectors
                                                 : tokens::kOctaves);
                        });
        Widgets().Combo("Phase function",
                        {"Cloud droplets (Mie)", "Henyey-Greenstein"},
                        0, [this](std::size_t index) {
                            Material(GetScene(), "/World/Looks/Cloud")
                                .SetPhaseFunction(std::string(
                                    index == 0
                                        ? tokens::kCloudDroplets
                                        : tokens::kHenyeyGreenstein));
                        });
        Widgets().Slider("Droplet radius (um)", 2.0, 20.0,
                         kDropletRadius, [this](double radius) {
                             Material(GetScene(),
                                      "/World/Looks/Cloud")
                                 .SetDropletRadius(radius);
                         });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            // The same view, orbiting and zooming about the cloud.
            viewer->LookAt({0.0, -6500.0, 1300.0},
                           {0.0, 0.0, 1700.0});
        }
    }

    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, Settled());
    }

    void TestFinal() override {
        Scene& scene = GetScene();
        const glm::dvec3 collected = demos::MeanLinearRgb(Settled());
        const glm::dvec3 octaves =
            demos::MeanLinearRgb(WithScattering(tokens::kOctaves));
        ScatterBy(tokens::kCollectors);
        Material(scene, "/World/Looks/Cloud")
            .SetPhaseFunction(std::string(tokens::kHenyeyGreenstein));
        const glm::dvec3 henyey =
            demos::MeanLinearRgb(GetRenderer().Render(scene));
        Material(scene, "/World/Looks/Cloud")
            .SetPhaseFunction(std::string(tokens::kCloudDroplets));
        scene.Set("/World/Cloud", tokens::kVisibility,
                  std::string(tokens::kInvisible));
        const glm::dvec3 clear =
            demos::MeanLinearRgb(GetRenderer().Render(scene));
        scene.Set("/World/Cloud", tokens::kVisibility,
                  std::string(tokens::kInherited));
        std::printf(
            "Mean linear RGB by collectors %.3f %.3f %.3f, octaves "
            "%.3f %.3f %.3f, Henyey-Greenstein %.3f %.3f %.3f, clear "
            "%.3f %.3f %.3f\n",
            collected.r, collected.g, collected.b, octaves.r,
            octaves.g, octaves.b, henyey.r, henyey.g, henyey.b,
            clear.r, clear.g, clear.b);
        Check(glm::length(collected - clear) > 0.01,
              "the cloud shows");
        Check(
            glm::length(collected - octaves) > 1e-4,
            "the collectors light the cloud otherwise than octaves");
        Check(
            glm::length(collected - henyey) > 1e-4,
            "the droplets scatter otherwise than Henyey-Greenstein");
    }

  private:
    void PlaceSun() {
        const double a = azimuth_ * std::numbers::pi / 180.0;
        const double e = elevation_ * std::numbers::pi / 180.0;
        DistantLight(GetScene(), "/World/Sun")
            .SetTransform(DistantLightTransform(
                {std::sin(a) * std::cos(e), std::cos(a) * std::cos(e),
                 std::sin(e)}));
    }

    void ScatterBy(std::string_view mode) {
        RenderSettings(GetScene(), "/Render/Settings")
            .SetVolumeMultipleScattering(std::string(mode));
    }

    Tensor WithScattering(std::string_view mode) {
        ScatterBy(mode);
        return GetRenderer().Render(GetScene());
    }

    /// The view once the collectors' transport table is ready, read
    /// or simulated off the frame's thread (octaves meanwhile): until
    /// it differs from the octaves' beyond the frames' noise.
    Tensor Settled() {
        const Tensor octaves = WithScattering(tokens::kOctaves);
        Tensor image = WithScattering(tokens::kCollectors);
        const auto start = std::chrono::steady_clock::now();
        const auto apart = [&] {
            const auto* a =
                static_cast<const std::uint8_t*>(image.Data());
            const auto* b =
                static_cast<const std::uint8_t*>(octaves.Data());
            const std::int64_t count = image.Shape()[0] *
                                       image.Shape()[1] *
                                       image.Shape()[2];
            double sum = 0.0;
            for (std::int64_t i = 0; i < count; ++i) {
                sum += std::abs(int(a[i]) - int(b[i]));
            }
            return sum / static_cast<double>(count) >= 1.0;
        };
        while (!apart() && std::chrono::steady_clock::now() - start <
                               std::chrono::seconds(120)) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(200));
            image = WithScattering(tokens::kCollectors);
        }
        return image;
    }

    double azimuth_ = kSunAzimuth;
    double elevation_ = kSunElevation;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    CloudDropletsDemo, "volumes", "cloud_droplets",
    "A back-lit cumulus: cloud droplets' Mie scattering "
    "and multiple scattering by collectors.");
