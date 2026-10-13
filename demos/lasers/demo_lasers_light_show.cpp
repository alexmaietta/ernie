// SPDX-License-Identifier: BSD-3-Clause
// A laser show: three projectors scan Lissajous figures through a
// hazy hall faster than the camera sees, so each draws a hollow cone
// of light in the haze and its figure on the far wall, the figures
// turning as their phases drift.
//
// References:
// - J. A. Lissajous, "Memoire sur l'etude optique des mouvements
//   vibratoires", Annales de Chimie et de Physique 51:147-231, 1857
//   (the figures).

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
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
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr glm::ivec2 kResolution{320, 200};
constexpr double kDensity = 0.08;  // 1/m, the haze's extinction
constexpr glm::dvec3 kStage{0.0, 0.0, 2.0};  // m, the projectors' row
constexpr double kWall = 16.0;  // m, the far wall along +y
constexpr double kDrift = 0.5;  // radians a second, the phase drift

/// A projector: its colour's wavelength (nm), power (W), place along
/// the row, the figure's full angles (degrees), frequencies and
/// phase.
struct Projector {
    const char* path;
    double wavelength;
    double power;
    double x;
    glm::dvec2 fieldOfView;
    glm::dvec2 frequencies;
    double phase;
};

constexpr std::array<Projector, 3> kProjectors{{
    {"/World/Red", 638.0, 1.0, -1.5, {24.0, 24.0}, {1.0, 1.0}, 90.0},
    {"/World/Green", 532.0, 1.0, 0.0, {30.0, 20.0}, {3.0, 2.0}, 90.0},
    {"/World/Blue", 450.0, 1.5, 1.5, {24.0, 18.0}, {5.0, 4.0}, 0.0},
}};

/// A hall's haze, 12 m x 18 m x 6 m in 0.25 m voxels, thinning
/// upwards, its billows soft.
std::vector<std::uint8_t> MakeHaze() {
    constexpr double kVoxel = 0.25;
    const glm::ivec3 shape(48, 72, 24);
    const glm::dvec3 low(-6.0, -1.0, 0.0);
    const GradientNoise noise(11);
    std::vector<float> density(
        static_cast<std::size_t>(shape.x * shape.y * shape.z), 0.0f);
    for (int i = 0; i < shape.x; ++i) {
        for (int j = 0; j < shape.y; ++j) {
            for (int k = 0; k < shape.z; ++k) {
                const glm::dvec3 p =
                    low + kVoxel * (glm::dvec3(i, j, k) + 0.5);
                const double value =
                    std::clamp(0.75 +
                                   0.4 * noise.Fbm(0.3 * p + 7.0, 3),
                               0.0, 1.0) *
                    std::exp(-p.z / 8.0);
                density[static_cast<std::size_t>(
                    (i * shape.y + j) * shape.z + k)] =
                    static_cast<float>(value);
            }
        }
    }
    return DenseNanoVdbGrid(density, shape, kVoxel,
                            low + 0.5 * kVoxel, "density");
}

class LightShowDemo : public app::Demo {
  public:
    LightShowDemo(app::Viewer* viewer,
                  const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 120;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Material::Define(scene, "/World/Looks/Floor")
            .SetAlbedo(glm::dvec3(0.05));
        Material::Define(scene, "/World/Looks/Wall")
            .SetAlbedo(glm::dvec3(0.7));
        GroundPlane::Define(scene, "/World/Floor")
            .SetMaterialBinding("/World/Looks/Floor");
        Cube::Define(scene, "/World/Wall")
            .SetSize(1.0)
            .SetMaterialBinding("/World/Looks/Wall");
        SetXform(scene, "/World/Wall",
                 {.translate = glm::dvec3(0.0, kWall, 3.0),
                  .scale = glm::dvec3(12.0, 0.2, 6.0)});
        for (std::size_t i = 0; i < kProjectors.size(); ++i) {
            const Projector& p = kProjectors[i];
            const glm::dvec3 aperture =
                kStage + glm::dvec3(p.x, 0.0, 0.0);
            LaserLight::Define(scene, p.path)
                .SetWavelength(p.wavelength)
                .SetPower(p.power)
                .SetWaistRadius(glm::dvec2(0.002))
                .SetPattern("scan")
                .SetPatternFieldOfView(p.fieldOfView)
                .SetScanFrequencies(p.frequencies)
                .SetScanPhase(p.phase)
                .SetTransform(LookAtTransform(
                    aperture, {aperture.x, kWall, 3.2},
                    {0.0, 0.0, 1.0}));
            phases_[i] = p.phase;
        }
        const std::vector<std::uint8_t> haze = MakeHaze();
        Volume::Define(scene, "/World/Haze")
            .SetField("density", "/World/Haze/Density")
            .SetMaterialBinding("/World/Looks/Haze");
        NanoVDBField::Define(scene, "/World/Haze/Density")
            .SetBuffer(Tensor::Copy<std::uint8_t>(
                haze, {static_cast<std::int64_t>(haze.size())}));
        Material::Define(scene, "/World/Looks/Haze")
            .SetDensity(kDensity)
            .SetVolumeAlbedo(glm::dvec3(0.95))
            .SetAnisotropy(0.5);
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(16.0)
            .SetSensorNoise(false)
            .SetExposureTime(1.0 / 30.0)
            .SetExposureFStop(2.0)
            .SetExposureIso(400.0)
            .SetClippingRange({0.05, 1e4})
            .SetTransform(LookAtTransform(
                {5.5, -3.0, 1.6}, {0.0, 8.0, 2.8}, {0.0, 0.0, 1.0}));
        const Path var("/Render/Camera/SpectralRadiance");
        RenderVar::Define(scene, var)
            .SetSourceName("SpectralRadiance");
        RenderProduct::Define(scene, "/Render/Camera")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(kResolution)
            .SetOrderedVars({var});
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(kResolution)
            .SetAmbient(0.0);
        Widgets().Slider("Haze density (1/m)", 0.0, 0.3, kDensity,
                         [this](double density) {
                             Material(GetScene(), "/World/Looks/Haze")
                                 .SetDensity(density);
                         });
        Widgets().Slider("Phase drift (rad/s)", 0.0, 3.0, kDrift,
                         [this](double drift) { drift_ = drift; });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            viewer->SetLighting(app::Lighting::Stage);
            app::ViewportDisplay display =
                viewer->GetViewportDisplay();
            display.lights = false;
            display.grid = false;
            viewer->SetViewportDisplay(display);
        }
    }

    void Step() override {
        // Each figure turns at its own rate.
        for (std::size_t i = 0; i < kProjectors.size(); ++i) {
            phases_[i] += glm::degrees(drift_ * frameDt) *
                          (1.0 + 0.5 * static_cast<double>(i));
            LaserLight(GetScene(), kProjectors[i].path)
                .SetScanPhase(std::fmod(phases_[i], 360.0));
        }
    }

    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, GetRenderer().Render(GetScene()));
    }

    void TestFinal() override {
        const std::vector<double> lit = Capture(true, kDensity);
        const std::vector<double> clear = Capture(true, 0.0);
        const std::vector<double> dark = Capture(false, kDensity);
        // What the haze adds: the cones of light the scans sweep.
        std::vector<double> cones(lit.size());
        for (std::size_t i = 0; i < lit.size(); ++i) {
            cones[i] = lit[i] - clear[i];
        }
        const double coneThreshold = 0.05 * Percentile(cones, 99.9);
        const double wallThreshold = 0.05 * Percentile(clear, 99.9);
        std::size_t swept = 0;
        std::size_t drawn = 0;
        for (std::size_t i = 0; i < lit.size(); ++i) {
            swept += cones[i] > coneThreshold ? 1u : 0u;
            drawn += clear[i] > wallThreshold ? 1u : 0u;
        }
        const double pixels = static_cast<double>(lit.size());
        const double darkest =
            *std::max_element(dark.begin(), dark.end());
        std::printf("the scans' cones light %.1f%% of the pixels in "
                    "the haze; "
                    "their figures %.1f%% on the wall; with the "
                    "projectors off, "
                    "the brightest pixel is %.3g\n",
                    100.0 * static_cast<double>(swept) / pixels,
                    100.0 * static_cast<double>(drawn) / pixels,
                    darkest);
        Check(static_cast<double>(swept) > 0.05 * pixels,
              "the cones show in the haze");
        Check(static_cast<double>(drawn) > 0.001 * pixels &&
                  static_cast<double>(drawn) < 0.2 * pixels,
              "the figures draw lines on the wall");
        Check(darkest == 0.0, "nothing else lights the hall");
    }

  private:
    std::array<double, kProjectors.size()> phases_{};
    double drift_ = kDrift;

    /// The value `percent` of the way up `values` sorted.
    static double Percentile(std::vector<double> values,
                             double percent) {
        const auto at = static_cast<std::ptrdiff_t>(
            percent / 100.0 * static_cast<double>(values.size() - 1));
        std::nth_element(values.begin(), values.begin() + at,
                         values.end());
        return values[static_cast<std::size_t>(at)];
    }

    /// The radiance summed over the bands, with the projectors on or
    /// off and the haze at `density`.
    std::vector<double> Capture(bool lasers, double density) {
        Scene& scene = GetScene();
        for (const Projector& p : kProjectors) {
            scene.Set(p.path, tokens::kVisibility,
                      std::string(lasers ? tokens::kInherited
                                         : tokens::kInvisible));
        }
        Material(scene, "/World/Looks/Haze").SetDensity(density);
        const std::vector<Path> products{Path("/Render/Camera")};
        RenderProductSetOutputs outputs =
            GetRenderer().Step(scene, products);
        Material(scene, "/World/Looks/Haze").SetDensity(kDensity);
        const Tensor bands = outputs.at(Path("/Render/Camera"))
                                 .vars.at("SpectralRadiance")
                                 .Map(MapDevice::Cpu)
                                 .tensor.Contiguous();
        const auto count = static_cast<std::size_t>(bands.Shape()[0]);
        const auto pixels =
            static_cast<std::size_t>(kResolution.x * kResolution.y);
        std::vector<double> radiance(pixels, 0.0);
        const auto* values = bands.As<float>();
        for (std::size_t b = 0; b < count; ++b) {
            for (std::size_t i = 0; i < pixels; ++i) {
                radiance[i] +=
                    static_cast<double>(values[b * pixels + i]);
            }
        }
        return radiance;
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    LightShowDemo, "lasers", "light_show",
    "A laser show scanning Lissajous figures in haze.");
