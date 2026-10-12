// SPDX-License-Identifier: BSD-3-Clause
// Laser beams made visible by drifting dust at night: a green grid of
// beamlets, a red line and a blue spot cross the haze to a wall, the
// dust scattering a little of each beam towards the camera, and a
// rock in the way leaves a dark channel behind it.
//
// References:
// - C. F. Bohren and D. R. Huffman, "Absorption and Scattering of
//   Light by Small Particles", Wiley, 1983 (a beam seen side-on by
//   the light its path scatters, forward-peaked for dust).

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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr glm::ivec2 kResolution{320, 200};
constexpr double kDensity = 0.15;  // 1/m, the dust's extinction
constexpr glm::dvec3 kRig{0.0, 0.0, 1.2};  // m, the lasers' aperture
// The wind's direction, rising a little.
constexpr glm::dvec3 kGust{1.0, 0.0, 0.15};

/// Billowing dust over 12 m x 12 m, 3 m deep, in 0.2 m voxels,
/// carried `drift` m by the wind.
std::vector<std::uint8_t>
MakeDust(glm::dvec3 drift = glm::dvec3(0.0)) {
    constexpr double kVoxel = 0.2;
    const glm::ivec3 shape(60, 60, 15);
    const glm::dvec3 low(-6.0, -2.0, 0.0);
    const GradientNoise noise(5);
    std::vector<float> density(
        static_cast<std::size_t>(shape.x) * shape.y * shape.z, 0.0f);
    for (int i = 0; i < shape.x; ++i) {
        for (int j = 0; j < shape.y; ++j) {
            for (int k = 0; k < shape.z; ++k) {
                const glm::dvec3 p =
                    low + kVoxel * (glm::dvec3(i, j, k) + 0.5);
                const double value =
                    std::clamp(
                        0.55 + 0.6 * noise.Fbm(
                                         0.4 * (p - drift) + 3.0, 4),
                        0.0, 1.0) *
                    std::exp(-p.z / 2.0);
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

class VisibleBeamsDemo : public app::Demo {
  public:
    VisibleBeamsDemo(app::Viewer* viewer,
                     const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Material::Define(scene, "/World/Looks/Ground")
            .SetAlbedo(glm::dvec3(0.15));
        Material::Define(scene, "/World/Looks/Wall")
            .SetAlbedo(glm::dvec3(0.6));
        Material::Define(scene, "/World/Looks/Rock")
            .SetAlbedo(glm::dvec3(0.25));
        GroundPlane::Define(scene, "/World/Ground")
            .SetMaterialBinding("/World/Looks/Ground");
        Cube::Define(scene, "/World/Wall")
            .SetSize(1.0)
            .SetMaterialBinding("/World/Looks/Wall");
        SetXform(scene, "/World/Wall",
                 {.translate = glm::dvec3(0.0, 9.0, 1.5),
                  .scale = glm::dvec3(10.0, 0.2, 3.0)});
        // A rock in the green grid's left beams.
        Sphere::Define(scene, "/World/Rock")
            .SetRadius(0.35)
            .SetMaterialBinding("/World/Looks/Rock");
        SetXform(scene, "/World/Rock",
                 {.translate = glm::dvec3(-0.45, 4.0, 0.95)});
        LaserLight::Define(scene, "/World/Grid")
            .SetWavelength(532.0)
            .SetPower(0.2)
            .SetWaistRadius(glm::dvec2(0.002))
            .SetPattern("grid")
            .SetGridColumns(5)
            .SetGridRows(3)
            .SetGridPitch(glm::dvec2(4.0, 3.0))
            .SetTransform(LookAtTransform(kRig, {0.0, 9.0, 1.2},
                                          {0.0, 0.0, 1.0}));
        LaserLight::Define(scene, "/World/Line")
            .SetWavelength(638.0)
            .SetPower(0.3)
            .SetWaistRadius(glm::dvec2(0.002))
            .SetPattern("line")
            .SetFanAngle(30.0)
            .SetTransform(LookAtTransform(kRig, {0.0, 9.0, 0.3},
                                          {0.0, 0.0, 1.0}));
        LaserLight::Define(scene, "/World/Spot")
            .SetWavelength(450.0)
            .SetPower(0.5)
            .SetWaistRadius(glm::dvec2(0.004))
            .SetTransform(LookAtTransform(kRig, {2.5, 9.0, 2.4},
                                          {0.0, 0.0, 1.0}));
        // Dust scattering most of what it stops, forwards.
        const std::vector<std::uint8_t> dust = MakeDust();
        Volume::Define(scene, "/World/Dust")
            .SetField("density", "/World/Dust/Density")
            .SetMaterialBinding("/World/Looks/Dust");
        NanoVDBField::Define(scene, "/World/Dust/Density")
            .SetBuffer(Tensor::Copy<std::uint8_t>(
                dust, {static_cast<std::int64_t>(dust.size())}));
        Material::Define(scene, "/World/Looks/Dust")
            .SetDensity(kDensity)
            .SetVolumeAlbedo(glm::dvec3(0.9))
            .SetAnisotropy(0.6);
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(18.0)
            .SetSensorNoise(false)
            .SetExposureTime(1.0 / 30.0)
            .SetExposureFStop(2.0)
            .SetExposureIso(1600.0)
            .SetClippingRange({0.05, 1e4})
            .SetTransform(LookAtTransform(
                {4.5, 0.0, 4.5}, {0.0, 4.2, 1.0}, {0.0, 0.0, 1.0}));
        std::vector<Path> vars;
        for (const char* name : {"SpectralRadiance", "PrimId"}) {
            const Path var = Path("/Render/Camera").Child(name);
            RenderVar::Define(scene, var).SetSourceName(name);
            vars.push_back(var);
        }
        RenderProduct::Define(scene, "/Render/Camera")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(kResolution)
            .SetOrderedVars(vars);
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(kResolution)
            .SetAmbient(0.0);
        Widgets().Slider("Dust density (1/m)", 0.0, 1.0, kDensity,
                         [this](double density) {
                             Material(GetScene(), "/World/Looks/Dust")
                                 .SetDensity(density);
                         });
        // Wind carrying the dust through the beams, which then
        // flicker.
        Widgets().Slider("Wind (m/s)", 0.0, 3.0, wind_,
                         [this](double wind) { wind_ = wind; });
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
        if (wind_ > 0.0) {
            drift_ += wind_ * frameDt * kGust;
            const std::vector<std::uint8_t> dust = MakeDust(drift_);
            NanoVDBField(GetScene(), "/World/Dust/Density")
                .SetBuffer(Tensor::Copy<std::uint8_t>(
                    dust, {static_cast<std::int64_t>(dust.size())}));
        }
    }

    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, GetRenderer().Render(GetScene()));
    }

    void TestFinal() override {
        const auto [lit, ids] = Capture(true, kDensity);
        const auto [clear, clearIds] = Capture(true, 0.0);
        const auto [dark, darkIds] = Capture(false, kDensity);
        // What the dust adds to the frame in clear air: the beams.
        double brightest = 0.0;
        for (std::size_t i = 0; i < lit.size(); ++i) {
            brightest = std::max(brightest, lit[i] - clear[i]);
        }
        std::size_t visible = 0;
        for (std::size_t i = 0; i < lit.size(); ++i) {
            visible += lit[i] - clear[i] > 0.05 * brightest ? 1 : 0;
        }
        const double share = static_cast<double>(visible) /
                             static_cast<double>(lit.size());
        const double darkest =
            *std::max_element(dark.begin(), dark.end());
        std::printf(
            "the beams light %.1f%% of the pixels in the dust; with "
            "the lasers off, the brightest pixel is %.3g\n",
            100.0 * share, darkest);
        Check(share > 0.003, "the beams show in the dust");
        Check(darkest == 0.0, "nothing else lights it");
    }

  private:
    double wind_ = 0.0;      ///< m/s
    glm::dvec3 drift_{0.0};  ///< m the wind has carried the dust

    /// The radiance over the bands and the prim ids, with the lasers
    /// on or off and the dust at `density`.
    std::pair<std::vector<double>, std::vector<std::uint32_t>>
    Capture(bool lasers, double density) {
        Scene& scene = GetScene();
        for (const char* path :
             {"/World/Grid", "/World/Line", "/World/Spot"}) {
            scene.Set(path, tokens::kVisibility,
                      std::string(lasers ? tokens::kInherited
                                         : tokens::kInvisible));
        }
        Material(scene, "/World/Looks/Dust").SetDensity(density);
        const std::vector<Path> products{Path("/Render/Camera")};
        RenderProductSetOutputs outputs =
            GetRenderer().Step(scene, products);
        Material(scene, "/World/Looks/Dust").SetDensity(kDensity);
        auto& vars = outputs.at(Path("/Render/Camera")).vars;
        const Tensor bands = vars.at("SpectralRadiance")
                                 .Map(MapDevice::Cpu)
                                 .tensor.Contiguous();
        const Tensor prims =
            vars.at("PrimId").Map(MapDevice::Cpu).tensor.Contiguous();
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
        const auto* id = prims.As<std::uint32_t>();
        return {radiance,
                std::vector<std::uint32_t>(id, id + pixels)};
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    VisibleBeamsDemo, "lasers", "visible_beams",
    "Laser beams made visible by drifting dust at night.");
