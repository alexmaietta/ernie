// SPDX-License-Identifier: BSD-3-Clause
// A rocket engine's plume in sunlight: hot gas glowing as a black
// body at the nozzle, cooling to soot that shadows and dims what is
// behind.

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

/// The plume's hottest, at the nozzle.
constexpr double kPeakKelvin = 3000.0;

struct PlumeGrids {
    std::vector<std::uint8_t> density;
    std::vector<std::uint8_t> temperature;  ///< Over 0 to 1.
};

/// A rocket engine's plume from a nozzle 1.2 m across at the origin
/// down -z for 12 m, in 0.1 m voxels: widening as it goes, hottest at
/// the nozzle with shock diamonds, cooling to soot downstream.
PlumeGrids MakePlume() {
    constexpr double kVoxel = 0.1;
    const glm::ivec3 shape(48, 48, 120);
    const glm::dvec3 low(-2.4, -2.4, -12.0);
    const std::size_t count =
        static_cast<std::size_t>(shape.x) * shape.y * shape.z;
    std::vector<float> density(count, 0.0f);
    std::vector<float> temperature(count, 0.0f);
    for (int i = 0; i < shape.x; ++i) {
        for (int j = 0; j < shape.y; ++j) {
            for (int k = 0; k < shape.z; ++k) {
                const glm::dvec3 p =
                    low + kVoxel * (glm::dvec3(i, j, k) + 0.5);
                const double s = std::max(-p.z, 0.0);  // Downstream.
                const double radius = 0.6 + 0.12 * s;
                const double core =
                    std::exp(-2.0 * (p.x * p.x + p.y * p.y) /
                             (radius * radius));
                const double diamonds =
                    1.0 +
                    0.35 *
                        std::cos(2.0 * std::numbers::pi * s / 1.4) *
                        std::exp(-s / 3.0);
                const double fade =
                    std::clamp(s / 0.3, 0.0, 1.0) *
                    std::clamp((12.0 - s) / 2.0, 0.0, 1.0);
                const double hot =
                    core * diamonds * std::exp(-s / 5.0) * fade;
                const double soot =
                    core * (0.8 + 0.4 * (1.0 - std::exp(-s / 4.0))) *
                    std::exp(-s / 14.0) * fade;
                const std::size_t at =
                    (static_cast<std::size_t>(i) * shape.y + j) *
                        shape.z +
                    k;
                // Nothing below 1e-3, so the grids stay sparse.
                density[at] =
                    soot > 1e-3 ? static_cast<float>(soot) : 0.0f;
                temperature[at] =
                    hot > 1e-3 ? static_cast<float>(hot) : 0.0f;
            }
        }
    }
    const glm::dvec3 origin = low + 0.5 * kVoxel;
    return {
        DenseNanoVdbGrid(density, shape, kVoxel, origin, "density"),
        DenseNanoVdbGrid(temperature, shape, kVoxel, origin,
                         "temperature")};
}

Tensor Bytes(const std::vector<std::uint8_t>& bytes) {
    return Tensor::Copy<std::uint8_t>(
        bytes, {static_cast<std::int64_t>(bytes.size())});
}

/// The image's mean linear RGB over columns [left, right) and rows
/// [top, bottom).
glm::dvec3 MeanIn(const Tensor& image, std::int64_t left,
                  std::int64_t right, std::int64_t top,
                  std::int64_t bottom) {
    const std::int64_t cols = image.Shape()[1];
    const std::uint8_t* pixels = image.As<std::uint8_t>();
    glm::dvec3 sum(0.0);
    for (std::int64_t r = top; r < bottom; ++r) {
        for (std::int64_t c = left; c < right; ++c) {
            const std::uint8_t* p = pixels + 4 * (r * cols + c);
            sum += demos::SrgbToLinear(glm::dvec3(p[0], p[1], p[2]) /
                                       255.0);
        }
    }
    return sum / static_cast<double>((right - left) * (bottom - top));
}

class PlumeDemo : public app::Demo {
  public:
    PlumeDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // The rocket above its nozzle at the origin, 1.2 m across.
        Cylinder::Define(scene, "/World/Rocket/Body")
            .SetRadius(0.6)
            .SetHeight(6.0)
            .SetMaterialBinding("/World/Looks/Paint");
        SetXform(scene, "/World/Rocket/Body",
                 {.translate = glm::dvec3(0.0, 0.0, 3.4)});
        Cone::Define(scene, "/World/Rocket/Nose")
            .SetRadius(0.6)
            .SetHeight(1.6)
            .SetMaterialBinding("/World/Looks/Paint");
        SetXform(scene, "/World/Rocket/Nose",
                 {.translate = glm::dvec3(0.0, 0.0, 7.2)});
        Cone::Define(scene, "/World/Rocket/Nozzle")
            .SetRadius(0.4)
            .SetHeight(0.8)
            .SetMaterialBinding("/World/Looks/Metal");
        SetXform(scene, "/World/Rocket/Nozzle",
                 {.translate = glm::dvec3(0.0, 0.0, 0.4)});
        Material::Define(scene, "/World/Looks/Paint")
            .SetAlbedo(glm::dvec3(0.7));
        Material::Define(scene, "/World/Looks/Metal")
            .SetAlbedo(glm::dvec3(0.25));

        // The plume's density and temperature, built here as NanoVDB
        // grids.
        const PlumeGrids grids = MakePlume();
        Volume::Define(scene, "/World/Plume")
            .SetField("density", "/World/Plume/Density")
            .SetField("temperature", "/World/Plume/Temperature")
            .SetMaterialBinding("/World/Looks/Exhaust");
        NanoVDBField::Define(scene, "/World/Plume/Density")
            .SetBuffer(Bytes(grids.density));
        NanoVDBField::Define(scene, "/World/Plume/Temperature")
            .SetBuffer(Bytes(grids.temperature));
        // Soot: mostly absorbing, so it glows as a black body where
        // hot (Kirchhoff) and scatters a little sunlight.
        Material::Define(scene, "/World/Looks/Exhaust")
            .SetDensity(1.5)
            .SetVolumeAlbedo(glm::dvec3(0.3))
            .SetAnisotropy(0.4);
        SetPeakTemperature(kPeakKelvin);

        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetShadowEnable(true)
            .SetTransform(DistantLightTransform({-0.5, -1.0, 0.6}));
        // Daylight exposure, from the side 40 m off.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(35.0)
            .SetSensorNoise(false)
            .SetExposureIso(100.0)
            .SetExposureFStop(8.0)
            .SetExposureTime(1.0 / 500.0)
            .SetTransform(LookAtTransform({0.0, -40.0, -1.0},
                                          {0.0, 0.0, -1.5},
                                          {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({320, 480})
            .SetAmbient(0.0);

        Widgets().Slider(
            "Peak temperature (K)", 800.0, 3500.0, kPeakKelvin,
            [this](double kelvin) { SetPeakTemperature(kelvin); });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, GetRenderer().Render(GetScene()));
    }

    void TestFinal() override {
        const Tensor image = GetRenderer().Render(GetScene());
        const std::int64_t rows = image.Shape()[0];
        const std::int64_t cols = image.Shape()[1];
        // The brightest spot down the plume's axis (a shock diamond),
        // and the empty sky beside it.
        glm::dvec3 core(0.0);
        for (std::int64_t row = rows / 4; row + 8 <= rows; row += 4) {
            const glm::dvec3 spot = MeanIn(
                image, cols / 2 - 4, cols / 2 + 4, row, row + 8);
            if (spot.r > core.r) {
                core = spot;
            }
        }
        const glm::dvec3 sky = MeanIn(image, 0, 16, rows - 32, rows);
        std::printf("Plume core %.4f %.4f %.4f, sky %.4f %.4f %.4f\n",
                    core.r, core.g, core.b, sky.r, sky.g, sky.b);
        Check(sky.r + sky.g + sky.b < 0.01, "space is black");
        Check(core.r > 0.2, "the hot core glows");
        Check(core.r > core.b, "the glow is warm, as a black body's");
    }

  private:
    void SetPeakTemperature(double kelvin) {
        Material(GetScene(), "/World/Looks/Exhaust")
            .SetTemperatureScale(kelvin);
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    PlumeDemo, "volumes", "plume",
    "A rocket engine's plume in sunlight: black-body glow and soot.");
