// SPDX-License-Identifier: BSD-3-Clause
// A patch of lunar terrain whose every pixel has its own albedo and
// thermal inertia: a diurnal table baked here by the regolith model
// over both, read at each pixel's albedo map and rock abundance map.
// By day the dark mare runs hotter than the bright highland; at night
// the rocky ejecta of fresh craters hold the day's heat and glow
// against fines cooled to about 100 K, as Diviner's night-time maps
// show.
//
// References:
// - J. L. Bandfield et al., "Lunar surface rock abundance and
// regolith
//   fines temperatures derived from LRO Diviner Radiometer data", JGR
//   Planets 116(E12), 2011.
// - P. O. Hayne et al., "Global regolith thermophysical properties of
//   the Moon from the Diviner Lunar Radiometer Experiment", JGR
//   Planets 122(12), 2017.

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
#include "ernie/util/regolith.h"
#include "util.h"

#include <glm/geometric.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kHalf = 100.0;  // m: the patch spans -kHalf to kHalf
                                 // in x (east) and y (north)
constexpr int kGrid = 128;       // quads along a side
constexpr int kMap =
    128;  // texels along a side of the albedo and rock maps
// Craters: centre, radius (m) and freshness (0 old, 1 fresh).
struct Crater {
    double x;
    double y;
    double radius;
    double fresh;
};
constexpr std::array<Crater, 5> kCraters = {
    Crater{-40.0, 30.0, 22.0, 1.0}, Crater{45.0, -35.0, 16.0, 0.9},
    Crater{30.0, 50.0, 12.0, 0.2}, Crater{-55.0, -50.0, 18.0, 0.1},
    Crater{5.0, -5.0, 9.0, 0.7}};
constexpr double kFines =
    50.0;  // J m^-2 K^-1 s^-1/2: regolith fines' thermal inertia
constexpr double kRock = 1500.0;  // bare rock's
constexpr double kStartHour =
    16.0;  // local solar time: lit; the night is the slider's

constexpr glm::dvec2 kResolution(400.0, 300.0);
constexpr double kFocalLength = 24.0;  // mm
constexpr double kAperture = 20.955;   // mm
constexpr glm::dvec3 kEye(0.0, -230.0, 140.0);
constexpr glm::dvec3 kTarget(0.0, 0.0, -10.0);

/// The terrain's height (m): each crater a bowl with a raised rim.
double Height(double x, double y) {
    double z = 0.0;
    for (const Crater& crater : kCraters) {
        const double r =
            std::hypot(x - crater.x, y - crater.y) / crater.radius;
        z += r < 1.0 ? 0.2 * crater.radius * (r * r - 1.0) +
                           0.04 * crater.radius
                     : 0.04 * crater.radius / (r * r * r);
    }
    return z;
}

/// The albedo and thermal inertia maps, float32 (kMap, kMap), row 0
/// at the north edge: dark mare to the west, bright highland to the
/// east, fresh craters' rays brighter and their ejecta rocky.
std::pair<std::vector<float>, std::vector<float>> Maps() {
    std::vector<float> albedo;
    std::vector<float> inertia;
    const double step = 2.0 * kHalf / kMap;
    for (int row = 0; row < kMap; ++row) {
        const double y = kHalf - (row + 0.5) * step;
        for (int col = 0; col < kMap; ++col) {
            const double x = -kHalf + (col + 0.5) * step;
            double a = 0.115 + 0.035 * std::tanh(x / 25.0);
            double rock = 0.0;
            for (const Crater& crater : kCraters) {
                const double r =
                    std::max(std::hypot(x - crater.x, y - crater.y) /
                                 crater.radius,
                             0.5);
                a += 0.08 * crater.fresh *
                     (r > 1.0 ? 1.0 / (r * r) : 1.0);
                rock = std::max(rock,
                                crater.fresh *
                                    std::min(1.0, 1.0 / (r * r * r)));
            }
            albedo.push_back(static_cast<float>(a));
            inertia.push_back(
                static_cast<float>(kFines + (kRock - kFines) * rock));
        }
    }
    return {albedo, inertia};
}

/// Towards the Sun at local solar time `hour` on the equator, the
/// pole north (+y): rising in the east (+x).
glm::dvec3 SunDirection(double hour) {
    const double a = (hour / 24.0 - 0.5) * 2.0 * M_PI;
    return {-std::sin(a), 0.0, std::cos(a)};
}

/// Where the cameras see the ground at (x, y), as (column, row).
glm::ivec2 Project(double x, double y) {
    const glm::dvec3 point(x, y, Height(x, y));
    const glm::dvec3 forward = glm::normalize(kTarget - kEye);
    const glm::dvec3 right =
        glm::normalize(glm::cross(forward, {0.0, 0.0, 1.0}));
    const glm::dvec3 up = glm::cross(right, forward);
    const glm::dvec3 d = point - kEye;
    const double pixels = kFocalLength / kAperture * kResolution.x;
    const double depth = glm::dot(d, forward);
    return {static_cast<int>(0.5 * kResolution.x +
                             pixels * glm::dot(d, right) / depth),
            static_cast<int>(0.5 * kResolution.y -
                             pixels * glm::dot(d, up) / depth)};
}

class RocksThermalDemo : public app::Demo {
  public:
    RocksThermalDemo(app::Viewer* viewer,
                     const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Xform::Define(scene, "/World/Looks");
        // The table over every inertia and albedo the maps hold.
        const DiurnalTable table = DiurnalTemperatureTable(
            RegolithProperties{}, 7, 48,
            {25.0f, 50.0f, 100.0f, 200.0f, 400.0f, 800.0f, 1600.0f},
            {0.05f, 0.1f, 0.15f, 0.25f});
        const auto [albedo, inertia] = Maps();
        const Path look("/World/Looks/Regolith");
        Material::Define(scene, look)
            .SetThermalModel("diurnal")
            .SetThermalModelParams({0.0f, 1.0f, 0.0f})
            .SetThermalModelData(table.ModelData())
            .SetThermalInertiaTexture(
                Tensor::Copy<float>(inertia, {kMap, kMap}));
        Shader::Define(scene, look.Child("Surface"))
            .SetInfoId(std::string(tokens::kUsdPreviewSurface))
            .SetRoughness(0.9);
        Connect(scene, look, tokens::kSurfaceOutput,
                look.Child("Surface"), tokens::kSurfaceOutput);
        std::vector<float> rgb;
        for (const float a : albedo) {
            rgb.insert(rgb.end(), {a, a, a});
        }
        Shader::Define(scene, look.Child("Albedo"))
            .SetInfoId(std::string(tokens::kUsdUVTexture))
            .SetSourceColorSpace("raw")
            .SetTexture(Tensor::Copy<float>(rgb, {kMap, kMap, 3}));
        Connect(scene, look.Child("Surface"), tokens::kDiffuseColor,
                look.Child("Albedo"), "outputs:rgb");
        DefineTerrain(look);
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun));
        PlaceSun(kStartHour);

        // Visible light by day; a microbolometer over 8 to 14 um,
        // ironbow over the range the last frame showed (80 to 400 K
        // before one).
        DefineCamera(kVisibleCamera, kVisibleProduct, {380.0, 720.0},
                     17, {"LdrColor"});
        DefineCamera(kThermalCamera, kThermalProduct,
                     {8000.0, 14000.0}, 6,
                     {"LdrColor", "Temperature"});
        Camera(scene, kVisibleCamera)
            .SetExposureFStop(16.0)
            .SetExposureTime(0.004)
            .SetExposureIso(100.0);
        const Path chain("/Render/ThermalChain");
        SensorFilterPrim::Define(scene, chain.Child("Bolometer"))
            .SetFilterType("bolometer");
        SensorFilterPrim::Define(scene, chain.Child("Display"))
            .SetFilterType("thermalDisplay");
        for (const auto& [name, value] :
             {std::pair{"automatic", 1.0f}, std::pair{"low", 80.0f},
              std::pair{"high", 400.0f},
              std::pair{"palette", 2.0f}}) {
            scene.Set(chain.Child("Display"),
                      "inputs:" + std::string(name), value);
        }
        SensorFilterPrim::Define(scene, chain.Child("Exposure"))
            .SetFilterType("exposure");
        RenderProduct(scene, kThermalProduct)
            .SetFilters({chain.Child("Bolometer"),
                         chain.Child("Display"),
                         chain.Child("Exposure")});
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(kThermalCamera)
            .SetResolution(kResolution)
            .SetAmbient(0.0)
            .SetThermalEnable(true)
            .SetThermalBackgroundTemperature(3.0);

        Widgets().Combo(
            "View", {"Visible", "Thermal camera", "Temperature"}, 1,
            [this](std::size_t index) { Show(index); });
        Widgets().Slider("Local time (h)", 0.0, 24.0, kStartHour,
                         [this](double hour) { PlaceSun(hour); });
        Show(1);
    }

    /// The visible and thermal cameras' images and false-colour
    /// Temperature, side by side.
    void SaveOutput(const std::filesystem::path& path) override {
        const Images images = RenderAll();
        const std::array<Tensor, 3> row = {
            Rgb(images.visible), Rgb(images.thermal),
            demos::Heatmap(images.kelvin, Rows(), Cols(), 80.0,
                           400.0)};
        app::WritePng(path, demos::SideBySide(row));
    }

    void TestFinal() override {
        const auto kelvinAt = [&](const Images& images, double x,
                                  double y) {
            const glm::ivec2 pixel = Project(x, y);
            return images.kelvin[static_cast<std::size_t>(
                pixel.y * Cols() + pixel.x)];
        };
        // 10 pm: the fresh crater's rocky ejecta against old fines (a
        // frame first for the display's gain to take the night's
        // range).
        PlaceSun(22.0);
        RenderAll();
        const Images night = RenderAll();
        const double rocky =
            kelvinAt(night, -40.0, 30.0 - 22.0 * 1.15);
        const double fines = kelvinAt(night, -60.0, -85.0);
        std::printf("%.0f h: rocky ejecta %.1f K, fines %.1f K\n",
                    hour_, rocky, fines);
        Check(fines < 130.0, "fines should be near 100 K at night");
        Check(rocky > fines + 40.0,
              "rocks should hold the day's heat");
        // The camera sees them: textured ground emits by its own
        // albedo.
        const auto brightness = [&](double x, double y) {
            const glm::ivec2 pixel = Project(x, y);
            const auto* rgba =
                static_cast<const std::uint8_t*>(
                    night.thermal.Data()) +
                static_cast<std::size_t>(pixel.y * Cols() + pixel.x) *
                    4;
            return (rgba[0] + rgba[1] + rgba[2]) / 3.0;
        };
        const double glow = brightness(-40.0, 30.0 - 22.0 * 1.15);
        const double dark = brightness(-60.0, -85.0);
        std::printf("thermal camera: rocks %.0f, fines %.0f\n", glow,
                    dark);
        Check(glow > dark + 40.0,
              "the rocks should glow to the camera");
        // Noon: the dark mare (west) hotter than the bright highland.
        PlaceSun(12.0);
        const Images noon = RenderAll();
        const double mare = kelvinAt(noon, -80.0, -10.0);
        const double highland = kelvinAt(noon, 80.0, 10.0);
        std::printf("noon: mare %.1f K, highland %.1f K\n", mare,
                    highland);
        Check(mare > 360.0 && mare < 400.0,
              "the mare should be near 390 K");
        Check(mare > highland + 4.0,
              "darker ground should be hotter");
        PlaceSun(kStartHour);
    }

  private:
    static constexpr const char* kVisibleCamera = "/World/Visible";
    static constexpr const char* kThermalCamera = "/World/Thermal";
    static constexpr const char* kVisibleProduct = "/Render/Visible";
    static constexpr const char* kThermalProduct = "/Render/Thermal";

    struct Images {
        Tensor visible;              // uint8 (rows, cols, 4)
        Tensor thermal;              // uint8 (rows, cols, 4)
        std::vector<double> kelvin;  // rows x cols
    };

    double hour_ = kStartHour;

    /// The patch: a grid of quads over the heights, with texture
    /// coordinates across it (its smooth normals the renderer's).
    void DefineTerrain(const Path& look) {
        std::vector<double> points;
        std::vector<float> st;
        for (int i = 0; i <= kGrid; ++i) {
            const double y = -kHalf + 2.0 * kHalf * i / kGrid;
            for (int j = 0; j <= kGrid; ++j) {
                const double x = -kHalf + 2.0 * kHalf * j / kGrid;
                points.insert(points.end(), {x, y, Height(x, y)});
                st.insert(
                    st.end(),
                    {static_cast<float>((x + kHalf) / (2.0 * kHalf)),
                     static_cast<float>((y + kHalf) /
                                        (2.0 * kHalf))});
            }
        }
        std::vector<std::int64_t> indices;
        for (int i = 0; i < kGrid; ++i) {
            for (int j = 0; j < kGrid; ++j) {
                const std::int64_t first = i * (kGrid + 1) + j;
                indices.insert(indices.end(),
                               {first, first + 1, first + kGrid + 2,
                                first + kGrid + 1});
            }
        }
        const std::int64_t count = (kGrid + 1) * (kGrid + 1);
        Scene& scene = GetScene();
        Mesh::Define(scene, "/World/Terrain")
            .SetPoints(Tensor::Copy<double>(points, {count, 3}))
            .SetFaceVertexCounts(Tensor::Copy<std::int64_t>(
                std::vector<std::int64_t>(kGrid * kGrid, 4),
                {kGrid * kGrid}))
            .SetFaceVertexIndices(Tensor::Copy<std::int64_t>(
                indices, {static_cast<std::int64_t>(indices.size())}))
            .SetMaterialBinding(look);
        scene.Write("/World/Terrain", "primvars:st",
                    Tensor::Copy<float>(st, {count, 2}));
    }

    void PlaceSun(double hour) {
        hour_ = hour;
        DistantLight(GetScene(), "/World/Sun")
            .SetTransform(DistantLightTransform(SunDirection(hour)));
    }

    void DefineCamera(const char* camera, const char* product,
                      glm::dvec2 range, int bands,
                      std::vector<std::string> vars) {
        Scene& scene = GetScene();
        Camera::Define(scene, camera)
            .SetFocalLength(kFocalLength)
            .SetHorizontalAperture(kAperture)
            .SetVerticalAperture(kAperture * kResolution.y /
                                 kResolution.x)
            .SetSensorNoise(false)
            .SetSpectralRange(range)
            .SetSpectralBands(bands)
            .SetTransform(
                LookAtTransform(kEye, kTarget, {0.0, 0.0, 1.0}));
        std::vector<Path> ordered;
        for (const std::string& var : vars) {
            const Path path = Path(product).Child(var);
            RenderVar::Define(scene, path).SetSourceName(var);
            ordered.push_back(path);
        }
        RenderProduct::Define(scene, product)
            .SetCamera(Path(camera))
            .SetResolution(kResolution)
            .SetOrderedVars(ordered);
    }

    void Show(std::size_t index) {
        if (GetViewer() == nullptr) {
            return;
        }
        GetViewer()->SetActiveCamera(
            Path(index == 0 ? kVisibleCamera : kThermalCamera));
        GetViewer()->SetViewVar(index == 2 ? "Temperature"
                                           : "LdrColor");
    }

    static std::int64_t Rows() {
        return static_cast<std::int64_t>(kResolution.y);
    }
    static std::int64_t Cols() {
        return static_cast<std::int64_t>(kResolution.x);
    }

    Images RenderAll() {
        const std::array<Path, 2> products = {Path(kVisibleProduct),
                                              Path(kThermalProduct)};
        RenderProductSetOutputs outputs =
            GetRenderer().Step(GetScene(), products);
        const auto map = [&](const char* product, const char* var) {
            return outputs.at(Path(product))
                .vars.at(var)
                .Map(MapDevice::Cpu)
                .tensor;
        };
        Images images{map(kVisibleProduct, "LdrColor"),
                      map(kThermalProduct, "LdrColor"),
                      {}};
        const Tensor kelvin = map(kThermalProduct, "Temperature");
        const auto* data = static_cast<const float*>(kelvin.Data());
        images.kelvin.assign(data, data + Rows() * Cols());
        return images;
    }

    /// An image's colour channels alone, for side by side with the
    /// heatmap.
    static Tensor Rgb(const Tensor& image) {
        Tensor rgb = Tensor::Zeros(DType::UInt8, {Rows(), Cols(), 3});
        const auto* in =
            static_cast<const std::uint8_t*>(image.Data());
        auto* out = static_cast<std::uint8_t*>(rgb.Data());
        const auto channels = image.Shape()[2];
        for (std::int64_t i = 0; i < Rows() * Cols(); ++i) {
            std::copy_n(in + i * channels, 3, out + i * 3);
        }
        return rgb;
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    RocksThermalDemo, "thermal", "rocks",
    "Lunar terrain with per-pixel albedo and rock abundance on a "
    "diurnal table baked over both: dark mare hotter by day, rocky "
    "ejecta glowing at night, through visible and 8 to 14 um "
    "cameras.");
