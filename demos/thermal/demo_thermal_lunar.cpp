// SPDX-License-Identifier: BSD-3-Clause
// Temperatures computed as the scene renders: a small Moon whose
// surface follows a diurnal table baked by a regolith model (hot at
// noon, the afternoon warmer than the morning, the night near 100 K),
// and a spacecraft whose panels balance the sunlight they absorb
// against what they radiate at once, with no solver running. Move the
// Sun and both follow; the bare aluminium panel is the hottest yet
// looks the coldest to the 8 to 14 um camera, mirroring cold space.
//
// References:
// - P. O. Hayne et al., "Global regolith thermophysical properties of
//   the Moon from the Diviner Lunar Radiometer Experiment", JGR
//   Planets 122(12), 2017.
// - D. G. Gilmore (ed.), "Spacecraft Thermal Control Handbook", 2nd
//   ed., The Aerospace Press, 2002.

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

#include <glm/geometric.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kRadius = 10.0;  // m: the Moon, small enough to share the frame
constexpr glm::dvec3 kCraft(-5.0, -26.0, 4.0);  // the spacecraft's body centre
struct Panel {
    const char* name;
    double offset;
    const char* finish;
};
constexpr std::array<Panel, 2> kPanels = {
    Panel{"White", -2.2, "white_coating"},
    Panel{"Aluminium", 2.2, "aluminium"}};
constexpr float kInternal = 150.0f;  // W/m^2 the body's electronics give off
constexpr float kSpace = 3.0f;       // K
constexpr double kSunElevation = 0.15;  // z of the Sun's direction, before normalizing
constexpr double kAzimuth = 20.0;  // degrees: the Sun right of the camera, its start

constexpr glm::dvec2 kResolution(400.0, 300.0);
constexpr double kFocalLength = 24.0;  // mm
constexpr double kAperture = 20.955;   // mm
constexpr glm::dvec3 kEye(0.0, -45.0, 0.0);
constexpr glm::dvec3 kTarget(0.0, 0.0, 0.0);

/// Towards the Sun at `azimuth` degrees from behind the camera,
/// positive to the right: 0 is a full Moon.
glm::dvec3 SunDirection(double azimuth) {
    const double a = glm::radians(azimuth);
    return glm::normalize(
        glm::dvec3(std::sin(a), -std::cos(a), kSunElevation));
}

/// A 2 x 1 m panel beside the body, `offset` along x, turned to face
/// `facing`.
Tensor PanelPoints(double offset, glm::dvec3 facing) {
    const glm::dvec3 centre = kCraft + glm::dvec3(offset, 0.0, 0.0);
    const glm::dvec3 u =
        glm::normalize(glm::cross(glm::dvec3(0.0, 0.0, 1.0), facing));
    const glm::dvec3 v = glm::cross(facing, u);
    std::vector<double> points;
    for (const auto& [a, b] : {std::pair{-1.0, -1.0}, std::pair{1.0, -1.0},
                               std::pair{1.0, 1.0}, std::pair{-1.0, 1.0}}) {
        const glm::dvec3 p = centre + a * 0.75 * u + b * 0.5 * v;
        points.insert(points.end(), {p.x, p.y, p.z});
    }
    return Tensor::Copy<double>(points, {4, 3});
}

/// Where the cameras see `point`, as (column, row).
glm::ivec2 Project(glm::dvec3 point) {
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

class LunarThermalDemo : public app::Demo {
  public:
    LunarThermalDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Xform::Define(scene, "/World/Looks");
        // The Moon's diurnal table about its pole, +z.
        Material::Define(scene, "/World/Looks/Regolith")
            .SetAlbedo(glm::dvec3(0.12))
            .SetThermalModel("diurnal")
            .SetThermalModelParams({0.0f, 0.0f, 1.0f});
        Sphere::Define(scene, "/World/Moon")
            .SetRadius(kRadius)
            .SetMaterialBinding("/World/Looks/Regolith");
        // The body black and warmed by its electronics, the panels open
        // behind, each with its finish's absorptivity and emissivity.
        Material::Define(scene, "/World/Looks/Body")
            .SetReflectanceLibrary("black_paint")
            .SetThermalModel("balance")
            .SetThermalModelParams({-1.0f, -1.0f, kSpace, kInternal, 1.0f});
        Cube::Define(scene, "/World/Body")
            .SetSize(1.0)
            .SetMaterialBinding("/World/Looks/Body");
        SetXform(scene, "/World/Body", {.translate = kCraft});
        for (const Panel& panel : kPanels) {
            const Path look("/World/Looks/" + std::string(panel.name));
            Material::Define(scene, look)
                .SetReflectanceLibrary(panel.finish)
                .SetThermalModel("balance")
                .SetThermalModelParams({-1.0f, -1.0f, kSpace, 0.0f, 2.0f});
            const std::vector<std::int64_t> indices{0, 1, 2, 3};
            Mesh::Define(scene, Path("/World/" + std::string(panel.name)))
                .SetPoints(PanelPoints(panel.offset, SunDirection(0.0)))
                .SetFaceVertexCounts(Tensor::Copy<std::int64_t>(
                    std::vector<std::int64_t>{4}, {1}))
                .SetFaceVertexIndices(
                    Tensor::Copy<std::int64_t>(indices, {4}))
                .SetMaterialBinding(look);
        }
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun));
        PlaceSun(kAzimuth);

        // Visible light for sunlit regolith; the thermal camera a
        // microbolometer over 8 to 14 um, ironbow from 80 to 400 K.
        DefineCamera(kVisibleCamera, kVisibleProduct, {380.0, 720.0}, 17,
                     {"LdrColor"});
        DefineCamera(kThermalCamera, kThermalProduct, {8000.0, 14000.0},
                     6, {"LdrColor", "Temperature"});
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
             {std::pair{"automatic", 0.0f}, std::pair{"low", 80.0f},
              std::pair{"high", 400.0f}, std::pair{"palette", 2.0f}}) {
            scene.Set(chain.Child("Display"), "inputs:" + std::string(name),
                      value);
        }
        SensorFilterPrim::Define(scene, chain.Child("Exposure"))
            .SetFilterType("exposure");
        RenderProduct(scene, kThermalProduct)
            .SetFilters({chain.Child("Bolometer"), chain.Child("Display"),
                         chain.Child("Exposure")});
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(kThermalCamera)
            .SetResolution(kResolution)
            .SetAmbient(0.0)
            .SetThermalEnable(true)
            .SetThermalBackgroundTemperature(kSpace);

        Widgets().Combo("View",
                        {"Visible", "Thermal camera", "Temperature"}, 1,
                        [this](std::size_t index) { Show(index); });
        Widgets().Slider("Sun azimuth (deg)", -80.0, 80.0, kAzimuth,
                         [this](double azimuth) { PlaceSun(azimuth); });
        Show(1);
    }

    /// The visible and thermal cameras' images and false-colour
    /// Temperature, side by side.
    void SaveOutput(const std::filesystem::path& path) override {
        const Images images = RenderAll();
        const std::array<Tensor, 3> row = {
            Rgb(images.visible), Rgb(images.thermal),
            demos::Heatmap(images.kelvin, Rows(), Cols(), 80.0, 400.0)};
        app::WritePng(path, demos::SideBySide(row));
    }

    void TestFinal() override {
        const Images images = RenderAll();
        const auto kelvinAt = [&](glm::ivec2 pixel) {
            return images.kelvin[static_cast<std::size_t>(
                pixel.y * Cols() + pixel.x)];
        };
        // The Temperature var on the Moon's equator `longitude` degrees
        // right of the sub-camera point.
        const auto surface = [&](double longitude) {
            const double a = glm::radians(longitude);
            return kelvinAt(Project(
                kRadius * glm::dvec3(std::sin(a), -std::cos(a), 0.0)));
        };
        const double noon = surface(azimuth_);
        const int equator = Project(glm::dvec3(0.0)).y;
        double night = std::numeric_limits<double>::max();
        for (std::int64_t c = 0; c < Cols(); ++c) {
            const double kelvin = kelvinAt({static_cast<int>(c), equator});
            if (kelvin > 0.0) {
                night = std::min(night, kelvin);
            }
        }
        std::printf("sun at %.0f deg: noon %.1f K, night side %.1f K\n",
                    azimuth_, noon, night);
        Check(std::abs(noon - 390.0) < 10.0,
              "the subsolar point should be near 390 K");
        Check(night < 130.0, "the night side should be near 100 K");
        // The same sun 45 degrees either side of noon: afternoon (east
        // of the subsolar point, right) warmer than morning.
        const double afternoon = surface(azimuth_ + 45.0);
        const double morning = surface(azimuth_ - 45.0);
        std::printf("morning %.1f K, afternoon %.1f K\n", morning,
                    afternoon);
        Check(afternoon > morning + 0.5, "the afternoon should be warmer");
        // The panels: aluminium absorbs little but radiates less still,
        // hot; white paint radiates well, cold; yet the camera sees the
        // aluminium darker, its low emissivity mirroring space.
        std::array<std::pair<double, double>, kPanels.size()> reads{};
        const auto* thermal =
            static_cast<const std::uint8_t*>(images.thermal.Data());
        for (std::size_t i = 0; i < kPanels.size(); ++i) {
            const glm::ivec2 pixel =
                Project(kCraft + glm::dvec3(kPanels[i].offset, 0.0, 0.0));
            const auto at = static_cast<std::size_t>(pixel.y * Cols() +
                                                     pixel.x) *
                            4;
            reads[i] = {kelvinAt(pixel),
                        (thermal[at] + thermal[at + 1] + thermal[at + 2]) /
                            3.0};
            std::printf("%s panel: %.1f K, thermal %.0f\n",
                        kPanels[i].name, reads[i].first, reads[i].second);
        }
        Check(reads[1].first > reads[0].first + 100.0,
              "bare aluminium should run far hotter than white paint");
        Check(reads[1].second < reads[0].second,
              "yet it should look the colder at 8 to 14 um");
        const double body = kelvinAt(Project(kCraft));
        std::printf("body %.1f K\n", body);
        Check(body > 250.0 && body < 400.0, "the body should sit between");
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

    double azimuth_ = kAzimuth;

    /// Moves the Sun, the panels turning to face it.
    void PlaceSun(double azimuth) {
        azimuth_ = azimuth;
        Scene& scene = GetScene();
        const glm::dvec3 direction = SunDirection(azimuth);
        DistantLight(scene, "/World/Sun")
            .SetTransform(DistantLightTransform(direction));
        for (const Panel& panel : kPanels) {
            Mesh(scene, Path("/World/" + std::string(panel.name)))
                .SetPoints(PanelPoints(panel.offset, direction));
        }
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
        GetViewer()->SetViewVar(index == 2 ? "Temperature" : "LdrColor");
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
        const auto* in = static_cast<const std::uint8_t*>(image.Data());
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
    LunarThermalDemo, "thermal", "lunar",
    "A Moon on a diurnal temperature table and a spacecraft in "
    "zero-capacitance balance, their temperatures computed as they "
    "render, through visible and 8 to 14 um cameras.");
