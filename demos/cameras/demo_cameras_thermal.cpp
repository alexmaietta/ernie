// SPDX-License-Identifier: BSD-3-Clause
// Measured finishes at one temperature and black spheres from 270 to
// 370 K, through a visible camera, an 8 to 14 um camera and the
// Temperature var: white paint glows as black paint does, while bare
// aluminium mirrors the cold sky.
//
// References:
// - S. K. Meerdink et al., "The ECOSTRESS spectral library version
//   1.0", Remote Sensing of Environment 230:111196, 2019.
// - M. Vollmer and K.-P. Moellmann, "Infrared Thermal Imaging",
//   2nd ed., Wiley-VCH, 2018.

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
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

// The back row, each at kBlockTemperature: all but bare aluminium
// emit nearly as a black body does at 8 to 14 um, weathered copper's
// oxide among them.
constexpr std::array<const char*, 7> kFinishes = {
    "black_paint", "olive_paint", "white_coating", "pine_wood",
    "brick",       "copper",      "aluminium"};
// The front row's black-painted spheres, in kelvin.
constexpr std::array<double, 6> kRamp = {270.0, 290.0, 310.0,
                                         330.0, 350.0, 370.0};
constexpr double kBlockTemperature = 330.0;
constexpr double kGroundTemperature = 295.0;
constexpr double kSkyTemperature = 230.0;  // a clear sky's, 8 to 14 um

constexpr glm::dvec2 kResolution(400.0, 300.0);
constexpr double kFocalLength = 24.0;  // mm
constexpr double kAperture = 20.955;   // mm
constexpr glm::dvec3 kEye(0.0, -5.5, 2.6);
constexpr glm::dvec3 kTarget(0.0, 0.2, 1.2);  // the sky above the horizon

glm::dvec3 FinishCentre(std::size_t i) {
    return {0.7 * (static_cast<double>(i) - 3.0), 0.8, 0.25};
}

glm::dvec3 RampCentre(std::size_t i) {
    return {0.7 * (static_cast<double>(i) - 2.5), -0.6, 0.25};
}

class ThermalDemo : public app::Demo {
  public:
    ThermalDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Xform::Define(scene, "/World/Looks");
        // Concrete, whose emissivity is its own and not the default
        // material's.
        Material::Define(scene, "/World/Looks/Concrete")
            .SetReflectanceLibrary("concrete");
        GroundPlane::Define(scene, "/World/Ground")
            .SetChecker(true)
            .SetCheckerColor0(glm::dvec3(1.0))
            .SetCheckerColor1(glm::dvec3(0.8))
            .SetMaterialBinding("/World/Looks/Concrete");
        for (std::size_t i = 0; i < kFinishes.size(); ++i) {
            const Path look("/World/Looks/" + std::string(kFinishes[i]));
            Material::Define(scene, look)
                .SetReflectanceLibrary(kFinishes[i])
                .SetThermalTemperature(kBlockTemperature);
            const Path block("/World/Block" + std::to_string(i));
            Cube::Define(scene, block).SetSize(0.5).SetMaterialBinding(
                look);
            SetXform(scene, block, {.translate = FinishCentre(i)});
        }
        for (std::size_t i = 0; i < kRamp.size(); ++i) {
            const Path look("/World/Looks/Sphere" + std::to_string(i));
            Material::Define(scene, look)
                .SetReflectanceLibrary("black_paint")
                .SetThermalTemperature(kRamp[i]);
            const Path ball("/World/Sphere" + std::to_string(i));
            Sphere::Define(scene, ball).SetRadius(0.25).SetMaterialBinding(
                look);
            SetXform(scene, ball, {.translate = RampCentre(i)});
        }
        // The Sun 40 degrees up behind the camera.
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1000.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(0.53)
            .SetTransform(DistantLightTransform(
                glm::normalize(glm::dvec3(-0.4, -0.8, 0.75))));
        // A blue sky, scattered sunlight falling as 1/lambda^4 to none
        // past 1.1 um; the thermal camera sees its background
        // temperature.
        DomeLight::Define(scene, "/World/Sky")
            .SetSpectrum(std::vector<float>{
                380.0f,  0.22f,  450.0f,  0.17f, 500.0f, 0.13f,
                550.0f,  0.10f,  600.0f,  0.075f, 700.0f, 0.04f,
                850.0f,  0.017f, 1100.0f, 0.0f,  20000.0f, 0.0f});

        // Visible light by the sunny 16 rule; the thermal camera a
        // microbolometer over 8 to 14 um, its brightness temperature in
        // the ironbow palette from 265 to 375 K.
        DefineCamera(kVisibleCamera, kVisibleProduct, {380.0, 720.0},
                     17, {"LdrColor"});
        DefineCamera(kThermalCamera, kThermalProduct,
                     {8000.0, 14000.0}, 6,
                     {"LdrColor", "Temperature", "BrightnessTemperature"});
        Camera(scene, kVisibleCamera)
            .SetExposureFStop(16.0)
            .SetExposureTime(0.01)
            .SetExposureIso(100.0);
        const Path chain("/Render/ThermalChain");
        SensorFilterPrim::Define(scene, chain.Child("Bolometer"))
            .SetFilterType("bolometer");
        SensorFilterPrim::Define(scene, chain.Child("Display"))
            .SetFilterType("thermalDisplay");
        for (const auto& [name, value] :
             {std::pair{"automatic", 0.0f}, std::pair{"low", 265.0f},
              std::pair{"high", 375.0f}, std::pair{"palette", 2.0f}}) {
            scene.Set(chain.Child("Display"), "inputs:" + std::string(name),
                      value);
        }
        SensorFilterPrim::Define(scene, chain.Child("Exposure"))
            .SetFilterType("exposure");
        RenderProduct(scene, kThermalProduct)
            .SetFilters({chain.Child("Bolometer"), chain.Child("Display"),
                         chain.Child("Exposure")});
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(kVisibleCamera)
            .SetResolution(kResolution)
            .SetAmbient(0.0)
            .SetThermalEnable(true)
            .SetThermalDefaultTemperature(kGroundTemperature)
            .SetThermalBackgroundTemperature(kSkyTemperature);

        Widgets().Combo("View",
                        {"Visible", "Thermal camera", "Temperature",
                         "Brightness temperature"},
                        1, [this](std::size_t index) { Show(index); });
        Widgets().Slider("Blocks (K)", 250.0, 400.0, kBlockTemperature,
                         [this](double kelvin) {
                             for (const char* finish : kFinishes) {
                                 Material(GetScene(),
                                          Path("/World/Looks/" +
                                               std::string(finish)))
                                     .SetThermalTemperature(kelvin);
                             }
                         });
        Show(1);
    }

    /// The visible and thermal camera's images, and false-colour
    /// Temperature and BrightnessTemperature, side by side.
    void SaveOutput(const std::filesystem::path& path) override {
        const Images images = RenderAll();
        const std::array<Tensor, 4> row = {
            Rgb(images.visible), Rgb(images.thermal),
            demos::Heatmap(images.kelvin, Rows(), Cols(),
                           kRamp.front(), kRamp.back()),
            demos::Heatmap(images.brightness, Rows(), Cols(),
                           kRamp.front(), kRamp.back())};
        app::WritePng(path, demos::SideBySide(row));
    }

    void TestFinal() override {
        const Images images = RenderAll();
        const auto at = [&](glm::dvec3 centre, double kelvin) {
            return Sample(images, centre, kelvin);
        };
        std::array<Sampled, kFinishes.size()> finish{};
        for (std::size_t i = 0; i < kFinishes.size(); ++i) {
            finish[i] = at(FinishCentre(i), kBlockTemperature);
            std::printf("%s: visible %.3f thermal %.3f\n", kFinishes[i],
                        finish[i].visible, finish[i].thermal);
            Check(finish[i].count > 0,
                  std::string(kFinishes[i]) +
                      " should read its temperature in the "
                      "Temperature var");
        }
        const Sampled& black = finish[0];
        const Sampled& white = finish[2];
        const Sampled& copper = finish[5];
        const Sampled& aluminium = finish[6];
        Check(white.visible > 3.0 * black.visible,
              "white paint should be the brighter in the visible");
        Check(std::abs(white.thermal - black.thermal) <
                  0.1 * black.thermal,
              "white and black paint should glow alike at 8 to 14 um");
        Check(std::abs(copper.thermal - black.thermal) <
                  0.1 * black.thermal,
              "weathered copper's oxide should glow as paint does");
        Check(aluminium.thermal < 0.6 * black.thermal,
              "bare aluminium should look cold: it mirrors the sky");
        const Sampled ground =
            at({1.0, -1.0, 0.0}, kGroundTemperature);
        Check(ground.count > 0,
              "the ground should read the default temperature");
        Check(aluminium.thermal < ground.thermal,
              "hot aluminium should look colder than the cooler ground");
        double previous = 0.0;
        for (std::size_t i = 0; i < kRamp.size(); ++i) {
            const Sampled sphere = at(RampCentre(i), kRamp[i]);
            std::printf("sphere at %.0f K: thermal %.3f\n", kRamp[i],
                        sphere.thermal);
            Check(sphere.count > 0 && sphere.thermal >= previous,
                  "the spheres should brighten with temperature");
            previous = sphere.thermal;
        }
        Check(at(RampCentre(kRamp.size() - 1), kRamp.back()).thermal >
                  1.5 * at(RampCentre(0), kRamp.front()).thermal,
              "370 K should far outshine 270 K");
        Check(images.kelvin.front() == 0.0,
              "the sky should have no temperature");
        // What the camera reads, for an emissivity of 1: black paint
        // near its 330 K, bare aluminium far colder.
        const auto brightnessAt = [&](glm::dvec3 point) {
            const glm::dvec2 pixel = Project(point);
            return images.brightness[static_cast<std::size_t>(
                static_cast<std::int64_t>(pixel.y) * Cols() +
                static_cast<std::int64_t>(pixel.x))];
        };
        const double paintReads = brightnessAt(FinishCentre(0));
        const double aluminiumReads = brightnessAt(FinishCentre(6));
        std::printf("brightness temperature: black paint %.1f K, bare "
                    "aluminium %.1f K\n",
                    paintReads, aluminiumReads);
        Check(std::abs(paintReads - kBlockTemperature) < 6.0,
              "black paint should read near its temperature");
        Check(aluminiumReads < 300.0,
              "bare aluminium should read far colder than it is");
        Check(Sky(images.thermal) < ground.thermal,
              "the sky should be the coldest in the thermal image");
    }

  private:
    static constexpr const char* kVisibleCamera = "/World/Visible";
    static constexpr const char* kThermalCamera = "/World/Thermal";
    static constexpr const char* kVisibleProduct = "/Render/Visible";
    static constexpr const char* kThermalProduct = "/Render/Thermal";

    struct Images {
        Tensor visible;                  // uint8 (rows, cols, 4)
        Tensor thermal;                  // uint8 (rows, cols, 4)
        std::vector<double> kelvin;      // rows x cols
        std::vector<double> brightness;  // rows x cols
    };

    /// Mean brightness (0 to 1) of the pixels around a point that
    /// read `kelvin`, and their count.
    struct Sampled {
        double visible = 0.0;
        double thermal = 0.0;
        int count = 0;
    };

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
        GetViewer()->SetViewVar(index == 2   ? "Temperature"
                                : index == 3 ? "BrightnessTemperature"
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
                      {},
                      {}};
        const Tensor kelvin = map(kThermalProduct, "Temperature");
        const auto* data = static_cast<const float*>(kelvin.Data());
        images.kelvin.assign(data, data + Rows() * Cols());
        const Tensor brightness =
            map(kThermalProduct, "BrightnessTemperature");
        const auto* read = static_cast<const float*>(brightness.Data());
        images.brightness.assign(read, read + Rows() * Cols());
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

    /// Where the cameras see `point`, as (column, row).
    static glm::dvec2 Project(glm::dvec3 point) {
        const glm::dvec3 forward = glm::normalize(kTarget - kEye);
        const glm::dvec3 right =
            glm::normalize(glm::cross(forward, {0.0, 0.0, 1.0}));
        const glm::dvec3 up = glm::cross(right, forward);
        const glm::dvec3 d = point - kEye;
        const double pixels = kFocalLength / kAperture * kResolution.x;
        const double depth = glm::dot(d, forward);
        return {0.5 * kResolution.x + pixels * glm::dot(d, right) / depth,
                0.5 * kResolution.y - pixels * glm::dot(d, up) / depth};
    }

    /// The mean brightness (0 to 1) of an image's top row.
    static double Sky(const Tensor& image) {
        const auto* data = static_cast<const std::uint8_t*>(image.Data());
        double sum = 0.0;
        for (std::int64_t c = 0; c < Cols(); ++c) {
            for (std::int64_t k = 0; k < 3; ++k) {
                sum += data[c * 4 + k];
            }
        }
        return sum / static_cast<double>(Cols()) / 765.0;
    }

    static Sampled Sample(const Images& images, glm::dvec3 point,
                          double kelvin) {
        const glm::dvec2 centre = Project(point);
        const auto* visible =
            static_cast<const std::uint8_t*>(images.visible.Data());
        const auto* thermal =
            static_cast<const std::uint8_t*>(images.thermal.Data());
        Sampled sampled;
        for (int r = -4; r <= 4; ++r) {
            for (int c = -4; c <= 4; ++c) {
                const auto row = static_cast<std::int64_t>(centre.y) + r;
                const auto col = static_cast<std::int64_t>(centre.x) + c;
                const auto pixel =
                    static_cast<std::size_t>(row * Cols() + col);
                if (std::abs(images.kelvin[pixel] - kelvin) > 0.01) {
                    continue;
                }
                for (std::size_t k = 0; k < 3; ++k) {
                    sampled.visible += visible[pixel * 4 + k] / 765.0;
                    sampled.thermal += thermal[pixel * 4 + k] / 765.0;
                }
                ++sampled.count;
            }
        }
        if (sampled.count > 0) {
            sampled.visible /= sampled.count;
            sampled.thermal /= sampled.count;
        }
        return sampled;
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    ThermalDemo, "cameras", "thermal",
    "Paints and metals at one temperature and spheres from 270 to "
    "370 K through visible and 8 to 14 um cameras, and the "
    "Temperature var.");
