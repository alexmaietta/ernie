// SPDX-License-Identifier: BSD-3-Clause
// An engine firing on a test stand through visible, 3 to 5 um and
// Temperature views: its exhaust a soot and temperature field glowing
// by Planck's law, its chamber and bell hot metal at their own
// temperatures.
//
// References:
// - M. Vollmer and K.-P. Moellmann, "Infrared Thermal Imaging",
//   2nd ed., Wiley-VCH, 2018.
// - F. S. Simmons, "Rocket Exhaust Plume Phenomenology", Aerospace
//   Press, 2000 (plume radiance and shock diamonds).

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/volume_schemas.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <glm/geometric.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

// The exhaust's grid: 0.125 m voxels from the nozzle's exit at the
// origin of x, along +x, about the engine's axis 1.5 m up.
constexpr std::int64_t kNx = 96;
constexpr std::int64_t kNy = 32;
constexpr std::int64_t kNz = 32;
constexpr double kVoxel = 0.125;
constexpr glm::dvec3 kGridOrigin(0.0, -2.0, -0.5);
constexpr double kAxisHeight = 1.5;
constexpr double kAmbient = 295.0;  // the air, the ground's too
constexpr double kPeakRise = 1700.0;  // above the air, at the nozzle
constexpr double kChamberKelvin = 700.0;
constexpr double kBellKelvin = 900.0;

constexpr glm::dvec2 kResolution(480.0, 270.0);
constexpr double kFocalLength = 28.0;  // mm
constexpr double kAperture = 20.955;   // mm
constexpr glm::dvec3 kEye(4.0, -20.0, 3.0);
constexpr glm::dvec3 kTarget(4.0, 0.0, 1.3);

/// The exhaust at a point, `s` metres downstream and `radial` off its
/// axis: its soot (relative density) and its temperature rise in
/// kelvin, which falls away downstream through bright shock diamonds.
glm::dvec2 Exhaust(double s, double radial) {
    const double width = 0.28 + 0.07 * s;
    const double core = std::exp(-(radial * radial) / (width * width));
    const double fade = std::clamp((11.5 - s) / 2.0, 0.0, 1.0);
    const double soot = core * (0.25 + 0.6 * (1.0 - std::exp(-s / 3.0))) /
                        (1.0 + 0.1 * s) * fade;
    const double diamonds =
        1.0 + 0.35 * std::exp(-s / 2.5) * std::cos(7.0 * s);
    const double rise = kPeakRise * std::exp(-s / 4.0) * core * diamonds;
    return {soot, rise * fade};
}

class ThermalPlumeDemo : public app::Demo {
  public:
    ThermalPlumeDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Xform::Define(scene, "/World/Looks");
        Material::Define(scene, "/World/Looks/Concrete")
            .SetReflectanceLibrary("concrete");
        Material::Define(scene, "/World/Looks/Steel")
            .SetReflectanceLibrary("galvanized_steel");
        GroundPlane::Define(scene, "/World/Ground")
            .SetMaterialBinding("/World/Looks/Concrete");
        // The stand, and the engine on it: chamber and bell at their
        // own temperatures.
        Cube::Define(scene, "/World/Stand")
            .SetSize(1.0)
            .SetMaterialBinding("/World/Looks/Concrete");
        SetXform(scene, "/World/Stand",
                 {.translate = glm::dvec3(-1.6, 0.0, 0.55),
                  .scale = glm::dvec3(2.4, 1.6, 1.1)});
        Cylinder::Define(scene, "/World/Chamber")
            .SetAxis("X")
            .SetRadius(0.3)
            .SetHeight(1.2)
            .SetMaterialBinding("/World/Looks/Steel")
            .SetThermalTemperature(kChamberKelvin);
        SetXform(scene, "/World/Chamber",
                 {.translate = glm::dvec3(-1.5, 0.0, kAxisHeight)});
        // The bell's apex towards the chamber, its mouth at the origin.
        Cone::Define(scene, "/World/Bell")
            .SetAxis("X")
            .SetRadius(0.42)
            .SetHeight(0.9)
            .SetMaterialBinding("/World/Looks/Steel")
            .SetThermalTemperature(kBellKelvin);
        SetXform(scene, "/World/Bell",
                 {.translate = glm::dvec3(-0.45, 0.0, kAxisHeight),
                  .rotate = glm::dvec3(0.0, 0.0, 180.0)});

        // The exhaust: soot whose absorption glows at its temperature.
        std::vector<float> soot(kNx * kNy * kNz);
        std::vector<float> rise(soot.size());
        for (std::int64_t i = 0; i < kNx; ++i) {
            for (std::int64_t j = 0; j < kNy; ++j) {
                for (std::int64_t k = 0; k < kNz; ++k) {
                    const glm::dvec3 p =
                        kGridOrigin +
                        kVoxel * glm::dvec3(static_cast<double>(i),
                                            static_cast<double>(j),
                                            static_cast<double>(k));
                    const glm::dvec2 at = Exhaust(
                        p.x, std::hypot(p.y, p.z - kAxisHeight));
                    const auto cell =
                        static_cast<std::size_t>((i * kNy + j) * kNz + k);
                    soot[cell] = at.x > 1e-4 ? static_cast<float>(at.x)
                                             : 0.0f;
                    rise[cell] = at.x > 1e-4 ? static_cast<float>(at.y)
                                             : 0.0f;
                }
            }
        }
        const Path plume("/World/Plume");
        Volume::Define(scene, plume)
            .SetField("density", plume.Child("Density"))
            .SetField("temperature", plume.Child("Temperature"))
            .SetMaterialBinding("/World/Looks/Exhaust");
        const std::array<std::pair<const char*, const std::vector<float>*>,
                         2>
            fields{{{"Density", &soot}, {"Temperature", &rise}}};
        for (const auto& [name, values] : fields) {
            DenseField::Define(scene, plume.Child(name))
                .SetVoxelSize(kVoxel)
                .SetValues(Tensor::Copy<float>(*values, {kNx, kNy, kNz}));
            SetXform(scene, plume.Child(name), {.translate = kGridOrigin});
        }
        Material::Define(scene, "/World/Looks/Exhaust")
            .SetDensity(4.0)
            .SetVolumeAlbedo(glm::dvec3(0.3))
            .SetTemperature(kAmbient)
            .SetTemperatureScale(1.0);

        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1000.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(0.53)
            .SetTransform(DistantLightTransform(
                glm::normalize(glm::dvec3(-0.3, -0.7, 0.8))));
        // A blue sky, scattered sunlight to none past 1.1 um.
        DomeLight::Define(scene, "/World/Sky")
            .SetSpectrum(std::vector<float>{
                380.0f,  0.22f,  450.0f,  0.17f, 500.0f, 0.13f,
                550.0f,  0.10f,  600.0f,  0.075f, 700.0f, 0.04f,
                850.0f,  0.017f, 1100.0f, 0.0f,  20000.0f, 0.0f});

        // Visible light by the sunny 16 rule; a 3 to 5 um camera of
        // flat response exposed for the exhaust, the cool ground dark.
        DefineCamera(kVisibleCamera, kVisibleProduct, {380.0, 720.0}, 17,
                     {"LdrColor"});
        DefineCamera(kInfraredCamera, kInfraredProduct, {3000.0, 5000.0},
                     8, {"LdrColor", "Temperature"});
        Camera(scene, kVisibleCamera)
            .SetExposureFStop(16.0)
            .SetExposureTime(0.01)
            .SetExposureIso(100.0);
        Camera(scene, kInfraredCamera)
            .SetSensorType("mono")
            .SetSensorQeMono(
                std::vector<float>{3000.0f, 0.8f, 5000.0f, 0.8f})
            .SetSensorFullWell(1e12)
            .SetExposureFStop(4.0)
            .SetExposureTime(1e-6);
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(kVisibleCamera)
            .SetResolution(kResolution)
            .SetAmbient(0.0)
            .SetThermalEnable(true)
            .SetThermalDefaultTemperature(kAmbient)
            .SetThermalBackgroundTemperature(230.0);

        Widgets().Combo("View", {"Visible", "3-5 um", "Temperature"}, 1,
                        [this](std::size_t index) { Show(index); });
        Widgets().Slider("Exhaust rise (K)", 200.0, 3000.0, kPeakRise,
                         [this](double kelvin) {
                             Material(GetScene(), "/World/Looks/Exhaust")
                                 .SetTemperatureScale(kelvin / kPeakRise);
                         });
        Show(1);
    }

    /// The visible, 3 to 5 um and false-colour Temperature images side
    /// by side.
    void SaveOutput(const std::filesystem::path& path) override {
        const Images images = RenderAll();
        const std::array<Tensor, 3> row = {
            Rgb(images.visible), Rgb(images.infrared),
            demos::Heatmap(images.kelvin, Rows(), Cols(), 250.0, 1300.0)};
        app::WritePng(path, demos::SideBySide(row));
    }

    void TestFinal() override {
        const Images images = RenderAll();
        const auto kelvinAt = [&](glm::dvec3 point) {
            const glm::ivec2 pixel = Pixel(point);
            return images.kelvin[static_cast<std::size_t>(
                pixel.y * Cols() + pixel.x)];
        };
        // At the first shock diamond, 2 pi / 7 m out.
        const double nozzle = kelvinAt({0.9, 0.0, kAxisHeight});
        const double tail = kelvinAt({9.0, 0.0, kAxisHeight});
        const double ground = kelvinAt({4.0, -4.0, 0.0});
        const double chamber = kelvinAt({-1.5, 0.0, kAxisHeight + 0.2});
        std::printf("Temperature: nozzle %.0f K, tail %.0f K, ground "
                    "%.0f K, chamber %.0f K\n",
                    nozzle, tail, ground, chamber);
        // A mean through the cooler sheath about the core, weighted by
        // what each part holds back.
        Check(nozzle > 800.0,
              "the exhaust by the nozzle should read over 800 K");
        Check(tail < nozzle - 300.0 && tail > kAmbient,
              "the exhaust should cool downstream");
        Check(std::abs(ground - kAmbient) < 0.5,
              "the ground should read the air's temperature");
        Check(std::abs(chamber - kChamberKelvin) < 0.5,
              "the chamber should read its own temperature");
        const double plumeIr = Brightness(images.infrared, {1.0, 0.0,
                                                           kAxisHeight});
        const double groundIr = Brightness(images.infrared, {4.0, -4.0,
                                                            0.0});
        const double plumeVisible =
            Brightness(images.visible, {1.0, 0.0, kAxisHeight});
        const double groundVisible =
            Brightness(images.visible, {4.0, -4.0, 0.0});
        std::printf("3-5 um: exhaust %.2f ground %.2f; visible: "
                    "exhaust %.2f ground %.2f\n",
                    plumeIr, groundIr, plumeVisible, groundVisible);
        Check(plumeIr > groundIr + 0.3,
              "the exhaust should outshine the ground at 3 to 5 um");
        Check(plumeVisible > groundVisible,
              "the exhaust should glow in the visible");
    }

  private:
    static constexpr const char* kVisibleCamera = "/World/Visible";
    static constexpr const char* kInfraredCamera = "/World/Infrared";
    static constexpr const char* kVisibleProduct = "/Render/Visible";
    static constexpr const char* kInfraredProduct = "/Render/Infrared";

    struct Images {
        Tensor visible;              // uint8 (rows, cols, 4)
        Tensor infrared;             // uint8 (rows, cols, 4)
        std::vector<double> kelvin;  // rows x cols
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
            Path(index == 0 ? kVisibleCamera : kInfraredCamera));
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
                                              Path(kInfraredProduct)};
        RenderProductSetOutputs outputs =
            GetRenderer().Step(GetScene(), products);
        const auto map = [&](const char* product, const char* var) {
            return outputs.at(Path(product))
                .vars.at(var)
                .Map(MapDevice::Cpu)
                .tensor;
        };
        Images images{map(kVisibleProduct, "LdrColor"),
                      map(kInfraredProduct, "LdrColor"),
                      {}};
        const Tensor kelvin = map(kInfraredProduct, "Temperature");
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

    /// The pixel (column, row) the cameras see `point` in.
    static glm::ivec2 Pixel(glm::dvec3 point) {
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

    /// The mean brightness (0 to 1) of the 3 x 3 pixels about where
    /// the cameras see `point`.
    static double Brightness(const Tensor& image, glm::dvec3 point) {
        const glm::ivec2 centre = Pixel(point);
        const auto* data = static_cast<const std::uint8_t*>(image.Data());
        double sum = 0.0;
        for (int r = -1; r <= 1; ++r) {
            for (int c = -1; c <= 1; ++c) {
                const auto pixel = static_cast<std::size_t>(
                    (centre.y + r) * Cols() + centre.x + c);
                for (std::size_t k = 0; k < 3; ++k) {
                    sum += data[pixel * 4 + k];
                }
            }
        }
        return sum / (27.0 * 255.0);
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    ThermalPlumeDemo, "thermal", "plume",
    "An engine firing on a test stand: its glowing exhaust and hot "
    "metal through visible, 3 to 5 um and Temperature views.");
