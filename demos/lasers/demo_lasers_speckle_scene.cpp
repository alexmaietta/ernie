// SPDX-License-Identifier: BSD-3-Clause
// Laser speckle in a room, through a phone-sized camera: an expanded
// green beam floods a table and a red pointer marks the wall. Rough
// plaster, paper and wood sparkle with full-contrast grain, brushed
// metal with partial grain, and a polished ball with none; the grain
// grows as the lens stops down and boils as the camera moves.
//
// References:
// - J. W. Goodman, "Speckle Phenomena in Optics", Roberts and
// Company,
//   2007 (grain size about lambda N on the sensor, contrast
//   1 / sqrt(M), partially developed speckle from smooth surfaces).

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr glm::ivec2 kResolution{480, 360};
constexpr double kPixel = 2e-3;  // mm, a phone camera's pixel
constexpr double kFocal = 1.0;   // mm
constexpr double kFNumber = 4.0;
constexpr glm::dvec3 kEye{0.7, -0.5, 1.35};  // m, the camera
constexpr glm::dvec3 kTarget{0.0, 1.5, 0.9};
constexpr const char* kLasers[] = {"/World/Beam", "/World/Pointer"};

/// A surface's look: its albedo and rms height (m), which makes its
/// speckle fully developed (rough), partial or absent (polished).
struct Look {
    const char* name;
    glm::dvec3 albedo;
    double height;
};

constexpr Look kLooks[] = {{"Plaster", {0.8, 0.78, 0.74}, 1e-5},
                           {"Wood", {0.45, 0.3, 0.18}, 1e-5},
                           {"Paper", {0.85, 0.85, 0.8}, 1e-5},
                           {"Brushed", {0.6, 0.6, 0.62}, 2.5e-8},
                           {"Polished", {0.9, 0.9, 0.9}, 1e-9}};

class SpeckleSceneDemo : public app::Demo {
  public:
    SpeckleSceneDemo(app::Viewer* viewer,
                     const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 120;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        for (const Look& look : kLooks) {
            Material::Define(scene,
                             Path("/World/Looks").Child(look.name))
                .SetAlbedo(look.albedo)
                .SetSpeckleRoughness(look.height);
        }
        GroundPlane::Define(scene, "/World/Floor")
            .SetMaterialBinding("/World/Looks/Wood");
        Cube::Define(scene, "/World/Wall")
            .SetSize(1.0)
            .SetMaterialBinding("/World/Looks/Plaster");
        SetXform(scene, "/World/Wall",
                 {.translate = glm::dvec3(0.0, 2.5, 1.5),
                  .scale = glm::dvec3(6.0, 0.1, 3.0)});
        Cube::Define(scene, "/World/Table")
            .SetSize(1.0)
            .SetMaterialBinding("/World/Looks/Wood");
        SetXform(scene, "/World/Table",
                 {.translate = glm::dvec3(0.0, 1.5, 0.725),
                  .scale = glm::dvec3(1.6, 0.8, 0.05)});
        Cube::Define(scene, "/World/Box")
            .SetSize(0.25)
            .SetMaterialBinding("/World/Looks/Paper");
        SetXform(scene, "/World/Box",
                 {.translate = glm::dvec3(-0.5, 1.5, 0.875)});
        Cylinder::Define(scene, "/World/Can")
            .SetRadius(0.1)
            .SetHeight(0.3)
            .SetMaterialBinding("/World/Looks/Brushed");
        SetXform(scene, "/World/Can",
                 {.translate = glm::dvec3(0.0, 1.5, 0.9)});
        Sphere::Define(scene, "/World/Ball")
            .SetRadius(0.12)
            .SetMaterialBinding("/World/Looks/Polished");
        SetXform(scene, "/World/Ball",
                 {.translate = glm::dvec3(0.5, 1.5, 0.87)});
        // A green beam expanded over the table's objects and a red
        // pointer's dot on the wall, both keeping their polarization.
        LaserLight::Define(scene, kLasers[0])
            .SetWavelength(532.0)
            .SetLinewidth(0.0)
            .SetPower(0.25)
            .SetWaistRadius(glm::dvec2(0.45, 0.25))
            .SetPolarizationDegree(1.0)
            .SetSpeckle(true)
            .SetTransform(LookAtTransform({-0.2, 0.3, 1.6}, kTarget,
                                          {0.0, 0.0, 1.0}));
        LaserLight::Define(scene, kLasers[1])
            .SetWavelength(650.0)
            .SetLinewidth(0.0)
            .SetPower(0.005)
            .SetWaistRadius(glm::dvec2(0.02))
            .SetPolarizationDegree(1.0)
            .SetSpeckle(true)
            .SetTransform(LookAtTransform({0.3, 0.2, 1.2},
                                          {-0.6, 2.45, 1.45},
                                          {0.0, 0.0, 1.0}));
        // A dim lamp over the table, so the room reads around the
        // lasers.
        SphereLight::Define(scene, "/World/Lamp")
            .SetRadius(0.1)
            .SetPhotometricPower(600.0)
            .SetEnableColorTemperature(true)
            .SetColorTemperature(3000.0);
        SetXform(scene, "/World/Lamp",
                 {.translate = glm::dvec3(0.0, 1.0, 2.6)});
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(kFocal)
            .SetHorizontalAperture(kPixel * kResolution.x)
            .SetVerticalAperture(kPixel * kResolution.y)
            .SetExposureIso(800.0)
            .SetSensorNoise(false)
            .SetClippingRange({0.02, 100.0});
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
        LookAt();
        StopDown(kFNumber);
        Widgets().Slider("f-number", 1.8, 16.0, kFNumber,
                         [this](double n) { StopDown(n); });
        Widgets().Checkbox("Speckle", true,
                           [this](bool on) { Speckle(on); });
        Widgets().Slider("Camera orbit (deg/s)", 0.0, 5.0, 0.0,
                         [this](double degrees) {
                             orbit_ =
                                 degrees * std::numbers::pi / 180.0;
                         });
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
        if (orbit_ > 0.0) {
            angle_ += orbit_ * frameDt;
            LookAt();
        }
    }

    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, GetRenderer().Render(GetScene()));
    }

    void TestFinal() override {
        const std::map<std::string, double> contrasts = Contrasts();
        std::printf("speckle contrast:");
        for (const auto& [name, contrast] : contrasts) {
            std::printf(" %s %.2f", name.c_str(), contrast);
        }
        std::printf("\n");
        for (const char* rough : {"Wall", "Box"}) {
            Check(contrasts.contains(rough) &&
                      contrasts.at(rough) > 0.3,
                  std::string(rough) + " speckles fully");
        }
        Check(contrasts.contains("Ball") &&
                  contrasts.at("Ball") < 0.1,
              "the polished ball does not");
        Check(contrasts.contains("Can") &&
                  contrasts.at("Ball") < contrasts.at("Can") &&
                  contrasts.at("Can") < contrasts.at("Box"),
              "the brushed can speckles in part");
    }

  private:
    double orbit_ = 0.0;  ///< radians a second the camera circles
    double angle_ = 0.0;

    /// Sets the lens to f-number `n`, the exposure time keeping the
    /// image as bright.
    void StopDown(double n) {
        Camera(GetScene(), "/World/Camera")
            .SetExposureFStop(n)
            .SetExposureTime(2e-3 * n * n);
    }

    void Speckle(bool on) {
        for (const char* path : kLasers) {
            LaserLight(GetScene(), path).SetSpeckle(on);
        }
    }

    /// The camera, turned `angle_` about the table's centre.
    void LookAt() {
        const double c = std::cos(angle_);
        const double s = std::sin(angle_);
        const glm::dvec2 from(kEye.x - kTarget.x, kEye.y - kTarget.y);
        const glm::dvec3 eye(kTarget.x + c * from.x - s * from.y,
                             kTarget.y + s * from.x + c * from.y,
                             kEye.z);
        Camera(GetScene(), "/World/Camera")
            .SetTransform(
                LookAtTransform(eye, kTarget, {0.0, 0.0, 1.0}));
    }

    /// The lasers' radiance summed over the bands (the lamp off),
    /// with or without speckle, and the prim id per pixel.
    std::pair<std::vector<double>, std::vector<std::uint32_t>>
    Capture(bool speckle) {
        Scene& scene = GetScene();
        Speckle(speckle);
        scene.Set("/World/Lamp", tokens::kVisibility,
                  std::string(tokens::kInvisible));
        const std::vector<Path> products{Path("/Render/Camera")};
        RenderProductSetOutputs outputs =
            GetRenderer().Step(scene, products);
        scene.Set("/World/Lamp", tokens::kVisibility,
                  std::string(tokens::kInherited));
        Speckle(true);
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

    /// Each lit surface's speckle contrast (deviation over mean of
    /// the light with speckle over without), over its pixels the
    /// lasers light brightly.
    std::map<std::string, double> Contrasts() {
        const auto [speckled, ids] = Capture(true);
        const auto [plain, plainIds] = Capture(false);
        std::map<std::string, double> result;
        for (const char* name :
             {"Wall", "Box", "Can", "Ball", "Table"}) {
            const Path path = Path("/World").Child(name);
            double brightest = 0.0;
            for (std::size_t i = 0; i < ids.size(); ++i) {
                if (GetRenderer().PrimPath(ids[i]) == path) {
                    brightest = std::max(brightest, plain[i]);
                }
            }
            double sum = 0.0;
            double squares = 0.0;
            std::size_t count = 0;
            for (std::size_t i = 0; i < ids.size(); ++i) {
                if (plain[i] > 0.2 * brightest && brightest > 0.0 &&
                    GetRenderer().PrimPath(ids[i]) == path) {
                    const double ratio = speckled[i] / plain[i];
                    sum += ratio;
                    squares += ratio * ratio;
                    ++count;
                }
            }
            if (count >= 20) {
                const double mean = sum / static_cast<double>(count);
                const double variance =
                    squares / static_cast<double>(count) -
                    mean * mean;
                result[name] =
                    std::sqrt(std::max(variance, 0.0)) / mean;
            }
        }
        return result;
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(SpeckleSceneDemo, "lasers", "speckle_scene",
                    "Laser speckle across a room's surfaces.");
