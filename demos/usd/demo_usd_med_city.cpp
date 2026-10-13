// SPDX-License-Identifier: BSD-3-Clause
// Pixar's Point Instanced Med City, imported with ImportUsd: one
// UsdGeomPointInstancer placing 40,000 buildings of eight prototypes
// over Mediterranean hills, in centimetres (USD's fallback), under
// the sun. The asset (0.5 MB from openusd.org, under Pixar's license
// in its LICENSE.txt) is fetched once into the cache.

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/usd_import.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr const char* kUrl =
    "https://openusd.org/files/PointInstancedMedCity.zip";

/// A view: name, eye and target, metres.
struct View {
    const char* name;
    glm::dvec3 eye;
    glm::dvec3 target;
};

const std::array<View, 4> kViews = {{
    {"Over the hills", {0.9, -2.4, 1.4}, {0.9, 0.15, 0.05}},
    {"Rooftops", {0.55, -0.45, 0.42}, {1.0, 0.2, 0.15}},
    {"Along the hillside", {-0.35, 0.15, 0.3}, {0.9, 0.2, 0.12}},
    {"Overhead", {0.89, 0.15, 3.2}, {0.89, 0.16, 0.0}},
}};

constexpr const char* kBuildings = "/MediterraneanHills/Buildings";
constexpr std::size_t kInstances = 40000;

std::filesystem::path AssetDirectory() {
    return demos::CacheDir() / "usd" / "med_city" /
           "PointInstancedMedCity";
}

/// Downloads and unpacks the asset into the cache (checking each
/// file's CRC-32), through a directory renamed when complete.
void FetchAsset(
    const std::function<void(const std::string&)>& report) {
    const std::filesystem::path root = demos::CacheDir() / "usd";
    const std::filesystem::path archive =
        root / "PointInstancedMedCity.zip";
    report(std::string("Fetching ") + kUrl);
    demos::Download(kUrl, archive, report);
    const std::filesystem::path partial = root / "med_city.part";
    std::filesystem::remove_all(partial);
    demos::ExtractZip(archive, partial);
    std::filesystem::rename(partial, root / "med_city");
    std::filesystem::remove(archive);
}

class MedCityDemo : public app::Demo {
  public:
    MedCityDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        DistantLight::Define(scene, "/World/Sun").SetAngle(0.53);
        PlaceSun();
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetClippingRange({0.002, 20.0})
            .SetSensorNoise(false)
            .SetExposureMode(std::string(tokens::kAuto));
        // A little blue fill for the shaded sides.
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({960, 600})
            .SetAmbient(0.08)
            .SetAmbientColor({0.55, 0.7, 1.0});

        std::string why;
        if (!UsdImportAvailable(&why)) {
            Report(why, true);
            return;
        }
        if (!std::filesystem::is_directory(AssetDirectory()) &&
            demos::CanDownload()) {
            try {
                FetchAsset([this](const std::string& text) {
                    Report(text, false);
                });
            } catch (const std::exception& error) {
                Report(error.what(), true);
            }
        }
        if (!std::filesystem::is_directory(AssetDirectory())) {
            Report("The Med City is not cached and cannot be fetched",
                   true);
            return;
        }
        result_ =
            ImportUsd(scene, Path("/World/City"),
                      AssetDirectory() / "PointInstancedMedCity.usd");
        for (const std::string& warning : result_->warnings) {
            Report(warning, true);
        }
        Report("Imported the Med City: " +
                   std::to_string(result_->prims.size()) + " prims",
               false);
        Show(0);

        std::vector<std::string> names;
        for (const View& view : kViews) {
            names.emplace_back(view.name);
        }
        Widgets().Combo(
            "View", std::move(names), 0,
            [this](std::size_t i) { Show(static_cast<int>(i)); });
        Widgets().Slider("Sun azimuth (deg)", -180.0, 180.0, azimuth_,
                         [this](double degrees) {
                             azimuth_ = degrees;
                             PlaceSun();
                         });
        Widgets().Slider("Sun elevation (deg)", 2.0, 85.0, elevation_,
                         [this](double degrees) {
                             elevation_ = degrees;
                             PlaceSun();
                         });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    /// The views side by side.
    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, demos::SideBySide(RenderAll()));
    }

    void TestFinal() override {
        if (!result_) {
            std::printf("Skipped: no USD importer or no Med City\n");
            return;
        }
        // One instancer of every building, its prototypes shared.
        const Scene& scene = GetScene();
        const Path buildings = result_->prims.at(kBuildings);
        Check(scene.TypeOf(buildings) == tokens::kPointInstancer,
              "the buildings should be a point instancer");
        const Tensor positions =
            scene.Read(buildings, tokens::kPositions)->Contiguous();
        const auto count =
            static_cast<std::size_t>(positions.Elements() / 3);
        std::size_t meshes = 0;
        for (const auto& [source, path] : result_->prims) {
            meshes += scene.Has(path) &&
                      scene.TypeOf(path) == tokens::kMesh;
        }
        std::printf("%zu instances, %zu meshes\n", count, meshes);
        Check(count == kInstances, "every building should be placed");
        Check(meshes == 8, "prototypes should be shared");
        // Centimetres into metres: the hills about 2 m across.
        const glm::dmat4 world = scene.WorldTransform(buildings);
        const auto* xyz = static_cast<const float*>(positions.Data());
        glm::dvec3 low(std::numeric_limits<double>::infinity());
        glm::dvec3 high(-std::numeric_limits<double>::infinity());
        for (std::size_t i = 0; i < count; ++i) {
            const glm::dvec3 at(world *
                                glm::dvec4(xyz[3 * i], xyz[3 * i + 1],
                                           xyz[3 * i + 2], 1.0));
            low = glm::min(low, at);
            high = glm::max(high, at);
        }
        std::printf("city spans %.2f x %.2f m\n", high.x - low.x,
                    high.y - low.y);
        Check(high.x - low.x > 1.5 && high.x - low.x < 3.0,
              "the city should be in metres");
        std::printf("mean of each view:");
        double least = 255.0;
        for (const Tensor& image : RenderAll()) {
            const double mean = MeanRgb(image);
            least = std::min(least, mean);
            std::printf(" %.1f", mean);
        }
        std::printf("\n");
        Check(least > 20.0, "every view should show the lit city");
    }

  private:
    void Report(const std::string& text, bool error) {
        if (app::Viewer* viewer = GetViewer()) {
            viewer->Log(error ? app::LogLevel::Error
                              : app::LogLevel::Info,
                        text);
        } else {
            std::printf("%s\n", text.c_str());
        }
    }

    /// Points the sun from its azimuth and elevation.
    void PlaceSun() {
        const double a = glm::radians(azimuth_);
        const double e = glm::radians(elevation_);
        const glm::dvec3 toward(std::cos(e) * std::cos(a),
                                std::cos(e) * std::sin(a),
                                std::sin(e));
        DistantLight(GetScene(), "/World/Sun")
            .SetTransform(LookAtTransform(toward, {0.0, 0.0, 0.0},
                                          {0.0, 0.0, 1.0}));
    }

    void Show(int index) {
        view_ = index;
        const View& view = kViews[static_cast<std::size_t>(index)];
        Camera(GetScene(), "/World/Camera")
            .SetTransform(LookAtTransform(view.eye, view.target,
                                          {0.0, 0.0, 1.0}));
    }

    /// The scene from each view, RGBA uint8; leaves the view as it
    /// was.
    std::vector<Tensor> RenderAll() {
        const int current = view_;
        std::vector<Tensor> images;
        for (int i = 0; i < static_cast<int>(kViews.size()); ++i) {
            Show(i);
            images.push_back(GetRenderer().Render(GetScene()));
        }
        Show(current);
        return images;
    }

    /// The mean of an RGBA uint8 image's colour channels.
    static double MeanRgb(const Tensor& image) {
        const Tensor pixels = image.Contiguous();
        const auto* bytes =
            static_cast<const std::uint8_t*>(pixels.Data());
        const auto count =
            static_cast<std::size_t>(pixels.Elements() / 4);
        double sum = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            sum += bytes[4 * i] + bytes[4 * i + 1] + bytes[4 * i + 2];
        }
        return sum / (3.0 * static_cast<double>(count));
    }

    std::optional<UsdImportResult> result_;
    double azimuth_ = -60.0;
    double elevation_ = 35.0;
    int view_ = 0;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    MedCityDemo, "usd", "med_city",
    "Pixar's Point Instanced Med City: 40,000 buildings "
    "of one USD point instancer.");
