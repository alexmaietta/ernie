// SPDX-License-Identifier: BSD-3-Clause
// Pixar's Kitchen Set, the OpenUSD reference scene, imported with
// ImportUsd: hundreds of assets composed through references and
// payloads, plain or with USD's native instancing, in centimetres,
// under two ceiling lights. The set (2.7 MB from openusd.org, under
// Pixar's license in its LICENSE.txt) is fetched once into the cache.

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/stage_schemas.h"
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
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr const char* kUrl =
    "https://openusd.org/files/Kitchen_set.zip";

/// A view: name, eye and target (metres), or none from overhead with
/// the ceiling cut away.
struct View {
    const char* name;
    std::optional<std::pair<glm::dvec3, glm::dvec3>> eyeTarget;
};

const std::array<View, 4> kViews = {{
    {"Into the kitchen", {{{3.3, -2.3, 1.7}, {0.0, 0.4, 0.8}}}},
    {"Stove and fridge", {{{0.0, -1.6, 1.4}, {2.5, 1.0, 0.9}}}},
    {"Breakfast table", {{{-1.0, 0.8, 1.5}, {2.0, -1.0, 0.8}}}},
    {"Cutaway", std::nullopt},
}};

constexpr const char* kCeiling =
    "/Kitchen_set/Arch_grp/Kitchen_1/Geom/Ceiling";
constexpr double kCeilingHeight = 2.54;  ///< Metres.
constexpr double kCutawayHeight =
    9.0;  ///< Metres, the overhead camera's.
/// The ceiling lights' x and y, metres.
constexpr std::array<glm::dvec2, 2> kLights = {
    {{-0.3, -0.6}, {2.0, -0.6}}};

std::filesystem::path SetDirectory() {
    return demos::CacheDir() / "usd" / "kitchen_set" / "Kitchen_set";
}

/// Downloads and unpacks the set into the cache (checking each file's
/// CRC-32), through a directory renamed when complete.
void FetchSet(const std::function<void(const std::string&)>& report) {
    const std::filesystem::path root = demos::CacheDir() / "usd";
    const std::filesystem::path archive = root / "Kitchen_set.zip";
    report(std::string("Fetching ") + kUrl);
    demos::Download(kUrl, archive, report);
    const std::filesystem::path partial = root / "kitchen_set.part";
    std::filesystem::remove_all(partial);
    demos::ExtractZip(archive, partial);
    std::filesystem::rename(partial, root / "kitchen_set");
    std::filesystem::remove(archive);
}

class KitchenDemo : public app::Demo {
  public:
    KitchenDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // Two lights just under the ceiling, facing down.
        for (std::size_t i = 0; i < kLights.size(); ++i) {
            const glm::dvec2 at = kLights[i];
            RectLight::Define(scene, LightPath(i))
                .SetWidth(1.2)
                .SetHeight(0.6)
                .SetPhotometricPower(4000.0)
                .SetShadowEnable(true)
                .SetTransform(LookAtTransform(
                    {at.x, at.y, kCeilingHeight - 0.09},
                    {at.x, at.y, 0.0}, {0.0, 1.0, 0.0}));
        }
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(18.0)
            .SetClippingRange({0.02, 50.0})
            .SetSensorNoise(false)
            .SetExposureMode(std::string(tokens::kAuto));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({960, 600})
            .SetAmbient(0.0);

        std::string why;
        if (!UsdImportAvailable(&why)) {
            Report(why, true);
            return;
        }
        if (!std::filesystem::is_directory(SetDirectory()) &&
            demos::CanDownload()) {
            try {
                FetchSet([this](const std::string& text) {
                    Report(text, false);
                });
            } catch (const std::exception& error) {
                Report(error.what(), true);
            }
        }
        if (!std::filesystem::is_directory(SetDirectory())) {
            Report(
                "The Kitchen Set is not cached and cannot be fetched",
                true);
            return;
        }
        Load();
        Show(0);

        std::vector<std::string> names;
        for (const View& view : kViews) {
            names.emplace_back(view.name);
        }
        Widgets().Combo(
            "View", std::move(names), 0,
            [this](std::size_t i) { Show(static_cast<int>(i)); });
        // The instanced set's native instances as instancers of
        // shared prototypes, or the plain set's copies.
        Widgets().Checkbox("Instancers", false,
                           [this](bool on) { Instancing(on); });
        // The set's meshes are Catmull-Clark surfaces (USD's
        // fallback), refined by OpenSubdiv, or their control meshes
        // drawn as they are; 2^level segments along each control
        // edge.
        Widgets().Checkbox(
            "Subdivision surfaces", true,
            [this](bool on) { SubdivisionSurfaces(on); });
        Widgets().Combo("Subdivision level", {"0", "1", "2", "3"}, 1,
                        [this](std::size_t level) {
                            Stage(GetScene(), Path::Root())
                                .SetSubdivisionRefineLevel(
                                    static_cast<int>(level));
                        });
        Widgets().Slider("Light power (lm)", 500.0, 20000.0, 4000.0,
                         [this](double lumens) { Power(lumens); });
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
            std::printf(
                "Skipped: no USD importer or no Kitchen Set\n");
            return;
        }
        // Centimetres into metres: the ceiling at 2.54 m.
        const Scene& scene = GetScene();
        const Path ceiling =
            result_->prims.at(std::string(kCeiling) + "/pPlane5");
        const Tensor points =
            scene.Read(ceiling, tokens::kPoints)->Contiguous();
        const auto* xyz = static_cast<const float*>(points.Data());
        const double height =
            (scene.WorldTransform(ceiling) *
             glm::dvec4(xyz[0], xyz[1], xyz[2], 1.0))
                .z;
        std::printf("ceiling at %.3f m\n", height);
        Check(std::abs(height - kCeilingHeight) < 0.01,
              "the set should be in metres");
        // Native instances become instancers sharing their
        // prototypes, so the instanced set imports fewer meshes than
        // the plain one.
        const auto count = [&](std::string_view type) {
            std::size_t n = 0;
            for (const auto& [source, path] : result_->prims) {
                n += GetScene().Has(path) &&
                     GetScene().TypeOf(path) == type;
            }
            return n;
        };
        const std::size_t plain = count("Mesh");
        Instancing(true);
        std::printf(
            "meshes: %zu plain, %zu instanced, in %zu instancers\n",
            plain, count("Mesh"), count("PointInstancer"));
        Check(count("PointInstancer") > 0,
              "instances should be instanced");
        Check(count("Mesh") < plain, "prototypes should be shared");
        std::printf("mean of each view:");
        double least = 255.0;
        for (const Tensor& image : RenderAll()) {
            const double mean = MeanRgb(image);
            least = std::min(least, mean);
            std::printf(" %.1f", mean);
        }
        std::printf("\n");
        Check(least > 5.0, "every view should be lit");
        // Without subdivision surfaces, the control meshes as they
        // are.
        SubdivisionSurfaces(false);
        for (const auto& [path, scheme] : schemes_) {
            Check(Mesh(GetScene(), path).GetSubdivisionScheme() ==
                      tokens::kNone,
                  "every mesh should draw its control mesh");
        }
        const double cage = MeanRgb(RenderAll().front());
        std::printf("control meshes: mean %.1f\n", cage);
        Check(cage > 5.0, "the control meshes should be lit");
    }

  private:
    static Path LightPath(std::size_t i) {
        return Path("/World/Light" + std::to_string(i));
    }

    void Report(const std::string& text, bool error) {
        if (app::Viewer* viewer = GetViewer()) {
            viewer->Log(error ? app::LogLevel::Error
                              : app::LogLevel::Info,
                        text);
        } else {
            std::printf("%s\n", text.c_str());
        }
    }

    /// Imports the set, natively instanced or not.
    void Load() {
        Scene& scene = GetScene();
        scene.Remove(Path("/World/Kitchen"));
        const std::string name = instanced_
                                     ? "Kitchen_set_instanced.usd"
                                     : "Kitchen_set.usd";
        result_ = ImportUsd(scene, Path("/World/Kitchen"),
                            SetDirectory() / name);
        for (const std::string& warning : result_->warnings) {
            Report(warning, true);
        }
        Report("Imported " + name + ": " +
                   std::to_string(result_->prims.size()) + " prims",
               false);
        schemes_.clear();
        for (const auto& [source, path] : result_->prims) {
            if (scene.Has(path) &&
                scene.TypeOf(path) == tokens::kMesh) {
                schemes_.emplace_back(
                    path, Mesh(scene, path).GetSubdivisionScheme());
            }
        }
        SubdivisionSurfaces(subdivided_);
    }

    void Instancing(bool on) {
        instanced_ = on;
        Load();
        Show(view_);
    }

    void SubdivisionSurfaces(bool on) {
        subdivided_ = on;
        for (const auto& [path, scheme] : schemes_) {
            Mesh(GetScene(), path)
                .SetSubdivisionScheme(
                    on ? scheme : std::string(tokens::kNone));
        }
    }

    void Power(double lumens) {
        for (std::size_t i = 0; i < kLights.size(); ++i) {
            RectLight(GetScene(), LightPath(i))
                .SetPhotometricPower(lumens);
        }
    }

    void Show(int index) {
        view_ = index;
        const View& view = kViews[static_cast<std::size_t>(index)];
        // The cutaway clips the ceiling with the near plane, whether
        // the ceiling is drawn as itself or as instances.
        Camera camera(GetScene(), "/World/Camera");
        if (view.eyeTarget) {
            camera.SetClippingRange({0.02, 50.0})
                .SetTransform(LookAtTransform(view.eyeTarget->first,
                                              view.eyeTarget->second,
                                              {0.0, 0.0, 1.0}));
        } else {
            camera
                .SetClippingRange(
                    {kCutawayHeight - kCeilingHeight + 0.05, 50.0})
                .SetTransform(LookAtTransform(
                    {0.7, -0.3, kCutawayHeight}, {0.7, -0.29, 0.0},
                    {0.0, 1.0, 0.0}));
        }
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
    /// Each imported mesh's subdivision scheme, to restore.
    std::vector<std::pair<Path, std::string>> schemes_;
    bool instanced_ = false;
    bool subdivided_ = true;
    int view_ = 0;
};

}  // namespace

ERNIE_REGISTER_DEMO(KitchenDemo, "usd", "kitchen",
                    "Pixar's Kitchen Set imported from USD, plain or "
                    "natively instanced.");
