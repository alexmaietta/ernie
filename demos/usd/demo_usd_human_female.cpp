// SPDX-License-Identifier: BSD-3-Clause
// Pixar's HumanFemale from the UsdSkelExamples, imported with
// ImportUsd: a skinned character whose skeleton, animation and blend
// shapes Ernie poses on the GPU at the timeline's time code, walking
// or idling, at three levels of detail. The examples (38 MB from
// openusd.org, under Pixar's license in HumanFemale/LICENSE.txt) are
// fetched once into the cache.

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
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
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr const char* kUrl =
    "https://openusd.org/files/UsdSkelExamples.zip";
/// Each clip: its name and file.
constexpr std::array<std::pair<const char*, const char*>, 2> kClips{{
    {"Walk", "HumanFemale.walk.usd"},
    {"Keep alive", "HumanFemale.keepAlive.usd"},
}};
/// The asset's geometricLOD variants.
constexpr std::array<const char*, 3> kLods{"full", "medium", "low"};
constexpr const char* kHuman = "/World/Human";
constexpr const char* kGroup =
    "/HumanFemale_Group";  ///< Its variant sets'.

std::filesystem::path AssetDirectory() {
    return demos::CacheDir() / "usd" / "skel_examples" /
           "UsdSkelExamples" / "HumanFemale";
}

/// Downloads and unpacks the examples into the cache (checking each
/// file's CRC-32), through a directory renamed when complete.
void FetchExamples(
    const std::function<void(const std::string&)>& report) {
    const std::filesystem::path root = demos::CacheDir() / "usd";
    const std::filesystem::path archive =
        root / "UsdSkelExamples.zip";
    report(std::string("Fetching ") + kUrl);
    demos::Download(kUrl, archive, report);
    const std::filesystem::path partial = root / "skel_examples.part";
    std::filesystem::remove_all(partial);
    demos::ExtractZip(archive, partial);
    std::filesystem::rename(partial, root / "skel_examples");
    std::filesystem::remove(archive);
}

class HumanFemaleDemo : public app::Demo {
  public:
    HumanFemaleDemo(app::Viewer* viewer,
                    const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        DistantLight::Define(scene, "/World/Sun")
            .SetAngle(0.53)
            .SetTransform(LookAtTransform(
                {1.0, -2.0, 2.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}));
        Material::Define(scene, "/Looks/Floor")
            .SetAlbedo(glm::dvec3(0.6));
        GroundPlane::Define(scene, "/World/Floor")
            .SetHeight(-0.04)
            .SetChecker(true)
            .SetCheckerSize(0.5)
            .SetMaterialBinding(Path("/Looks/Floor"));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetClippingRange({0.01, 50.0})
            .SetSensorNoise(false)
            .SetExposureMode(std::string(tokens::kAuto))
            .SetTransform(LookAtTransform({2.8, -1.9, 0.9},
                                          {0.0, -0.3, 0.65},
                                          {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({600, 800})
            .SetAmbient(0.15);

        std::string why;
        if (!UsdImportAvailable(&why)) {
            Report(why, true);
            return;
        }
        if (!std::filesystem::is_directory(AssetDirectory()) &&
            demos::CanDownload()) {
            try {
                FetchExamples([this](const std::string& text) {
                    Report(text, false);
                });
            } catch (const std::exception& error) {
                Report(error.what(), true);
            }
        }
        if (!std::filesystem::is_directory(AssetDirectory())) {
            Report("The UsdSkelExamples are not cached and cannot be "
                   "fetched",
                   true);
            return;
        }
        Load();

        std::vector<std::string> clips;
        for (const auto& [name, file] : kClips) {
            clips.emplace_back(name);
        }
        Widgets().Combo("Clip", std::move(clips), 0,
                        [this](std::size_t i) {
                            clip_ = i;
                            Load();
                        });
        Widgets().Combo("Level of detail",
                        {kLods.begin(), kLods.end()}, 0,
                        [this](std::size_t i) {
                            lod_ = i;
                            Load();
                        });
        // Dual quaternions keep volume where joints twist (wrists,
        // elbows).
        Widgets().Combo(
            "Skinning", {"Linear", "Dual quaternion"}, 0,
            [this](std::size_t i) { ChooseMethod(i == 1); });
        Widgets().Combo("Subdivision level", {"0", "1", "2"}, 1,
                        [this](std::size_t level) {
                            Stage(GetScene(), Path::Root())
                                .SetSubdivisionRefineLevel(
                                    static_cast<int>(level));
                        });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            viewer->GetTimeline().Play();
        }
    }

    /// Four moments of the clip side by side.
    void SaveOutput(const std::filesystem::path& path) override {
        const Stage stage(GetScene(), Path::Root());
        std::vector<double> times;
        for (int i = 0; i < 4; ++i) {
            times.push_back(
                stage.GetStartTimeCode() +
                (stage.GetEndTimeCode() - stage.GetStartTimeCode()) *
                    i / 3.0);
        }
        app::WritePng(path, demos::SideBySide(RenderAt(times)));
    }

    void TestFinal() override {
        if (!result_) {
            std::printf(
                "Skipped: no USD importer or no UsdSkelExamples\n");
            return;
        }
        const auto count = [&](std::string_view type) {
            std::size_t n = 0;
            for (const auto& [source, path] : result_->prims) {
                n += GetScene().Has(path) &&
                     GetScene().TypeOf(path) == type;
            }
            return n;
        };
        const Stage stage(GetScene(), Path::Root());
        std::printf("%zu skinned meshes, %zu skeleton, %zu "
                    "animation, %zu blend "
                    "shapes, time codes %g to %g\n",
                    Skinned().size(), count(tokens::kSkeleton),
                    count(tokens::kSkelAnimation),
                    count(tokens::kBlendShape),
                    stage.GetStartTimeCode(), stage.GetEndTimeCode());
        Check(count(tokens::kSkeleton) == 1 &&
                  count(tokens::kSkelAnimation) == 1,
              "the skeleton and its animation should be imported");
        Check(Skinned().size() == 89,
              "every mesh of the full level should be skinned");
        Check(stage.GetStartTimeCode() == 101.0 &&
                  stage.GetEndTimeCode() == 129.0,
              "the walk's range should be the timeline's");
        // She moves: the walk's start and middle differ, both lit.
        const std::vector<Tensor> images = RenderAt({101.0, 115.0});
        const double moved = Changed(images[0], images[1]);
        std::printf(
            "pixels changed mid-walk: %.3f; means %.1f, %.1f\n",
            moved, MeanRgb(images[0]), MeanRgb(images[1]));
        Check(moved > 0.01, "the walk should move her");
        Check(std::min(MeanRgb(images[0]), MeanRgb(images[1])) > 5.0,
              "she should be lit");
        // Dual quaternions pose her too.
        ChooseMethod(true);
        Check(MeanRgb(RenderAt({115.0}).front()) > 5.0,
              "dual quaternion skinning should draw her");
        // The idle clip's own range; the low level's fewer meshes.
        clip_ = 1;
        lod_ = 2;
        Load();
        std::printf("keep alive, low: %zu meshes, to %g\n",
                    Skinned().size(), stage.GetEndTimeCode());
        Check(stage.GetEndTimeCode() == 250.0,
              "the idle clip's range should be taken");
        Check(Skinned().size() == 22,
              "the low level should have fewer meshes");
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

    /// Imports the clip at the level of detail, its range the
    /// timeline's.
    void Load() {
        Scene& scene = GetScene();
        scene.Remove(Path(kHuman));
        UsdImportOptions options;
        options.variants.push_back(
            {kGroup, "geometricLOD", kLods[lod_]});
        const std::string name = kClips[clip_].second;
        result_ = ImportUsd(scene, Path(kHuman),
                            AssetDirectory() / name, options);
        for (const std::string& warning : result_->warnings) {
            Report(warning, true);
        }
        if (result_->timeCodeRange) {
            Stage(scene, Path::Root())
                .SetStartTimeCode(result_->timeCodeRange->x)
                .SetEndTimeCode(result_->timeCodeRange->y)
                .SetTimeCodesPerSecond(result_->timeCodesPerSecond)
                .SetFramesPerSecond(result_->framesPerSecond)
                .SetTimeCode(result_->timeCodeRange->x);
        }
        ChooseMethod(dualQuaternion_);
        Report("Imported " + name + " (" + kLods[lod_] +
                   "): " + std::to_string(Skinned().size()) +
                   " skinned meshes",
               false);
    }

    std::vector<Path> Skinned() {
        std::vector<Path> meshes;
        const Scene& scene = GetScene();
        for (const auto& [source, path] : result_->prims) {
            if (scene.Has(path) &&
                scene.TypeOf(path) == tokens::kMesh &&
                !Mesh(scene, path).GetSkeleton().IsEmpty()) {
                meshes.push_back(path);
            }
        }
        return meshes;
    }

    void ChooseMethod(bool dualQuaternion) {
        dualQuaternion_ = dualQuaternion;
        for (const Path& path : Skinned()) {
            Mesh(GetScene(), path)
                .SetSkinningMethod(std::string(
                    dualQuaternion ? tokens::kDualQuaternion
                                   : tokens::kClassicLinear));
        }
    }

    /// The scene at each time code, RGBA uint8; leaves the time as it
    /// was.
    std::vector<Tensor> RenderAt(const std::vector<double>& times) {
        Stage stage(GetScene(), Path::Root());
        const double current = stage.GetTimeCode();
        std::vector<Tensor> images;
        for (const double t : times) {
            stage.SetTimeCode(t);
            images.push_back(GetRenderer().Render(GetScene()));
        }
        stage.SetTimeCode(current);
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

    /// The share of pixels whose colour changed by more than 16
    /// levels.
    static double Changed(const Tensor& a, const Tensor& b) {
        const Tensor pa = a.Contiguous();
        const Tensor pb = b.Contiguous();
        const auto* x = static_cast<const std::uint8_t*>(pa.Data());
        const auto* y = static_cast<const std::uint8_t*>(pb.Data());
        const auto count =
            static_cast<std::size_t>(pa.Elements() / 4);
        std::size_t changed = 0;
        for (std::size_t i = 0; i < count; ++i) {
            int most = 0;
            for (std::size_t c = 0; c < 3; ++c) {
                most = std::max(most, std::abs(int(x[4 * i + c]) -
                                               int(y[4 * i + c])));
            }
            changed += most > 16 ? 1 : 0;
        }
        return static_cast<double>(changed) /
               static_cast<double>(count);
    }

    std::optional<UsdImportResult> result_;
    std::size_t clip_ = 0;
    std::size_t lod_ = 0;
    bool dualQuaternion_ = false;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    HumanFemaleDemo, "usd", "human_female",
    "Pixar's HumanFemale: a UsdSkel character skinned on the "
    "GPU, walking or idling on the timeline.");
