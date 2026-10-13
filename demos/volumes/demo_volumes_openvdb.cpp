// SPDX-License-Identifier: BSD-3-Clause
// OpenVDB's sample volumes side by side, read from their OpenVDB
// files
// (.vdb) as they are: Ernie reads each with its built-in OpenVDB and
// walks the grids on the GPU as NanoVDB. Smoke and a second smoke
// column scatter the sun; fire, and the explosion when switched on,
// glow as black bodies at their temperature grids. The samples (60
// MB, the explosion 76 MB more) are fetched once from openvdb.org
// into the cache.

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
#include "util.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr const char* kUrl =
    "https://media.githubusercontent.com/media/"
    "AcademySoftwareFoundation/openvdb-website/master/download/"
    "models/";
const Path kSamples("/World/Samples");

/// A sample: its file, its active bounds in its own units (Y up), the
/// height it stands in metres, where it stands, and its material.
struct Sample {
    const char* name;
    glm::dvec3 low;
    glm::dvec3 high;
    double height;
    glm::dvec2 at;
    double density;
    double albedo;
    double anisotropy;
    /// Kelvin a unit of its temperature grid, or 0 without one.
    double kelvin;
    bool optional;  ///< Fetched when switched on.
};

constexpr std::array<Sample, 4> kSampleTable{{
    {"smoke",
     {-33.6, -12.1, -26.0},
     {18.8, 93.1, 26.9},
     4.0,
     {-4.5, 0.0},
     2.5,
     0.9,
     0.3,
     0.0,
     false},
    // Its temperature grid peaks near 45: 40 K a unit.
    {"fire",
     {-19.1, -8.2, -18.7},
     {19.9, 80.9, 18.4},
     3.0,
     {0.0, 0.0},
     5.0,
     0.3,
     0.0,
     40.0,
     false},
    {"smoke2",
     {-6.9, -1.2, -7.9},
     {12.1, 59.7, 9.8},
     5.0,
     {4.5, 0.0},
     30.0,
     0.95,
     0.3,
     0.0,
     false},
    // 76 MB, behind the others. Its temperature grid peaks near 0.9:
    // 3000 K a unit.
    {"explosion",
     {-85.5, -35.5, -108.5},
     {114.5, 235.5, 120.5},
     6.0,
     {0.0, 9.0},
     0.8,
     0.4,
     0.0,
     3000.0,
     true},
}};

std::filesystem::path SampleFile(const std::string& name) {
    return demos::CacheDir() / "openvdb" / (name + ".vdb");
}

class OpenVdbDemo : public app::Demo {
  public:
    OpenVdbDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Xform::Define(scene, kSamples);
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetShadowEnable(true)
            .SetTransform(DistantLightTransform({-0.6, -1.0, 0.7}));
        Material::Define(scene, "/World/Looks/Ground")
            .SetAlbedo(glm::dvec3(0.06));
        GroundPlane::Define(scene, "/World/Ground")
            .SetChecker(true)
            .SetCheckerSize(1.0)
            .SetMaterialBinding(Path("/World/Looks/Ground"));
        // Daylight exposure: auto exposure would meter on the fire.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(28.0)
            .SetSensorNoise(false)
            .SetExposureIso(100.0)
            .SetExposureFStop(8.0)
            .SetExposureTime(1.0 / 250.0)
            .SetTransform(LookAtTransform(
                {0.5, -11.0, 2.4}, {0.5, 0.0, 2.1}, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({960, 400})
            .SetAmbient(0.1);
        for (const Sample& sample : kSampleTable) {
            if (!sample.optional) {
                Show(sample);
            }
        }

        Widgets().Checkbox("Explosion (76 MB)", false,
                           [this](bool on) { SetExplosion(on); });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, GetRenderer().Render(GetScene()));
    }

    void TestFinal() override {
        if (shown_.empty()) {
            std::printf("Skipped: no OpenVDB samples\n");
            return;
        }
        Check(OpenVdbAvailable(), "Ernie should read OpenVDB");
        Scene& scene = GetScene();
        GetRenderer().Render(scene);
        const Tensor full = GetRenderer().Render(scene);
        for (const std::string& name : shown_) {
            const Path path = kSamples.Child(name);
            scene.Set(path, tokens::kVisibility,
                      std::string(tokens::kInvisible));
            const Tensor without = GetRenderer().Render(scene);
            scene.Set(path, tokens::kVisibility,
                      std::string(tokens::kInherited));
            const double covers = Changed(full, without);
            std::printf("%s: covers %.3f of the view\n", name.c_str(),
                        covers);
            Check(covers > 0.005, name + " should show");
        }
        // In the dark the fire still glows at its temperature.
        scene.Set("/World/Sun", tokens::kVisibility,
                  std::string(tokens::kInvisible));
        RenderSettings(scene, "/Render/Settings").SetAmbient(0.0);
        const double dark = MeanRgb(GetRenderer().Render(scene));
        scene.Set(kSamples.Child("fire"), tokens::kVisibility,
                  std::string(tokens::kInvisible));
        const double unlit = MeanRgb(GetRenderer().Render(scene));
        std::printf(
            "in the dark: mean %.2f with fire, %.2f without\n", dark,
            unlit);
        Check(dark > unlit + 0.05, "the fire should glow");
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

    /// Fetches the sample once and stands it on the ground.
    void Show(const Sample& sample) {
        const std::string name = sample.name;
        const std::filesystem::path file = SampleFile(name);
        const std::string url = kUrl + name + ".vdb";
        if (!std::filesystem::is_regular_file(file) &&
            demos::CanDownload()) {
            try {
                Report("Fetching " + url, false);
                demos::Download(url, file,
                                [this](const std::string& text) {
                                    Report(text, false);
                                });
            } catch (const std::exception& error) {
                Report(error.what(), true);
            }
        }
        if (!std::filesystem::is_regular_file(file)) {
            Report(name + ".vdb is not cached and cannot be fetched",
                   true);
            return;
        }
        Scene& scene = GetScene();
        const double scale =
            sample.height / (sample.high.y - sample.low.y);
        const Path look("/World/Looks/" + name);
        Material material = Material::Define(scene, look);
        material.SetDensity(sample.density)
            .SetVolumeAlbedo(glm::dvec3(sample.albedo))
            .SetAnisotropy(sample.anisotropy);
        if (sample.kelvin > 0.0) {
            material.SetTemperatureField("temperature")
                .SetTemperatureScale(sample.kelvin);
        }
        // Y up to Z up (a quarter turn about x takes y to z and z to
        // -y), centred on x, its foot on the ground.
        const Path path = kSamples.Child(name);
        Volume volume = Volume::Define(scene, path);
        volume.SetMaterialBinding(look);
        SetXform(
            scene, path,
            {.translate = glm::dvec3(
                 sample.at.x -
                     0.5 * (sample.low.x + sample.high.x) * scale,
                 sample.at.y +
                     0.5 * (sample.low.z + sample.high.z) * scale,
                 -sample.low.y * scale),
             .rotate = glm::dvec3(90.0, 0.0, 0.0),
             .scale = glm::dvec3(scale)});
        std::vector<std::string> fields{"density"};
        if (sample.kelvin > 0.0) {
            fields.emplace_back("temperature");
        }
        for (const std::string& field : fields) {
            std::string child = field;
            child.front() = static_cast<char>(std::toupper(
                static_cast<unsigned char>(child.front())));
            const Path fieldPath = path.Child(child);
            OpenVDBAsset::Define(scene, fieldPath)
                .SetFilePath(file.string())
                .SetFieldName(field);
            volume.SetField(field, fieldPath);
        }
        shown_.push_back(name);
    }

    void SetExplosion(bool on) {
        if (on) {
            Show(kSampleTable.back());
        } else {
            GetScene().Remove(kSamples.Child("explosion"));
            std::erase(shown_, std::string("explosion"));
        }
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
                most = std::max(
                    most, std::abs(static_cast<int>(x[4 * i + c]) -
                                   static_cast<int>(y[4 * i + c])));
            }
            changed += most > 16;
        }
        return static_cast<double>(changed) /
               static_cast<double>(count);
    }

    std::vector<std::string> shown_;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    OpenVdbDemo, "volumes", "openvdb",
    "OpenVDB's sample volumes read from their .vdb files: smoke "
    "scattering the sun, fire and an explosion glowing at their "
    "temperatures.");
