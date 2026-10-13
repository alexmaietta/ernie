// SPDX-License-Identifier: BSD-3-Clause
// OpenVDB's smoke sample in a USD shot, imported with ImportUsd: a
// UsdVol Volume reading the OpenVDB file through Ernie's built-in
// OpenVDB, an ember light linked to light the smoke alone with its
// intensity sampled over time, a sun whose shadows leave out a post,
// and the shot's camera and render settings. The shot is written in
// code beside the sample (2.6 MB from openvdb.org, fetched once into
// the cache), as .usda or packed with it into a .usdz.

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/stage_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/usd_import.h"
#include "ernie/scene/volume_schemas.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr const char* kUrl =
    "https://media.githubusercontent.com/media/"
    "AcademySoftwareFoundation/openvdb-website/master/download/"
    "models/"
    "smoke.vdb";
constexpr const char* kShot = "/World/Shot";
constexpr std::array<const char*, 2> kFormats{"usda", "usdz"};

// The smoke spans about -34 to 19 by -12 to 93 by -26 to 27 of the
// sample's units, Y up; at 5 cm a unit it stands 5 m tall.
constexpr std::string_view kLayer = R"(#usda 1.0
(
    defaultPrim = "Shot"
    upAxis = "Y"
    metersPerUnit = 0.05
    startTimeCode = 1
    endTimeCode = 48
    timeCodesPerSecond = 24
)

def Xform "Shot"
{
    def Volume "Smoke"
    {
        rel field:density = </Shot/Smoke/Density>

        def OpenVDBAsset "Density"
        {
            asset filePath = @smoke.vdb@
            token fieldName = "density"
        }
    }

    def Cube "Ground"
    {
        double size = 1
        color3f[] primvars:displayColor = [(0.45, 0.42, 0.38)]
        double3 xformOp:translate = (0, -14, 0)
        float3 xformOp:scale = (400, 2, 400)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }

    def Cube "Post"
    {
        double size = 1
        color3f[] primvars:displayColor = [(0.3, 0.32, 0.36)]
        double3 xformOp:translate = (48, 17, 10)
        float3 xformOp:scale = (8, 62, 8)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }

    def DistantLight "Sun" (
        prepend apiSchemas = ["CollectionAPI:shadowLink"]
    )
    {
        float inputs:intensity = 6
        float inputs:angle = 0.53
        rel collection:shadowLink:excludes = </Shot/Post>
        float3 xformOp:rotateXYZ = (-35, 40, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }

    def SphereLight "Ember" (
        prepend apiSchemas = ["CollectionAPI:lightLink"]
    )
    {
        float inputs:intensity.timeSamples = {
            1: 60,
            12: 220,
            24: 80,
            36: 260,
            48: 60,
        }
        color3f inputs:color = (1, 0.45, 0.15)
        float inputs:radius = 4
        uniform bool collection:lightLink:includeRoot = 0
        rel collection:lightLink:includes = </Shot/Smoke>
        double3 xformOp:translate = (-8, 4, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}

def Camera "ShotCam"
{
    float focalLength = 24
    float2 clippingRange = (1, 5000)
    double3 xformOp:translate = (10, 45, 190)
    float3 xformOp:rotateXYZ = (-4, 8, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
}

def Scope "Render"
{
    def RenderSettings "Settings"
    {
        rel camera = </ShotCam>
        uniform int2 resolution = (960, 540)
    }
}
)";

std::filesystem::path AssetDirectory() {
    return demos::CacheDir() / "usd" / "openvdb";
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>()};
}

/// The CRC-32 a zip entry records.
std::uint32_t Crc32(std::string_view bytes) {
    std::uint32_t crc = 0xffffffffu;
    for (const char c : bytes) {
        crc ^= static_cast<std::uint8_t>(c);
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

/// Writes `files` (name and bytes, the layer first) as a .usdz: an
/// uncompressed zip whose files start on 64-byte boundaries, padded
/// in each entry's extra field, as USD requires.
void WriteUsdz(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, std::string>>& files) {
    std::string zip;
    std::string directory;
    const auto u16 = [](std::string& out, std::size_t v) {
        out.push_back(static_cast<char>(v & 0xff));
        out.push_back(static_cast<char>((v >> 8) & 0xff));
    };
    const auto u32 = [&](std::string& out, std::size_t v) {
        u16(out, v & 0xffff);
        u16(out, (v >> 16) & 0xffff);
    };
    for (const auto& [name, bytes] : files) {
        const std::size_t offset = zip.size();
        const std::uint32_t crc = Crc32(bytes);
        const std::size_t pad =
            (64 - (offset + 30 + name.size() + 4) % 64) % 64;
        u32(zip, 0x04034b50u);
        for (const std::size_t v : {20, 0, 0, 0, 0}) {
            u16(zip, v);
        }
        u32(zip, crc);
        u32(zip, bytes.size());
        u32(zip, bytes.size());
        u16(zip, name.size());
        u16(zip, 4 + pad);
        zip += name;
        u16(zip, 0x1986);
        u16(zip, pad);
        zip.append(pad, '\0');
        zip += bytes;
        u32(directory, 0x02014b50u);
        for (const std::size_t v : {20, 20, 0, 0, 0, 0}) {
            u16(directory, v);
        }
        u32(directory, crc);
        u32(directory, bytes.size());
        u32(directory, bytes.size());
        u16(directory, name.size());
        for (const std::size_t v : {0, 0, 0, 0}) {
            u16(directory, v);
        }
        u32(directory, 0);
        u32(directory, offset);
        directory += name;
    }
    const std::size_t start = zip.size();
    zip += directory;
    u32(zip, 0x06054b50u);
    u16(zip, 0);
    u16(zip, 0);
    u16(zip, files.size());
    u16(zip, files.size());
    u32(zip, directory.size());
    u32(zip, start);
    u16(zip, 0);
    const std::filesystem::path partial = path.string() + ".part";
    std::ofstream(partial, std::ios::binary)
        .write(zip.data(), static_cast<std::streamsize>(zip.size()));
    std::filesystem::rename(partial, path);
}

class SmokeDemo : public app::Demo {
  public:
    SmokeDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // USD has no standard volume shader, so the smoke takes
        // Ernie's.
        Material::Define(scene, "/Looks/Smoke")
            .SetDensity(density_)
            .SetVolumeAlbedo(glm::dvec3(0.9))
            .SetPhaseFunction("henyeyGreenstein")
            .SetAnisotropy(0.3);

        std::string why;
        if (!UsdImportAvailable(&why)) {
            Report(why, true);
            return;
        }
        const std::filesystem::path sample =
            AssetDirectory() / "smoke.vdb";
        if (!std::filesystem::is_regular_file(sample) &&
            demos::CanDownload()) {
            try {
                Report(std::string("Fetching ") + kUrl, false);
                demos::Download(kUrl, sample,
                                [this](const std::string& text) {
                                    Report(text, false);
                                });
            } catch (const std::exception& error) {
                Report(error.what(), true);
            }
        }
        if (!std::filesystem::is_regular_file(sample)) {
            Report("OpenVDB's smoke sample is not cached and cannot "
                   "be fetched",
                   true);
            return;
        }
        WriteShot(sample);
        Load();

        Widgets().Combo("Format", {"USDA", "USDZ (packed)"}, 0,
                        [this](std::size_t i) {
                            format_ = i;
                            Load();
                        });
        Widgets().Slider("Density", 0.1, 4.0, density_,
                         [this](double density) {
                             density_ = density;
                             Material(GetScene(), "/Looks/Smoke")
                                 .SetDensity(density);
                         });
        if (viewer != nullptr) {
            viewer->GetTimeline().Play();
        }
    }

    /// The ember dim, then bright.
    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, demos::SideBySide(RenderAt({1.0, 36.0})));
    }

    void TestFinal() override {
        if (!result_) {
            std::printf(
                "Skipped: no USD importer or no smoke sample\n");
            return;
        }
        Scene& scene = GetScene();
        const Path smoke = PrimOf("/Shot/Smoke");
        const Path density = PrimOf("/Shot/Smoke/Density");
        Check(scene.TypeOf(smoke) == tokens::kVolume &&
                  scene.TypeOf(density) == tokens::kOpenVDBAsset,
              "the Volume and its field should be imported");
        Check(OpenVDBAsset(scene, density)
                  .GetFilePath()
                  .ends_with("smoke.vdb"),
              "the field should read the OpenVDB sample");
        Check(SphereLight(scene, PrimOf("/Shot/Ember"))
                      .GetLightLinkIncludes() ==
                  std::vector<Path>{smoke},
              "the ember should light the smoke alone");
        Check(RenderSettings(scene, PrimOf("/Render/Settings"))
                      .GetCamera() ==
                  std::vector<Path>{PrimOf("/ShotCam")},
              "the settings should name the shot's camera");

        // The smoke shows, and the ember brightens it as it flares.
        const std::vector<Tensor> lit = RenderAt({1.0, 36.0});
        scene.Set(smoke, tokens::kVisibility,
                  std::string(tokens::kInvisible));
        const Tensor bare = RenderAt({1.0}).front();
        scene.Set(smoke, tokens::kVisibility,
                  std::string(tokens::kInherited));
        const double shown = Changed(lit[0], bare);
        std::printf("smoke covers %.3f of the view; means %.1f bare, "
                    "%.1f dim, %.1f bright\n",
                    shown, MeanRgb(bare), MeanRgb(lit[0]),
                    MeanRgb(lit[1]));
        Check(shown > 0.05, "the smoke should show");
        Check(MeanRgb(lit[1]) > MeanRgb(lit[0]),
              "the ember should flare");

        // Packed: the sample comes out of the .usdz and renders the
        // same (exposure settled on each).
        const Tensor before = RenderAt({1.0, 1.0}).back();
        format_ = 1;
        Load();
        Check(result_->warnings.empty(),
              "the package should import without warnings");
        const std::string packed =
            OpenVDBAsset(scene, PrimOf("/Shot/Smoke/Density"))
                .GetFilePath();
        std::printf("packed sample read from %s\n", packed.c_str());
        Check(packed.ends_with("smoke.vdb") &&
                  packed.find("usd_packages") != std::string::npos,
              "the sample should be unpacked into the cache");
        const Tensor again = RenderAt({1.0, 1.0}).back();
        Check(std::abs(MeanRgb(again) - MeanRgb(before)) < 1.0,
              "the packed shot should render as the plain one");
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

    /// The shot's layer beside the sample, and both in a .usdz,
    /// written when they change (Ernie unpacks a package once per
    /// version).
    static void WriteShot(const std::filesystem::path& sample) {
        const std::filesystem::path layer =
            AssetDirectory() / "smoke_shot.usda";
        const std::filesystem::path package =
            AssetDirectory() / "smoke_shot.usdz";
        if (std::filesystem::is_regular_file(package) &&
            ReadFile(layer) == kLayer) {
            return;
        }
        WriteUsdz(package, {{"smoke_shot.usda", std::string(kLayer)},
                            {"smoke.vdb", ReadFile(sample)}});
        std::ofstream(layer, std::ios::binary)
            .write(kLayer.data(),
                   static_cast<std::streamsize>(kLayer.size()));
    }

    /// Imports the shot in the chosen format, its range the
    /// timeline's and its camera the viewer's.
    void Load() {
        Scene& scene = GetScene();
        scene.Remove(Path(kShot));
        const std::filesystem::path file =
            AssetDirectory() /
            (std::string("smoke_shot.") + kFormats[format_]);
        result_ = ImportUsd(scene, Path(kShot), file);
        for (const std::string& warning : result_->warnings) {
            Report(warning, true);
        }
        if (result_->timeCodeRange) {
            Stage(scene, Path::Root())
                .SetStartTimeCode(result_->timeCodeRange->x)
                .SetEndTimeCode(result_->timeCodeRange->y)
                .SetTimeCodesPerSecond(result_->timeCodesPerSecond)
                .SetTimeCode(result_->timeCodeRange->x);
        }
        scene.Set(PrimOf("/Shot/Smoke"), tokens::kMaterialBinding,
                  Path("/Looks/Smoke"));
        const Path camera = PrimOf("/ShotCam");
        Camera(scene, camera)
            .SetExposureMode(std::string(tokens::kAuto))
            .SetSensorNoise(false);
        if (app::Viewer* viewer = GetViewer()) {
            viewer->SetActiveCamera(camera);
        }
        Report("Imported " + file.filename().string(), false);
    }

    /// The Ernie prim the shot's USD prim became.
    Path PrimOf(const std::string& usdPath) const {
        return result_->prims.at(usdPath);
    }

    /// The scene at each time code, RGBA uint8.
    std::vector<Tensor> RenderAt(const std::vector<double>& times) {
        Stage stage(GetScene(), Path::Root());
        std::vector<Tensor> images;
        for (const double t : times) {
            stage.SetTimeCode(t);
            images.push_back(GetRenderer().Render(GetScene()));
        }
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
                most = std::max(
                    most, std::abs(static_cast<int>(x[4 * i + c]) -
                                   static_cast<int>(y[4 * i + c])));
            }
            changed += most > 16;
        }
        return static_cast<double>(changed) /
               static_cast<double>(count);
    }

    std::size_t format_ = 0;
    double density_ = 1.0;
    std::optional<UsdImportResult> result_;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    SmokeDemo, "usd", "smoke",
    "OpenVDB's smoke sample in a USD shot: a UsdVol volume, a light "
    "linked to it alone and the shot's render settings.");
