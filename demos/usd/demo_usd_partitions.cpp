// SPDX-License-Identifier: BSD-3-Clause
// Four cloned environments as Omniverse lays them out for a tiled
// camera, imported with ImportUsd: overlapping at the origin, kept
// apart by RTX scene partitions (`omni:scenePartition`), which become
// Ernie's worlds, and lit by Omniverse's photometric lights. The
// environments' cameras render as one batch, each seeing its own.

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/usd_import.h"
#include "ernie/session/resolve.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr int kEnvironments = 4;
constexpr int kWidth = 320;
constexpr int kHeight = 240;
const Path kRoot("/World/Lab");
const Path kProduct("/Render/Environments");

// Each environment a table, a block of its own colour in the same
// spot as the others', and a lamp of its own power and colour; the
// floor and the dome are every environment's.
constexpr std::string_view kLayer = R"(#usda 1.0
(
    defaultPrim = "Lab"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Lab"
{
    def Cube "Floor"
    {
        double size = 1
        color3f[] primvars:displayColor = [(0.5, 0.5, 0.5)]
        double3 xformOp:translate = (0, 0, -0.05)
        float3 xformOp:scale = (20, 20, 0.1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }

    def DomeLight "Sky" (
        prepend apiSchemas = ["PhotometricDomeLightAPI"]
    )
    {
        float photometric:illuminance = 400
    }

    def Xform "Envs"
    {
        def Xform "Env_0"
        {
            token omni:scenePartition = "env_0"
            def Cube "Table"
            {
                double size = 1
                color3f[] primvars:displayColor = [(0.45, 0.3, 0.18)]
                double3 xformOp:translate = (0, 0, 0.35)
                float3 xformOp:scale = (1.2, 0.8, 0.7)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            }
            def Cube "Block"
            {
                double size = 0.3
                color3f[] primvars:displayColor = [(0.8, 0.1, 0.1)]
                double3 xformOp:translate = (0, 0, 0.85)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def SphereLight "Lamp" (
                prepend apiSchemas = ["PhotometricAreaLightAPI"]
            )
            {
                float photometric:power = 1600
                token physical:illuminant = "blackbody"
                float inputs:colorTemperature = 2700
                float inputs:radius = 0.05
                double3 xformOp:translate = (-0.5, -0.4, 1.6)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def Xform "Rig"
            {
                float xformOp:rotateZ = 0
                uniform token[] xformOpOrder = ["xformOp:rotateZ"]
                def Camera "Camera"
                {
                    float focalLength = 24
                    double3 xformOp:translate = (0, -2.6, 1.5)
                    float3 xformOp:rotateXYZ = (75, 0, 0)
                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
                }
            }
        }
        def Xform "Env_1"
        {
            token omni:scenePartition = "env_1"
            def Cube "Table"
            {
                double size = 1
                color3f[] primvars:displayColor = [(0.45, 0.3, 0.18)]
                double3 xformOp:translate = (0, 0, 0.35)
                float3 xformOp:scale = (1.2, 0.8, 0.7)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            }
            def Cube "Block"
            {
                double size = 0.36
                color3f[] primvars:displayColor = [(0.1, 0.7, 0.15)]
                double3 xformOp:translate = (0, 0, 0.88)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def SphereLight "Lamp" (
                prepend apiSchemas = ["PhotometricAreaLightAPI"]
            )
            {
                float photometric:power = 3200
                token physical:illuminant = "blackbody"
                float inputs:colorTemperature = 4000
                float inputs:radius = 0.05
                double3 xformOp:translate = (0.5, -0.4, 1.6)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def Xform "Rig"
            {
                float xformOp:rotateZ = 30
                uniform token[] xformOpOrder = ["xformOp:rotateZ"]
                def Camera "Camera"
                {
                    float focalLength = 24
                    double3 xformOp:translate = (0, -2.6, 1.5)
                    float3 xformOp:rotateXYZ = (75, 0, 0)
                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
                }
            }
        }
        def Xform "Env_2"
        {
            token omni:scenePartition = "env_2"
            def Cube "Table"
            {
                double size = 1
                color3f[] primvars:displayColor = [(0.45, 0.3, 0.18)]
                double3 xformOp:translate = (0, 0, 0.35)
                float3 xformOp:scale = (1.2, 0.8, 0.7)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            }
            def Cube "Block"
            {
                double size = 0.42
                color3f[] primvars:displayColor = [(0.12, 0.2, 0.8)]
                double3 xformOp:translate = (0, 0, 0.91)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def SphereLight "Lamp" (
                prepend apiSchemas = ["PhotometricAreaLightAPI"]
            )
            {
                float photometric:power = 6400
                token physical:illuminant = "blackbody"
                float inputs:colorTemperature = 5500
                float inputs:radius = 0.05
                double3 xformOp:translate = (0, 0.5, 1.8)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def Xform "Rig"
            {
                float xformOp:rotateZ = 60
                uniform token[] xformOpOrder = ["xformOp:rotateZ"]
                def Camera "Camera"
                {
                    float focalLength = 24
                    double3 xformOp:translate = (0, -2.6, 1.5)
                    float3 xformOp:rotateXYZ = (75, 0, 0)
                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
                }
            }
        }
        def Xform "Env_3"
        {
            token omni:scenePartition = "env_3"
            def Cube "Table"
            {
                double size = 1
                color3f[] primvars:displayColor = [(0.45, 0.3, 0.18)]
                double3 xformOp:translate = (0, 0, 0.35)
                float3 xformOp:scale = (1.2, 0.8, 0.7)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            }
            def Cube "Block"
            {
                double size = 0.48
                color3f[] primvars:displayColor = [(0.8, 0.7, 0.1)]
                double3 xformOp:translate = (0, 0, 0.94)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def SphereLight "Lamp" (
                prepend apiSchemas = ["PhotometricAreaLightAPI"]
            )
            {
                float photometric:power = 12800
                token physical:illuminant = "blackbody"
                float inputs:colorTemperature = 6500
                float inputs:radius = 0.05
                double3 xformOp:translate = (0, -0.6, 1.8)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def Xform "Rig"
            {
                float xformOp:rotateZ = 90
                uniform token[] xformOpOrder = ["xformOp:rotateZ"]
                def Camera "Camera"
                {
                    float focalLength = 24
                    double3 xformOp:translate = (0, -2.6, 1.5)
                    float3 xformOp:rotateXYZ = (75, 0, 0)
                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
                }
            }
        }
    }

    # Pebbles on the tables, two in each environment's partition.
    def PointInstancer "Pebbles"
    {
        point3f[] positions = [(-0.45, 0.25, 0.73), (0.45, 0.25, 0.73), (-0.45, -0.25, 0.73), (0.45, -0.25, 0.73), (-0.25, 0.3, 0.73), (0.25, 0.3, 0.73), (-0.25, -0.3, 0.73), (0.25, -0.3, 0.73)]
        int[] protoIndices = [0, 0, 0, 0, 0, 0, 0, 0]
        rel prototypes = </Lab/Pebbles/Pebble>
        token[] primvars:omni:scenePartition = ["env_0", "env_0", "env_1", "env_1", "env_2", "env_2", "env_3", "env_3"] (
            interpolation = "vertex"
        )
        def Sphere "Pebble"
        {
            double radius = 0.06
            color3f[] primvars:displayColor = [(0.85, 0.85, 0.8)]
        }
    }

    # Sees every partition: what the environments would look like
    # without them.
    def Camera "Overview"
    {
        float focalLength = 24
        double3 xformOp:translate = (0, -2.6, 1.5)
        float3 xformOp:rotateXYZ = (75, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }
}
)";

/// Each environment's block colour, as its dominant channels.
constexpr std::array<const char*, kEnvironments> kColours = {
    "red", "green", "blue", "yellow"};

class PartitionsDemo : public app::Demo {
  public:
    PartitionsDemo(app::Viewer* viewer,
                   const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Xform::Define(GetScene(), "/World");
        std::string why;
        if (!UsdImportAvailable(&why)) {
            Report(why, true);
            return;
        }
        Load();
        std::vector<std::string> cameras;
        for (int i = 0; i < kEnvironments; ++i) {
            cameras.push_back("env_" + std::to_string(i));
        }
        cameras.emplace_back("Overview (every partition)");
        Widgets().Combo("Camera", cameras, 0,
                        [this](std::size_t i) { Look(i); });
        Look(0);
    }

    /// The four environments' views, then the overview's.
    void SaveOutput(const std::filesystem::path& path) override {
        if (!result_) {
            return;
        }
        std::vector<Tensor> images = RenderViews();
        images.push_back(GetRenderer().Render(GetScene()));
        app::WritePng(path, demos::SideBySide(images));
    }

    void TestFinal() override {
        if (!result_) {
            std::printf("Skipped: no USD importer\n");
            return;
        }
        Scene& scene = GetScene();
        Check(result_->warnings.empty(),
              "the lab should import without warnings");

        // Each environment is a world, and its camera and lamp in it.
        const auto names = scene.Get<std::vector<std::string>>(
            kRoot, "ernie:usd:scenePartitions", {});
        Check(names.size() == kEnvironments,
              "each partition should be a world");
        for (int i = 0; i < kEnvironments; ++i) {
            const std::string env = "env_" + std::to_string(i);
            const auto world = static_cast<std::int32_t>(
                std::find(names.begin(), names.end(), env) -
                names.begin());
            Check(WorldOf(scene, PrimOf(Env(i))) == world,
                  env + " should be its partition's world");
            Check(WorldOf(scene, CameraOf(i)) == world,
                  env + "'s camera should inherit its world");
            const Light lamp(scene, PrimOf(Env(i) + "/Lamp"));
            Check(lamp.IsPhotometric() &&
                      lamp.GetPhysicalIlluminant() == "blackbody",
                  env + "'s lamp should keep its photometric units");
        }
        Check(Light(scene, PrimOf("/Lab/Sky")).IsPhotometric(),
              "the dome should be photometric");
        Check(scene.Read(PrimOf("/Lab/Pebbles"), tokens::kWorlds) !=
                  nullptr,
              "each pebble should be in its partition");

        // Every view sees its own block in the middle, though the
        // blocks share the spot.
        const std::vector<Tensor> views = RenderViews();
        for (int i = 0; i < kEnvironments; ++i) {
            const std::uint8_t* centre =
                views[static_cast<std::size_t>(i)]
                    .As<std::uint8_t>() +
                (kHeight / 2 * kWidth + kWidth / 2) * 4;
            const int r = centre[0];
            const int g = centre[1];
            const int b = centre[2];
            std::printf("env_%d centre (%d, %d, %d)\n", i, r, g, b);
            const std::string colour =
                kColours[static_cast<std::size_t>(i)];
            const bool seen =
                colour == "red"     ? r > g + 30 && r > b + 30
                : colour == "green" ? g > r + 30 && g > b + 30
                : colour == "blue"  ? b > r + 30 && b > g + 30
                                    : r > b + 30 && g > b + 30;
            Check(seen, "env_" + std::to_string(i) +
                            "'s camera should see its own " + colour +
                            " block alone");
        }
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

    /// Writes the layer to the cache and imports it, with a product
    /// of the environments' cameras.
    void Load() {
        Scene& scene = GetScene();
        const std::filesystem::path file =
            demos::CacheDir() / "usd" / "partitions.usda";
        std::filesystem::create_directories(file.parent_path());
        std::ofstream(file, std::ios::binary)
            .write(kLayer.data(),
                   static_cast<std::streamsize>(kLayer.size()));
        result_ = ImportUsd(scene, kRoot, file);
        for (const std::string& warning : result_->warnings) {
            Report(warning, true);
        }
        std::vector<Path> cameras;
        for (int i = 0; i < kEnvironments; ++i) {
            cameras.push_back(CameraOf(i));
        }
        cameras.push_back(PrimOf("/Lab/Overview"));
        // A batch meters no exposure: an indoor one, about EV 8.
        for (const Path& camera : cameras) {
            Camera(scene, camera)
                .SetExposureFStop(2.8)
                .SetExposureTime(1.0 / 30.0)
                .SetExposureIso(100.0)
                .SetSensorNoise(false);
        }
        cameras.pop_back();
        const Path var = kProduct.Child("LdrColor");
        RenderVar::Define(scene, var).SetSourceName("LdrColor");
        RenderProduct::Define(scene, kProduct)
            .SetCamera(cameras)
            .SetResolution(glm::dvec2(kWidth, kHeight))
            .SetOrderedVars({var});
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera({PrimOf("/Lab/Overview")})
            .SetResolution(glm::dvec2(kWidth, kHeight));
        Report("Imported " + file.filename().string() + " with " +
                   std::to_string(kEnvironments) + " partitions",
               false);
    }

    /// Looks through environment `i`'s camera, or the overview's.
    void Look(std::size_t i) {
        if (app::Viewer* viewer = GetViewer()) {
            viewer->SetActiveCamera(
                i < kEnvironments ? CameraOf(static_cast<int>(i))
                                  : PrimOf("/Lab/Overview"));
        }
    }

    /// The environments' views, one batch, as RGBA uint8 images.
    std::vector<Tensor> RenderViews() {
        const Path products[] = {kProduct};
        RenderProductSetOutputs outputs =
            GetRenderer().Step(GetScene(), products);
        const Tensor batch = outputs.at(kProduct)
                                 .vars.at("LdrColor")
                                 .Map(MapDevice::Cpu)
                                 .tensor.Contiguous();
        const auto* bytes = batch.As<std::uint8_t>();
        const std::size_t view =
            static_cast<std::size_t>(kWidth) * kHeight * 4;
        std::vector<Tensor> images;
        for (int i = 0; i < kEnvironments; ++i) {
            images.push_back(Tensor::Copy<std::uint8_t>(
                std::span(bytes + static_cast<std::size_t>(i) * view,
                          view),
                {kHeight, kWidth, 4}));
        }
        return images;
    }

    static std::string Env(int i) {
        return "/Lab/Envs/Env_" + std::to_string(i);
    }

    Path CameraOf(int i) const {
        return PrimOf(Env(i) + "/Rig/Camera");
    }

    /// The Ernie prim the lab's USD prim became.
    Path PrimOf(const std::string& usdPath) const {
        return result_->prims.at(usdPath);
    }

    std::optional<UsdImportResult> result_;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    PartitionsDemo, "usd", "partitions",
    "Cloned environments kept apart by RTX scene partitions, as "
    "worlds, and lit by Omniverse's photometric lights.");
