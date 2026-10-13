// SPDX-License-Identifier: BSD-3-Clause
// Four worlds in one place, each an Earth-sized adaptive sphere seen
// by one view of a batched camera from its own altitude: every sphere
// is refined only towards its own world's view, so each horizon lies
// where the true sphere's does. LOD levels are tinted.

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

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kEarth = 6371000.0;
constexpr std::int64_t kSize = 256;
constexpr double kFocal = 24.0;
const Path kCamera("/World/Camera");
const Path kProduct("/Render/Batch");

// Each world's view: its altitude, metres, and its sphere's colour.
struct World {
    double altitude;
    glm::dvec3 colour;
};
constexpr std::array<World, 4> kWorlds = {{
    {100000.0, {0.30, 0.45, 0.80}},
    {10000.0, {0.35, 0.60, 0.30}},
    {1000.0, {0.75, 0.60, 0.35}},
    {100.0, {0.70, 0.70, 0.70}},
}};

/// The view of world `w`: over the north pole at its altitude,
/// looking level along +x.
glm::dmat4 PoseOf(std::size_t w) {
    const glm::dvec3 eye(0.0, 0.0, kEarth + kWorlds[w].altitude);
    return LookAtTransform(eye, eye + glm::dvec3(1.0, 0.0, 0.0),
                           {0.0, 0.0, 1.0});
}

/// The image row of the true sphere's horizon from world `w`'s view:
/// acos(R / (R + h)) below the centre.
double HorizonRow(std::size_t w) {
    const double dip =
        std::acos(kEarth / (kEarth + kWorlds[w].altitude));
    const double halfHeight = 0.5 * 15.2908 / kFocal;
    return 0.5 * kSize * (1.0 + std::tan(dip) / halfHeight);
}

class BatchedAdaptiveSpheresDemo : public app::Demo {
  public:
    BatchedAdaptiveSpheresDemo(app::Viewer* viewer,
                               const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        DistantLight::Define(scene, "/World/Sun")
            .SetIntensity(1.0)
            .SetTransform(DistantLightTransform({0.3, -0.4, 0.85}));
        std::vector<float> poses;
        std::vector<std::int32_t> worlds;
        for (std::size_t w = 0; w < kWorlds.size(); ++w) {
            const std::string env = "/Envs/Env" + std::to_string(w);
            Xform::Define(scene, env).SetWorld(static_cast<int>(w));
            const std::string look = env + "/Look";
            Material::Define(scene, look);
            Shader::Define(scene, look + "/Surface")
                .SetInfoId(tokens::kUsdPreviewSurface)
                .SetDiffuseColor(kWorlds[w].colour);
            Connect(scene, look, tokens::kSurfaceOutput,
                    look + "/Surface", tokens::kSurfaceOutput);
            Sphere::Define(scene, env + "/Planet")
                .SetRadius(kEarth)
                .SetTessellation(std::string(tokens::kAdaptiveSphere))
                .SetMaterialBinding(look);
            const glm::dmat4 pose = PoseOf(w);
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    poses.push_back(
                        static_cast<float>(pose[col][row]));
                }
            }
            worlds.push_back(static_cast<std::int32_t>(w));
        }
        const auto views = static_cast<std::int64_t>(kWorlds.size());
        Camera::Define(scene, kCamera)
            .SetFocalLength(kFocal)
            .SetClippingRange(glm::dvec2(
                1.0, std::numeric_limits<double>::infinity()))
            .SetPoses(Tensor::Copy<float>(poses, {views, 4, 4}))
            .SetWorlds(Tensor::Copy<std::int32_t>(worlds, {views}));
        const Path ldr = kProduct.Child("LdrColor");
        RenderVar::Define(scene, ldr).SetSourceName("LdrColor");
        RenderProduct::Define(scene, kProduct)
            .SetCamera(kCamera)
            .SetResolution(glm::dvec2(kSize, kSize))
            .SetOrderedVars({ldr});
        // The viewer looks down on the worlds, all in one place.
        Camera::Define(scene, "/World/Overview")
            .SetFocalLength(24.0)
            .SetClippingRange(glm::dvec2(
                1.0, std::numeric_limits<double>::infinity()))
            .SetTransform(LookAtTransform({0.0, -2.0e7, 1.0e7},
                                          {0.0, 0.0, 0.0},
                                          {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera("/World/Overview")
            .SetResolution(glm::dvec2(1280.0, 720.0))
            .SetLodShowLevels(true);
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Overview");
        }
    }

    void SaveOutput(const std::filesystem::path& path) override {
        // The four views side by side.
        const Tensor images = Views();
        const auto* from =
            static_cast<const std::uint8_t*>(images.Data());
        const auto count = static_cast<std::int64_t>(kWorlds.size());
        Tensor sheet =
            Tensor::Zeros(DType::UInt8, {kSize, count * kSize, 4});
        auto* to = static_cast<std::uint8_t*>(sheet.Data());
        for (std::int64_t v = 0; v < count; ++v) {
            for (std::int64_t y = 0; y < kSize; ++y) {
                std::memcpy(to + (y * count * kSize + v * kSize) * 4,
                            from + ((v * kSize + y) * kSize) * 4,
                            kSize * 4);
            }
        }
        app::WritePng(path, sheet);
    }

    void TestFinal() override {
        // Down each view's middle column, the first row that is not
        // sky is the true sphere's horizon.
        const Tensor images = Views();
        const auto* p =
            static_cast<const std::uint8_t*>(images.Data());
        for (std::size_t w = 0; w < kWorlds.size(); ++w) {
            std::int64_t row = 0;
            while (row < kSize) {
                const auto* texel =
                    p +
                    ((static_cast<std::int64_t>(w) * kSize + row) *
                         kSize +
                     kSize / 2) *
                        4;
                if (texel[0] + texel[1] + texel[2] > 0) {
                    break;
                }
                ++row;
            }
            std::printf("world %zu, %.0f m up: horizon at row %lld, "
                        "the sphere's %.1f\n",
                        w, kWorlds[w].altitude,
                        static_cast<long long>(row), HorizonRow(w));
            Check(std::abs(static_cast<double>(row) - HorizonRow(w)) <
                      1.5,
                  "each world's horizon is its true sphere's");
        }
    }

  private:
    /// Every view's LdrColor, uint8 (views, kSize, kSize, 4).
    Tensor Views() {
        const Path products[] = {kProduct};
        RenderProductSetOutputs outputs =
            GetRenderer().Step(GetScene(), products);
        return outputs.at(kProduct)
            .vars.at("LdrColor")
            .Map(MapDevice::Cpu)
            .tensor;
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(BatchedAdaptiveSpheresDemo, "batched",
                    "adaptive_spheres",
                    "Four worlds' adaptive spheres, each refined "
                    "only towards its own "
                    "world's view of a batched camera.");
