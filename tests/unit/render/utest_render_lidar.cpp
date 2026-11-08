// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <numbers>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/render/render_output.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/scene.h"
#include "support/devices.h"

namespace ernie {
namespace {

using device::Device;
using test::ForEachDevice;

constexpr float kWallDistance = 5;
const std::vector<std::string> kProducts = {"/Render/Scan"};

Ref<Scene> BeamAtWall(Device& device, Float2 divergence) {
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<Plane>("/Wall", {.translate = {0, 0, -kWallDistance}, .width = 0, .length = 0});
    scene->Define<Lidar>("/Lidar", {.scanType = ScanType::SolidState,
                                    .firingsPerScan = 1,
                                    .emitterElevations = {0},
                                    .returnMode = ReturnMode::Mean,
                                    .beamDivergence = divergence});
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>(
        "/Render/Points", {.sourceName = RenderVarSource::PointCloud, .channels = {"Coordinates"}});
    scene->Define<RenderProduct>("/Render/Scan",
                                 {.camera = {"/Lidar"}, .orderedVars = {"/Render/Points"}});
    return scene;
}

std::vector<float> ScanPoint(Renderer& renderer) {
    RenderOutput output = renderer.Step(kProducts, 0);
    output.Wait();
    const RenderVarOutput& points = output["/Render/Scan"].frames.at(0).Source("PointCloud");
    return points.channels.at("Coordinates").ToHost<float>();
}

TEST(LidarTest, DivergentBeamsCenterOnTheirEmitter) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = BeamAtWall(device, {20, 20});
        Renderer renderer(scene);
        for (const uint32_t samples : {2u, 3u, 4u, 7u}) {
            scene->Select("/Lidar").Get(Lidar::kBeamSamples).Set(samples);
            const std::vector<float> point = ScanPoint(renderer);
            ASSERT_GE(point.size(), 3u) << samples << " samples";
            EXPECT_NEAR(point[0], 0, 0.01) << samples << " samples";
            EXPECT_NEAR(point[1], 0, 0.01) << samples << " samples";
            EXPECT_NEAR(point[2], -kWallDistance, 0.1) << samples << " samples";
        }
    });
}

TEST(LidarTest, DustPointsReturnWhereTheBeamMeetsThem) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = BeamAtWall(device, {0, 0});
        const Prim dust =
            scene->Define<Points>("/Dust", {.points = {{0, 0, -2}, {3, 0, -2}}, .widths = {0.5f}});
        Renderer renderer(scene);
        const auto range = [&] {
            const std::vector<float> point = ScanPoint(renderer);
            return point.size() >= 3 ? point[2] : 0.0f;
        };
        EXPECT_NEAR(range(), -1.75f, 1e-3f);

        dust.Set(Points::kPoints, std::vector<Float3>{{0, 0, -3}, {3, 0, -2}});
        EXPECT_NEAR(range(), -2.75f, 1e-3f);

        dust.Set(Points::kNormals, std::vector<Float3>{{0, 0, 1}});
        EXPECT_NEAR(range(), -3, 1e-3f);

        dust.Set(Points::kDisplayOpacity, 0.0f);
        EXPECT_NEAR(range(), -kWallDistance, 1e-3f);

        dust.Set(Points::kDisplayOpacity, 1.0f);
        dust.Set(Points::kPoints, std::vector<Float3>{{3, 0, -2}});
        EXPECT_NEAR(range(), -kWallDistance, 1e-3f);
    });
}

}
}
