// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/render/render_output.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/primvar.h"
#include "ernie/scene/rotation.h"
#include "ernie/scene/scene.h"
#include "support/devices.h"

namespace ernie {
namespace {

using device::Device;
using test::ForEachDevice;

constexpr uint32_t kSide = 3;

float CenterVelocity(Renderer& renderer) {
    RenderOutput output = renderer.Step(std::vector<std::string>{"/Render/Scan"}, 1.0 / 30);
    output.Wait();
    const std::vector<float> velocity =
        output["/Render/Scan"].frames.at(0).Source("RadialVelocity").tensor.ToHost<float>();
    return velocity[(kSide / 2) * kSide + kSide / 2];
}

TEST(SurfaceTest, RadarSeesTheMotionOfTheNearestMovingAncestor) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim radar =
            scene->Define<Radar>("/Radar", {.fieldOfView = {2, 2}, .rays = {kSide, kSide}});
        const Prim cart = scene->Define<Xform>("/Cart", {.translate = {0, 0, -10}});
        scene->Define<Xform>("/Cart/Body");
        scene->Define<Cube>("/Cart/Body/Panel", {.size = 4});
        scene->Define<Scope>("/Render");
        scene->Define<RenderVar>("/Render/Velocity",
                                 {.sourceName = RenderVarSource::RadialVelocity});
        scene->Define<RenderProduct>("/Render/Scan",
                                     {.camera = {"/Radar"}, .orderedVars = {"/Render/Velocity"}});
        Renderer renderer(scene);
        EXPECT_NEAR(CenterVelocity(renderer), 0, 1e-4f);

        scene->Apply<MotionAPI>(cart, {.motionVelocity = {0, 0, -3}});
        EXPECT_NEAR(CenterVelocity(renderer), 3, 1e-3f);

        scene->Apply<MotionAPI>(radar, {.motionVelocity = {0, 0, -1}});
        EXPECT_NEAR(CenterVelocity(renderer), 2, 1e-3f);
    });
}

struct Colors {
    Float3 first;
    Float3 second;
};

Colors CenterColors(Renderer& renderer) {
    RenderOutput output = renderer.Step(std::vector<std::string>{"/Render/View"}, 1.0 / 30);
    output.Wait();
    const std::vector<float> color =
        output["/Render/View"].frames.at(0).Source("HdrColor").tensor.ToHost<float>();
    const auto at = [&](uint32_t camera) {
        const uint32_t pixel = ((camera * kSide + kSide / 2) * kSide + kSide / 2) * 4;
        return Float3{color[pixel], color[pixel + 1], color[pixel + 2]};
    };
    return {at(0), at(1)};
}

Ref<Scene> TwoTiles(Device& device) {
    const Ref<Scene> scene = Scene::Create(device);
    for (const auto& [path, x] : {std::pair{"/CamA", 0.0}, std::pair{"/CamB", 10.0}}) {
        scene->Define<Camera>(path, {.translate = {x, 0, 5},
                                     .orient = LookAt({x, 0, 5}, {x, 0, 0}, {0, 1, 0}),
                                     .focalLength = 400});
    }
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Color", {.sourceName = RenderVarSource::HdrColor});
    scene->Define<RenderProduct>("/Render/View", {.camera = {"/CamA", "/CamB"},
                                                  .resolution = {kSide, kSide},
                                                  .orderedVars = {"/Render/Color"}});
    scene->Find("/Render/View")->Set(MinimalAPI::kMode, MinimalMode::DisplayColor);
    return scene;
}

const Mesh::Desc kTile = {.points = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}},
                          .faceVertexIndices = {0, 1, 2, 3},
                          .faceVertexCounts = {4}};

TEST(SurfaceTest, DisplayColorsFollowThePrimAndEachInstancerCopy) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = TwoTiles(device);
        Mesh::Desc red = kTile;
        red.displayColor = {1, 0, 0};
        const Prim tile = scene->Define<Mesh>("/Tile", red);
        scene->Define<PointInstancer>(
            "/Inst", {.prototypes = {"/Inst/Protos/Tile"},
                      .protoIndices = {0},
                      .positions = {{10, 0, 0}},
                      .primvars = {Primvar::Create("displayColor", Interpolation::Vertex,
                                                   std::vector<Float3>{{0, 0, 1}})}});
        scene->Define<Scope>("/Inst/Protos");
        scene->Define<Mesh>("/Inst/Protos/Tile", kTile);
        Renderer renderer(scene);
        Colors colors = CenterColors(renderer);
        EXPECT_GT(colors.first.x, 0);
        EXPECT_EQ(colors.first.z, 0);
        EXPECT_EQ(colors.second.x, 0);
        EXPECT_GT(colors.second.z, 0);

        tile.Set(Mesh::kDisplayColor, Float3{0, 1, 0});
        colors = CenterColors(renderer);
        EXPECT_EQ(colors.first.x, 0);
        EXPECT_GT(colors.first.y, 0);
    });
}

}
}
