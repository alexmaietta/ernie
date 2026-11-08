// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
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
constexpr uint32_t kCenter = (kSide / 2) * kSide + kSide / 2;

void DefineCameras(Scene& scene, const std::vector<std::pair<std::string, Float3>>& cameras,
                   const std::vector<std::string>& vars) {
    std::vector<std::string> paths;
    for (const auto& [path, at] : cameras) {
        const Double3 eye = {at.x, at.y, at.z};
        const Double3 target = {at.x, at.y, 0};
        scene.Define<Camera>(
            path, {.translate = eye, .orient = LookAt(eye, target, {0, 1, 0}), .focalLength = 400});
        paths.push_back(path);
    }
    scene.Define<Scope>("/Render");
    std::vector<std::string> ordered;
    for (const std::string& var : vars) {
        ordered.push_back("/Render/" + var);
        scene.Define<RenderVar>(ordered.back(), {.sourceName = var});
    }
    scene.Define<RenderProduct>(
        "/Render/View", {.camera = paths, .resolution = {kSide, kSide}, .orderedVars = ordered});
}

template <class T> std::vector<T> Centers(Renderer& renderer, const char* source, uint32_t width) {
    RenderOutput output = renderer.Step(std::vector<std::string>{"/Render/View"}, 1.0 / 30);
    output.Wait();
    const std::vector<T> values =
        output["/Render/View"].frames.at(0).Source(source).tensor.ToHost<T>();
    const uint32_t cameras = uint32_t(values.size() / (kSide * kSide * width));
    std::vector<T> centers;
    for (uint32_t camera = 0; camera < cameras; ++camera) {
        for (uint32_t c = 0; c < width; ++c) {
            centers.push_back(values[(camera * kSide * kSide + kCenter) * width + c]);
        }
    }
    return centers;
}

void DefineLook(Scene& scene, const std::string& path, Float3 color) {
    const Prim material = scene.Define<Material>(path);
    const Prim surface = scene.Define<Shader>(path + "/Surface", {.id = "UsdPreviewSurface"});
    surface.Set(UsdPreviewSurface::kDiffuseColor, color);
    material.Connect("surface", surface, "surface");
}

const Mesh::Desc kTwoTiles = {.points = {{-1, -1, 0},
                                         {1, -1, 0},
                                         {1, 1, 0},
                                         {-1, 1, 0},
                                         {9, -1, 0},
                                         {11, -1, 0},
                                         {11, 1, 0},
                                         {9, 1, 0}},
                              .faceVertexIndices = {0, 1, 2, 3, 4, 5, 6, 7},
                              .faceVertexCounts = {4, 4}};

TEST(MeshRenderTest, SubsetsBindMaterialsToTheirFaces) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        DefineCameras(*scene, {{"/CamA", {0, 0, 5}}, {"/CamB", {10, 0, 5}}}, {"HdrColor"});
        scene->Find("/Render/View")->Set(MinimalAPI::kMode, MinimalMode::TextureDiffuse);
        scene->Define<Scope>("/Looks");
        DefineLook(*scene, "/Looks/Red", {1, 0, 0});
        DefineLook(*scene, "/Looks/Green", {0, 1, 0});
        Mesh::Desc tiles = kTwoTiles;
        tiles.materialBinding = {"/Looks/Red"};
        scene->Define<Mesh>("/Tiles", tiles);
        const Prim top = scene->Define<GeomSubset>(
            "/Tiles/Top",
            {.materialBinding = {"/Looks/Green"}, .indices = {1}, .familyName = "materialBind"});
        Renderer renderer(scene);
        std::vector<float> colors = Centers<float>(renderer, "HdrColor", 4);
        EXPECT_GT(colors[0], 0);
        EXPECT_EQ(colors[1], 0);
        EXPECT_EQ(colors[4], 0);
        EXPECT_GT(colors[5], 0);

        top.Set(GeomSubset::kFamilyName, std::string("other"));
        colors = Centers<float>(renderer, "HdrColor", 4);
        EXPECT_GT(colors[4], 0);
        EXPECT_EQ(colors[5], 0);

        top.Set(GeomSubset::kFamilyName, std::string("materialBind"));
        top.Set(GeomSubset::kIndices, std::vector<int32_t>{0});
        colors = Centers<float>(renderer, "HdrColor", 4);
        EXPECT_EQ(colors[0], 0);
        EXPECT_GT(colors[1], 0);
        EXPECT_GT(colors[4], 0);
        EXPECT_EQ(colors[5], 0);
    });
}

TEST(MeshRenderTest, LabeledSubsetsSegmentTheirFaces) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        DefineCameras(*scene, {{"/CamA", {0, 0, 5}}, {"/CamB", {10, 0, 5}}},
                      {"SemanticSegmentation"});
        const Prim tiles = scene->Define<Mesh>("/Tiles", kTwoTiles);
        scene->Apply<SemanticsLabelsAPI>(tiles, "class", {.semanticsLabels = {"body"}});
        scene->Define<GeomSubset>(
            "/Tiles/Wheel",
            {.indices = {1}, .familyName = "parts", .semanticsLabels = {{"class", {"wheel"}}}});
        scene->Require(std::vector<Token>{scene->Token("class")});
        Renderer renderer(scene);
        const std::vector<uint32_t> ids = Centers<uint32_t>(renderer, "SemanticSegmentation", 1);
        ASSERT_EQ(ids.size(), 2u);
        EXPECT_GE(ids[0], 2u);
        EXPECT_GE(ids[1], 2u);
        EXPECT_NE(ids[0], ids[1]);
    });
}

Ref<Scene> FrontAndBack(Device& device) {
    const Ref<Scene> scene = Scene::Create(device);
    DefineCameras(*scene, {{"/Front", {0, 0, 5}}, {"/Back", {0, 0, -5}}}, {"DistanceToCamera"});
    scene->Define<Mesh>("/Tile", {.points = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}},
                                  .faceVertexIndices = {0, 1, 2, 3},
                                  .faceVertexCounts = {4}});
    return scene;
}

TEST(MeshRenderTest, CameraRaysSkipTheBacksOfSingleSidedMeshes) {
    ForEachDevice([](Device& device) {
        for (const RenderMode mode : {RenderMode::RealTime, RenderMode::Minimal}) {
            const Ref<Scene> scene = FrontAndBack(device);
            scene->Select("/Render/View").Get<RenderMode>("renderMode").Set(mode);
            const Prim tile = *scene->Find("/Tile");
            Renderer renderer(scene);
            const auto seen = [&] {
                const std::vector<float> distances =
                    Centers<float>(renderer, "DistanceToCamera", 1);
                return std::pair{std::abs(distances[0] - 5) < 1e-3f,
                                 std::abs(distances[1] - 5) < 1e-3f};
            };
            EXPECT_EQ(seen(), std::pair(true, true));

            tile.Set(Mesh::kDoubleSided, false);
            EXPECT_EQ(seen(), std::pair(true, false));

            tile.Set(Mesh::kOrientation, Orientation::LeftHanded);
            EXPECT_EQ(seen(), std::pair(false, true));
        }
    });
}

Float3 CenterNormal(Renderer& renderer) {
    const std::vector<float> normal = Centers<float>(renderer, "Normal", 3);
    return {normal[0], normal[1], normal[2]};
}

TEST(MeshRenderTest, PolygonalMeshesWithoutNormalsShadeFaceted) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        DefineCameras(*scene, {{"/Cam", {-0.5f, 0, 5}}}, {"Normal"});
        const Prim roof = scene->Define<Mesh>(
            "/Roof",
            {.points = {{-1, -1, 0}, {0, -1, 1}, {1, -1, 0}, {-1, 1, 0}, {0, 1, 1}, {1, 1, 0}},
             .faceVertexIndices = {0, 1, 4, 3, 1, 2, 5, 4},
             .faceVertexCounts = {4, 4},
             .subdivisionScheme = SubdivisionScheme::None});
        Renderer renderer(scene);
        Float3 normal = CenterNormal(renderer);
        EXPECT_NEAR(normal.x, -std::sqrt(0.5f), 1e-4f);
        EXPECT_NEAR(normal.z, std::sqrt(0.5f), 1e-4f);

        roof.Set(Mesh::kSubdivisionScheme, SubdivisionScheme::Bilinear);
        normal = CenterNormal(renderer);
        EXPECT_NEAR(normal.x, -std::sqrt(0.5f), 1e-4f);

        roof.Set(Mesh::kSubdivisionScheme, SubdivisionScheme::CatmullClark);
        normal = CenterNormal(renderer);
        EXPECT_NEAR(normal.x, -0.5257f, 2e-3f);
        EXPECT_NEAR(normal.z, 0.8507f, 2e-3f);
    });
}

TEST(MeshRenderTest, IndexedNormalsPrimvarOverridesNormals) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        DefineCameras(*scene, {{"/Cam", {0, 0, 5}}}, {"Normal"});
        const float half = std::sqrt(0.5f);
        const Prim tile = scene->Define<Mesh>(
            "/Tile", {.points = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}},
                      .normals = std::vector<Float3>(4, Float3{0, 0, 1}),
                      .faceVertexIndices = {0, 1, 2, 3},
                      .faceVertexCounts = {4},
                      .subdivisionScheme = SubdivisionScheme::None});
        Renderer renderer(scene);
        EXPECT_NEAR(CenterNormal(renderer).z, 1, 1e-4f);

        tile.Set(Mesh::kPrimvars, std::vector<Primvar>{Primvar::Create(
                                      "normals", Interpolation::Vertex,
                                      std::vector<Float3>{{half, 0, half}}, {0, 0, 0, 0})});
        const Float3 normal = CenterNormal(renderer);
        EXPECT_NEAR(normal.x, half, 1e-4f);
        EXPECT_NEAR(normal.z, half, 1e-4f);
    });
}

}
}
