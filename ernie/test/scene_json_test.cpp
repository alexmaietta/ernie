// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/scene_json.h"

#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/resolve.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using ernie::DType;
using ernie::LoadScene;
using ernie::Path;
using ernie::Scene;
namespace tokens = ernie::tokens;

namespace {

/// The description of `test_scenes_load_from_descriptions` in `test/python/test_schema.py`.
constexpr const char* kDescription = R"({
  "/World": {
    "type": "Xform",
    "Rock": {"type": "Sphere", "radius": 0.5, "translate": [0, 0, 1]},
    "Sun": {"type": "DistantLight", "angle": 0.53, "inputs:intensity": 2.0},
    "Ground": {"type": "Terrain", "heights": [[0, 1], [2, 3.5]], "cell_size": 2.0},
    "Quad": {"type": "Mesh", "points": [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
             "face_vertex_counts": [3], "face_vertex_indices": [0, 1, 2]},
    "Group": {"Child": {"type": "Cube", "size": 2.0}}
  },
  "Render/Settings": {"type": "RenderSettings", "camera": "/World/Camera",
                      "resolution": [640, 480]}
})";

/// The message `LoadScene` throws for `json`, or "" if it does not throw.
std::string ErrorOf(const std::string& json) {
    Scene scene;
    try {
        LoadScene(scene, json);
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return {};
}

void ExpectNear(const glm::dmat4& actual, const glm::dmat4& expected) {
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            EXPECT_NEAR(actual[col][row], expected[col][row], 1e-9) << col << ", " << row;
        }
    }
}

}  // namespace

TEST(LoadScene, LoadsTheDescriptionPythonLoads) {
    Scene scene;
    LoadScene(scene, kDescription);

    EXPECT_EQ(ernie::Sphere(scene, "/World/Rock").GetRadius(), 0.5);
    EXPECT_EQ(scene.Get<glm::dvec3>("/World/Rock", tokens::kTranslate, {}),
              glm::dvec3(0.0, 0.0, 1.0));
    EXPECT_EQ(ernie::DistantLight(scene, "/World/Sun").GetIntensity(), 2.0);
    EXPECT_EQ(ernie::DistantLight(scene, "/World/Sun").GetAngle(), 0.53);

    const ernie::Terrain ground(scene, "/World/Ground");
    EXPECT_EQ(ground.GetCellSize(), 2.0);
    ASSERT_NE(ground.GetHeights(), nullptr);
    EXPECT_EQ(ground.GetHeights()->Type(), DType::Float32);
    EXPECT_EQ(ground.GetHeights()->Shape(), (std::vector<std::int64_t>{2, 2}));
    EXPECT_EQ(ground.GetHeights()->As<float>()[3], 3.5f);

    const ernie::Mesh quad(scene, "/World/Quad");
    ASSERT_NE(quad.GetFaceVertexIndices(), nullptr);
    EXPECT_EQ(quad.GetFaceVertexIndices()->Type(), DType::Int32);
    // Points are float32 though written as whole numbers: the declaration picks the type.
    EXPECT_EQ(quad.GetPoints()->Type(), DType::Float32);
    EXPECT_EQ(quad.GetFaceVertexCounts()->Shape(), (std::vector<std::int64_t>{1}));
    EXPECT_EQ(quad.GetPoints()->Shape(), (std::vector<std::int64_t>{3, 3}));

    EXPECT_EQ(scene.TypeOf("/World/Group"), "");
    EXPECT_EQ(ernie::Cube(scene, "/World/Group/Child").GetSize(), 2.0);

    const ernie::RenderSettings settings(scene, "/Render/Settings");
    EXPECT_EQ(settings.GetCamera(), std::vector<Path>{Path("/World/Camera")});
    EXPECT_EQ(settings.GetResolution(), glm::dvec2(640.0, 480.0));
}

TEST(LoadScene, ExistingPrimsKeepTheirType) {
    Scene scene;
    LoadScene(scene, kDescription);
    LoadScene(scene, R"({"/World/Rock": {"radius": 0.75}, "/World/Sun": {"angle": 1.0}})");
    EXPECT_EQ(scene.TypeOf("/World/Rock"), "Sphere");
    EXPECT_EQ(ernie::Sphere(scene, "/World/Rock").GetRadius(), 0.75);
    EXPECT_EQ(ernie::DistantLight(scene, "/World/Sun").GetAngle(), 1.0);
    EXPECT_EQ(ernie::DistantLight(scene, "/World/Sun").GetIntensity(), 2.0);
}

TEST(LoadScene, SetsTransforms) {
    Scene scene;
    LoadScene(scene, R"({
      "/A": {"translate": [1, 2, 3], "orient": [0.7071067811865476, 0, 0, 0.7071067811865476],
             "scale": 2},
      "/B": {"rotate": [0, 0, 90], "rotation_order": "ZYX", "scale": [1, 2, 3]},
      "/C": {"transform": [[1, 0, 0, 4], [0, 1, 0, 5], [0, 0, 1, 6], [0, 0, 0, 1]]},
      "/D": {"untyped": "custom", "user:count": 3, "user:tags": ["a", "b"]}
    })");

    // [w, x, y, z]: a quarter turn about +z.
    const glm::dquat quarter = glm::angleAxis(glm::radians(90.0), glm::dvec3(0.0, 0.0, 1.0));
    ExpectNear(scene.LocalTransform("/A"),
               glm::translate(glm::dmat4(1.0), glm::dvec3(1.0, 2.0, 3.0)) *
                   glm::mat4_cast(quarter) * glm::scale(glm::dmat4(1.0), glm::dvec3(2.0)));
    EXPECT_NE(scene.Get("/B", "xformOp:rotateZYX"), nullptr);
    EXPECT_EQ(scene.Get<glm::dvec3>("/B", tokens::kScale, {}), glm::dvec3(1.0, 2.0, 3.0));
    // Rows: the translation is the last column.
    EXPECT_EQ(scene.LocalTransform("/C")[3], glm::dvec4(4.0, 5.0, 6.0, 1.0));

    // Untyped prims take any attribute, converted as Python converts it.
    EXPECT_EQ(scene.TypeOf("/D"), "");
    EXPECT_EQ(scene.Get<std::string>("/D", "untyped", ""), "custom");
    EXPECT_EQ(scene.Get<int>("/D", "user:count", 0), 3);
    EXPECT_EQ(scene.Get<std::vector<std::string>>("/D", "user:tags", {}),
              (std::vector<std::string>{"a", "b"}));
}

TEST(LoadScene, LoadsAFileThatResolvesForRendering) {
    Scene scene;
    ernie::LoadSceneFile(scene, std::filesystem::path(ERNIE_TEST_FIXTURES) / "scene_basic.json");

    const ernie::RenderPrims prims = ernie::ResolveRenderPrims(scene, "/Render/Settings");
    EXPECT_EQ(prims.camera, Path("/World/Camera"));
    EXPECT_EQ(prims.light, Path("/World/Sun"));
    EXPECT_EQ(prims.ground, Path("/World/Ground"));
    EXPECT_TRUE(ernie::GroundPlane(scene, "/World/Ground").GetChecker());

    const ernie::FrameParams frame = ernie::FrameParamsOf(scene, prims, 640, 480);
    ExpectNear(frame.camera.transform,
               ernie::LookAtTransform({0.0, -10.0, 5.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}));
    EXPECT_EQ(frame.camera.focalLengthMm, 35.0);
    EXPECT_NEAR(frame.camera.nearPlane, 0.1, 1e-12);
    EXPECT_GT(frame.sunDirection.z, 0.0f);
    EXPECT_FLOAT_EQ(frame.ambient, 0.2f);
}

TEST(LoadScene, ErrorsNameThePrimAndTheKey) {
    const std::string syntax = ErrorOf("{\n  \"/X\": {\"type\": \"Sphere\",}\n}");
    EXPECT_NE(syntax.find("does not parse"), std::string::npos) << syntax;
    EXPECT_NE(syntax.find("line 2"), std::string::npos) << syntax;

    const std::string misspelt = ErrorOf(R"({"/X": {"type": "Sphere", "radus": 1.0}})");
    EXPECT_NE(misspelt.find("'/X', \"radus\""), std::string::npos) << misspelt;
    EXPECT_NE(misspelt.find("did you mean 'radius'"), std::string::npos) << misspelt;

    const std::string wrongType = ErrorOf(R"({"/X": {"type": "Sphere", "radius": "big"}})");
    EXPECT_NE(wrongType.find("'/X', \"radius\""), std::string::npos) << wrongType;

    const std::string child =
        ErrorOf(R"({"/X": {"Y": {"type": "DistantLight", "angle": [1, 2]}}})");
    EXPECT_NE(child.find("'/X/Y', \"angle\""), std::string::npos) << child;

    EXPECT_NE(ErrorOf("[1, 2]").find("object of prim paths"), std::string::npos);
    EXPECT_NE(ErrorOf(R"({"/X": 1})").find("'/X'"), std::string::npos);
    EXPECT_NE(ErrorOf(R"({"/X": {"orient": [1, 0, 0]}})").find("[w, x, y, z]"), std::string::npos);
    EXPECT_NE(
        ErrorOf(R"({"/X": {"type": "Terrain", "heights": [[0, 1], [2]]}})").find("rectangular"),
        std::string::npos);
    EXPECT_NE(ErrorOf(R"({"/X": {"user:big": 4000000000}})").find("32 bits"), std::string::npos);

    Scene scene;
    EXPECT_THROW(ernie::LoadSceneFile(scene, std::filesystem::path(ERNIE_TEST_FIXTURES) /
                                                 "no_such_scene.json"),
                 std::invalid_argument);
}
