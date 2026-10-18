// SPDX-License-Identifier: BSD-3-Clause
/// @file json_test.cpp
/// @brief Tests of scene files: loading through the typed API, and saving back exactly.
#include "ernie/scene/json.h"
#include "ernie/scene/schemas/geom.h"
#include "ernie/scene/schemas/mesh.h"
#include "ernie/scene/schemas/shade.h"
#include "ernie/scene/xform.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie {
namespace {

constexpr std::string_view kScene = R"({
  "prims": {
    "/World": {"type": "Xform", "xform": {"translate": [0, 0, 1], "scale": 2}},
    "/World/Rock": {
      "type": "Sphere", "apiSchemas": ["MaterialBindingAPI"],
      "radius": 0.5, "material:binding": "/Looks/Basalt", "primvars:displayColor": [0.2, 0.3, 0.4]
    },
    "/World/Tri": {
      "type": "Mesh",
      "points": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
      "faceVertexCounts": [3], "faceVertexIndices": [0, 1, 2],
      "xformOp:transform": [[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      "xformOpOrder": ["xformOp:transform"]
    },
    "/World/Group": {"ernie:note": "rocks", "ernie:count": 3, "ernie:offset": [1, 2]}
  }
})";

TEST(Json, LoadsPrimsThroughTheTypedApi) {
    Scene scene;
    LoadScene(scene, kScene);
    EXPECT_EQ(Sphere(scene, "/World/Rock").GetRadius(), 0.5);
    EXPECT_EQ(MaterialBindingAPI(scene, "/World/Rock").GetMaterialBinding(), Path("/Looks/Basalt"));
    EXPECT_EQ(Gprim(scene, "/World/Rock").GetDisplayColor(), glm::dvec3(0.2, 0.3, 0.4));
    EXPECT_EQ(scene.WorldTransform(*scene.Find("/World/Rock"))[3], glm::dvec4(0, 0, 1, 1));
    EXPECT_EQ(scene.LocalTransform(*scene.Find("/World/Tri"))[3], glm::dvec4(5, 0, 0, 1))
        << "matrices are rows, translation last";

    const Mesh tri(scene, "/World/Tri");
    EXPECT_EQ(tri.GetPoints().Shape(), (std::vector<std::int64_t>{3, 3}));
    EXPECT_EQ(tri.GetPoints().Type(), DType::Float32);
    EXPECT_EQ(tri.GetFaceVertexIndices().Type(), DType::Int32) << "integer attributes";

    EXPECT_TRUE(scene.TypeName(*scene.Find("/World/Group")).empty());
    EXPECT_EQ(scene.Get("/World/Group", "ernie:note"), Value(std::string("rocks")));
    EXPECT_EQ(scene.Get("/World/Group", "ernie:count"), Value(3));
    EXPECT_EQ(scene.Get("/World/Group", "ernie:offset"), Value(glm::dvec2(1, 2)));
}

TEST(Json, SavedScenesLoadBackTheSame) {
    Scene scene;
    LoadScene(scene, kScene);
    scene.Set("/World/Group", "ernie:weights", std::vector<float>{0.1f, 0.25f});
    const std::string saved = SaveScene(scene);
    EXPECT_NE(saved.find("[0.1, 0.25]"), std::string::npos) << "floats in their shortest form";
    EXPECT_NE(saved.find(std::string(kSceneSchemaUrl)), std::string::npos);

    Scene again;
    LoadScene(again, saved);
    EXPECT_EQ(SaveScene(again), saved);
    EXPECT_EQ(again.Prims(), scene.Prims());
}

TEST(Json, ArraysAreHandedOverWithoutCopies) {
    Scene scene;
    const std::vector<float> xyz{0, 0, 0, 1, 1, 1};
    const Tensor points = Tensor::Copy<float>(xyz, {2, 3});
    LoadScene(scene, R"({"prims": {"/Cloud": {"type": "Points", "points": {"$array": 0}}}})",
              std::span(&points, 1));
    EXPECT_EQ(Points(scene, "/Cloud").GetPoints().Data(), points.Data());
    EXPECT_THROW(LoadScene(scene, R"({"prims": {"/Cloud": {"points": {"$array": 1}}}})",
                           std::span(&points, 1)),
                 std::invalid_argument);
}

TEST(Json, ErrorsNameThePrimAndTheKey) {
    Scene scene;
    const auto message = [&](std::string_view json) {
        try {
            LoadScene(scene, json);
        } catch (const std::invalid_argument& error) {
            return std::string(error.what());
        }
        return std::string();
    };
    EXPECT_NE(message("{").find("Not JSON"), std::string::npos);
    EXPECT_NE(message(R"({"World": {}})").find("\"prims\""), std::string::npos);
    EXPECT_NE(
        message(R"({"prims": {"/Ball": {"type": "Sphere", "radius": -1}}})").find("'/Ball' radius"),
        std::string::npos)
        << "out of range";
    EXPECT_NE(message(R"({"prims": {"/Ball": {"type": "Sphere", "colour": 1}}})").find("colour"),
              std::string::npos)
        << "undeclared";
    EXPECT_NE(message(R"({"prims": {"/Ball": {"type": "Teapot"}}})").find("'/Ball'"),
              std::string::npos);
    EXPECT_NE(message(R"({"prims": {"/Ball": {"xform": {"spin": 1}}}})").find("spin"),
              std::string::npos);
}

TEST(Json, NonFiniteNumbersCannotBeSaved) {
    Scene scene;
    Sphere::Define(scene, "/Ball");
    scene.Set("/Ball", "primvars:displayOpacity", 0.5);
    scene.Set("/Ball", "primvars:ernie:heat", std::numeric_limits<double>::infinity());
    EXPECT_THROW(SaveScene(scene), std::invalid_argument);
}

}  // namespace
}  // namespace ernie
