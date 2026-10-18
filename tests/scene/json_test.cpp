// SPDX-License-Identifier: BSD-3-Clause
/// @file json_test.cpp
/// @brief Tests of scene files: loading fixtures through the typed API, saving them back to their
/// golden files, and validation's reports on broken ones.
#include "ernie/scene/json.h"
#include "ernie/scene/reference.h"
#include "ernie/scene/schema/camera.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/mesh.h"
#include "ernie/scene/schema/shade.h"
#include "ernie/util/file.h"

#include "tests/common.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie {
namespace {

std::string Fixture(const char* name) {
    return ReadTextFile(test::FixturePath(std::string("scenes/") + name));
}

/// Compares text with a golden fixture; with ERNIE_UPDATE_GOLDEN set, rewrites the fixture.
void ExpectGolden(const std::string& text, const char* name) {
    if (std::getenv("ERNIE_UPDATE_GOLDEN") != nullptr) {
        WriteTextFile(test::FixturePath(std::string("scenes/") + name), text);
    }
    EXPECT_EQ(text, Fixture(name)) << name << " is stale: rerun with ERNIE_UPDATE_GOLDEN=1";
}

std::string Messages(const std::vector<SceneProblem>& problems) {
    std::string text;
    for (const SceneProblem& problem : problems) {
        text += problem.Message() + "\n";
    }
    return text;
}

TEST(Json, FixturesLoadThroughTheTypedApi) {
    Scene scene;
    LoadSceneFile(scene, test::FixturePath("scenes/rover.json"));
    EXPECT_EQ(Cylinder(scene, "/World/Rover/Wheel").GetAxis(), Axis::Y);
    EXPECT_EQ(MaterialBindingAPI(scene, "/World/Rover").GetMaterialBinding(), Path("/Looks/Paint"));
    EXPECT_EQ(BasisCurves(scene, "/World/Rover/Cable").GetType(), "linear")
        << "an attribute named type, apart from typeName";
    EXPECT_NEAR(scene.WorldTransform(*scene.Find("/World/Rover/Mast"))[3].z, 1.5, 1e-12)
        << "matrices are rows, translation last";
    EXPECT_EQ(scene.WorldTransform(*scene.Find("/World/Rover"))[3], glm::dvec4(2, -1, 0, 1));

    const Mesh body(scene, "/World/Rover/Body");
    EXPECT_EQ(body.GetPoints().Shape(), (std::vector<std::int64_t>{4, 3}));
    EXPECT_EQ(body.GetPoints().Type(), DType::Float32);
    EXPECT_EQ(body.GetFaceVertexIndices().Type(), DType::Int32) << "integer attributes";
    EXPECT_EQ(scene.Get("/Lighting", "ernie:note"),
              Value(std::string("custom data on a grouping")));
}

TEST(Json, SavedScenesMatchTheirGoldenFiles) {
    Scene scene;
    LoadScene(scene, Fixture("rover.json"));
    const std::string saved = SaveScene(scene);
    ExpectGolden(saved, "rover.saved.json");

    Scene again;
    LoadScene(again, saved);
    EXPECT_EQ(SaveScene(again), saved) << "saving is a fixed point";
}

TEST(Json, ValidScenesHaveNoProblems) {
    EXPECT_EQ(Messages(ValidateScene(Fixture("rover.json"))), "");
    EXPECT_EQ(Messages(ValidateScene(Fixture("rover.saved.json"))), "");
}

TEST(Json, ValidationReportsEveryProblemOfABrokenScene) {
    ExpectGolden(Messages(ValidateScene(Fixture("invalid/problems.json"))),
                 "invalid/problems.expected");
    EXPECT_TRUE(test::Contains(Messages(ValidateScene("{")), "Not JSON"));
    EXPECT_TRUE(test::Contains(Messages(ValidateScene(R"({"World": {}})")), "\"prims\""));
}

TEST(Json, LoadingStopsAtTheFirstProblem) {
    Scene scene;
    EXPECT_THROW(LoadScene(scene, Fixture("invalid/problems.json")), std::invalid_argument);
}

TEST(Json, ArraysAreHandedOverWithoutCopies) {
    Scene scene;
    const std::vector<float> xyz{0, 0, 0, 1, 1, 1};
    const Tensor points = Tensor::Copy<float>(xyz, {2, 3});
    LoadScene(scene, R"({"prims": {"/Cloud": {"typeName": "Points", "points": {"$array": 0}}}})",
              std::span(&points, 1));
    EXPECT_EQ(Points(scene, "/Cloud").GetPoints().Data(), points.Data());
    EXPECT_THROW(LoadScene(scene, R"({"prims": {"/Cloud": {"points": {"$array": 1}}}})",
                           std::span(&points, 1)),
                 std::invalid_argument);
}

TEST(Json, NonFiniteNumbersCannotBeSaved) {
    Scene scene;
    Sphere::Define(scene, "/Ball");
    scene.Set("/Ball", "primvars:ernie:heat", std::numeric_limits<double>::infinity());
    EXPECT_THROW(SaveScene(scene), std::invalid_argument);
}

}  // namespace
}  // namespace ernie
