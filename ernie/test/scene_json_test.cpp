// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/scene_json.h"

#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/resolve.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using ernie::DType;
using ernie::LoadScene;
using ernie::Path;
using ernie::Scene;
using ernie::Tensor;
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

namespace {

/// The elements of a CPU tensor as doubles, in row-major order.
std::vector<double> NumbersOf(const ernie::Tensor& tensor) {
    const ernie::Tensor contiguous = tensor.Contiguous();
    std::vector<double> numbers;
    for (std::int64_t i = 0; i < contiguous.Elements(); ++i) {
        const auto index = static_cast<std::size_t>(i);
        switch (contiguous.Type()) {
        case DType::Int32:
            numbers.push_back(contiguous.As<std::int32_t>()[index]);
            break;
        case DType::Float32:
            numbers.push_back(contiguous.As<float>()[index]);
            break;
        case DType::Float64:
            numbers.push_back(contiguous.As<double>()[index]);
            break;
        default:
            ADD_FAILURE() << "unexpected element type";
        }
    }
    return numbers;
}

bool IsSession(const Path& path) {
    return path.String().starts_with(tokens::kSessionRoot);
}

/// Expects `copy` to hold the prims of `scene` outside the session, of the same types, with the
/// same attributes: declared values equal as their declared types, others equal, and tensors of
/// the same shape and numbers.
void ExpectEquivalent(const Scene& scene, const Scene& copy) {
    std::vector<Path> prims;
    for (const Path& path : scene.Prims()) {
        if (!IsSession(path)) {
            prims.push_back(path);
        }
    }
    ASSERT_EQ(copy.Prims(), prims);
    for (const Path& path : prims) {
        SCOPED_TRACE(path.String());
        EXPECT_EQ(copy.TypeOf(path), scene.TypeOf(path));
        ASSERT_EQ(copy.Attributes(path), scene.Attributes(path));
        const ernie::PrimTypeDef* type = ernie::FindPrimType(scene.TypeOf(path));
        for (const std::string& name : scene.Attributes(path)) {
            SCOPED_TRACE(name);
            if (const ernie::Tensor* tensor = scene.Read(path, name)) {
                const ernie::Tensor* loaded = copy.Read(path, name);
                ASSERT_NE(loaded, nullptr);
                EXPECT_EQ(loaded->Shape(), tensor->Shape());
                EXPECT_EQ(NumbersOf(*loaded), NumbersOf(*tensor));
                continue;
            }
            const ernie::Value* value = scene.Get(path, name);
            const ernie::Value* loaded = copy.Get(path, name);
            ASSERT_NE(value, nullptr);
            ASSERT_NE(loaded, nullptr);
            const ernie::AttributeDef* def = type != nullptr ? type->Find(name) : nullptr;
            if (def != nullptr && !def->meta.lenient) {
                EXPECT_EQ(def->Coerce(*loaded), def->Coerce(*value));
            } else {
                EXPECT_EQ(*loaded, *value);
            }
        }
    }
}

/// A scene with prims of every family, transforms, relationships, tensors and the viewer's
/// session prims.
Scene RichScene() {
    Scene scene;
    scene.Define("/World", "Xform");
    ernie::XformOps ops;
    ops.translate = glm::dvec3(1.0, 2.0, 3.0);
    ops.orient = glm::angleAxis(glm::radians(30.0), glm::normalize(glm::dvec3(1.0, 2.0, 3.0)));
    ops.scale = glm::dvec3(0.5, 2.0, 1.25);
    ernie::SetXform(scene, "/World", ops);

    // Geometry: shapes, a mesh with per-vertex colours, an instancer with a list of prototypes.
    ernie::Sphere::Define(scene, "/World/Rock").SetRadius(0.1).SetMaterialBinding("/Looks/Stone");
    const float points[] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.1f};
    const int counts[] = {3};
    const int indices[] = {0, 1, 2};
    const float colors[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.25f};
    ernie::Mesh::Define(scene, "/World/Quad")
        .SetPoints(ernie::Tensor::Copy(std::span<const float>(points), {3, 3}))
        .SetFaceVertexCounts(ernie::Tensor::Copy(std::span<const int>(counts), {1}))
        .SetFaceVertexIndices(ernie::Tensor::Copy(std::span<const int>(indices), {3}));
    scene.Write("/World/Quad", tokens::kDisplayColor,
                ernie::Tensor::Copy(std::span<const float>(colors), {3, 3}));
    ernie::Cube::Define(scene, "/World/Instancer/Box").SetSize(0.25);
    ernie::Sphere::Define(scene, "/World/Instancer/Ball");
    const float positions[] = {0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f};
    const int proto[] = {0, 1};
    ernie::PointInstancer::Define(scene, "/World/Instancer")
        .SetPrototypes({Path("/World/Instancer/Box"), Path("/World/Instancer/Ball")})
        .SetPositions(ernie::Tensor::Copy(std::span<const float>(positions), {2, 3}))
        .SetProtoIndices(ernie::Tensor::Copy(std::span<const int>(proto), {2}));

    // A matrix transform, lights, a camera, a terrain.
    scene.Define("/World/Pivot", "Xform");
    glm::dmat4 matrix(1.0);
    matrix[3] = glm::dvec4(4.0, 5.0, 6.0, 1.0);
    matrix[0][1] = 0.3;
    scene.Set("/World/Pivot", tokens::kTransform, matrix);
    scene.Set("/World/Pivot", tokens::kXformOpOrder,
              std::vector<std::string>{std::string(tokens::kTransform)});
    ernie::DistantLight::Define(scene, "/World/Sun")
        .SetAngle(0.53)
        .SetIntensity(2.0)
        .SetColor(glm::dvec3(1.0, 0.9, 0.8));
    ernie::SphereLight::Define(scene, "/World/Lamp").SetRadius(0.2);
    ernie::Camera::Define(scene, "/World/Camera")
        .SetFocalLength(35.0)
        .SetClippingRange(glm::dvec2(0.1, 1000.0));
    const float heights[] = {0.0f, 1.0f, 2.0f, 3.5f, 4.0f, 5.0f};
    ernie::Terrain::Define(scene, "/World/Ground")
        .SetHeights(ernie::Tensor::Copy(std::span<const float>(heights), {2, 3}))
        .SetCellSize(2.0);

    // Render settings: one camera, a list of products, a lenient bool wireframe.
    ernie::RenderProduct::Define(scene, "/Render/Product");
    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetCamera("/World/Camera")
        .SetProducts({Path("/Render/Product")})
        .SetResolution(glm::dvec2(640.0, 480.0));
    scene.Set("/Render/Settings", tokens::kWireframe, true);

    // An unregistered material and an untyped prim of undeclared values and an undeclared tensor.
    scene.Define("/Looks/Stone", "Material");
    scene.Define("/Custom", "");
    scene.Set("/Custom", "user:count", 3);
    scene.Set("/Custom", "user:speed", 2.5);
    scene.Set("/Custom", "user:flag", true);
    scene.Set("/Custom", "user:size", glm::ivec2(4, 5));
    scene.Set("/Custom", "user:dir", glm::dvec3(0.1, 0.2, 0.3));
    scene.Set("/Custom", "user:tint", glm::vec4(0.25f, 0.5f, 0.75f, 1.0f));
    scene.Set("/Custom", "user:name", std::string("rock \"one\""));
    scene.Set("/Custom", "user:tags", std::vector<std::string>{"a", "b"});
    scene.Set("/Custom", "user:frame", matrix);
    const float samples[] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
    scene.Write("/Custom", "user:samples",
                ernie::Tensor::Copy(std::span<const float>(samples), {3, 2}));
    scene.Define("/Custom/Empty", "");

    // The viewer's own prims.
    ernie::Camera::Define(scene, Path(std::string(tokens::kSessionRoot) + "/Perspective"));
    return scene;
}

}  // namespace

TEST(SaveScene, RoundTripsThroughLoadScene) {
    const Scene scene = RichScene();
    const std::string text = ernie::SaveScene(scene);
    Scene loaded;
    LoadScene(loaded, text);
    ExpectEquivalent(scene, loaded);
    EXPECT_EQ(ernie::SaveScene(loaded), text) << "saving what loads gives the same text";

    EXPECT_FALSE(loaded.Has(Path(tokens::kSessionRoot)));
    EXPECT_EQ(text.find("__session"), std::string::npos);
    ExpectNear(loaded.LocalTransform("/World"), scene.LocalTransform("/World"));
    ExpectNear(loaded.LocalTransform("/World/Pivot"), scene.LocalTransform("/World/Pivot"));
    EXPECT_TRUE(loaded.Get("/World", tokens::kOrient)->Is<glm::dquat>());
    EXPECT_EQ(loaded.Read("/World/Quad", tokens::kDisplayColor)->Type(), DType::Float32);
    EXPECT_EQ(loaded.Read("/World/Quad", "faceVertexIndices")->Type(), DType::Int32);
    EXPECT_EQ(ernie::PointInstancer(loaded, "/World/Instancer").GetPrototypes(),
              (std::vector<Path>{"/World/Instancer/Box", "/World/Instancer/Ball"}));
    EXPECT_EQ(ernie::Sphere(loaded, "/World/Rock").GetMaterialBinding(), Path("/Looks/Stone"));
    EXPECT_TRUE(loaded.Get("/Render/Settings", tokens::kWireframe)->Is<bool>());
}

/// A value of `def`'s kind other than its default, to see it come back.
ernie::Value SampleOf(const ernie::AttributeDef& def) {
    using ernie::AttributeKind;
    switch (def.kind) {
    case AttributeKind::Bool:
        return !*def.fallback.Get<bool>();
    case AttributeKind::Int:
        return *def.fallback.Get<int>() + 1;
    case AttributeKind::Double:
        return *def.fallback.Get<double>() + 0.25;
    case AttributeKind::Vec2:
        return glm::dvec2(0.25, 0.5);
    case AttributeKind::Vec3:
        return glm::dvec3(0.25, 0.5, 0.75);
    case AttributeKind::Vec4:
        return glm::dvec4(0.25, 0.5, 0.75, 0.1);
    case AttributeKind::Token:
        return def.meta.options.empty() ? std::string("sample")
                                        : std::string(def.meta.options.back());
    case AttributeKind::Relationship:
        return Path("/Target");
    case AttributeKind::Relationships:
        return std::vector<Path>{"/Target", "/Other"};
    case AttributeKind::FloatArray:
        return std::vector<float>{0.25f, 0.5f, 0.75f, 1.0f, 1.25f};
    case AttributeKind::Quat:
        return glm::normalize(glm::dquat(0.5, 0.5, -0.5, 0.25));
    case AttributeKind::Matrix:
        return glm::translate(glm::dmat4(1.0), glm::dvec3(1.0, 2.0, 3.0));
    case AttributeKind::Tensor:
        break;
    }
    return {};
}

TEST(SaveScene, EveryDeclaredAttributeRoundTripsAsItsKind) {
    // One prim of every registered type with every attribute it declares set, each array
    // attribute's array form on a second prim, and every transform op: what loads back is what
    // was saved, by kind, not by how the JSON looks.
    Scene scene;
    for (const ernie::PrimTypeDef* type : ernie::PrimTypes()) {
        if (type->abstract) {
            continue;
        }
        const Path path("/" + std::string(type->name));
        const Path arrays("/" + std::string(type->name) + "Arrays");
        scene.Define(path, type->name);
        scene.Define(arrays, type->name);
        for (const ernie::AttributeDef* def : type->AllAttributes()) {
            SCOPED_TRACE(std::string(type->name) + " " + std::string(def->name));
            if (def->kind == ernie::AttributeKind::Tensor) {
                scene.Write(
                    path, def->name,
                    def->meta.integer
                        ? Tensor::Copy<std::int32_t>(std::array<std::int32_t, 3>{1, 2, 3}, {3})
                        : Tensor::Copy<float>(std::array<float, 6>{1, 2, 3, 4, 5, 6.5f}, {2, 3}));
                continue;
            }
            scene.Set(path, def->name, SampleOf(*def));
            if (def->meta.array) {
                scene.Write(arrays, def->name,
                            Tensor::Copy<float>(std::array<float, 6>{1, 2, 3, 4, 5, 6.5f}, {2, 3}));
            }
        }
    }
    scene.Define("/Ops", "Xform");
    scene.Set("/Ops", tokens::kTranslate, glm::dvec3(1.0, 2.0, 3.0));
    scene.Set("/Ops", std::string(tokens::kOrient) + ":pivot",
              glm::normalize(glm::dquat(0.5, 0.5, -0.5, 0.25)));
    scene.Set("/Ops", tokens::kRotateX, 30.0);
    scene.Set("/Ops", tokens::kRotateZYX, glm::dvec3(10.0, 20.0, 30.0));
    scene.Set("/Ops", tokens::kScale, glm::dvec3(2.0));
    scene.Set("/Ops", tokens::kTransform, glm::translate(glm::dmat4(1.0), glm::dvec3(1.0)));
    scene.Write("/Ops", tokens::kOrient,
                Tensor::Copy<float>(std::array<float, 8>{0, 0, 0, 1, 0, 0, 1, 0}, {2, 4}));

    const std::string text = ernie::SaveScene(scene);
    Scene loaded;
    LoadScene(loaded, text);
    ExpectEquivalent(scene, loaded);
    EXPECT_EQ(ernie::SaveScene(loaded), text);
    EXPECT_TRUE(loaded.Get("/Ops", std::string(tokens::kOrient) + ":pivot")->Is<glm::dquat>());
    EXPECT_TRUE(loaded.Get("/Ops", tokens::kTransform)->Is<glm::dmat4>());
    EXPECT_EQ(loaded.Read("/MeshArrays", tokens::kDisplayColor)->Shape(),
              (std::vector<std::int64_t>{2, 3}));
}

TEST(SaveScene, ArraysAndValuesAreCheckedByKind) {
    Scene scene;
    scene.Define("/Mesh", "Mesh");
    scene.Define("/Group", "");
    const Tensor rows = Tensor::Copy<float>(std::array<float, 6>{1, 2, 3, 4, 5, 6}, {2, 3});
    // An array in place of a value only where one is declared to take it.
    EXPECT_NO_THROW(scene.Write("/Mesh", tokens::kDisplayColor, rows));
    EXPECT_THROW(scene.Write("/Mesh", tokens::kMaterialBinding, rows), std::invalid_argument);
    // Transform ops by their kind, on untyped prims too: four numbers are not a quaternion.
    EXPECT_THROW(scene.Set("/Group", tokens::kOrient, glm::vec4(0, 0, 0, 1)),
                 std::invalid_argument);
    EXPECT_THROW(scene.Set("/Group", "xformOp:translate:pivot", 1.0), std::invalid_argument);
    EXPECT_NO_THROW(scene.Set("/Group", tokens::kScale, 2.0));
    EXPECT_THROW(LoadScene(scene, R"({"/Group": {"xformOp:orient": [1, 0, 0]}})"),
                 std::invalid_argument);
    LoadScene(scene, R"({"/Group": {"xformOp:orient:pivot": [1, 0, 0, 0]}})");
    EXPECT_EQ(*scene.Get("/Group", "xformOp:orient:pivot")->Get<glm::dquat>(),
              glm::dquat(1, 0, 0, 0));
}

TEST(SaveScene, UntypedPrimsAreCheckedForWhatEveryPrimHas) {
    Scene scene;
    scene.Define("/Group", "");
    scene.Define("/Custom", "Fiducial");  // A type that is not registered.
    for (const char* path : {"/Group", "/Custom"}) {
        SCOPED_TRACE(path);
        // Imageable's attributes, which Ernie reads on every prim, by their kind.
        EXPECT_THROW(scene.Set(path, tokens::kWorld, std::string("two")), std::invalid_argument);
        EXPECT_THROW(scene.Set(path, tokens::kVisibility, 0.5), std::invalid_argument);
        EXPECT_NO_THROW(scene.Set(path, tokens::kWorld, 2));
        EXPECT_NO_THROW(scene.Set(path, tokens::kVisibility, std::string(tokens::kInvisible)));
        // Any other name, as before.
        EXPECT_NO_THROW(scene.Set(path, "radius", std::string("big")));
        EXPECT_NO_THROW(scene.Set(path, tokens::kDisplayColor, glm::dvec4(1.0)));
    }
}

TEST(LoadScene, TakesArraysBesideTheJsonWithoutCopying) {
    const Tensor points =
        Tensor::Copy<float>(std::array<float, 9>{0, 0, 0, 1, 0, 0, 0, 1, 0}, {3, 3});
    const Tensor colors = Tensor::Copy<double>(std::array<double, 6>{1, 0, 0, 0, 1, 0}, {2, 3});
    const Tensor up = Tensor::Copy<double>(std::array<double, 3>{0, 0, 2}, {3});
    const std::array<Tensor, 3> arrays{points, colors, up};
    Scene scene;
    LoadScene(scene,
              R"({"/Mesh": {"type": "Mesh", "points": {"$array": 0},
                            "display_color": {"$array": 1}, "translate": {"$array": 2},
                            "user:offset": {"$array": 2}}})",
              arrays);
    // Where an array goes, written as it is: shared, its element type kept.
    EXPECT_EQ(scene.Read("/Mesh", tokens::kPoints)->Data(), points.Data());
    EXPECT_EQ(scene.Read("/Mesh", tokens::kDisplayColor)->Type(), DType::Float64);
    // Elsewhere, read as the numbers it holds.
    EXPECT_EQ(*scene.Get("/Mesh", tokens::kTranslate)->Get<glm::dvec3>(), glm::dvec3(0, 0, 2));
    EXPECT_EQ(*scene.Get("/Mesh", "user:offset")->Get<glm::dvec3>(), glm::dvec3(0, 0, 2));
    try {
        LoadScene(scene, R"({"/Mesh": {"points": {"$array": 3}}})", arrays);
        FAIL() << "an index past the arrays loads";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("of the 3 arrays"), std::string::npos)
            << error.what();
    }
}

TEST(SaveScene, WritesStablePrettyJson) {
    Scene scene;
    ernie::Sphere::Define(scene, "/B").SetRadius(0.1);
    scene.Define("/A/Child", "");
    scene.Set("/A", "user:z", 1);
    scene.Set("/A", "user:a", glm::vec3(0.1f, 0.2f, 0.3f));
    EXPECT_EQ(ernie::SaveScene(scene), R"({
  "/A": {
    "user:a": [0.1, 0.2, 0.3],
    "user:z": 1,
    "Child": {}
  },
  "/B": {
    "type": "Sphere",
    "radius": 0.1
  }
}
)");
    EXPECT_EQ(ernie::SaveScene(Scene()), "{}\n");
}

TEST(SaveScene, FilesRoundTripAndErrorsNameThePrimAndAttribute) {
    const Scene scene = RichScene();
    const std::filesystem::path file =
        std::filesystem::path(::testing::TempDir()) / "ernie_save_scene.json";
    ernie::SaveSceneFile(scene, file);
    Scene loaded;
    ernie::LoadSceneFile(loaded, file);
    ExpectEquivalent(scene, loaded);
    std::filesystem::remove(file);
    EXPECT_THROW(ernie::SaveSceneFile(scene, std::filesystem::path(::testing::TempDir()) /
                                                 "no_such_dir" / "scene.json"),
                 std::invalid_argument);

    const auto errorOf = [](const Scene& broken) -> std::string {
        try {
            ernie::SaveScene(broken);
        } catch (const std::invalid_argument& error) {
            return error.what();
        }
        return {};
    };

    Scene gpu;
    float dummy = 0.0f;
    ernie::Mesh::Define(gpu, "/G").SetPoints(ernie::Tensor(
        &dummy, DType::Float32, {1, 3}, {}, ernie::Device{ernie::DeviceKind::Cuda, 0}));
    const std::string gpuError = errorOf(gpu);
    EXPECT_NE(gpuError.find("'/G', \"points\""), std::string::npos) << gpuError;
    EXPECT_NE(gpuError.find("GPU"), std::string::npos) << gpuError;

    Scene nan;
    nan.Define("/N", "");
    nan.Set("/N", "user:x", std::numeric_limits<double>::quiet_NaN());
    EXPECT_NE(errorOf(nan).find("'/N', \"user:x\""), std::string::npos);

    Scene misread;
    misread.Define("/M", "");
    misread.Set("/M", "Child", 1);
    misread.Define("/M/Child", "");
    EXPECT_NE(errorOf(misread).find("'/M/Child'"), std::string::npos) << errorOf(misread);
    Scene transformKey;
    transformKey.Define("/T", "");
    transformKey.Set("/T", "scale", 2.0);
    EXPECT_NE(errorOf(transformKey).find("'/T', \"scale\""), std::string::npos);
}
