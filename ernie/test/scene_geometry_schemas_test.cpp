// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tokens.h"
#include "ernie/util/tensor.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

using ernie::AttributeKind;
using ernie::Path;
using ernie::PrimTypeDef;
using ernie::Scene;
using ernie::Tensor;
using ernie::Value;
namespace tokens = ernie::tokens;

TEST(GeometrySchemas, ShapesReadUsdGeomFallbacks) {
    Scene scene;
    EXPECT_EQ(ernie::Cube::Define(scene, "/Cube").GetSize(), 2.0);
    EXPECT_EQ(ernie::Sphere::Define(scene, "/Sphere").GetRadius(), 1.0);
    const ernie::Cylinder cylinder = ernie::Cylinder::Define(scene, "/Cylinder");
    EXPECT_EQ(cylinder.GetRadius(), 1.0);
    EXPECT_EQ(cylinder.GetHeight(), 2.0);
    EXPECT_EQ(cylinder.GetAxis(), "Z");
    const ernie::Cone cone = ernie::Cone::Define(scene, "/Cone");
    EXPECT_EQ(cone.GetRadius(), 1.0);
    EXPECT_EQ(cone.GetHeight(), 2.0);
    const ernie::Capsule capsule = ernie::Capsule::Define(scene, "/Capsule");
    EXPECT_EQ(capsule.GetRadius(), 0.5);
    EXPECT_EQ(capsule.GetHeight(), 1.0);
    const ernie::Plane plane = ernie::Plane::Define(scene, "/Plane");
    EXPECT_EQ(plane.GetWidth(), 2.0);
    EXPECT_EQ(plane.GetLength(), 2.0);
    EXPECT_EQ(plane.GetAxis(), "Z");
    EXPECT_EQ(plane.GetDisplayColor(), glm::dvec3(0.5));
    EXPECT_TRUE(plane.GetMaterialBinding().IsEmpty());
    EXPECT_EQ(scene.TypeOf("/Plane"), tokens::kPlane);

    const ernie::GroundPlane ground = ernie::GroundPlane::Define(scene, "/Ground");
    EXPECT_EQ(ground.GetHeight(), 0.0);
    EXPECT_FALSE(ground.GetChecker());
    EXPECT_EQ(ground.GetCheckerSize(), 1.0);
    EXPECT_EQ(ground.GetCheckerColor0(), glm::dvec3(1.0));
    EXPECT_EQ(ground.GetCheckerColor1(), glm::dvec3(0.5));
    EXPECT_EQ(ernie::Points::Define(scene, "/Cloud").GetWidths(), 1.0);
    EXPECT_EQ(ernie::Mesh::Define(scene, "/Mesh").GetPoints(), nullptr);
    EXPECT_EQ(ernie::PointInstancer::Define(scene, "/Instancer").GetProtoIndices(), nullptr);
    EXPECT_NO_THROW(ernie::Volume::Define(scene, "/Volume"));
}

TEST(GeometrySchemas, ViewsSetByUsdName) {
    Scene scene;
    ernie::Capsule capsule = ernie::Capsule::Define(scene, "/Capsule");
    capsule.SetRadius(0.25).SetHeight(3.0).SetAxis("X");
    EXPECT_TRUE(capsule.HasRadius());
    EXPECT_FALSE(ernie::Capsule(scene, "/Capsule").HasDisplayColor());
    EXPECT_EQ(*scene.Get("/Capsule", tokens::kGeomRadius), Value(0.25));
    EXPECT_EQ(*scene.Get("/Capsule", tokens::kGeomHeight), Value(3.0));
    EXPECT_EQ(capsule.GetAxis(), "X");
    // Authored as another numeric type, it still reads as the declared one.
    scene.Define("/Cube", tokens::kCube).Set(tokens::kSize, 3);
    EXPECT_EQ(ernie::Cube(scene, "/Cube").GetSize(), 3.0);

    ernie::GroundPlane ground = ernie::GroundPlane::Define(scene, "/Ground");
    ground.SetHeight(-2.5).SetChecker(true).SetCheckerColor0(glm::dvec3(0.2, 0.3, 0.4));
    ground.SetMaterialBinding("/Looks/Grass");
    EXPECT_EQ(*scene.Get("/Ground", tokens::kGroundHeight), Value(-2.5));
    EXPECT_TRUE(ground.GetChecker());
    EXPECT_EQ(ground.GetCheckerColor0(), glm::dvec3(0.2, 0.3, 0.4));
    EXPECT_EQ(ground.GetMaterialBinding(), Path("/Looks/Grass"));

    // Bulk arrays are tensors, written with `Scene::Write`.
    const std::array<float, 9> points{0, 0, 0, 1, 0, 0, 0, 1, 0};
    const std::array<std::int64_t, 1> counts{3};
    ernie::Mesh mesh = ernie::Mesh::Define(scene, "/Mesh");
    mesh.SetPoints(Tensor::Copy<float>(points, {3, 3}))
        .SetFaceVertexCounts(Tensor::Copy<std::int64_t>(counts, {1}));
    ASSERT_NE(mesh.GetPoints(), nullptr);
    EXPECT_EQ(mesh.GetPoints()->Elements(), 9);
    EXPECT_EQ(scene.Read("/Mesh", tokens::kFaceVertexCounts), mesh.GetFaceVertexCounts());
    EXPECT_TRUE(mesh.HasPoints());
    EXPECT_FALSE(mesh.HasNormals());

    ernie::Points cloud = ernie::Points::Define(scene, "/Cloud");
    cloud.SetWidths(0.1);
    EXPECT_EQ(*scene.Get("/Cloud", tokens::kWidths), Value(0.1));
}

TEST(GeometrySchemas, ViewsCheckTypes) {
    Scene scene;
    ernie::Cylinder::Define(scene, "/Cylinder");
    ernie::Mesh::Define(scene, "/Mesh");
    ernie::Points::Define(scene, "/Cloud");
    ernie::GroundPlane::Define(scene, "/Ground");
    EXPECT_NO_THROW(ernie::AxisShape(scene, "/Cylinder"));
    EXPECT_NO_THROW(ernie::Gprim(scene, "/Cylinder"));
    EXPECT_THROW(ernie::Cone(scene, "/Cylinder"), std::invalid_argument);
    EXPECT_NO_THROW(ernie::PointBased(scene, "/Mesh"));
    EXPECT_NO_THROW(ernie::PointBased(scene, "/Cloud"));
    EXPECT_THROW(ernie::AxisShape(scene, "/Mesh"), std::invalid_argument);
    EXPECT_THROW(ernie::Gprim(scene, "/Ground"), std::invalid_argument);
    EXPECT_NO_THROW(ernie::Imageable(scene, "/Ground"));
    EXPECT_THROW(ernie::Mesh(scene, "/Missing"), std::invalid_argument);

    const ernie::AxisShape shape(scene, "/Cylinder");
    EXPECT_EQ(shape.PrimType().name, "Cylinder");
    EXPECT_EQ(shape.GetAxis(), "Z");
    const Scene& constScene = scene;
    ernie::Cylinder readOnly(constScene, "/Cylinder");
    EXPECT_EQ(readOnly.GetHeight(), 2.0);
    EXPECT_THROW(readOnly.SetHeight(1.0), std::logic_error);
}

TEST(GeometrySchemas, RegistryDescribesTheTypes) {
    const PrimTypeDef& capsule = ernie::GetPrimType("Capsule");
    EXPECT_FALSE(capsule.abstract);
    ASSERT_NE(capsule.base, nullptr);
    EXPECT_EQ(capsule.base->name, "AxisShape");
    EXPECT_TRUE(capsule.base->abstract);
    EXPECT_TRUE(capsule.IsA(ernie::GetPrimType("Gprim")));
    EXPECT_EQ(capsule.Find(tokens::kGeomRadius)->fallback, Value(0.5));
    EXPECT_EQ(ernie::GetPrimType("Cylinder").Find(tokens::kGeomRadius)->fallback, Value(1.0));
    const ernie::AttributeDef* axis = capsule.Find(tokens::kAxis);
    ASSERT_NE(axis, nullptr);
    EXPECT_EQ(axis->kind, AttributeKind::Token);
    EXPECT_EQ(axis->meta.options.size(), 3u);
    EXPECT_EQ(capsule.Find(tokens::kRadius), nullptr) << "a light's inputs:radius";

    const PrimTypeDef& mesh = ernie::GetPrimType(tokens::kMesh);
    EXPECT_EQ(mesh.base->name, "PointBased");
    EXPECT_EQ(mesh.Find(tokens::kPoints)->kind, AttributeKind::Tensor);
    EXPECT_EQ(mesh.FindByPyName("face_vertex_indices")->name, tokens::kFaceVertexIndices);
    EXPECT_EQ(mesh.Find(tokens::kDisplayColor)->kind, AttributeKind::Vec3);
    const PrimTypeDef& points = ernie::GetPrimType(tokens::kGeomPoints);
    EXPECT_TRUE(points.IsA(ernie::GetPrimType("PointBased")));
    EXPECT_EQ(points.Find(tokens::kWidths)->fallback, Value(1.0));

    const PrimTypeDef& instancer = ernie::GetPrimType(tokens::kPointInstancer);
    EXPECT_FALSE(instancer.IsA(ernie::GetPrimType("Gprim")));
    EXPECT_EQ(instancer.Find(tokens::kOrientations)->kind, AttributeKind::Tensor);
    EXPECT_EQ(instancer.Find(tokens::kPrototypes), nullptr) << "a list of paths, read by name";
    EXPECT_TRUE(ernie::GetPrimType(tokens::kVolume).attributes.empty());

    const PrimTypeDef& ground = ernie::GetPrimType(tokens::kGroundPlane);
    EXPECT_EQ(ground.base->name, "Imageable");
    EXPECT_EQ(ground.Find(tokens::kCheckerColor1)->fallback, Value(glm::dvec3(0.5)));
    EXPECT_EQ(ground.Find(tokens::kChecker)->label, "Checkerboard");
    EXPECT_EQ(ground.Find(tokens::kMaterialBinding)->kind, AttributeKind::Relationship);
    EXPECT_EQ(ground.FindByPyName("checker_size")->name, tokens::kCheckerSize);
}
