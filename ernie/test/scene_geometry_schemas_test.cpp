// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/resolve.h"
#include "ernie/util/tensor.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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
    EXPECT_EQ(ernie::Sphere::Define(scene, "/Sphere").GetRadius(),
              1.0);
    const ernie::Cylinder cylinder =
        ernie::Cylinder::Define(scene, "/Cylinder");
    EXPECT_EQ(cylinder.GetRadius(), 1.0);
    EXPECT_EQ(cylinder.GetHeight(), 2.0);
    EXPECT_EQ(cylinder.GetAxis(), "Z");
    const ernie::Cone cone = ernie::Cone::Define(scene, "/Cone");
    EXPECT_EQ(cone.GetRadius(), 1.0);
    EXPECT_EQ(cone.GetHeight(), 2.0);
    const ernie::Capsule capsule =
        ernie::Capsule::Define(scene, "/Capsule");
    EXPECT_EQ(capsule.GetRadius(), 0.5);
    EXPECT_EQ(capsule.GetHeight(), 1.0);
    const ernie::Plane plane = ernie::Plane::Define(scene, "/Plane");
    EXPECT_EQ(plane.GetWidth(), 2.0);
    EXPECT_EQ(plane.GetLength(), 2.0);
    EXPECT_EQ(plane.GetAxis(), "Z");
    EXPECT_EQ(plane.GetDisplayColor(), glm::dvec3(0.5));
    EXPECT_TRUE(plane.GetMaterialBinding().IsEmpty());
    EXPECT_EQ(scene.TypeOf("/Plane"), tokens::kPlane);

    const ernie::GroundPlane ground =
        ernie::GroundPlane::Define(scene, "/Ground");
    EXPECT_EQ(ground.GetHeight(), 0.0);
    EXPECT_FALSE(ground.GetChecker());
    EXPECT_EQ(ground.GetCheckerSize(), 1.0);
    EXPECT_EQ(ground.GetCheckerColor0(), glm::dvec3(1.0));
    EXPECT_EQ(ground.GetCheckerColor1(), glm::dvec3(0.5));
    EXPECT_EQ(ernie::Points::Define(scene, "/Cloud").GetWidths(),
              std::vector<float>{1.0f});
    EXPECT_EQ(ernie::Mesh::Define(scene, "/Mesh").GetPoints(),
              nullptr);
    EXPECT_EQ(ernie::PointInstancer::Define(scene, "/Instancer")
                  .GetProtoIndices(),
              nullptr);
    EXPECT_NO_THROW(ernie::Volume::Define(scene, "/Volume"));
}

TEST(GeometrySchemas, ViewsSetByUsdName) {
    Scene scene;
    ernie::Capsule capsule =
        ernie::Capsule::Define(scene, "/Capsule");
    capsule.SetRadius(0.25).SetHeight(3.0).SetAxis("X");
    EXPECT_TRUE(capsule.HasRadius());
    EXPECT_FALSE(ernie::Capsule(scene, "/Capsule").HasDisplayColor());
    EXPECT_EQ(*scene.Get("/Capsule", tokens::kGeomRadius),
              Value(0.25));
    EXPECT_EQ(*scene.Get("/Capsule", tokens::kGeomHeight),
              Value(3.0));
    EXPECT_EQ(capsule.GetAxis(), "X");
    // Authored as another numeric type, it still reads as the
    // declared one.
    scene.Define("/Cube", tokens::kCube).Set(tokens::kSize, 3);
    EXPECT_EQ(ernie::Cube(scene, "/Cube").GetSize(), 3.0);

    ernie::GroundPlane ground =
        ernie::GroundPlane::Define(scene, "/Ground");
    ground.SetHeight(-2.5).SetChecker(true).SetCheckerColor0(
        glm::dvec3(0.2, 0.3, 0.4));
    ground.SetMaterialBinding("/Looks/Grass");
    EXPECT_EQ(*scene.Get("/Ground", tokens::kGroundHeight),
              Value(-2.5));
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
    EXPECT_EQ(scene.Read("/Mesh", tokens::kFaceVertexCounts),
              mesh.GetFaceVertexCounts());
    EXPECT_TRUE(mesh.HasPoints());
    EXPECT_FALSE(mesh.HasNormals());

    ernie::Points cloud = ernie::Points::Define(scene, "/Cloud");
    cloud.SetWidths({0.1f});
    EXPECT_EQ(*scene.Get("/Cloud", tokens::kWidths),
              Value(std::vector<float>{0.1f}));
}

TEST(GeometrySchemas, ViewsCheckTypes) {
    Scene scene;
    ernie::Cylinder::Define(scene, "/Cylinder");
    ernie::Mesh::Define(scene, "/Mesh");
    ernie::Points::Define(scene, "/Cloud");
    ernie::GroundPlane::Define(scene, "/Ground");
    EXPECT_NO_THROW(ernie::AxisShape(scene, "/Cylinder"));
    EXPECT_NO_THROW(ernie::Gprim(scene, "/Cylinder"));
    EXPECT_THROW(ernie::Cone(scene, "/Cylinder"),
                 std::invalid_argument);
    EXPECT_NO_THROW(ernie::PointBased(scene, "/Mesh"));
    EXPECT_NO_THROW(ernie::PointBased(scene, "/Cloud"));
    EXPECT_THROW(ernie::AxisShape(scene, "/Mesh"),
                 std::invalid_argument);
    EXPECT_THROW(ernie::Gprim(scene, "/Ground"),
                 std::invalid_argument);
    EXPECT_NO_THROW(ernie::Imageable(scene, "/Ground"));
    EXPECT_THROW(ernie::Mesh(scene, "/Missing"),
                 std::invalid_argument);

    const ernie::AxisShape shape(scene, "/Cylinder");
    EXPECT_EQ(shape.PrimType().name, "Cylinder");
    EXPECT_EQ(shape.GetAxis(), "Z");
    const Scene& constScene = scene;
    ernie::Cylinder readOnly(constScene, "/Cylinder");
    EXPECT_EQ(readOnly.GetHeight(), 2.0);
    EXPECT_THROW(readOnly.SetHeight(1.0), std::logic_error);
}

TEST(GeometrySchemas, BoundablesDeclareTheirExtent) {
    Scene scene;
    ernie::Mesh mesh = ernie::Mesh::Define(scene, "/Mesh");
    ernie::PointInstancer instancer =
        ernie::PointInstancer::Define(scene, "/Instancer");
    ernie::GroundPlane::Define(scene, "/Ground");
    EXPECT_NO_THROW(ernie::Boundable(scene, "/Mesh"));
    EXPECT_NO_THROW(ernie::Boundable(scene, "/Instancer"));
    EXPECT_THROW(ernie::Boundable(scene, "/Ground"),
                 std::invalid_argument);
    EXPECT_TRUE(mesh.GetExtent().empty());

    // Six numbers as a value, or a (2, 3) array written in its place,
    // are the local bounds.
    mesh.SetExtent({-1, -2, -3, 1, 2, 3});
    ASSERT_TRUE(ernie::ExtentOf(scene, "/Mesh").has_value());
    EXPECT_EQ(ernie::ExtentOf(scene, "/Mesh")->first,
              glm::vec3(-1.0f, -2.0f, -3.0f));
    const std::array<float, 6> box{0, 0, 0, 4, 4, 4};
    instancer.Write(tokens::kExtent,
                    Tensor::Copy<float>(box, {2, 3}));
    EXPECT_TRUE(instancer.HasExtent());
    EXPECT_TRUE(instancer.GetExtent().empty())
        << "the declared value is not the array";
    EXPECT_EQ(ernie::ExtentOf(scene, "/Instancer")->second,
              glm::vec3(4.0f));
    mesh.SetExtent({1, 2});
    EXPECT_FALSE(ernie::ExtentOf(scene, "/Mesh").has_value());
}

TEST(GeometrySchemas, InstancersDeclarePrototypesAndWorlds) {
    Scene scene;
    ernie::Mesh::Define(scene, "/Protos/Rock");
    ernie::Mesh::Define(scene, "/Protos/Tree");
    ernie::PointInstancer instancer =
        ernie::PointInstancer::Define(scene, "/Instancer")
            .SetPrototypes({"/Protos/Rock", "/Protos/Tree"})
            .SetWorlds(Tensor::Copy<std::int32_t>(
                std::array<std::int32_t, 2>{0, 1}, {2}));
    EXPECT_EQ(instancer.GetPrototypes(),
              (std::vector<Path>{Path("/Protos/Rock"),
                                 Path("/Protos/Tree")}));
    EXPECT_EQ(instancer.GetWorlds()->Elements(), 2);
    // One prototype given alone is stored as a path, and still reads
    // as a list.
    instancer.SetPrototypes("/Protos/Rock");
    EXPECT_EQ(*scene.Get("/Instancer", tokens::kPrototypes),
              Value(Path("/Protos/Rock")));
    EXPECT_EQ(instancer.GetPrototypes(),
              std::vector<Path>{Path("/Protos/Rock")});
}

TEST(GeometrySchemas, PointsWidthsAreOneForAllOrOnePerPoint) {
    Scene scene;
    ernie::Points cloud = ernie::Points::Define(scene, "/Cloud");
    EXPECT_EQ(cloud.GetWidths(), std::vector<float>{1.0f});
    cloud.SetWidths({0.5f, 0.6f, 0.7f});
    EXPECT_EQ(cloud.GetWidths(),
              (std::vector<float>{0.5f, 0.6f, 0.7f}));
    // A single number, as written before widths were a list, reads as
    // a list of one.
    scene.Set("/Cloud", tokens::kWidths, 0.25);
    EXPECT_EQ(cloud.GetWidths(), std::vector<float>{0.25f});
    // Widths written as an array (an OrArray attribute) read back
    // through the view; the declared value is the fallback then.
    EXPECT_EQ(cloud.ReadWidths(), nullptr);
    cloud.WriteWidths(
        Tensor::Copy<float>(std::array<float, 3>{1, 2, 3}, {3}));
    ASSERT_NE(cloud.ReadWidths(), nullptr);
    EXPECT_EQ(cloud.ReadWidths()->Elements(), 3);
    EXPECT_EQ(cloud.GetWidths(), std::vector<float>{1.0f});
    EXPECT_TRUE(ernie::Points::Definition()
                    .Find(tokens::kWidths)
                    ->meta.array);
    EXPECT_FALSE(ernie::Points::Definition()
                     .Find(tokens::kMaterialBinding)
                     ->meta.array);
}

TEST(GeometrySchemas, DisplayColorsArePerPrimOrPerVertex) {
    Scene scene;
    ernie::Mesh mesh = ernie::Mesh::Define(scene, "/Mesh");
    mesh.SetDisplayColor({1.0, 0.0, 0.0});
    EXPECT_EQ(mesh.GetDisplayColor(), glm::dvec3(1.0, 0.0, 0.0));
    EXPECT_EQ(mesh.ReadDisplayColor(), nullptr);
    mesh.WriteDisplayColor(Tensor::Copy<float>(
        std::array<float, 6>{1, 0, 0, 0, 1, 0}, {2, 3}));
    ASSERT_NE(mesh.ReadDisplayColor(), nullptr);
    EXPECT_EQ(mesh.ReadDisplayColor()->Shape(),
              (std::vector<std::int64_t>{2, 3}));
    EXPECT_EQ(
        ernie::Mesh::Definition().Find(tokens::kDisplayColor)->kind,
        ernie::AttributeKind::Vec3);
}

TEST(GeometrySchemas, RegistryDescribesTheTypes) {
    const PrimTypeDef& capsule = ernie::GetPrimType("Capsule");
    EXPECT_FALSE(capsule.abstract);
    ASSERT_NE(capsule.base, nullptr);
    EXPECT_EQ(capsule.base->name, "AxisShape");
    EXPECT_TRUE(capsule.base->abstract);
    EXPECT_TRUE(capsule.IsA(ernie::GetPrimType("Gprim")));
    EXPECT_EQ(capsule.Find(tokens::kGeomRadius)->fallback,
              Value(0.5));
    EXPECT_EQ(ernie::GetPrimType("Cylinder")
                  .Find(tokens::kGeomRadius)
                  ->fallback,
              Value(1.0));
    const ernie::AttributeDef* axis = capsule.Find(tokens::kAxis);
    ASSERT_NE(axis, nullptr);
    EXPECT_EQ(axis->kind, AttributeKind::Token);
    EXPECT_EQ(axis->meta.options.size(), 3u);
    EXPECT_EQ(capsule.Find(tokens::kRadius), nullptr)
        << "a light's inputs:radius";

    const PrimTypeDef& mesh = ernie::GetPrimType(tokens::kMesh);
    EXPECT_EQ(mesh.base->name, "PointBased");
    EXPECT_EQ(mesh.Find(tokens::kPoints)->kind,
              AttributeKind::Tensor);
    EXPECT_EQ(mesh.FindByPyName("face_vertex_indices")->name,
              tokens::kFaceVertexIndices);
    EXPECT_EQ(mesh.Find(tokens::kDisplayColor)->kind,
              AttributeKind::Vec3);
    const PrimTypeDef& points =
        ernie::GetPrimType(tokens::kGeomPoints);
    EXPECT_TRUE(points.IsA(ernie::GetPrimType("PointBased")));
    EXPECT_EQ(points.Find(tokens::kWidths)->fallback,
              Value(std::vector<float>{1.0f}));

    EXPECT_EQ(ernie::Gprim::Definition().base,
              &ernie::Boundable::Definition());
    EXPECT_TRUE(ernie::Boundable::Definition().abstract);
    EXPECT_EQ(mesh.Find(tokens::kExtent)->kind,
              AttributeKind::FloatArray);
    const PrimTypeDef& instancer =
        ernie::GetPrimType(tokens::kPointInstancer);
    EXPECT_FALSE(instancer.IsA(ernie::GetPrimType("Gprim")));
    EXPECT_EQ(instancer.base, &ernie::Boundable::Definition());
    EXPECT_EQ(instancer.Find(tokens::kWorlds)->kind,
              AttributeKind::Tensor);
    EXPECT_EQ(instancer.Find(tokens::kOrientations)->kind,
              AttributeKind::Tensor);
    EXPECT_EQ(instancer.Find(tokens::kPrototypes)->kind,
              AttributeKind::Relationships);
    EXPECT_TRUE(
        ernie::GetPrimType(tokens::kVolume).attributes.empty());

    const PrimTypeDef& ground =
        ernie::GetPrimType(tokens::kGroundPlane);
    EXPECT_EQ(ground.base->name, "Imageable");
    EXPECT_EQ(ground.Find(tokens::kCheckerColor1)->fallback,
              Value(glm::dvec3(0.5)));
    EXPECT_EQ(ground.Find(tokens::kChecker)->label, "Checkerboard");
    EXPECT_EQ(ground.Find(tokens::kMaterialBinding)->kind,
              AttributeKind::Relationship);
    EXPECT_EQ(ground.FindByPyName("checker_size")->name,
              tokens::kCheckerSize);
}
