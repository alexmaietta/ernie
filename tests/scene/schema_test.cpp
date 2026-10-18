// SPDX-License-Identifier: BSD-3-Clause
/// @file schema_test.cpp
/// @brief Tests of the schema registry, extents and world bounds.
#include "ernie/scene/schema.h"
#include "ernie/scene/schemas/geom.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"

#include "tests/common.h"

#include <gtest/gtest.h>

#include <string>

namespace ernie {
namespace {

void ExpectBox(const Box& box, const glm::dvec3& min, const glm::dvec3& max) {
    EXPECT_TRUE(test::BoxNear(box, min, max));
}

TEST(Schema, RegistryListsTypesWithBasesAndAttributes) {
    const SchemaDef* sphere = FindSchema(tokens::kSphere);
    ASSERT_NE(sphere, nullptr);
    EXPECT_TRUE(sphere->IsA(Gprim::Definition()));
    EXPECT_TRUE(sphere->IsA(Imageable::Definition()));
    EXPECT_FALSE(sphere->IsA(Cube::Definition()));
    EXPECT_EQ(sphere->Find("visibility"), Imageable::Definition().Find("visibility"));
    EXPECT_EQ(sphere->AllAttributes().front()->name, "visibility") << "bases first";
    EXPECT_EQ(sphere->Find("radius")->fallback, Value(1.0));
    EXPECT_EQ(FindSchema("Teapot"), nullptr);
    for (const SchemaDef* schema : Schemas()) {
        EXPECT_EQ(FindSchema(schema->name), schema);
        if (schema->base != nullptr) {
            EXPECT_EQ(FindSchema(schema->base->name), schema->base) << "bases are registered";
        }
    }
}

TEST(Schema, EveryPrimTypeSaysHowItIsBounded) {
    for (const SchemaDef* schema : Schemas()) {
        if (schema->kind != SchemaKind::Concrete) {
            continue;
        }
        const ExtentRule rule = schema->Extent();
        EXPECT_NE(rule.kind, ExtentRule::Kind::Inherit)
            << schema->name << " declares no extent rule";
        if (rule.kind == ExtentRule::Kind::Computed) {
            EXPECT_NE(rule.compute, nullptr) << schema->name;
        }
    }
}

TEST(Schema, DeclarationsSayWhatTheirChangesStamp) {
    EXPECT_EQ(Imageable::Definition().Find("visibility")->meta.dirty, Dirty::Visibility);
    EXPECT_EQ(Xformable::Definition().Find("xformOpOrder")->meta.dirty, Dirty::Transform);
    EXPECT_EQ(Sphere::Definition().Find("radius")->meta.dirty, Dirty::Params);
    EXPECT_EQ(Cylinder::Definition().Find("axis")->meta.options,
              (std::vector<std::string_view>{"X", "Y", "Z"}))
        << "choices list their tokens";
}

TEST(Extent, ShapesAreBoundedAsUsdGeomComputesThem) {
    Scene scene;
    ExpectBox(*scene.Extent(Cube::Define(scene, "/Cube").SetSize(4.0).Id()), glm::dvec3(-2.0),
              glm::dvec3(2.0));
    ExpectBox(*scene.Extent(Sphere::Define(scene, "/Sphere").SetRadius(3.0).Id()), glm::dvec3(-3.0),
              glm::dvec3(3.0));
    ExpectBox(*scene.Extent(Cylinder::Define(scene, "/Cylinder").SetAxis(Axis::X).Id()),
              {-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0});
    ExpectBox(*scene.Extent(Cone::Define(scene, "/Cone").SetHeight(6.0).Id()), {-1.0, -1.0, -3.0},
              {1.0, 1.0, 3.0});
    ExpectBox(*scene.Extent(Capsule::Define(scene, "/Capsule").SetAxis(Axis::Y).Id()),
              {-0.5, -1.0, -0.5}, {0.5, 1.0, 0.5});
    ExpectBox(*scene.Extent(Plane::Define(scene, "/Plane").SetWidth(4.0).SetAxis(Axis::X).Id()),
              {0.0, -1.0, -2.0}, {0.0, 1.0, 2.0});
}

TEST(Extent, GroupingsAndTypelessPrimsHaveNoExtentOfTheirOwn) {
    Scene scene;
    EXPECT_FALSE(scene.Extent(Xform::Define(scene, "/World/Group").Id()));
    EXPECT_FALSE(scene.Extent(*scene.Find("/World")));
}

TEST(Extent, WorldBoundsGatherVisibleDescendantsThroughTheirTransforms) {
    Scene scene;
    Xform::Define(scene, "/World");
    SetXform(scene, "/World", {.translate = glm::dvec3(10.0, 0.0, 0.0)});
    Sphere::Define(scene, "/World/A");
    Cube::Define(scene, "/World/B").SetSize(2.0);
    SetXform(scene, "/World/B", {.translate = glm::dvec3(0.0, 5.0, 0.0), .scale = glm::dvec3(2.0)});
    ExpectBox(scene.WorldBounds("/World"), {8.0, -1.0, -2.0}, {12.0, 7.0, 2.0});
    SCOPED_TRACE("a prim's bounds include its ancestors' transforms");
    ExpectBox(scene.WorldBounds("/World/A"), {9.0, -1.0, -1.0}, {11.0, 1.0, 1.0});

    Sphere::Define(scene, "/World/Hidden").SetRadius(100.0).SetVisibility("invisible");
    Sphere::Define(scene, "/World/Guide").SetRadius(100.0).SetPurpose("guide");
    SCOPED_TRACE("invisible and guide prims are left out");
    ExpectBox(scene.WorldBounds("/World"), {8.0, -1.0, -2.0}, {12.0, 7.0, 2.0});
    EXPECT_TRUE(Box().Empty());
    EXPECT_TRUE(scene.WorldBounds("/World/Hidden").Empty());
}

TEST(Extent, ResettingTheStackDropsAncestorsFromWorldBounds) {
    Scene scene;
    Xform::Define(scene, "/World");
    SetXform(scene, "/World", {.translate = glm::dvec3(100.0)});
    Sphere::Define(scene, "/World/Ball");
    scene.Set("/World/Ball", tokens::kXformOpOrder,
              std::vector<std::string>{std::string(tokens::kResetXformStack)});
    ExpectBox(scene.WorldBounds("/World/Ball"), glm::dvec3(-1.0), glm::dvec3(1.0));
}

}  // namespace
}  // namespace ernie
