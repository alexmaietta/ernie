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

using test::BoxNear;
using test::Ints;

TEST(Schema, RegistryListsTypesWithBasesAndAttributes) {
    const SchemaDef* sphere = FindSchema(Sphere::kName);
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
    const auto extent = [&](const SchemaView& view) { return *scene.Extent(view.Id()); };
    EXPECT_TRUE(BoxNear(extent(Cube::Define(scene, "/Cube").SetSize(4.0)), glm::dvec3(-2.0),
                        glm::dvec3(2.0)));
    EXPECT_TRUE(BoxNear(extent(Sphere::Define(scene, "/Sphere").SetRadius(3.0)), glm::dvec3(-3.0),
                        glm::dvec3(3.0)));
    EXPECT_TRUE(BoxNear(extent(Cylinder::Define(scene, "/Cylinder").SetAxis(Axis::X)),
                        glm::dvec3(-1.0), glm::dvec3(1.0)));
    EXPECT_TRUE(BoxNear(extent(Cone::Define(scene, "/Cone").SetHeight(6.0)), {-1.0, -1.0, -3.0},
                        {1.0, 1.0, 3.0}));
    EXPECT_TRUE(BoxNear(extent(Capsule::Define(scene, "/Capsule").SetAxis(Axis::Y)),
                        {-0.5, -1.0, -0.5}, {0.5, 1.0, 0.5}));
    EXPECT_TRUE(BoxNear(extent(Plane::Define(scene, "/Plane").SetWidth(4.0).SetAxis(Axis::X)),
                        {0.0, -1.0, -2.0}, {0.0, 1.0, 2.0}));
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
    EXPECT_TRUE(BoxNear(scene.WorldBounds("/World"), {8.0, -1.0, -2.0}, {12.0, 7.0, 2.0}));
    EXPECT_TRUE(BoxNear(scene.WorldBounds("/World/A"), {9.0, -1.0, -1.0}, {11.0, 1.0, 1.0}))
        << "a prim's bounds include its ancestors' transforms";

    Sphere::Define(scene, "/World/Hidden").SetRadius(100.0).SetVisibility("invisible");
    Sphere::Define(scene, "/World/Guide").SetRadius(100.0).SetPurpose("guide");
    EXPECT_TRUE(BoxNear(scene.WorldBounds("/World"), {8.0, -1.0, -2.0}, {12.0, 7.0, 2.0}))
        << "invisible and guide prims are left out";
    EXPECT_TRUE(scene.WorldBounds("/World/Hidden").Empty());
}

TEST(Extent, ModelsWithoutLoadedDescendantsAreBoundedByTheirHint) {
    Scene scene;
    Xform::Define(scene, "/World");
    SetXform(scene, "/World", {.translate = glm::dvec3(10.0, 0.0, 0.0)});
    Xform::Define(scene, "/World/Rover");
    GeomModelAPI::Apply(scene, "/World/Rover")
        .SetExtentsHint({-1, -1, 0, 1, 1, 2, -5, -5, -5, 5, 5, 5});
    EXPECT_TRUE(BoxNear(scene.WorldBounds("/World"), {9.0, -1.0, 0.0}, {11.0, 1.0, 2.0}));
    EXPECT_TRUE(BoxNear(scene.BoundsInParent(*scene.Find("/World/Rover")), glm::dvec3(-5.0),
                        glm::dvec3(5.0)))
        << "bounds in the parent take every purpose's box";
    Sphere::Define(scene, "/World/Rover/Body").SetRadius(0.5);
    EXPECT_TRUE(BoxNear(scene.WorldBounds("/World"), {9.5, -0.5, -0.5}, {10.5, 0.5, 0.5}))
        << "loaded descendants replace the hint";
    Sphere(scene, "/World/Rover/Body").SetVisibility("invisible");
    EXPECT_TRUE(scene.WorldBounds("/World").Empty()) << "hidden, not unloaded";
}

TEST(Extent, SubsetsAreNotBounded) {
    Scene scene;
    Cube::Define(scene, "/Box");
    GeomSubset subset = GeomSubset::Define(scene, "/Box/Top");
    subset.SetIndices(Ints({4})).SetFamilyName("materialBind");
    EXPECT_EQ(subset.GetElementType(), "face");
    EXPECT_FALSE(scene.Extent(subset.Id()));
    EXPECT_TRUE(BoxNear(scene.WorldBounds("/Box"), glm::dvec3(-1.0), glm::dvec3(1.0)));
}

TEST(Extent, ResettingTheStackDropsAncestorsFromWorldBounds) {
    Scene scene;
    Xform::Define(scene, "/World");
    SetXform(scene, "/World", {.translate = glm::dvec3(100.0)});
    Sphere::Define(scene, "/World/Ball");
    scene.Set("/World/Ball", tokens::kXformOpOrder,
              std::vector<std::string>{std::string(tokens::kResetXformStack)});
    EXPECT_TRUE(BoxNear(scene.WorldBounds("/World/Ball"), glm::dvec3(-1.0), glm::dvec3(1.0)));
}

}  // namespace
}  // namespace ernie
