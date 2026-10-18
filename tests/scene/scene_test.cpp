// SPDX-License-Identifier: BSD-3-Clause
/// @file scene_test.cpp
/// @brief Tests of prims, attributes, typed views and applied API schemas over the columnar store.
#include "ernie/scene/scene.h"
#include "ernie/scene/schema/collection.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/shade.h"
#include "ernie/scene/tokens.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace ernie {
namespace {

TEST(Scene, DefinesPrimsWithTypelessAncestors) {
    Scene scene;
    const PrimId rock = scene.Define("/World/Rocks/Rock", Sphere::kName);
    EXPECT_EQ(scene.TypeName(rock), Sphere::kName);
    ASSERT_TRUE(scene.Has("/World/Rocks"));
    EXPECT_EQ(scene.TypeName(*scene.Find("/World/Rocks")), "");
    EXPECT_EQ(scene.Schema(*scene.Find("/World")), nullptr);
    EXPECT_EQ(scene.Children("/World"), std::vector<Path>{Path("/World/Rocks")});
    EXPECT_EQ(scene.Children(Path::Root()), std::vector<Path>{Path("/World")});

    EXPECT_THROW(scene.Define(Path::Root(), Xform::kName), std::invalid_argument);
    EXPECT_THROW(scene.Define("/World/A", "Teapot"), std::invalid_argument);
    EXPECT_THROW(scene.Define("/World/A", "Gprim"), std::invalid_argument) << "abstract";
}

TEST(Scene, RedefiningWithAnotherTypeKeepsWhatBothDeclare) {
    Scene scene;
    const PrimId prim = scene.Define("/Shape", Sphere::kName);
    scene.Set(prim, Sphere::kRadius, 3.0);
    scene.Set(prim, Gprim::kDisplayColor, glm::dvec3(1.0, 0.0, 0.0));
    EXPECT_EQ(scene.Define("/Shape", Cylinder::kName), prim);
    EXPECT_EQ(scene.TypeName(prim), Cylinder::kName);
    EXPECT_EQ(Cylinder(scene, "/Shape").GetRadius(), 3.0);
    EXPECT_EQ(Cylinder(scene, "/Shape").GetDisplayColor(), glm::dvec3(1.0, 0.0, 0.0));
    EXPECT_FALSE(Cylinder(scene, "/Shape").HasHeight());
}

TEST(Scene, RemovesASubtree) {
    Scene scene;
    scene.Define("/World/Rover/Wheel", Cylinder::kName);
    scene.Define("/World/Sun", Sphere::kName);
    EXPECT_TRUE(scene.Remove("/World/Rover"));
    EXPECT_FALSE(scene.Has("/World/Rover"));
    EXPECT_FALSE(scene.Has("/World/Rover/Wheel"));
    EXPECT_TRUE(scene.Has("/World/Sun"));
    EXPECT_FALSE(scene.Remove("/World/Rover"));
    EXPECT_THROW(scene.Remove(Path::Root()), std::invalid_argument);

    scene.Define("/World/Rover", Xform::kName);
    EXPECT_TRUE(scene.Children("/World/Rover").empty()) << "the path stays, its prims don't";
}

TEST(Scene, ListsPrimsOfATypeAndItsDerivedTypes) {
    Scene scene;
    scene.Define("/B", Sphere::kName);
    scene.Define("/A", Cube::kName);
    scene.Define("/G", Xform::kName);
    EXPECT_EQ(scene.Prims(Sphere::kName), std::vector<Path>{Path("/B")});
    EXPECT_EQ(scene.Prims("Gprim"), (std::vector<Path>{Path("/A"), Path("/B")}));
    EXPECT_EQ(scene.Prims().size(), 3u);
    EXPECT_THROW(scene.Prims("Teapot"), std::invalid_argument);
}

TEST(Scene, SetsAndGetsValuesByUsdName) {
    Scene scene;
    const PrimId prim = scene.Define("/Ball", Sphere::kName);
    EXPECT_FALSE(scene.Get(prim, Sphere::kRadius)) << "unauthored";
    scene.Set(prim, Sphere::kRadius, 2);
    EXPECT_EQ(scene.Get(prim, Sphere::kRadius), Value(2.0)) << "ints widen to doubles";
    scene.Set(prim, Gprim::kDisplayColor, glm::vec3(0.25f));
    EXPECT_EQ(scene.Get(prim, Gprim::kDisplayColor), Value(glm::dvec3(0.25)));
    scene.Set(prim, Imageable::kVisibility, "invisible");
    EXPECT_EQ(scene.Get(prim, Imageable::kVisibility), Value("invisible"));
    EXPECT_EQ(scene.Attributes("/Ball"),
              (std::vector<std::string>{"visibility", "primvars:displayColor", "radius"}));

    EXPECT_TRUE(scene.Clear(prim, Sphere::kRadius));
    EXPECT_FALSE(scene.Clear(prim, Sphere::kRadius));
    EXPECT_FALSE(scene.IsAuthored(prim, Sphere::kRadius));
}

TEST(Scene, SettingChecksNamesTypesAndLimits) {
    Scene scene;
    const PrimId prim = scene.Define("/Ball", Sphere::kName);
    EXPECT_THROW(scene.Set(prim, "teapot:spout", 1.0), std::invalid_argument);
    EXPECT_THROW(scene.Set(prim, Sphere::kRadius, "big"), std::invalid_argument);
    EXPECT_THROW(scene.Set(prim, Sphere::kRadius, -1.0), std::invalid_argument);
    EXPECT_THROW(scene.Set(prim, Imageable::kVisibility, "hidden"), std::invalid_argument);
    EXPECT_THROW(scene.Set("/Nowhere", Sphere::kRadius, 1.0), std::invalid_argument);
    try {
        scene.Set(prim, Gprim::kDisplayOpacity, 2.0);
        FAIL() << "out of range";
    } catch (const std::invalid_argument& error) {
        EXPECT_STREQ(error.what(), "'/Ball': 'primvars:displayOpacity' must be from 0 to 1, not 2");
    }
}

TEST(Scene, OpenNamespacesAndTypelessPrimsTakeCustomAttributes) {
    Scene scene;
    const PrimId group = scene.Define("/World/Group", Xform::kName);
    scene.Set(group, "user:mass", 12.5);
    EXPECT_EQ(scene.Get(group, "user:mass"), Value(12.5));
    EXPECT_THROW(scene.Set(group, "user:mass", "heavy"), std::invalid_argument)
        << "typed by its first value";
    const PrimId world = *scene.Find("/World");
    scene.Set(world, "anything", std::string("goes"));
    EXPECT_EQ(scene.Get(world, "anything"), Value("goes"));
}

TEST(Scene, ViewsReadFallbacksAndSetThroughTheScene) {
    Scene scene;
    Sphere ball = Sphere::Define(scene, "/Ball");
    EXPECT_EQ(ball.GetRadius(), 1.0);
    EXPECT_EQ(ball.GetDisplayColor(), glm::dvec3(0.5));
    EXPECT_EQ(ball.GetVisibility(), "inherited");
    ball.SetRadius(4.0).SetDoubleSided(true);
    EXPECT_EQ(ball.GetRadius(), 4.0);
    EXPECT_TRUE(ball.GetDoubleSided());
    EXPECT_TRUE(ball.HasRadius());
    EXPECT_THROW(ball.SetRadius(-4.0), std::invalid_argument);
    ball.ClearRadius();
    EXPECT_EQ(ball.GetRadius(), 1.0);

    Cylinder can = Cylinder::Define(scene, "/Can");
    EXPECT_EQ(can.GetAxis(), Axis::Z);
    can.SetAxis(Axis::X);
    EXPECT_EQ(can.GetAxis(), Axis::X);
    EXPECT_EQ(scene.Get(can.Id(), Cylinder::kAxis), Value("X"));
}

TEST(Scene, ViewsCheckTypesAndConstness) {
    Scene scene;
    scene.Define("/Ball", Sphere::kName);
    EXPECT_NO_THROW(Gprim(scene, "/Ball"));
    EXPECT_THROW(Cube(scene, "/Ball"), std::invalid_argument);
    EXPECT_THROW(Sphere(scene, "/Missing"), std::invalid_argument);
    const Scene& view = scene;
    Sphere ball(view, "/Ball");
    EXPECT_EQ(ball.GetRadius(), 1.0);
    EXPECT_THROW(ball.SetRadius(2.0), std::logic_error);
}

TEST(Scene, ChangesAreClassifiedForRenderers) {
    Scene scene;
    const PrimId ball = scene.Define("/Ball", Sphere::kName);
    const auto classify = [&](std::string_view attribute, const Value& value) {
        const std::uint64_t cursor = scene.Version();
        scene.Set(ball, attribute, value);
        const ChangeSet changes = scene.Store().ChangesSince(cursor);
        EXPECT_EQ(changes.values.size(), 1u) << attribute;
        return scene.Classify(changes.values.front());
    };
    EXPECT_EQ(classify(Sphere::kRadius, 2.0), Dirty::Params);
    EXPECT_EQ(classify(Imageable::kVisibility, "invisible"), Dirty::Visibility);
    EXPECT_EQ(classify(Xformable::kXformOpOrder, std::vector<std::string>{}), Dirty::Transform);
    EXPECT_EQ(classify(tokens::kTranslate, glm::dvec3(1.0)), Dirty::Transform) << "undeclared op";
    EXPECT_EQ(classify("primvars:wetness", 0.5), Dirty::Primvars) << "undeclared primvar";
    EXPECT_TRUE(Any(Dirty::Points | Dirty::Topology, Dirty::Topology));
    EXPECT_FALSE(Any(Dirty::Points, Dirty::Topology));
}

TEST(ApiSchema, ApplyingAddsItsAttributesAtTheirFallbacks) {
    Scene scene;
    const PrimId ball = scene.Define("/Ball", Sphere::kName);
    EXPECT_THROW(scene.Set(ball, MaterialBindingAPI::kMaterialBinding, Path("/Looks/Red")),
                 std::invalid_argument)
        << "not applied";
    MaterialBindingAPI binding = MaterialBindingAPI::Apply(scene, "/Ball");
    EXPECT_TRUE(scene.HasApi(ball, "MaterialBindingAPI"));
    EXPECT_EQ(binding.GetMaterialBinding(), Path());
    binding.SetMaterialBinding(Path("/Looks/Red"));
    EXPECT_EQ(scene.Get(ball, "material:binding"), Value(Path("/Looks/Red")));
    scene.Set(ball, "material:binding:preview", Path("/Looks/Fast"));
    EXPECT_EQ(Sphere(scene, "/Ball").GetRadius(), 1.0) << "the type's attributes stay";
}

TEST(ApiSchema, PrimsOfOneTypeAndApiSchemasShareABucket) {
    Scene scene;
    const PrimId a = scene.Define("/A", Sphere::kName);
    const PrimId b = scene.Define("/B", Sphere::kName);
    scene.ApplyApi(a, "MaterialBindingAPI");
    EXPECT_NE(scene.Store().PrimBucket(a), scene.Store().PrimBucket(b));
    scene.ApplyApi(b, "MaterialBindingAPI");
    EXPECT_EQ(scene.Store().PrimBucket(a), scene.Store().PrimBucket(b));
    scene.ApplyApi(a, "MaterialBindingAPI");
    EXPECT_EQ(scene.AppliedApis(a), std::vector<std::string>{"MaterialBindingAPI"})
        << "applying twice applies once";
}

TEST(ApiSchema, RetypingKeepsAppliedSchemasAndRemovingDropsTheirValues) {
    Scene scene;
    const PrimId prim = scene.Define("/Shape", Sphere::kName);
    MaterialBindingAPI::Apply(scene, "/Shape").SetMaterialBinding(Path("/Looks/Red"));
    scene.Define("/Shape", Cube::kName);
    EXPECT_EQ(MaterialBindingAPI(scene, "/Shape").GetMaterialBinding(), Path("/Looks/Red"));

    EXPECT_TRUE(scene.RemoveApi(prim, "MaterialBindingAPI"));
    EXPECT_FALSE(scene.RemoveApi(prim, "MaterialBindingAPI"));
    EXPECT_FALSE(scene.Get(prim, "material:binding"));
    EXPECT_THROW(MaterialBindingAPI(scene, "/Shape"), std::invalid_argument);
}

TEST(ApiSchema, MultipleApplySchemasNameTheirAttributesByInstance) {
    Scene scene;
    const PrimId world = scene.Define("/World", Xform::kName);
    CollectionAPI lights = CollectionAPI::Apply(scene, "/World", "lights");
    CollectionAPI::Apply(scene, "/World", "rocks").SetIncludes({Path("/World/Rocks")});
    lights.SetIncludes({Path("/World/Sun"), Path("/World/Lamp")})
        .SetExpansionRule(ExpansionRule::ExplicitOnly);

    EXPECT_EQ(scene.Get(world, "collection:lights:includes"),
              Value(std::vector<Path>{Path("/World/Sun"), Path("/World/Lamp")}));
    EXPECT_EQ(scene.Get(world, "collection:lights:expansionRule"), Value("explicitOnly"));
    EXPECT_EQ(CollectionAPI(scene, "/World", "rocks").GetIncludes(),
              std::vector<Path>{Path("/World/Rocks")});
    EXPECT_EQ(CollectionAPI(scene, "/World", "rocks").GetExpansionRule(),
              ExpansionRule::ExpandPrims);
    EXPECT_EQ(scene.AppliedApis(world).size(), 2u);
    EXPECT_THROW(CollectionAPI(scene, "/World", "plants"), std::invalid_argument);
}

TEST(ApiSchema, NamesMustBeKnownWithTheRightInstances) {
    Scene scene;
    const PrimId prim = scene.Define("/A", Xform::kName);
    EXPECT_THROW(scene.ApplyApi(prim, "TeapotAPI"), std::invalid_argument);
    EXPECT_THROW(scene.ApplyApi(prim, "CollectionAPI"), std::invalid_argument) << "no instance";
    EXPECT_THROW(scene.ApplyApi(prim, "CollectionAPI:two words"), std::invalid_argument);
    EXPECT_THROW(scene.ApplyApi(prim, "MaterialBindingAPI:extra"), std::invalid_argument);
    EXPECT_THROW(scene.ApplyApi(prim, "Sphere"), std::invalid_argument) << "a prim type";
    EXPECT_TRUE(scene.AppliedApis(prim).empty());
}

}  // namespace
}  // namespace ernie
