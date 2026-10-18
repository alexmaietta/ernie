// SPDX-License-Identifier: BSD-3-Clause
/// @file api_test.cpp
/// @brief Tests of applied API schemas: single-apply, multiple-apply and their buckets.
#include "ernie/scene/scene.h"
#include "ernie/scene/schemas/collection.h"
#include "ernie/scene/schemas/geom.h"
#include "ernie/scene/schemas/shade.h"
#include "ernie/scene/tokens.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace ernie {
namespace {

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
