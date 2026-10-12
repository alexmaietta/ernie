// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/session/light_links.h"

#include "ernie/scene/base_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/scene.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

using ernie::LinkCollection;
using ernie::LinkGroups;
using ernie::Path;
using ernie::Scene;

// The nearest listed ancestor decides, over the root.
TEST(LightLinks, CollectionsExpandTheirPrims) {
    LinkCollection collection;
    EXPECT_TRUE(collection.Everything());
    EXPECT_TRUE(collection.Contains("/World/Rock"));
    collection.excludes = {"/World/Earth"};
    collection.includes = {"/World/Earth/Pole"};
    EXPECT_FALSE(collection.Everything());
    EXPECT_FALSE(collection.Contains("/World/Earth"));
    EXPECT_FALSE(collection.Contains("/World/Earth/Ground"));
    EXPECT_TRUE(collection.Contains("/World/Earth/Pole/Flag"));
    EXPECT_TRUE(collection.Contains("/World/Moon"));
    collection = {false, {"/World/Moon"}, {}};
    EXPECT_TRUE(collection.Contains("/World/Moon/Surface"));
    EXPECT_FALSE(collection.Contains("/World/Earth"));
}

// Earthshine lights the Moon alone, and the Earth casts none of the
// sun's shadows: three groups, each light's masks over them.
TEST(LightLinks, GroupGeometryByTheLightsThatReachIt) {
    Scene scene;
    ernie::Xform::Define(scene, "/World");
    ernie::Sphere::Define(scene, "/World/Earth");
    ernie::Sphere::Define(scene, "/World/Moon");
    ernie::Sphere::Define(scene, "/World/Rock");
    ernie::DistantLight::Define(scene, "/World/Sun")
        .SetShadowLinkExcludes({Path("/World/Earth")});
    ernie::DistantLight::Define(scene, "/World/Earthshine")
        .SetLightLinkIncludeRoot(false)
        .SetLightLinkIncludes({Path("/World/Moon")});
    const std::vector<Path> lights = {"/World/Sun",
                                      "/World/Earthshine"};
    const std::vector<Path> geometry = {"/World/Earth", "/World/Moon",
                                        "/World/Rock"};
    const LinkGroups groups =
        ernie::LinkGroupsOf(scene, lights, geometry);
    const std::uint32_t earth = groups.GroupOf("/World/Earth");
    const std::uint32_t moon = groups.GroupOf("/World/Moon");
    const std::uint32_t rock = groups.GroupOf("/World/Rock");
    EXPECT_NE(earth, moon);
    EXPECT_NE(moon, rock);
    EXPECT_NE(earth, rock);
    const auto has = [](std::uint32_t mask, std::uint32_t group) {
        return ((mask >> group) & 1u) != 0;
    };
    // The sun lights all, shadowed by all but the Earth.
    for (const std::uint32_t g : {earth, moon, rock}) {
        EXPECT_TRUE(has(groups.LightMask("/World/Sun"), g));
    }
    EXPECT_FALSE(has(groups.ShadowMask("/World/Sun"), earth));
    EXPECT_TRUE(has(groups.ShadowMask("/World/Sun"), moon));
    // Earthshine lights the Moon alone.
    EXPECT_TRUE(has(groups.LightMask("/World/Earthshine"), moon));
    EXPECT_FALSE(has(groups.LightMask("/World/Earthshine"), earth));
    EXPECT_FALSE(has(groups.LightMask("/World/Earthshine"), rock));
    EXPECT_EQ(groups.ShadowMask("/World/Earthshine"), ~0u);
}

TEST(LightLinks, WithoutLinksEverythingIsGroupZero) {
    Scene scene;
    ernie::Sphere::Define(scene, "/Ball");
    ernie::SphereLight::Define(scene, "/Lamp");
    const std::vector<Path> lights = {"/Lamp"};
    const std::vector<Path> geometry = {"/Ball"};
    const LinkGroups groups =
        ernie::LinkGroupsOf(scene, lights, geometry);
    EXPECT_EQ(groups.GroupOf("/Ball"), 0u);
    EXPECT_EQ(groups.LightMask("/Lamp"), ~0u);
    EXPECT_EQ(groups.ShadowMask("/Lamp"), ~0u);
}
