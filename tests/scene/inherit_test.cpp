// SPDX-License-Identifier: BSD-3-Clause
/// @file inherit_test.cpp
/// @brief Tests of what edits reach beyond their prim: inherited attributes resolving from the
/// nearest authoring ancestor, and relationship targets reaching their holders.
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/instancer.h"
#include "ernie/scene/schema/shade.h"
#include "ernie/util/algorithm.h"

#include "tests/common.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie {
namespace {

std::string Visibility(const Scene& scene, const char* path) {
    return *scene.GetResolved(*scene.Find(path), Imageable::kVisibility)->Get<std::string>();
}

int World(const Scene& scene, const char* path) {
    return *scene.GetResolved(*scene.Find(path), Imageable::kWorld)->Get<int>();
}

/// @return Whether the last update changed `attribute`'s resolved value of the prim at `path`.
bool Changed(const Scene& scene, std::uint64_t cursor, const char* path,
             std::string_view attribute) {
    const PrimStore& store = scene.Store();
    const PrimId prim = *scene.Find(path);
    return test::ValueChanged(store.ChangesSince(cursor), store.PrimBucket(prim),
                              *store.Tokens().Find(ResolvedName(attribute)), store.PrimRow(prim));
}

using Paths = std::vector<std::string>;

/// @return The paths of the prims holding a relationship to `target`, sorted.
Paths Holders(const Scene& scene, const char* target) {
    Paths paths = Map(scene.Holders(Path(target)),
                      [&](PrimId prim) { return scene.PrimPath(prim).String(); });
    std::ranges::sort(paths);
    return paths;
}

TEST(Inherit, ValuesComeFromTheNearestAuthoringAncestor) {
    Scene scene;
    Xform::Define(scene, "/World").SetWorld(3);
    Xform::Define(scene, "/World/Rover").SetVisibility("invisible");
    Sphere::Define(scene, "/World/Rover/Wheel").SetVisibility("inherited");
    Sphere::Define(scene, "/World/Rover/Mast").SetWorld(5);
    Sphere::Define(scene, "/Sun");
    scene.Update();
    EXPECT_EQ(World(scene, "/World/Rover/Wheel"), 3);
    EXPECT_EQ(World(scene, "/World/Rover/Mast"), 5) << "its own value wins";
    EXPECT_EQ(World(scene, "/Sun"), -1) << "the fallback";
    EXPECT_EQ(Visibility(scene, "/World/Rover/Wheel"), "invisible")
        << "authoring \"inherited\" inherits";
    EXPECT_EQ(Visibility(scene, "/Sun"), "inherited");
    EXPECT_EQ(scene.Get(*scene.Find("/World/Rover/Wheel"), Imageable::kVisibility),
              Value(std::string("inherited")))
        << "authored values are kept apart from resolved ones";
}

TEST(Inherit, TypelessPrimsPassValuesThroughAndMaterialsAreBoundDown) {
    Scene scene;
    scene.Define("/World", "");
    scene.Set("/World", Imageable::kWorld, 2);
    scene.Define("/World/Group/Rock", "Mesh");
    Xform::Define(scene, "/World/Group");
    MaterialBindingAPI::Apply(scene, "/World/Group").SetMaterialBinding(Path("/Looks/Basalt"));
    scene.Update();
    EXPECT_EQ(World(scene, "/World/Group/Rock"), 2);
    EXPECT_EQ(
        scene.GetResolved(*scene.Find("/World/Group/Rock"), MaterialBindingAPI::kMaterialBinding),
        Value(Path("/Looks/Basalt")));
    EXPECT_FALSE(scene.GetResolved(*scene.Find("/World"), Imageable::kWorld))
        << "typeless prims hold no resolved values";
}

TEST(Inherit, EditsReachDescendantsAndStampOnlyRealChanges) {
    Scene scene;
    Xform::Define(scene, "/World");
    Xform::Define(scene, "/World/Rover");
    Sphere::Define(scene, "/World/Rover/Wheel");
    Sphere::Define(scene, "/World/Lander").SetVisibility("invisible");
    scene.Update();

    std::uint64_t cursor = scene.Version();
    Imageable(scene, "/World").SetVisibility("invisible");
    scene.Update();
    EXPECT_EQ(Visibility(scene, "/World/Rover/Wheel"), "invisible");
    EXPECT_TRUE(Changed(scene, cursor, "/World/Rover/Wheel", Imageable::kVisibility));
    EXPECT_FALSE(Changed(scene, cursor, "/World/Rover/Wheel", Imageable::kWorld))
        << "other attributes are untouched";

    cursor = scene.Version();
    scene.Clear(*scene.Find("/World"), Imageable::kVisibility);
    scene.Update();
    EXPECT_EQ(Visibility(scene, "/World/Rover/Wheel"), "inherited") << "unauthoring reverts";
    EXPECT_EQ(Visibility(scene, "/World/Lander"), "invisible") << "its own value stays";
    EXPECT_TRUE(Changed(scene, cursor, "/World/Rover/Wheel", Imageable::kVisibility));
}

TEST(Inherit, NewRetypedAndMovedPrimsResolveToo) {
    Scene scene;
    Xform::Define(scene, "/World").SetWorld(7);
    for (int i = 0; i < 100; ++i) {
        Sphere::Define(scene, Path("/World/Ball_" + std::to_string(i)));
    }
    scene.Update();
    scene.Remove("/World/Ball_0");  // Ball_99 moves into its row
    Cube::Define(scene, "/World/Ball_5");
    scene.ApplyApi(*scene.Find("/World/Ball_6"), "MaterialBindingAPI");
    Sphere::Define(scene, "/World/Late");
    scene.Update();
    for (const char* path : {"/World/Ball_99", "/World/Ball_5", "/World/Ball_6", "/World/Late"}) {
        EXPECT_EQ(World(scene, path), 7) << path;
    }
}

TEST(Inherit, ResolvedValuesCannotBeAuthored) {
    Scene scene;
    Xform::Define(scene, "/World");
    EXPECT_THROW(scene.Set("/World", ResolvedName(Imageable::kWorld), 1), std::invalid_argument);
    const ChangeSet::Values change{
        scene.Store().PrimBucket(*scene.Find("/World")),
        *scene.Store().Tokens().Find(ResolvedName(MaterialBindingAPI::kMaterialBinding)),
        {}};
    EXPECT_EQ(scene.Classify(change), Dirty::Material) << "dirty as the attribute is";
}

TEST(Holders, TargetsReachThePrimsHoldingThem) {
    Scene scene;
    Sphere::Define(scene, "/Protos/Ball");
    Cube::Define(scene, "/Protos/Box");
    PointInstancer::Define(scene, "/Rocks")
        .SetPrototypes({Path("/Protos/Ball"), Path("/Protos/Box")});
    PointInstancer::Define(scene, "/Pebbles").SetPrototypes({Path("/Protos/Ball")});
    scene.Update();
    EXPECT_EQ(Holders(scene, "/Protos/Ball"), (Paths{"/Pebbles", "/Rocks"}));
    EXPECT_EQ(Holders(scene, "/Protos/Box"), (Paths{"/Rocks"}));
    EXPECT_TRUE(Holders(scene, "/Nowhere").empty());

    PointInstancer(scene, "/Rocks").SetPrototypes({Path("/Protos/Box")});
    scene.Remove("/Pebbles");
    scene.Update();
    EXPECT_EQ(Holders(scene, "/Protos/Ball"), Paths{}) << "retargeted and removed holders drop";
    EXPECT_EQ(Holders(scene, "/Protos/Box"), (Paths{"/Rocks"}));
}

TEST(Holders, AMaterialReachesEveryPrimBoundToItThroughInheritance) {
    Scene scene;
    Material::Define(scene, "/Looks/Basalt");
    Xform::Define(scene, "/World");
    MaterialBindingAPI::Apply(scene, "/World").SetMaterialBinding(Path("/Looks/Basalt"));
    Sphere::Define(scene, "/World/Rock");
    Cube::Define(scene, "/World/Boulder");
    scene.Update();
    EXPECT_EQ(Holders(scene, "/Looks/Basalt"), (Paths{"/World", "/World/Boulder", "/World/Rock"}));

    scene.RemoveApi(*scene.Find("/World"), "MaterialBindingAPI");
    scene.Update();
    EXPECT_EQ(Holders(scene, "/Looks/Basalt"), Paths{}) << "unbound, though the slots live on";
}

}  // namespace
}  // namespace ernie
