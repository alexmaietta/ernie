// SPDX-License-Identifier: BSD-3-Clause
/// @file holder_index_test.cpp
/// @brief Tests of the reverse index from relationship targets to the prims holding them.
#include "ernie/scene/schemas/geom.h"
#include "ernie/scene/schemas/instancer.h"
#include "ernie/scene/schemas/shade.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace ernie {
namespace {

std::vector<std::string> Holders(const Scene& scene, const char* target) {
    std::vector<std::string> paths;
    for (const PrimId prim : scene.Holders(Path(target))) {
        paths.push_back(scene.PrimPath(prim).String());
    }
    std::ranges::sort(paths);
    return paths;
}

using Paths = std::vector<std::string>;

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
