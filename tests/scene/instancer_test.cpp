// SPDX-License-Identifier: BSD-3-Clause
/// @file instancer_test.cpp
/// @brief Tests of point instancers' extents: prototypes' bounds at each instance.
#include "ernie/scene/schemas/instancer.h"
#include "ernie/scene/xform.h"

#include "tests/common.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace ernie {
namespace {

using test::BoxNear;
using test::Floats;

/// A ball (radius 1) and a cube (size 2, moved 3 m up by its own transform), hidden as
/// prototypes usually are, under an instancer with two instances.
PointInstancer TwoInstances(Scene& scene) {
    PointInstancer instancer = PointInstancer::Define(scene, "/World/Rocks");
    Sphere::Define(scene, "/World/Rocks/Protos/Ball").SetVisibility("invisible");
    Cube::Define(scene, "/World/Rocks/Protos/Box").SetVisibility("invisible");
    SetXform(scene, "/World/Rocks/Protos/Box", {.translate = glm::dvec3(0.0, 0.0, 3.0)});
    instancer.SetPrototypes({Path("/World/Rocks/Protos/Ball"), Path("/World/Rocks/Protos/Box")})
        .SetProtoIndices(test::Ints({0, 1}))
        .SetPositions(Floats({0, 0, 0, 10, 0, 0}, {2, 3}));
    return instancer;
}

TEST(PointInstancer, IsBoundedByItsPrototypesAtEachInstance) {
    Scene scene;
    const PointInstancer rocks = TwoInstances(scene);
    EXPECT_TRUE(BoxNear(*scene.Extent(rocks.Id()), {-1.0, -1.0, -1.0}, {11.0, 1.0, 4.0}))
        << "hidden prototypes count, each with its own transform";
    EXPECT_TRUE(BoxNear(scene.WorldBounds("/World"), {-1.0, -1.0, -1.0}, {11.0, 1.0, 4.0}))
        << "prototypes under the instancer aren't added again";
}

TEST(PointInstancer, ScalesAndOrientationsPlaceThePrototypes) {
    Scene scene;
    PointInstancer rocks = TwoInstances(scene);
    rocks.SetScales(Floats({1, 1, 1, 2, 2, 2}, {2, 3}));
    EXPECT_TRUE(BoxNear(*scene.Extent(rocks.Id()), {-1.0, -2.0, -1.0}, {12.0, 2.0, 8.0}));

    const float half = static_cast<float>(std::sqrt(0.5));
    rocks.ClearScales()
        .SetQuaternionOrder(QuaternionOrder::Wxyz)
        .SetOrientations(Floats({1, 0, 0, 0, half, half, 0, 0}, {2, 4}));
    EXPECT_TRUE(BoxNear(*scene.Extent(rocks.Id()), {-1.0, -4.0, -1.0}, {11.0, 1.0, 1.0}, 1e-6))
        << "the box turned 90 degrees about x, its offset now along -y";
}

TEST(PointInstancer, AnInstancerWithoutInstancesBoundsNothing) {
    Scene scene;
    PointInstancer rocks = PointInstancer::Define(scene, "/Rocks");
    EXPECT_TRUE(scene.Extent(rocks.Id())->Empty());
    rocks.SetExtent({-5, -5, -5, 5, 5, 5});
    EXPECT_TRUE(scene.Extent(rocks.Id())->Empty())
        << "the authored extent stands in only for instances on a device";
}

}  // namespace
}  // namespace ernie
