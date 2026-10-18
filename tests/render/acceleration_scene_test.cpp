// SPDX-License-Identifier: BSD-3-Clause
/// @file acceleration_scene_test.cpp
/// @brief Tests of the device scene's acceleration structures under validation: a mesh and
/// analytic spheres traced through the top level each frame builds, following the scene as prims
/// move, come and go, with bottom levels compacted in later frames.
#include "ernie/scene/xform.h"

#include "tests/render/common.h"

#include <gtest/gtest.h>

#include <vector>

namespace ernie {
namespace {

using test::kMiss;

class AccelerationScene : public test::RayTracingTest {
  protected:
    void SetUp() override {
        test::RayTracingTest::SetUp();
        if (IsSkipped()) {
            return;
        }
        // A 20 m ground at z = 0, and a sphere of radius 2 at (3, 0, 4) under a moved parent.
        test::DefineSquare(scene_, "/Ground", 10, 0);
        scene_.Define("/Props", "Xform");
        SetXform(scene_, "/Props", {.translate = glm::dvec3(1, 0, 0)});
        const PrimId ball = scene_.Define("/Props/Ball", "Sphere");
        scene_.Set(ball, "radius", 2.0);
        SetXform(scene_, "/Props/Ball", {.translate = glm::dvec3(2, 0, 4)});
    }

    /// @return The hit distance of a ray down -z from (x, y, 100) for each (x, y), or `kMiss`.
    std::vector<float> Frame(const std::vector<glm::vec2>& from) {
        std::vector<test::Ray> rays;
        for (const glm::vec2& at : from) {
            rays.push_back({glm::vec4(at, 100, 1), glm::vec4(0, 0, -1, 0)});
        }
        std::vector<float> distances;
        for (const glm::vec4& hit : Trace(rays)) {
            distances.push_back(hit.x);
        }
        return distances;
    }
};

TEST_F(AccelerationScene, TracesMeshesAndSpheresWhereTheSceneHasThem) {
    const std::vector<float> hits = Frame({{0, 0}, {3, 0}, {50, 0}});
    EXPECT_NEAR(hits[0], 100.0f, 1e-4f) << "the ground";
    EXPECT_NEAR(hits[1], 94.0f, 1e-4f) << "the ball's top, exactly round";
    EXPECT_EQ(hits[2], kMiss) << "beyond the ground";
    EXPECT_EQ(Device().Instances().Count(), 2u);
    EXPECT_EQ(Device().Tracer()->Bottoms().Pending(), 0u);
}

TEST_F(AccelerationScene, FollowsPrimsMovingComingAndGoing) {
    Frame({{0, 0}});
    // Frames pass while the ground's compacted size comes back and it moves into its compact copy.
    for (int frame = 0; frame < 3; ++frame) {
        EXPECT_NEAR(Frame({{0, 0}})[0], 100.0f, 1e-4f) << "frame " << frame;
    }
    const BottomLevels::Statistics& totals = Device().Tracer()->Bottoms().Totals();
    EXPECT_EQ(totals.built, 2u) << "the ground and the unit sphere, once each";
    EXPECT_EQ(totals.compacted, 2u) << "both, a frame or more after their builds";
    SetXform(scene_, "/Props/Ball", {.translate = glm::dvec3(-4, 0, 4)});
    const PrimId pebble = scene_.Define("/Pebble", "Sphere");
    SetXform(scene_, "/Pebble", {.translate = glm::dvec3(5, 5, 1)});
    scene_.Remove("/Ground");
    const std::vector<float> hits = Frame({{-3, 0}, {3, 0}, {5, 5}, {0, -8}});
    EXPECT_NEAR(hits[0], 94.0f, 1e-4f) << "the ball, moved";
    EXPECT_EQ(hits[1], kMiss) << "where the ball was";
    EXPECT_NEAR(hits[2], 98.0f, 1e-4f) << "the pebble, sharing the ball's unit sphere";
    EXPECT_EQ(hits[3], kMiss) << "the ground, removed";
    EXPECT_TRUE(Device().Instances().Row(pebble));
}

}  // namespace
}  // namespace ernie
