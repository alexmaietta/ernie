// SPDX-License-Identifier: BSD-3-Clause
/// @file extent_cache_test.cpp
/// @brief Tests that cached extents follow edits, removals and additions.
#include "ernie/scene/schemas/geom.h"
#include "ernie/util/random.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace ernie {
namespace {

/// Spheres whose radii the test tracks, so every extent can be checked against its own.
struct Balls {
    Scene scene;
    std::vector<PrimId> prims;
    std::vector<double> radii;
    int next = 0;

    void Add(double radius) {
        Sphere ball = Sphere::Define(scene, "/Ball_" + std::to_string(next++));
        ball.SetRadius(radius);
        prims.push_back(ball.Id());
        radii.push_back(radius);
    }
    void Remove(std::size_t i) {
        scene.Remove(scene.PrimPath(prims[i]));
        prims.erase(prims.begin() + static_cast<std::ptrdiff_t>(i));
        radii.erase(radii.begin() + static_cast<std::ptrdiff_t>(i));
    }
    void ExpectExtents() const {
        for (std::size_t i = 0; i < prims.size(); ++i) {
            EXPECT_EQ(scene.Extent(prims[i])->max, glm::dvec3(radii[i])) << "ball " << i;
        }
    }
};

TEST(ExtentCache, CachedExtentsFollowEditsBeforeAndAfterAnUpdate) {
    Balls balls;
    for (int i = 0; i < 100; ++i) {
        balls.Add(1.0 + i);
    }
    balls.scene.Update();
    balls.ExpectExtents();

    Sphere(balls.scene, balls.scene.PrimPath(balls.prims[7])).SetRadius(42.0);
    balls.radii[7] = 42.0;
    balls.ExpectExtents();  // computed afresh: the bucket changed since the update
    balls.scene.Update();
    balls.ExpectExtents();
}

TEST(ExtentCache, RandomEditsRemovalsAndAdditionsKeepEveryExtentRight) {
    Balls balls;
    SplitMix64 random(3);
    for (int i = 0; i < 300; ++i) {
        balls.Add(1.0 + i);
    }
    for (int step = 0; step < 40; ++step) {
        for (int op = 0; op < 20; ++op) {
            const std::uint64_t pick = random.NextBits();
            const std::size_t i = pick % balls.prims.size();
            if (pick % 3 == 0) {
                balls.Remove(i);
            } else if (pick % 3 == 1) {
                balls.Add(static_cast<double>(pick % 1000));
            } else {
                const double radius = static_cast<double>(pick % 977);
                Sphere(balls.scene, balls.scene.PrimPath(balls.prims[i])).SetRadius(radius);
                balls.radii[i] = radius;
            }
        }
        balls.scene.Update();
        balls.ExpectExtents();
    }
}

}  // namespace
}  // namespace ernie
