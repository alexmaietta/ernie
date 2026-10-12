// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/virtual_shadow_map.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

using ernie::VirtualShadowLevel;
using ernie::VirtualShadowMap;

namespace {

glm::dvec3 Clip(const VirtualShadowLevel& level,
                const glm::dvec3& p) {
    return glm::dvec3(glm::dmat4(level.viewProjection) *
                      glm::dvec4(p, 1.0));
}

/// The virtual texel holding `p` (relative to the origin) in `level`.
glm::ivec2 TexelOf(const VirtualShadowLevel& level,
                   const glm::dvec3& p) {
    const glm::dvec3 clip = Clip(level, p);
    return glm::ivec2(glm::floor(
        (glm::dvec2(clip) * 0.5 + 0.5) *
        static_cast<double>(VirtualShadowMap::kVirtualTexels)));
}

const glm::dvec3 kToSun = glm::normalize(glm::dvec3(0.4, -0.3, 0.8));

}  // namespace

TEST(VirtualShadowMap, LevelsDoubleAndCentreOnTheCamera) {
    const glm::dvec3 eye(1234.5, -678.9, 12.0);
    const glm::dvec3 origin = eye + glm::dvec3(3.0, -2.0, 1.0);
    const auto levels =
        VirtualShadowMap::FitLevels(kToSun, eye, origin, -4, 20);
    ASSERT_EQ(levels.size(), 25u);
    for (std::size_t i = 0; i < levels.size(); ++i) {
        const VirtualShadowLevel& level = levels[i];
        EXPECT_EQ(level.level, -4 + static_cast<int>(i));
        EXPECT_DOUBLE_EQ(level.radius,
                         std::ldexp(1.0, level.level + 1));
        EXPECT_DOUBLE_EQ(level.texelWorldSize,
                         4.0 * level.radius / 16384.0);
        // Snapped to its radius, the camera is within half a radius
        // of the centre: a quarter of the way to the edge.
        const glm::dvec3 clip = Clip(level, eye - origin);
        EXPECT_LE(std::abs(clip.x), 0.25 + 1e-6) << "level " << i;
        EXPECT_LE(std::abs(clip.y), 0.25 + 1e-6) << "level " << i;
        EXPECT_NEAR(clip.z, 0.5, 1e-6) << "level " << i;
    }
}

TEST(VirtualShadowMap, NearerTheLightIsDeeper) {
    const auto levels = VirtualShadowMap::FitLevels(
        kToSun, glm::dvec3(0.0), glm::dvec3(0.0), 0, 0);
    const glm::dvec3 low = Clip(levels[0], glm::dvec3(0.0));
    const glm::dvec3 high = Clip(levels[0], kToSun * 1.0);
    EXPECT_GT(high.z, low.z);
    EXPECT_NEAR(high.x, low.x, 1e-6);
    EXPECT_NEAR(high.y, low.y, 1e-6);
}

// Pages hold still: a world point keeps its texel as the camera
// moves within a level's snap and as the origin rebases, on a
// planet's scale too.
TEST(VirtualShadowMap, TexelsHoldStillAsTheCameraAndOriginMove) {
    const glm::dvec3 world(6371008.8, 120.0, -45.0);
    const glm::dvec3 eye = world + glm::dvec3(0.3, 0.2, 2.0);
    const auto a =
        VirtualShadowMap::FitLevels(kToSun, eye, eye, 4, 4);
    const glm::dvec3 moved = eye + glm::dvec3(1.0, -0.5, 0.25);
    const glm::dvec3 rebased = moved + glm::dvec3(700.0, 300.0, 0.0);
    const auto b =
        VirtualShadowMap::FitLevels(kToSun, moved, rebased, 4, 4);
    EXPECT_EQ(TexelOf(a[0], world - eye),
              TexelOf(b[0], world - rebased));
}

TEST(VirtualShadowMap, LevelsSuitTheDistance) {
    EXPECT_EQ(VirtualShadowMap::LevelFor(1.0, 0.0), 0);
    EXPECT_EQ(VirtualShadowMap::LevelFor(1.9, 0.0), 0);
    EXPECT_EQ(VirtualShadowMap::LevelFor(2.0, 0.0), 1);
    EXPECT_EQ(VirtualShadowMap::LevelFor(0.3, 0.0), -2);
    EXPECT_EQ(VirtualShadowMap::LevelFor(1000.0, -1.0), 8);
    // A receiver's level holds it: within R of the camera.
    for (const double d : {0.7, 3.0, 50.0, 4.2e5}) {
        const double r =
            std::ldexp(1.0, VirtualShadowMap::LevelFor(d, 0.0) + 1);
        EXPECT_LT(d, r);
    }
}

// The Sun's penumbrae filter in a receiver's own level; a wide
// light's in as many coarser ones as span them, up to the most.
TEST(VirtualShadowMap, WideLightsFilterInCoarserLevels) {
    EXPECT_EQ(VirtualShadowMap::FilterLevelsFor(0.0), 0u);
    EXPECT_EQ(VirtualShadowMap::FilterLevelsFor(
                  std::tan(0.5 * 0.53 * std::numbers::pi / 180.0)),
              0u);
    // 4096 tan(5 degrees) is 358 texels: three levels of 64.
    EXPECT_EQ(VirtualShadowMap::FilterLevelsFor(
                  std::tan(5.0 * std::numbers::pi / 180.0)),
              3u);
    EXPECT_EQ(VirtualShadowMap::FilterLevelsFor(1.0),
              VirtualShadowMap::kMaxFilterLevels);
}

// A level draws only the casters over where its receivers can be:
// what the camera sees at the distances it suits, not what lies
// behind it; a level for distances nearer than the near plane draws
// nothing.
TEST(VirtualShadowMap, LevelsCullToTheirReceivers) {
    // 10 m up looking along +y, reversed-Z, the near plane 0.1 m
    // away.
    const glm::dvec3 eye(0.0, 0.0, 10.0);
    const glm::dmat4 view = glm::lookAtRH(
        eye, eye + glm::dvec3(0, 1, 0), glm::dvec3(0.0, 0.0, 1.0));
    glm::dmat4 reverse(1.0);
    reverse[2][2] = -1.0;
    reverse[3][2] = 1.0;
    const glm::dmat4 projection =
        reverse *
        glm::perspectiveRH_ZO(glm::radians(60.0), 1.0, 0.1, 1.0e4);
    const glm::mat4 inverse(glm::inverse(projection * view));
    const auto levels = VirtualShadowMap::FitLevels(
        kToSun, eye, glm::dvec3(0.0), -4, 22);
    constexpr float kBias = 1.0f;
    // Level -4 holds distances to 2^(-4) m, short of the near plane.
    EXPECT_FALSE(VirtualShadowMap::CasterCull(
        levels[0], inverse, glm::vec3(eye), kBias, 0, false, false));
    // Level 5 holds 16 to 32 m.
    const VirtualShadowLevel& level = levels[5 + 4];
    ASSERT_EQ(level.level, 5);
    const std::optional<glm::mat4> cull =
        VirtualShadowMap::CasterCull(level, inverse, glm::vec3(eye),
                                     kBias, 0, false, false);
    ASSERT_TRUE(cull);
    const auto inside = [&](const glm::dvec3& p) {
        const glm::dvec4 clip =
            glm::dmat4(*cull) * glm::dvec4(p, 1.0);
        return std::abs(clip.x) <= 1.0 && std::abs(clip.y) <= 1.0;
    };
    EXPECT_TRUE(inside(eye + glm::dvec3(0.0, 20.0, 0.0))) << "seen";
    EXPECT_TRUE(inside(eye + glm::dvec3(8.0, 20.0, -5.0)))
        << "seen off the axis";
    EXPECT_FALSE(inside(eye + glm::dvec3(0.0, -20.0, 0.0)))
        << "behind the camera";
    EXPECT_GT(std::abs((*cull)[0][0]),
              std::abs(level.viewProjection[0][0]))
        << "narrower than the level";
}
