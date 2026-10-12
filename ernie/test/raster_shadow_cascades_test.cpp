// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/renderer.h"
#include "ernie/raster/shadow_cascades.h"
#include "ernie/scene/orbit_controller.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using ernie::CameraView;
using ernie::CascadeSplits;
using ernie::FitCascades;
using ernie::LightParams;
using ernie::LightType;
using ernie::OrbitController;
using ernie::ShadowCascade;

namespace {

constexpr std::uint32_t kResolution = 1024;
const glm::vec3 kSceneMin(-500.0f, -500.0f, -30.0f);
const glm::vec3 kSceneMax(500.0f, 500.0f, 40.0f);

CameraView TestCamera() {
    OrbitController camera;
    camera.target = {50.0f, -20.0f, 0.0f};
    camera.distance = 400.0f;
    camera.yaw = 0.4f;
    camera.pitch = 0.5f;
    return camera.MakeCamera(8, 5).RasterView();
}

/// Whether `p` lies inside the clip volume of `view`.
bool Sees(const ShadowCascade& view, const glm::vec3& p) {
    const glm::vec4 clip =
        view.lightViewProjection * glm::vec4(p, 1.0f);
    if (clip.w <= 0.0f) {
        return false;
    }
    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
    return std::abs(ndc.x) <= 1.0f && std::abs(ndc.y) <= 1.0f &&
           ndc.z >= 0.0f && ndc.z <= 1.0f;
}

glm::vec3 Project(const glm::mat4& m, const glm::vec3& p) {
    const glm::vec4 clip = m * glm::vec4(p, 1.0f);
    return glm::vec3(clip) / clip.w;
}

glm::vec3 SliceCorner(const CameraView& camera, float depth, int i) {
    const float tanY = std::tan(0.5f * camera.verticalFov);
    const float tanX = tanY * camera.aspect;
    const glm::vec4 inView((i & 1 ? 1.0f : -1.0f) * tanX * depth,
                           (i & 2 ? 1.0f : -1.0f) * tanY * depth,
                           -depth, 1.0f);
    return glm::vec3(glm::inverse(camera.view) * inView);
}

}  // namespace

TEST(CascadeSplits, GrowEachCascadeByTheExponent) {
    // Exponent 3 over 40 m from 1 m: spans of 1, 3, 9 and 27 parts.
    const auto splits = CascadeSplits(1.0f, 41.0f, 4, 3.0f);
    ASSERT_EQ(splits.size(), 5u);
    EXPECT_FLOAT_EQ(splits[0], 1.0f);
    EXPECT_FLOAT_EQ(splits[1], 2.0f);
    EXPECT_FLOAT_EQ(splits[2], 5.0f);
    EXPECT_FLOAT_EQ(splits[3], 14.0f);
    EXPECT_FLOAT_EQ(splits[4], 41.0f);
    // Exponent 1: uniform.
    const auto uniform = CascadeSplits(1.0f, 1001.0f, 4, 1.0f);
    EXPECT_FLOAT_EQ(uniform[1], 251.0f);
    EXPECT_FLOAT_EQ(uniform[2], 501.0f);
}

TEST(FitCascades,
     CoversEveryFrustumSliceAndTheCastersTowardsTheLight) {
    const CameraView camera = TestCamera();
    const glm::vec3 toSun =
        glm::normalize(glm::vec3(0.6f, -0.3f, 0.5f));
    const auto cascades =
        FitCascades(camera, 1200.0f, toSun, kSceneMin, kSceneMax, 4,
                    3.0f, kResolution);
    ASSERT_EQ(cascades.size(), 4u);
    EXPECT_FLOAT_EQ(cascades.front().splitNear, camera.nearPlane);
    // The distance up to the next quarter stop.
    EXPECT_GE(cascades.back().splitFar, 1200.0f);
    EXPECT_LT(cascades.back().splitFar, 1200.0f * std::exp2(0.25f));

    for (std::size_t c = 0; c < cascades.size(); ++c) {
        const ShadowCascade& cascade = cascades[c];
        if (c > 0) {
            EXPECT_FLOAT_EQ(cascade.splitNear,
                            cascades[c - 1].splitFar);
            EXPECT_GT(cascade.texelWorldSize,
                      cascades[c - 1].texelWorldSize);
        }
        for (const float depth :
             {cascade.splitNear, cascade.splitFar}) {
            for (int i = 0; i < 4; ++i) {
                const glm::vec3 ndc =
                    Project(cascade.lightViewProjection,
                            SliceCorner(camera, depth, i));
                EXPECT_LE(std::fabs(ndc.x), 1.0f) << "cascade " << c;
                EXPECT_LE(std::fabs(ndc.y), 1.0f) << "cascade " << c;
                EXPECT_GE(ndc.z, 0.0f) << "cascade " << c;
                EXPECT_LE(ndc.z, 1.0f) << "cascade " << c;
            }
        }
        // Without depth clamping, no caster nearer the light is cut.
        for (int i = 0; i < 8; ++i) {
            const glm::vec3 corner(
                (i & 1) ? kSceneMax.x : kSceneMin.x,
                (i & 2) ? kSceneMax.y : kSceneMin.y,
                (i & 4) ? kSceneMax.z : kSceneMin.z);
            const float depth =
                Project(cascade.lightViewProjection, corner).z;
            EXPECT_GE(depth, 0.0f) << "cascade " << c;
        }
    }
}

TEST(FitCascades, SnapsToWholeTexelsSoShadowsDoNotShimmer) {
    CameraView camera = TestCamera();
    const glm::vec3 toSun =
        glm::normalize(glm::vec3(-0.2f, 0.4f, 0.8f));
    for (const float shift : {0.0f, 0.013f, 0.37f, 2.9f}) {
        camera.view = TestCamera().view *
                      glm::mat4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                                -shift, shift, 0.0f, 1.0f);
        for (const ShadowCascade& cascade :
             FitCascades(camera, 1000.0f, toSun, kSceneMin, kSceneMax,
                         3, 3.0f, kResolution)) {
            const glm::vec3 ndc =
                Project(cascade.lightViewProjection, glm::vec3(0.0f));
            const float texelX = (0.5f * ndc.x + 0.5f) * kResolution;
            const float texelY = (0.5f * ndc.y + 0.5f) * kResolution;
            EXPECT_NEAR(texelX, std::round(texelX), 1e-2f);
            EXPECT_NEAR(texelY, std::round(texelY), 1e-2f);
        }
    }
}

TEST(FitCascades, HandlesAnOverheadSun) {
    const auto cascades =
        FitCascades(TestCamera(), 800.0f, glm::vec3(0.0f, 0.0f, 1.0f),
                    kSceneMin, kSceneMax, 2, 3.0f, kResolution);
    ASSERT_EQ(cascades.size(), 2u);
    const glm::vec3 ndc = Project(cascades[0].lightViewProjection,
                                  glm::vec3(0.0f, 0.0f, 40.0f));
    EXPECT_TRUE(std::isfinite(ndc.x));
    EXPECT_GE(ndc.z, 0.0f);
}

// A flat scene straight under the sun spans no depth along it: the
// cascades still span at least kCascadeMinDepthHalf either way, so
// receivers in them lie well inside rather than on a clipping plane,
// clamped or not.
TEST(FitCascades, FlatScenesUnderAnOverheadSunKeepTheirDepth) {
    const CameraView camera = TestCamera();
    const glm::vec3 low(-10.0f, -10.0f, 0.0f);
    const glm::vec3 high(10.0f, 10.0f, 0.0f);
    for (const bool clamp : {false, true}) {
        for (const ShadowCascade& cascade : FitCascades(
                 camera, 800.0f, glm::vec3(0.0f, 0.0f, 1.0f), low,
                 high, 3, 3.0f, kResolution, 0.0f, clamp)) {
            // A receiver on the view's axis, mid-cascade.
            const glm::vec3 receiver =
                camera.eye +
                camera.forward *
                    (0.5f * (cascade.splitNear + cascade.splitFar));
            const float a =
                Project(cascade.lightViewProjection, receiver).z;
            const float b =
                Project(cascade.lightViewProjection,
                        receiver + glm::vec3(0.0f, 0.0f, 1.0f))
                    .z;
            EXPECT_GT(a, 0.0f) << "clamp " << clamp;
            EXPECT_LT(a, 1.0f) << "clamp " << clamp;
            // A metre towards the light is a small step in depth.
            EXPECT_LT(a - b,
                      1.0f / (2.0f * ernie::kCascadeMinDepthHalf) *
                          1.01f)
                << "clamp " << clamp;
            EXPECT_GT(a - b, 0.0f) << "clamp " << clamp;
        }
    }
}

TEST(WithShadows, GrowsTheBoxToWhereItsShadowsLand) {
    // A 2 m box on the ground under a sun 45 degrees up towards -y:
    // its top casts 2 m along +y.
    const glm::vec3 toSun =
        glm::normalize(glm::vec3(0.0f, -1.0f, 1.0f));
    const auto [low, high] =
        ernie::WithShadows({-1, -1, 0}, {1, 1, 2}, toSun);
    EXPECT_NEAR(high.y, 3.0f, 1e-5f);
    EXPECT_NEAR(low.y, -1.0f, 1e-5f);
    EXPECT_NEAR(low.z, 0.0f, 1e-5f);
    EXPECT_NEAR(high.z, 2.0f, 1e-5f);
    // A grazing sun stretches it at most `maxLength` diagonals; one
    // below the horizon not at all.
    const glm::vec3 grazing =
        glm::normalize(glm::vec3(0.0f, -1.0f, 1e-3f));
    const auto [grazingLow, grazingHigh] =
        ernie::WithShadows({-1, -1, 0}, {1, 1, 2}, grazing, 2.0f);
    EXPECT_LE(grazingHigh.y,
              1.0f + 2.0f * glm::length(glm::vec3(2, 2, 2)) + 1e-4f);
    const auto [belowLow, belowHigh] = ernie::WithShadows(
        {-1, -1, 0}, {1, 1, 2}, glm::vec3(0.0f, 0.0f, -1.0f));
    EXPECT_EQ(belowHigh, glm::vec3(1, 1, 2));
}

TEST(FitLightShadowViews, CoversTheSceneFromEachKindOfLight) {
    const glm::vec3 low(-10.0f, -10.0f, 0.0f);
    const glm::vec3 high(10.0f, 10.0f, 2.0f);
    LightParams distant;
    distant.type = LightType::Distant;
    distant.direction = glm::normalize(glm::vec3(1.0f, 0.0f, 1.0f));
    const std::vector<ShadowCascade> ortho =
        ernie::FitLightShadowViews(distant, glm::vec3(0.0f), low,
                                   high, 512);
    ASSERT_EQ(ortho.size(), 1u);
    for (int i = 0; i < 8; ++i) {
        const glm::vec3 corner((i & 1) ? high.x : low.x,
                               (i & 2) ? high.y : low.y,
                               (i & 4) ? high.z : low.z);
        EXPECT_TRUE(Sees(ortho[0], corner)) << "corner " << i;
    }
    EXPECT_NEAR(ortho[0].texelWorldSize,
                glm::length(high - low) / 512.0f, 1e-4f);

    LightParams spot;
    const glm::vec3 position(0.0f, 0.0f, 10.0f);
    spot.position = position;
    spot.direction = glm::vec3(0.0f, 0.0f, -1.0f);
    spot.coneAngle = 30.0f;
    const std::vector<ShadowCascade> one =
        ernie::FitLightShadowViews(spot, position, low, high, 512);
    ASSERT_EQ(one.size(), 1u);
    EXPECT_TRUE(Sees(one[0], {0.0f, 0.0f, 0.0f}));
    EXPECT_TRUE(Sees(one[0], {5.0f, 0.0f, 0.0f}))
        << "inside the 30 degree cone";
    EXPECT_FALSE(Sees(one[0], {9.0f, 0.0f, 0.0f})) << "outside it";
    EXPECT_NEAR(one[0].texelWorldSize,
                2.0f * std::tan(glm::radians(30.0f)) / 512.0f, 1e-6f);

    spot.coneAngle = 180.0f;
    const std::vector<ShadowCascade> cube =
        ernie::FitLightShadowViews(spot, position, low, high, 512);
    ASSERT_EQ(cube.size(), 6u);
    const glm::vec3 axes[] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                              {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
    for (std::size_t face = 0; face < 6; ++face) {
        EXPECT_TRUE(Sees(cube[face], position + 5.0f * axes[face]))
            << "face " << face;
        EXPECT_FALSE(Sees(cube[face], position - 5.0f * axes[face]))
            << "face " << face;
    }

    spot.shadows = false;
    EXPECT_TRUE(
        ernie::FitLightShadowViews(spot, position, low, high, 512)
            .empty());
}

TEST(FitLightShadowViews, LasersLookAlongTheirBeam) {
    // A beam 1 mm at its waist, slanted 45 degrees down from 10 m up:
    // its view sees along it, a point halfway nearer than the end.
    const glm::vec3 low(-10.0f, -10.0f, 0.0f);
    const glm::vec3 high(10.0f, 10.0f, 10.0f);
    LightParams laser;
    laser.type = LightType::Laser;
    const glm::vec3 position(-10.0f, 0.0f, 10.0f);
    laser.position = position;
    laser.direction = glm::normalize(glm::vec3(1.0f, 0.0f, -1.0f));
    laser.beam.waist = glm::vec2(0.001f);
    laser.beam.rayleigh = glm::vec2(5.0f);
    const std::vector<ShadowCascade> views =
        ernie::FitLightShadowViews(laser, position, low, high, 512);
    ASSERT_EQ(views.size(), 1u);
    const glm::vec3 halfway(-5.0f, 0.0f, 5.0f);
    const glm::vec3 end(0.0f, 0.0f, 0.0f);
    EXPECT_TRUE(Sees(views[0], halfway));
    EXPECT_TRUE(Sees(views[0], end));
    EXPECT_FALSE(Sees(views[0], {0.0f, 1.0f, 0.0f}))
        << "a metre off the beam";
    const auto depth = [&](const glm::vec3& p) {
        const glm::vec4 clip =
            views[0].lightViewProjection * glm::vec4(p, 1.0f);
        return clip.z / clip.w;
    };
    EXPECT_LT(depth(halfway), depth(end));
}

// Maps that clamp depth span only each cascade's slice in depth, so a
// planet-sized scene keeps depth precision: every slice corner lies
// within the depth range, which is about the slice's size.
TEST(FitCascades, ClampedCasterDepthsSpanOnlyTheSlice) {
    const CameraView camera = TestCamera();
    const glm::vec3 toSun =
        glm::normalize(glm::vec3(0.6f, -0.3f, 0.5f));
    const glm::vec3 planetMin(-6.4e6f, -6.4e6f, -12.8e6f);
    const glm::vec3 planetMax(6.4e6f, 6.4e6f, 40.0f);
    const auto cascades =
        FitCascades(camera, 1200.0f, toSun, planetMin, planetMax, 4,
                    3.0f, kResolution, 0.0f, true);
    ASSERT_EQ(cascades.size(), 4u);
    for (std::size_t c = 0; c < cascades.size(); ++c) {
        const ShadowCascade& cascade = cascades[c];
        // The depth one unit of the map's range covers, in metres.
        const glm::vec3 a =
            Project(cascade.lightViewProjection, glm::vec3(0.0f));
        const glm::vec3 b =
            Project(cascade.lightViewProjection, toSun);
        const float range = 1.0f / std::fabs(b.z - a.z);
        EXPECT_LT(range, 4.0f * cascade.splitFar +
                             2.0f * ernie::kCascadeMinDepthHalf)
            << "cascade " << c;
        for (const float depth :
             {cascade.splitNear, cascade.splitFar}) {
            for (int i = 0; i < 4; ++i) {
                EXPECT_TRUE(
                    Sees(cascade, SliceCorner(camera, depth, i)))
                    << "cascade " << c;
            }
        }
    }
}

// A shadow distance that drifts within a quarter stop, as the view
// turns, gives the same cascades: their snapped texels hold still.
TEST(FitCascades, DriftingDistancesKeepTheCascades) {
    const CameraView camera = TestCamera();
    const glm::vec3 toSun =
        glm::normalize(glm::vec3(0.6f, -0.3f, 0.5f));
    const auto fit = [&](float distance) {
        return FitCascades(camera, distance, toSun, kSceneMin,
                           kSceneMax, 4, 3.0f, kResolution);
    };
    const auto a = fit(1030.0f);
    const auto b = fit(1060.0f);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t c = 0; c < a.size(); ++c) {
        EXPECT_EQ(a[c].splitFar, b[c].splitFar) << "cascade " << c;
        EXPECT_EQ(a[c].lightViewProjection, b[c].lightViewProjection)
            << "cascade " << c;
    }
}
