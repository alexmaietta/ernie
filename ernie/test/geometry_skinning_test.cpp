// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/geometry/skinning.h"

#include <gtest/gtest.h>

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie {
namespace {

constexpr float kPi = 3.14159265358979f;

void ExpectNear(const glm::vec3& a, const glm::vec3& b,
                float tolerance = 1e-5f) {
    EXPECT_NEAR(a.x, b.x, tolerance);
    EXPECT_NEAR(a.y, b.y, tolerance);
    EXPECT_NEAR(a.z, b.z, tolerance);
}

TEST(SkinningTest, JointParentsFollowThePaths) {
    const std::vector<std::string> joints = {
        "Hips", "Hips/Spine", "Hips/Spine/Neck", "Hips/LegL"};
    EXPECT_EQ(JointParents(joints), (std::vector<int>{-1, 0, 1, 0}));
    const std::vector<std::string> orphan = {"Hips/Spine", "Hips"};
    EXPECT_THROW(JointParents(orphan), std::invalid_argument);
    const std::vector<std::string> twice = {"Hips", "Hips"};
    EXPECT_THROW(JointParents(twice), std::invalid_argument);
}

// An arm along x: the shoulder at the origin, the elbow at x = 1,
// bound in the rest pose; the elbow bent 90 degrees about z carries a
// point of the forearm around it.
TEST(SkinningTest, LinearBlendSkinningFollowsTheJoints) {
    const std::vector<int> parents = {-1, 0};
    const std::vector<glm::mat4> rest = {
        glm::mat4(1.0f),
        glm::translate(glm::mat4(1.0f), {1.0f, 0.0f, 0.0f})};
    const std::vector<glm::mat4> bind =
        SkelSpaceTransforms(parents, rest);
    const std::vector<glm::mat4> posed = {
        rest[0],
        JointTransform({1.0f, 0.0f, 0.0f},
                       glm::angleAxis(0.5f * kPi, glm::vec3(0, 0, 1)),
                       glm::vec3(1.0f))};
    const std::vector<glm::mat4> skinning =
        SkinningTransforms(SkelSpaceTransforms(parents, posed), bind);
    std::vector<float> points = {2.0f, 0.0f, 0.0f, 0.5f, 0.0f,
                                 0.0f, 1.0f, 0.0f, 0.0f};
    const std::vector<int> indices = {1, 0, 0, 0, 0, 1};
    std::vector<float> weights = {1.0f, 0.0f, 3.0f, 0.0f, 2.0f, 2.0f};
    NormalizeWeights(weights, 2);
    EXPECT_FLOAT_EQ(weights[2], 1.0f);
    EXPECT_FLOAT_EQ(weights[4], 0.5f);
    SkinPoints(SkinningMethod::Linear, glm::mat4(1.0f), skinning,
               indices, weights, 2, false, points);
    ExpectNear({points[0], points[1], points[2]}, {1.0f, 1.0f, 0.0f});
    ExpectNear({points[3], points[4], points[5]}, {0.5f, 0.0f, 0.0f});
    // On the elbow both joints agree.
    ExpectNear({points[6], points[7], points[8]}, {1.0f, 0.0f, 0.0f});

    // geomBindTransform moves the points into the skeleton's space
    // first; one set of influences moves every point rigidly.
    std::vector<float> rigid = {1.0f, 0.0f, 0.0f};
    const std::vector<int> one = {1};
    const std::vector<float> full = {1.0f};
    SkinPoints(SkinningMethod::Linear,
               glm::translate(glm::mat4(1.0f), {1.0f, 0.0f, 0.0f}),
               skinning, one, full, 1, true, rigid);
    ExpectNear({rigid[0], rigid[1], rigid[2]}, {1.0f, 1.0f, 0.0f});
    const std::vector<int> outOfRange = {2};
    EXPECT_THROW(SkinPoints(SkinningMethod::Linear, glm::mat4(1.0f),
                            skinning, outOfRange, full, 1, true,
                            rigid),
                 std::invalid_argument);
}

// A point between two joints, one twisted half a turn about the bone:
// linear blending collapses it onto the bone (the candy wrapper),
// dual quaternions keep its distance from the bone.
TEST(SkinningTest,
     DualQuaternionsKeepVolumeWhereLinearBlendingCollapses) {
    const std::vector<glm::mat4> skinning = {
        glm::mat4(1.0f), glm::mat4_cast(glm::angleAxis(
                             kPi * 0.999f, glm::vec3(1, 0, 0)))};
    const std::vector<int> indices = {0, 1};
    const std::vector<float> weights = {0.5f, 0.5f};
    std::vector<float> linear = {0.0f, 1.0f, 0.0f};
    std::vector<float> dual = linear;
    SkinPoints(SkinningMethod::Linear, glm::mat4(1.0f), skinning,
               indices, weights, 2, true, linear);
    SkinPoints(SkinningMethod::DualQuaternion, glm::mat4(1.0f),
               skinning, indices, weights, 2, true, dual);
    EXPECT_LT(std::hypot(linear[1], linear[2]), 0.01f);
    EXPECT_NEAR(std::hypot(dual[1], dual[2]), 1.0f, 1e-4f);
    EXPECT_NEAR(dual[0], 0.0f, 1e-5f);
}

// One joint each: both methods give the joint's transform, scale
// included (dual quaternions take it apart and put it back).
TEST(SkinningTest, DualQuaternionsMatchLinearBlendingForOneJoint) {
    const glm::mat4 m =
        glm::translate(glm::mat4(1.0f), {0.3f, -2.0f, 1.0f}) *
        glm::mat4_cast(glm::angleAxis(
            0.7f, glm::normalize(glm::vec3(1, 2, 3)))) *
        glm::scale(glm::mat4(1.0f), {2.0f, 0.5f, 1.5f});
    const std::vector<glm::mat4> skinning = {m};
    const std::vector<int> indices = {0};
    const std::vector<float> weights = {1.0f};
    std::vector<float> linear = {0.2f, 0.4f, -0.8f, 1.0f, 1.0f, 1.0f};
    std::vector<float> dual = linear;
    SkinPoints(SkinningMethod::Linear, glm::mat4(1.0f), skinning,
               indices, weights, 1, true, linear);
    SkinPoints(SkinningMethod::DualQuaternion, glm::mat4(1.0f),
               skinning, indices, weights, 1, true, dual);
    for (std::size_t i = 0; i < linear.size(); ++i) {
        EXPECT_NEAR(linear[i], dual[i], 1e-4f);
    }
}

TEST(SkinningTest, AnimationSamplesInterpolate) {
    const std::vector<double> times = {0.0, 10.0};
    const std::vector<float> translations = {0, 0, 0, 10, 0, 0};
    const glm::quat quarter =
        glm::angleAxis(0.5f * kPi, glm::vec3(0, 0, 1));
    const std::vector<float> rotations = {
        0, 0, 0, 1, quarter.x, quarter.y, quarter.z, quarter.w};
    const std::vector<float> weights = {0.0f, 1.0f};
    const AnimationPose half = SampleAnimation(
        times, 1, 1, translations, rotations, {}, weights, 5.0);
    ExpectNear(half.translations[0], {5.0f, 0.0f, 0.0f});
    EXPECT_NEAR(glm::angle(half.rotations[0]), 0.25f * kPi, 1e-5f);
    EXPECT_FLOAT_EQ(half.blendShapeWeights[0], 0.5f);
    EXPECT_TRUE(half.scales.empty());
    // Held outside the samples.
    ExpectNear(SampleAnimation(times, 1, 1, translations, rotations,
                               {}, weights, -3.0)
                   .translations[0],
               {0.0f, 0.0f, 0.0f});
    ExpectNear(SampleAnimation(times, 1, 1, translations, rotations,
                               {}, weights, 30.0)
                   .translations[0],
               {10.0f, 0.0f, 0.0f});
    EXPECT_THROW(
        SampleAnimation(times, 2, 1, translations, {}, {}, {}, 0.0),
        std::invalid_argument);
    const std::vector<std::string> from = {"b", "c", "a"};
    const std::vector<std::string> to = {"a", "b"};
    EXPECT_EQ(NameMapping(from, to), (std::vector<int>{1, -1, 0}));
}

TEST(SkinningTest, InbetweensSplitTheWeight) {
    using Weights = std::vector<std::pair<int, float>>;
    EXPECT_EQ(SubShapeWeights(0.7f, {}), (Weights{{0, 0.7f}}));
    const std::vector<float> half = {0.5f};
    const auto expect = [&](float weight, const Weights& expected) {
        const Weights got = SubShapeWeights(weight, half);
        ASSERT_EQ(got.size(), expected.size()) << weight;
        for (std::size_t i = 0; i < got.size(); ++i) {
            EXPECT_EQ(got[i].first, expected[i].first) << weight;
            EXPECT_NEAR(got[i].second, expected[i].second, 1e-6f)
                << weight;
        }
    };
    expect(0.25f, {{1, 0.5f}});
    expect(0.5f, {{1, 1.0f}});
    expect(0.75f, {{1, 0.5f}, {0, 0.5f}});
    expect(1.5f, {{1, -1.0f}, {0, 2.0f}});
    expect(-0.5f, {{1, -1.0f}});
    expect(0.0f, {});
}

TEST(SkinningTest, BlendShapesOffsetPoints) {
    std::vector<float> points = {0, 0, 0, 1, 1, 1};
    const std::vector<float> dense = {1, 0, 0, 0, 2, 0};
    ApplyBlendShape(points, dense, {}, 0.5f);
    EXPECT_EQ(points, (std::vector<float>{0.5f, 0, 0, 1, 2, 1}));
    const std::vector<float> sparse = {0, 0, 4};
    const std::vector<int> second = {1};
    ApplyBlendShape(points, sparse, second, 0.25f);
    EXPECT_EQ(points, (std::vector<float>{0.5f, 0, 0, 1, 2, 2}));
    const std::vector<int> beyond = {2};
    EXPECT_THROW(ApplyBlendShape(points, sparse, beyond, 1.0f),
                 std::invalid_argument);
}

}  // namespace
}  // namespace ernie
