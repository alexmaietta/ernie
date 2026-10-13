// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/scene.h"
#include "ernie/scene/scene_json.h"
#include "ernie/scene/stage_schemas.h"
#include "ernie/scene/tokens.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

namespace ernie {
namespace {

void SetTime(Scene& scene, double time) {
    Stage(scene, Path::Root()).SetTimeCode(time);
}

// A sampled attribute holds its value at the stage's time code:
// interpolated between samples, held beyond them, and evaluated again
// as the time code moves; GetAt reads any time without moving it.
TEST(TimeSamplesTest, AttributesTakeTheirValueAtTheTimeCode) {
    Scene scene;
    scene.Define("/Lamp", tokens::kSphereLight);
    scene.SetAt("/Lamp", tokens::kRadius, 10.0, 1.0);
    scene.SetAt("/Lamp", tokens::kRadius, 20.0, 3.0);
    EXPECT_EQ(scene.SamplesOf("/Lamp", tokens::kRadius).size(), 2u);
    EXPECT_EQ(scene.Get<double>("/Lamp", tokens::kRadius, 0.0), 1.0)
        << "held before the first sample";
    SetTime(scene, 15.0);
    EXPECT_DOUBLE_EQ(scene.Get<double>("/Lamp", tokens::kRadius, 0.0),
                     2.0);
    SetTime(scene, 30.0);
    EXPECT_DOUBLE_EQ(scene.Get<double>("/Lamp", tokens::kRadius, 0.0),
                     3.0);
    EXPECT_DOUBLE_EQ(
        *scene.GetAt("/Lamp", tokens::kRadius, 12.5).Get<double>(),
        1.5);
    EXPECT_DOUBLE_EQ(scene.Get<double>("/Lamp", tokens::kRadius, 0.0),
                     3.0);

    // Tokens hold; quaternions turn the short way at a steady rate.
    scene.Define("/Arm", tokens::kXform);
    scene.SetAt("/Arm", tokens::kVisibility, 0.0,
                std::string(tokens::kInherited));
    scene.SetAt("/Arm", tokens::kVisibility, 40.0,
                std::string(tokens::kInvisible));
    EXPECT_EQ(*scene.GetAt("/Arm", tokens::kVisibility, 39.0)
                   .Get<std::string>(),
              tokens::kInherited);
    const glm::dquat quarter =
        glm::angleAxis(0.5 * M_PI, glm::dvec3(0, 0, 1));
    scene.SetAt("/Arm", "ernie:spin", 0.0, glm::dquat(1, 0, 0, 0));
    scene.SetAt("/Arm", "ernie:spin", 10.0, quarter);
    EXPECT_NEAR(glm::angle(*scene.GetAt("/Arm", "ernie:spin", 5.0)
                                .Get<glm::dquat>()),
                0.25 * M_PI, 1e-9);

    // A value set without a time clears the samples.
    scene.Set("/Lamp", tokens::kRadius, 7.0);
    EXPECT_TRUE(scene.SamplesOf("/Lamp", tokens::kRadius).empty());
    SetTime(scene, 10.0);
    EXPECT_EQ(scene.Get<double>("/Lamp", tokens::kRadius, 0.0), 7.0);
}

// Sampled transforms move prims and their children, and a new time
// code reports only the prims whose values changed.
TEST(TimeSamplesTest, SampledTransformsMoveAndReportOnlyRealChanges) {
    Scene scene;
    scene.Define("/Root", tokens::kXform);
    scene.Define("/Root/Child", tokens::kXform);
    scene.Define("/Still", tokens::kXform);
    scene.SetAt("/Root", tokens::kTransform, 0.0, glm::dmat4(1.0));
    scene.SetAt(
        "/Root", tokens::kTransform, 10.0,
        glm::translate(glm::dmat4(1.0), glm::dvec3(10.0, 0.0, 0.0)));
    scene.SetAt(
        "/Root", tokens::kTransform, 20.0,
        glm::translate(glm::dmat4(1.0), glm::dvec3(10.0, 0.0, 0.0)));
    EXPECT_EQ(scene.AnimatedPrims(), std::vector<Path>{"/Root"});

    ChangeCursor cursor;
    scene.Changes(cursor);
    SetTime(scene, 5.0);
    EXPECT_NEAR(scene.WorldTransform("/Root/Child")[3].x, 5.0, 1e-12);
    std::vector<Path> moved;
    for (const Change& change : scene.Changes(cursor)) {
        if (Any(change.bits, Dirty::Transform)) {
            moved.push_back(change.path);
        }
    }
    EXPECT_EQ(moved, (std::vector<Path>{"/Root", "/Root/Child"}));
    // Between two equal samples nothing changes after the first move.
    SetTime(scene, 12.0);
    scene.Changes(cursor);
    SetTime(scene, 15.0);
    for (const Change& change : scene.Changes(cursor)) {
        EXPECT_FALSE(Any(change.bits, Dirty::Transform))
            << change.path.String();
    }

    // Removed prims stop being evaluated; clones keep their samples.
    scene.Clone("/Root", "/Copy");
    EXPECT_EQ(scene.SamplesOf("/Copy", tokens::kTransform).size(),
              3u);
    scene.Remove("/Root");
    EXPECT_EQ(scene.AnimatedPrims(), std::vector<Path>{"/Copy"});
}

// Float tensors blend between samples on the CPU; on a sample or
// beyond the last, the sample itself is shared; other types hold.
TEST(TimeSamplesTest, TensorsBlendOrShareTheirSamples) {
    Scene scene;
    scene.Define("/Cloth", tokens::kMesh);
    const Tensor flat = Tensor::Copy<float>(
        std::vector<float>{0, 0, 0, 1, 0, 0}, {2, 3});
    const Tensor raised = Tensor::Copy<float>(
        std::vector<float>{0, 0, 2, 1, 0, 4}, {2, 3});
    scene.WriteAt("/Cloth", tokens::kPoints, 0.0, flat);
    scene.WriteAt("/Cloth", tokens::kPoints, 10.0, raised);
    EXPECT_EQ(scene.Read("/Cloth", tokens::kPoints)->Data(),
              flat.Data());
    SetTime(scene, 5.0);
    // A consumer blending on the device reads where the time falls.
    const std::optional<SampleBracket> bracket =
        scene.BracketAt("/Cloth", tokens::kPoints);
    ASSERT_TRUE(bracket.has_value());
    EXPECT_EQ(bracket->lower, 0u);
    EXPECT_EQ(bracket->upper, 1u);
    EXPECT_DOUBLE_EQ(bracket->alpha, 0.5);
    const Tensor* half = scene.Read("/Cloth", tokens::kPoints);
    ASSERT_NE(half, nullptr);
    EXPECT_FLOAT_EQ(half->As<float>()[2], 1.0f);
    EXPECT_FLOAT_EQ(half->As<float>()[5], 2.0f);
    SetTime(scene, 10.0);
    EXPECT_EQ(scene.Read("/Cloth", tokens::kPoints)->Data(),
              raised.Data());

    scene.WriteAt("/Cloth", tokens::kFaceVertexCounts, 0.0,
                  Tensor::Copy<std::int32_t>(
                      std::vector<std::int32_t>{3}, {1}));
    scene.WriteAt("/Cloth", tokens::kFaceVertexCounts, 20.0,
                  Tensor::Copy<std::int32_t>(
                      std::vector<std::int32_t>{4}, {1}));
    EXPECT_EQ(scene.Read("/Cloth", tokens::kFaceVertexCounts)
                  ->As<std::int32_t>()[0],
              3);
}

// Samples save with the scene and load back, values and tensors
// alike.
TEST(TimeSamplesTest, SamplesRoundTripThroughSceneJson) {
    Scene scene;
    scene.Define("/Arm", tokens::kXform);
    scene.SetAt("/Arm", tokens::kTransform, 0.0, glm::dmat4(1.0));
    scene.SetAt(
        "/Arm", tokens::kTransform, 8.0,
        glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 4.0, 0.0)));
    scene.Define("/Cloth", tokens::kMesh);
    scene.WriteAt(
        "/Cloth", tokens::kPoints, 0.0,
        Tensor::Copy<float>(std::vector<float>{0, 0, 0}, {1, 3}));
    scene.WriteAt(
        "/Cloth", tokens::kPoints, 8.0,
        Tensor::Copy<float>(std::vector<float>{0, 0, 8}, {1, 3}));
    Scene loaded;
    LoadScene(loaded, SaveScene(scene));
    EXPECT_EQ(loaded.SamplesOf("/Arm", tokens::kTransform).size(),
              2u);
    EXPECT_EQ(loaded.SamplesOf("/Cloth", tokens::kPoints).size(), 2u);
    SetTime(loaded, 2.0);
    EXPECT_NEAR(loaded.WorldTransform("/Arm")[3].y, 1.0, 1e-12);
    EXPECT_FLOAT_EQ(
        loaded.Read("/Cloth", tokens::kPoints)->As<float>()[2], 2.0f);
}

}  // namespace
}  // namespace ernie
