// SPDX-License-Identifier: BSD-3-Clause
/// @file time_samples_test.cpp
/// @brief Tests of time samples: bracketing, interpolation, and the clock that evaluates them.
#include "ernie/scene/schemas/geom.h"
#include "ernie/scene/schemas/mesh.h"
#include "ernie/scene/time_samples.h"

#include <gtest/gtest.h>

#include <glm/gtc/quaternion.hpp>

#include <stdexcept>
#include <vector>

namespace ernie {
namespace {

std::vector<TimeSample> At(std::vector<double> times) {
    std::vector<TimeSample> samples;
    for (const double time : times) {
        samples.push_back({time, Value(time), {}});
    }
    return samples;
}

TEST(TimeSamples, BracketsHoldTheEndsAndSitOnSamples) {
    const std::vector<TimeSample> samples = At({0.0, 10.0, 20.0});
    EXPECT_EQ(Bracket(samples, -5.0), (SampleBracket{0, 0, 0.0}));
    EXPECT_EQ(Bracket(samples, 10.0), (SampleBracket{1, 1, 0.0}));
    EXPECT_EQ(Bracket(samples, 15.0), (SampleBracket{1, 2, 0.5}));
    EXPECT_EQ(Bracket(samples, 99.0), (SampleBracket{2, 2, 0.0}));
}

TEST(TimeSamples, NumbersMixLinearlyQuaternionsSpinAndTheRestHold) {
    EXPECT_EQ(Interpolate(Value(2.0), Value(4.0), 0.25), Value(2.5));
    EXPECT_EQ(Interpolate(Value(glm::dvec3(0.0)), Value(glm::dvec3(2.0)), 0.5),
              Value(glm::dvec3(1.0)));
    const glm::dquat quarter = glm::angleAxis(glm::radians(90.0), glm::dvec3(0, 0, 1));
    const glm::dquat half =
        *Interpolate(Value(glm::dquat(1, 0, 0, 0)), Value(quarter), 0.5).Get<glm::dquat>();
    EXPECT_NEAR(glm::degrees(glm::angle(half)), 45.0, 1e-9);
    EXPECT_EQ(Interpolate(Value(1), Value(3), 0.5), Value(1)) << "ints hold";
    EXPECT_EQ(Interpolate(Value(std::string("a")), Value(std::string("b")), 0.9),
              Value(std::string("a")));

    const std::vector<float> zeros{0, 0}, twos{2, 2};
    const auto mixed =
        Interpolate(Tensor::Copy<float>(zeros, {2}), Tensor::Copy<float>(twos, {2}), 0.5);
    ASSERT_TRUE(mixed);
    EXPECT_EQ(mixed->As<float>()[1], 1.0f);
    EXPECT_FALSE(
        Interpolate(Tensor::Copy<float>(zeros, {2}), Tensor::Copy<float>(twos, {1, 2}), 0.5))
        << "shapes differ";
}

TEST(TimeSamples, TheClockEvaluatesSamplesStampingOnlyChanges) {
    Scene scene;
    const PrimId ball = Sphere::Define(scene, "/Ball").Id();
    scene.SetAt(ball, Sphere::kRadius, 0.0, 1);
    scene.SetAt(ball, Sphere::kRadius, 10.0, 3.0);
    scene.SetAt(ball, Imageable::kPurpose, 0.0, "render");
    scene.SetAt(ball, Imageable::kPurpose, 10.0, "guide");
    EXPECT_EQ(Sphere(scene, "/Ball").GetRadius(), 1.0) << "the value at the clock";
    EXPECT_EQ(scene.GetAt(ball, Sphere::kRadius, 5.0), Value(2.0)) << "samples are checked and "
                                                                      "converted: 1 to 1.0";

    scene.SetTime(5.0);
    EXPECT_EQ(Sphere(scene, "/Ball").GetRadius(), 2.0);
    const std::uint64_t version = scene.Version();
    scene.SetTime(5.0);
    EXPECT_EQ(scene.Version(), version) << "nothing changed";
    scene.SetTime(7.0);
    EXPECT_EQ(scene.Get(ball, Imageable::kPurpose), Value(std::string("render"))) << "held";
    scene.SetTime(50.0);
    EXPECT_EQ(Sphere(scene, "/Ball").GetRadius(), 3.0) << "held past the last";

    EXPECT_THROW(scene.SetAt(ball, Sphere::kRadius, 20.0, -1.0), std::invalid_argument);
    scene.Set(ball, Sphere::kRadius, 9.0);
    EXPECT_TRUE(scene.Samples(ball, Sphere::kRadius).empty()) << "a value without a time";
    EXPECT_EQ(scene.Samples(ball, Imageable::kPurpose).size(), 2u);
    scene.Remove("/Ball");
    const PrimId again = Sphere::Define(scene, "/Ball").Id();
    EXPECT_TRUE(scene.Samples(again, Imageable::kPurpose).empty()) << "removed with the prim";
}

TEST(TimeSamples, ArraysMixOnTheHost) {
    Scene scene;
    const PrimId cloud = Points::Define(scene, "/Cloud").Id();
    const std::vector<float> a{0, 0, 0}, b{2, 4, 6};
    scene.WriteAt(cloud, PointBased::kPoints, 0.0, Tensor::Copy<float>(a, {1, 3}));
    scene.WriteAt(cloud, PointBased::kPoints, 2.0, Tensor::Copy<float>(b, {1, 3}));
    scene.SetTime(1.0);
    EXPECT_EQ(Points(scene, "/Cloud").GetPoints().As<float>()[2], 3.0f);
    const std::vector<float> pair{0, 0};
    EXPECT_THROW(scene.WriteAt(cloud, PointBased::kPoints, 3.0, Tensor::Copy<float>(pair, {1, 2})),
                 std::invalid_argument)
        << "points are (n, 3)";
}

}  // namespace
}  // namespace ernie
