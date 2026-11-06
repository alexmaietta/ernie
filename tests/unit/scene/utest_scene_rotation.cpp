// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/rotation.h"

#include <cmath>

#include <gtest/gtest.h>

#include "ernie/util/error.h"

namespace ernie {
namespace {

constexpr double kTolerance = 1e-12;
constexpr Double3 kX{1, 0, 0};
constexpr Double3 kY{0, 1, 0};
constexpr Double3 kZ{0, 0, 1};

double Dot(Double3 a, Double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Double3 Unit(Double3 v) {
    const double length = std::sqrt(Dot(v, v));
    return {v.x / length, v.y / length, v.z / length};
}

void ExpectNear(Double3 actual, Double3 expected) {
    EXPECT_NEAR(actual.x, expected.x, kTolerance);
    EXPECT_NEAR(actual.y, expected.y, kTolerance);
    EXPECT_NEAR(actual.z, expected.z, kTolerance);
}

void ExpectSameRotation(Quatd actual, Quatd expected) {
    const double sign = actual.x * expected.x + actual.y * expected.y + actual.z * expected.z +
                                    actual.w * expected.w <
                                0
                            ? -1
                            : 1;
    EXPECT_NEAR(actual.x, sign * expected.x, kTolerance);
    EXPECT_NEAR(actual.y, sign * expected.y, kTolerance);
    EXPECT_NEAR(actual.z, sign * expected.z, kTolerance);
    EXPECT_NEAR(actual.w, sign * expected.w, kTolerance);
}

void ExpectUnit(Quatd q) {
    EXPECT_NEAR(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w, 1, kTolerance);
}

TEST(RotationTest, LookAlongTurnsMinusZToTheDirectionWithYTowardUp) {
    for (const Double3 direction :
         {Double3{1, 0, 0}, Double3{0, -3, 0}, Double3{1, 2, -0.5}, Double3{-0.2, 0.1, 4}}) {
        SCOPED_TRACE(testing::Message()
                     << direction.x << ", " << direction.y << ", " << direction.z);
        const Quatd rotation = LookAlong(direction);
        ExpectUnit(rotation);
        ExpectNear(Rotate(rotation, {0, 0, -1}), Unit(direction));
        const Double3 up = Rotate(rotation, kY);
        EXPECT_NEAR(Dot(up, Unit(direction)), 0, kTolerance);
        EXPECT_GT(Dot(up, kZ), 0);
        EXPECT_NEAR(Rotate(rotation, kX).z, 0, kTolerance);
    }
}

TEST(RotationTest, LookAlongTheUpDirectionPicksAnotherUp) {
    for (const Double3 direction : {Double3{0, 0, -2}, Double3{0, 0, 1}}) {
        const Quatd rotation = LookAlong(direction);
        ExpectUnit(rotation);
        ExpectNear(Rotate(rotation, {0, 0, -1}), Unit(direction));
    }
    const Quatd sideways = LookAlong({0, 1, 0}, {0, 1, 0});
    ExpectNear(Rotate(sideways, {0, 0, -1}), kY);
}

TEST(RotationTest, LookAtLooksFromTheEyeToTheTarget) {
    const Double3 eye{1, 2, 3};
    const Double3 target{-4, 0, 1};
    const Double3 up{0, 1, 0};
    ExpectSameRotation(LookAt(eye, target, up), LookAlong({-5, -2, -2}, up));
}

TEST(RotationTest, AzimuthAndElevationLookBackAtTheOrigin) {
    ExpectNear(Rotate(FromAzimuthElevation(0, 0), {0, 0, -1}), {0, -1, 0});
    ExpectNear(Rotate(FromAzimuthElevation(90, 0), {0, 0, -1}), {-1, 0, 0});
    ExpectNear(Rotate(FromAzimuthElevation(0, 90), {0, 0, -1}), {0, 0, -1});
    const double half = std::sqrt(0.5);
    ExpectNear(Rotate(FromAzimuthElevation(180, 45), {0, 0, -1}), {0, half, -half});
}

TEST(RotationTest, AxisAngleTurnsCounterclockwiseInDegrees) {
    ExpectNear(Rotate(FromAxisAngle({0, 0, 5}, 90), kX), kY);
    ExpectNear(Rotate(FromAxisAngle(kX, 90), kY), kZ);
    ExpectNear(Rotate(FromAxisAngle(kY, -90), kZ), {-1, 0, 0});
    ExpectSameRotation(FromAxisAngle(kZ, 0), Quatd{});
    ExpectUnit(FromAxisAngle({1, 1, 1}, 33));
}

TEST(RotationTest, ZeroLengthDirectionsHaveNoRotation) {
    EXPECT_THROW(LookAlong({0, 0, 0}), Error);
    EXPECT_THROW(LookAlong(kX, {0, 0, 0}), Error);
    EXPECT_THROW(LookAt(kY, kY), Error);
    EXPECT_THROW(FromAxisAngle({0, 0, 0}, 45), Error);
}

TEST(RotationTest, AxesRoundTripInEveryBranch) {
    for (const Quatd rotation :
         {Quatd{}, FromAxisAngle({1, 2, 3}, 40), FromAxisAngle(kX, 170), FromAxisAngle(kY, 170),
          FromAxisAngle(kZ, 170), FromAxisAngle({1, -1, 0.2}, 180)}) {
        SCOPED_TRACE(testing::Message() << rotation.x << ", " << rotation.y << ", " << rotation.z
                                        << ", " << rotation.w);
        ExpectSameRotation(
            FromAxes(Rotate(rotation, kX), Rotate(rotation, kY), Rotate(rotation, kZ)), rotation);
    }
}

TEST(RotationTest, EulerDegreesTurnAboutXThenYThenZ) {
    ExpectSameRotation(FromEulerDegrees({30, 0, 0}), FromAxisAngle(kX, 30));
    ExpectSameRotation(FromEulerDegrees({0, 30, 0}), FromAxisAngle(kY, 30));
    ExpectSameRotation(FromEulerDegrees({0, 0, 30}), FromAxisAngle(kZ, 30));
    ExpectSameRotation(
        FromEulerDegrees({10, -20, 70}),
        Multiply(FromAxisAngle(kZ, 70), Multiply(FromAxisAngle(kY, -20), FromAxisAngle(kX, 10))));

    for (const Double3 degrees : {Double3{10, -20, 70}, Double3{-170, 45, 120}, Double3{0, 0, 0}}) {
        ExpectNear(ToEulerDegrees(FromEulerDegrees(degrees)), degrees);
    }
    EXPECT_NEAR(ToEulerDegrees(FromEulerDegrees({0, 90, 0})).y, 90, 1e-6);
}

TEST(RotationTest, QuaternionsComposeInvertAndNormalize) {
    const Quatd a = FromAxisAngle({1, 2, 3}, 50);
    const Quatd b = FromAxisAngle({-2, 0, 1}, 125);
    const Double3 v{0.3, -1.2, 2};
    ExpectNear(Rotate(Multiply(a, b), v), Rotate(a, Rotate(b, v)));
    ExpectNear(Rotate(Conjugate(a), Rotate(a, v)), v);
    ExpectSameRotation(Multiply(a, Conjugate(a)), Quatd{});

    ExpectSameRotation(Normalize(Quatd{0, 0, 2, 0}), Quatd{0, 0, 1, 0});
    ExpectSameRotation(Normalize(Quatd{0, 0, 0, 0}), Quatd{});
    ExpectNear(Rotate(Quatd{}, v), v);
}

}
}
