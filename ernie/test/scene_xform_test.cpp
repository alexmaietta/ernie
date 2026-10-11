// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/edit_history.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <gtest/gtest.h>

#include <array>
#include <random>
#include <string>
#include <vector>

using ernie::DType;
using ernie::EditHistory;
using ernie::Path;
using ernie::RotationOrder;
using ernie::Scene;
using ernie::Tensor;
using ernie::Value;
using ernie::XformOps;
namespace tokens = ernie::tokens;

namespace {

constexpr std::array<RotationOrder, 6> kOrders = {
    RotationOrder::XYZ, RotationOrder::XZY, RotationOrder::YXZ,
    RotationOrder::YZX, RotationOrder::ZXY, RotationOrder::ZYX};

/// The axes of `order` in the order they apply, spelled out
/// independently of xform.cpp.
std::array<int, 3> Axes(RotationOrder order) {
    switch (order) {
    case RotationOrder::XYZ:
        return {0, 1, 2};
    case RotationOrder::XZY:
        return {0, 2, 1};
    case RotationOrder::YXZ:
        return {1, 0, 2};
    case RotationOrder::YZX:
        return {1, 2, 0};
    case RotationOrder::ZXY:
        return {2, 0, 1};
    case RotationOrder::ZYX:
        return {2, 1, 0};
    }
    return {0, 1, 2};
}

glm::dvec3 Unit(int axis) {
    glm::dvec3 v(0.0);
    v[axis] = 1.0;
    return v;
}

/// The reference Euler rotation: glm::rotate about each axis, the
/// first-applied rightmost.
glm::dmat4 ReferenceEuler(glm::dvec3 degrees, RotationOrder order) {
    const auto [i, j, k] = Axes(order);
    const auto rotation = [&](int axis) {
        return glm::rotate(glm::dmat4(1.0),
                           glm::radians(degrees[axis]), Unit(axis));
    };
    return rotation(k) * rotation(j) * rotation(i);
}

double MaxError(const glm::dmat4& a, const glm::dmat4& b) {
    double error = 0.0;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            error =
                std::max(error, std::abs(a[col][row] - b[col][row]));
        }
    }
    return error;
}

#define EXPECT_MAT_NEAR(a, b, tolerance)                             \
    EXPECT_LE(MaxError((a), (b)), (tolerance))

glm::dvec3 Apply(const glm::dmat4& m, glm::dvec3 point) {
    return glm::dvec3(m * glm::dvec4(point, 1.0));
}

void ExpectVecNear(glm::dvec3 a, glm::dvec3 b,
                   double tolerance = 1e-12) {
    EXPECT_NEAR(a.x, b.x, tolerance);
    EXPECT_NEAR(a.y, b.y, tolerance);
    EXPECT_NEAR(a.z, b.z, tolerance);
}

/// A uniformly random rotation (Shoemake's subgroup algorithm through
/// a random unit quaternion).
glm::dquat RandomRotation(std::mt19937& random) {
    std::normal_distribution<double> normal;
    return glm::normalize(glm::dquat(normal(random), normal(random),
                                     normal(random), normal(random)));
}

glm::dmat4 Trs(glm::dvec3 translate, const glm::dmat3& rotation,
               glm::dvec3 scale) {
    return glm::translate(glm::dmat4(1.0), translate) *
           glm::dmat4(rotation) * glm::scale(glm::dmat4(1.0), scale);
}

Scene OnePrim(const Path& path = Path("/World/Rock")) {
    Scene scene;
    ernie::Xform::Define(scene, path);
    return scene;
}

std::vector<std::string>
Order(std::initializer_list<std::string_view> ops) {
    return {ops.begin(), ops.end()};
}

}  // namespace

// Euler angles

TEST(Xform, EulerRotationMatchesAxisRotationsInEveryOrder) {
    std::mt19937 random(7);
    std::uniform_real_distribution<double> angle(-180.0, 180.0);
    for (const RotationOrder order : kOrders) {
        for (int n = 0; n < 200; ++n) {
            const glm::dvec3 degrees(angle(random), angle(random),
                                     angle(random));
            EXPECT_MAT_NEAR(
                glm::dmat4(ernie::EulerRotation(degrees, order)),
                ReferenceEuler(degrees, order), 1e-12);
        }
    }
}

TEST(Xform, EulerRotationsAreRightHandedAboutEachAxis) {
    for (const RotationOrder order : kOrders) {
        // A quarter turn about x takes y to z; about y, z to x; about
        // z, x to y.
        ExpectVecNear(ernie::EulerRotation({90.0, 0.0, 0.0}, order) *
                          glm::dvec3(0, 1, 0),
                      {0, 0, 1});
        ExpectVecNear(ernie::EulerRotation({0.0, 90.0, 0.0}, order) *
                          glm::dvec3(0, 0, 1),
                      {1, 0, 0});
        ExpectVecNear(ernie::EulerRotation({0.0, 0.0, 90.0}, order) *
                          glm::dvec3(1, 0, 0),
                      {0, 1, 0});
    }
}

TEST(Xform, EulerOrderSaysWhichAxisTurnsFirst) {
    // x then y: (0, 1, 0) goes to z under x, and z goes to x under y.
    ExpectVecNear(
        ernie::EulerRotation({90.0, 90.0, 0.0}, RotationOrder::XYZ) *
            glm::dvec3(0, 1, 0),
        {1, 0, 0});
    // y then x: (0, 1, 0) stays under y, then goes to z under x.
    ExpectVecNear(
        ernie::EulerRotation({90.0, 90.0, 0.0}, RotationOrder::YXZ) *
            glm::dvec3(0, 1, 0),
        {0, 0, 1});
    // Yaw, pitch, roll (ZYX) differs from XYZ for the same angles.
    EXPECT_GT(MaxError(glm::dmat4(ernie::EulerRotation(
                           {10, 20, 30}, RotationOrder::ZYX)),
                       glm::dmat4(ernie::EulerRotation(
                           {10, 20, 30}, RotationOrder::XYZ))),
              0.1);
}

TEST(Xform, EulerAnglesRecoverTheAnglesInEveryOrder) {
    std::mt19937 random(11);
    std::uniform_real_distribution<double> outer(-179.0, 179.0);
    std::uniform_real_distribution<double> middle(-89.0, 89.0);
    for (const RotationOrder order : kOrders) {
        const int j = Axes(order)[1];
        for (int n = 0; n < 200; ++n) {
            glm::dvec3 degrees(outer(random), outer(random),
                               outer(random));
            degrees[j] = middle(random);
            ExpectVecNear(
                ernie::EulerAngles(
                    ernie::EulerRotation(degrees, order), order),
                degrees, 1e-9);
        }
    }
}

TEST(Xform, EulerAnglesReproduceAnyRotationInEveryOrder) {
    std::mt19937 random(13);
    for (const RotationOrder order : kOrders) {
        for (int n = 0; n < 500; ++n) {
            const glm::dmat3 rotation =
                glm::mat3_cast(RandomRotation(random));
            const glm::dvec3 degrees =
                ernie::EulerAngles(rotation, order);
            EXPECT_LE(std::abs(degrees[Axes(order)[1]]), 90.0 + 1e-9);
            EXPECT_MAT_NEAR(
                glm::dmat4(ernie::EulerRotation(degrees, order)),
                glm::dmat4(rotation), 1e-9);
        }
    }
}

TEST(Xform, EulerAnglesHandleGimbalLock) {
    std::mt19937 random(17);
    std::uniform_real_distribution<double> angle(-170.0, 170.0);
    for (const RotationOrder order : kOrders) {
        const auto [i, j, k] = Axes(order);
        for (const double lock : {90.0, -90.0}) {
            for (int n = 0; n < 50; ++n) {
                glm::dvec3 degrees(0.0);
                degrees[i] = angle(random);
                degrees[j] = lock;
                degrees[k] = angle(random);
                const glm::dmat3 rotation =
                    ernie::EulerRotation(degrees, order);
                const glm::dvec3 found =
                    ernie::EulerAngles(rotation, order);
                EXPECT_NEAR(found[j], lock, 1e-6);
                EXPECT_EQ(found[k], 0.0)
                    << "the last angle is 0 at gimbal lock";
                EXPECT_MAT_NEAR(
                    glm::dmat4(ernie::EulerRotation(found, order)),
                    glm::dmat4(rotation), 1e-9);
            }
        }
    }
}

TEST(Xform, RotateOpsNameTheirOrder) {
    EXPECT_EQ(ernie::RotateOp(RotationOrder::XYZ),
              tokens::kRotateXYZ);
    EXPECT_EQ(ernie::RotateOp(RotationOrder::XZY),
              tokens::kRotateXZY);
    EXPECT_EQ(ernie::RotateOp(RotationOrder::YXZ),
              tokens::kRotateYXZ);
    EXPECT_EQ(ernie::RotateOp(RotationOrder::YZX),
              tokens::kRotateYZX);
    EXPECT_EQ(ernie::RotateOp(RotationOrder::ZXY),
              tokens::kRotateZXY);
    EXPECT_EQ(ernie::RotateOp(RotationOrder::ZYX),
              tokens::kRotateZYX);
}

// Composing ops

TEST(Xform, APrimWithoutOpsIsAtTheIdentity) {
    const Scene scene = OnePrim();
    EXPECT_EQ(scene.LocalTransform(Path("/World/Rock")),
              glm::dmat4(1.0));
}

TEST(Xform, WithoutAnOrderTheTransformOpAloneIsTheLocalTransform) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    const glm::dmat4 m =
        glm::translate(glm::dmat4(1.0), glm::dvec3(1, 2, 3));
    scene.Set(rock, tokens::kTransform, m);
    // Ops the missing order does not list are ignored, as in USD.
    scene.Set(rock, tokens::kTranslate, glm::dvec3(100, 0, 0));
    EXPECT_EQ(scene.LocalTransform(rock), m);
}

TEST(Xform, OpsOutsideTheOrderAreIgnored) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.Set(rock, tokens::kTranslate, glm::dvec3(1, 2, 3));
    scene.Set(rock, tokens::kScale, glm::dvec3(5.0));
    scene.Set(rock, tokens::kXformOpOrder,
              Order({tokens::kTranslate}));
    EXPECT_EQ(scene.LocalTransform(rock),
              glm::translate(glm::dmat4(1.0), glm::dvec3(1, 2, 3)));
}

TEST(Xform, AnEmptyOrderIsTheIdentityEvenWithATransformOp) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.Set(rock, tokens::kTransform,
              glm::scale(glm::dmat4(1.0), glm::dvec3(2.0)));
    scene.Set(rock, tokens::kXformOpOrder,
              std::vector<std::string>{});
    EXPECT_EQ(scene.LocalTransform(rock), glm::dmat4(1.0));
    // An empty number list, as Python passes [], counts as an empty
    // order too.
    scene.Set(rock, tokens::kXformOpOrder, std::vector<float>{});
    EXPECT_EQ(scene.LocalTransform(rock), glm::dmat4(1.0));
}

TEST(Xform, AnOpWithoutAValueIsTheIdentity) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.Set(rock, tokens::kTranslate, glm::dvec3(1, 0, 0));
    scene.Set(rock, tokens::kXformOpOrder,
              Order({tokens::kTranslate, tokens::kRotateZ}));
    EXPECT_EQ(scene.LocalTransform(rock),
              glm::translate(glm::dmat4(1.0), glm::dvec3(1, 0, 0)));
}

TEST(Xform, TranslateMovesInDoublePrecision) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    // A point on the Moon's far side: 1737.4 km out, with a
    // millimetre that float would lose.
    const glm::dvec3 offset(1737400.0005, -2.25, 1e7 + 0.001);
    scene.Set(rock, tokens::kTranslate, offset);
    scene.Set(rock, tokens::kXformOpOrder,
              Order({tokens::kTranslate}));
    EXPECT_EQ(glm::dvec3(scene.LocalTransform(rock)[3]), offset);
    // A float vector works too.
    scene.Set(rock, tokens::kTranslate, glm::vec3(1.5f, 2.5f, 3.5f));
    ExpectVecNear(glm::dvec3(scene.LocalTransform(rock)[3]),
                  {1.5, 2.5, 3.5});
}

TEST(Xform, ScaleTakesThreeNumbersOrOne) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.Set(rock, tokens::kXformOpOrder, Order({tokens::kScale}));
    scene.Set(rock, tokens::kScale, glm::dvec3(1, 2, 3));
    EXPECT_EQ(scene.LocalTransform(rock),
              glm::scale(glm::dmat4(1.0), glm::dvec3(1, 2, 3)));
    scene.Set(rock, tokens::kScale, 4.0);
    EXPECT_EQ(scene.LocalTransform(rock),
              glm::scale(glm::dmat4(1.0), glm::dvec3(4.0)));
    scene.Set(rock, tokens::kScale, 2.0f);
    EXPECT_EQ(scene.LocalTransform(rock),
              glm::scale(glm::dmat4(1.0), glm::dvec3(2.0)));
}

TEST(Xform, SingleAxisRotationsTakeDegrees) {
    const std::pair<std::string_view, int> ops[] = {
        {tokens::kRotateX, 0},
        {tokens::kRotateY, 1},
        {tokens::kRotateZ, 2}};
    for (const auto& [op, axis] : ops) {
        Scene scene = OnePrim();
        const Path rock("/World/Rock");
        scene.Set(rock, op, 30.0);
        scene.Set(rock, tokens::kXformOpOrder, Order({op}));
        EXPECT_MAT_NEAR(scene.LocalTransform(rock),
                        glm::rotate(glm::dmat4(1.0),
                                    glm::radians(30.0), Unit(axis)),
                        1e-15)
            << op;
        scene.Set(rock, op, 30);  // An int is an angle too.
        EXPECT_MAT_NEAR(scene.LocalTransform(rock),
                        glm::rotate(glm::dmat4(1.0),
                                    glm::radians(30.0), Unit(axis)),
                        1e-15)
            << op;
    }
}

TEST(Xform, EveryEulerOpComposesItsOrder) {
    const std::pair<std::string_view, RotationOrder> ops[] = {
        {tokens::kRotateXYZ, RotationOrder::XYZ},
        {tokens::kRotateXZY, RotationOrder::XZY},
        {tokens::kRotateYXZ, RotationOrder::YXZ},
        {tokens::kRotateYZX, RotationOrder::YZX},
        {tokens::kRotateZXY, RotationOrder::ZXY},
        {tokens::kRotateZYX, RotationOrder::ZYX}};
    const glm::dvec3 degrees(12.0, -34.0, 56.0);
    for (const auto& [op, order] : ops) {
        Scene scene = OnePrim();
        const Path rock("/World/Rock");
        scene.Set(rock, op, degrees);
        scene.Set(rock, tokens::kXformOpOrder, Order({op}));
        EXPECT_MAT_NEAR(scene.LocalTransform(rock),
                        ReferenceEuler(degrees, order), 1e-12)
            << op;
    }
}

TEST(Xform, OrientRotatesByAQuaternionAndNormalizesIt) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    const glm::dquat q = glm::angleAxis(
        glm::radians(40.0), glm::normalize(glm::dvec3(1, 2, 3)));
    scene.Set(rock, tokens::kOrient, q);
    scene.Set(rock, tokens::kXformOpOrder, Order({tokens::kOrient}));
    EXPECT_MAT_NEAR(scene.LocalTransform(rock),
                    glm::dmat4(glm::mat3_cast(q)), 1e-14);
    // A quaternion off unit length still rotates without scaling.
    scene.Set(rock, tokens::kOrient, q * 3.0);
    EXPECT_MAT_NEAR(scene.LocalTransform(rock),
                    glm::dmat4(glm::mat3_cast(q)), 1e-14);
}

TEST(Xform, TheFirstOpInTheOrderIsOutermost) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.Set(rock, tokens::kTranslate, glm::dvec3(10, 0, 0));
    scene.Set(rock, tokens::kRotateZ, 90.0);
    scene.Set(rock, tokens::kScale, glm::dvec3(2.0));
    // translate, rotate, scale: (1, 0, 0) scales to (2, 0, 0), turns
    // to (0, 2, 0), moves.
    scene.Set(rock, tokens::kXformOpOrder,
              Order({tokens::kTranslate, tokens::kRotateZ,
                     tokens::kScale}));
    ExpectVecNear(Apply(scene.LocalTransform(rock), {1, 0, 0}),
                  {10, 2, 0});
    // scale, rotate, translate: (1, 0, 0) moves to (11, 0, 0), turns,
    // then scales.
    scene.Set(rock, tokens::kXformOpOrder,
              Order({tokens::kScale, tokens::kRotateZ,
                     tokens::kTranslate}));
    ExpectVecNear(Apply(scene.LocalTransform(rock), {1, 0, 0}),
                  {0, 22, 0});
}

TEST(Xform, ManyOpsMultiplyInOrder) {
    std::mt19937 random(19);
    std::uniform_real_distribution<double> number(-3.0, 3.0);
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    const glm::dvec3 t(number(random), number(random),
                       number(random));
    const glm::dvec3 e(30 * number(random), 30 * number(random),
                       30 * number(random));
    const glm::dquat q = RandomRotation(random);
    const glm::dvec3 s(1.5, 0.5, 2.0);
    const glm::dmat4 m =
        glm::translate(glm::dmat4(1.0), glm::dvec3(0, 0, 7));
    scene.Set(rock, tokens::kTranslate, t);
    scene.Set(rock, tokens::kRotateZYX, e);
    scene.Set(rock, tokens::kOrient, q);
    scene.Set(rock, tokens::kScale, s);
    scene.Set(rock, tokens::kTransform, m);
    scene.Set(
        rock, tokens::kXformOpOrder,
        Order({tokens::kTranslate, tokens::kRotateZYX,
               tokens::kTransform, tokens::kOrient, tokens::kScale}));
    const glm::dmat4 expected =
        glm::translate(glm::dmat4(1.0), t) *
        ReferenceEuler(e, RotationOrder::ZYX) * m *
        glm::dmat4(glm::mat3_cast(q)) *
        glm::scale(glm::dmat4(1.0), s);
    EXPECT_MAT_NEAR(scene.LocalTransform(rock), expected, 1e-12);
}

TEST(Xform, AnOpMayRepeatUnderASuffix) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.Set(rock, tokens::kTranslate, glm::dvec3(1, 0, 0));
    scene.Set(rock, "xformOp:translate:offset", glm::dvec3(0, 2, 0));
    scene.Set(
        rock, tokens::kXformOpOrder,
        Order({tokens::kTranslate, "xformOp:translate:offset"}));
    ExpectVecNear(Apply(scene.LocalTransform(rock), {0, 0, 0}),
                  {1, 2, 0});
}

TEST(Xform, InvertedPivotsRotateAboutAPoint) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    const glm::dvec3 pivot(5, 5, 0);
    scene.Set(rock, "xformOp:translate:pivot", pivot);
    scene.Set(rock, tokens::kRotateZ, 90.0);
    // USD's pivot pattern: translate to the pivot, rotate, and
    // translate back.
    scene.Set(rock, tokens::kXformOpOrder,
              Order({"xformOp:translate:pivot", tokens::kRotateZ,
                     "!invert!xformOp:translate:pivot"}));
    const glm::dmat4 local = scene.LocalTransform(rock);
    ExpectVecNear(Apply(local, pivot), pivot);
    ExpectVecNear(Apply(local, pivot + glm::dvec3(1, 0, 0)),
                  pivot + glm::dvec3(0, 1, 0));
}

TEST(Xform, InvertingAnyOpUndoesIt) {
    std::mt19937 random(23);
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.Set(rock, tokens::kTranslate, glm::dvec3(1, -2, 3));
    scene.Set(rock, tokens::kRotateXYZ, glm::dvec3(10, 20, 30));
    scene.Set(rock, tokens::kOrient, RandomRotation(random));
    scene.Set(rock, tokens::kScale, glm::dvec3(2, 3, 4));
    for (const std::string_view op :
         {tokens::kTranslate, tokens::kRotateXYZ, tokens::kOrient,
          tokens::kScale}) {
        scene.Set(rock, tokens::kXformOpOrder,
                  Order({op, "!invert!" + std::string(op)}));
        EXPECT_MAT_NEAR(scene.LocalTransform(rock), glm::dmat4(1.0),
                        1e-12)
            << op;
    }
}

// The transform stack

TEST(Xform, WorldTransformsComposeOpsDownTheHierarchy) {
    Scene scene;
    ernie::Xform::Define(scene, "/World/Robot/Arm");
    scene.GetPrim(Path("/World"))
        .SetXform({.translate = glm::dvec3(100, 0, 0)});
    scene.GetPrim(Path("/World/Robot"))
        .SetXform({.rotate = glm::dvec3(0, 0, 90)});
    scene.GetPrim(Path("/World/Robot/Arm"))
        .SetXform({.translate = glm::dvec3(1, 0, 0)});
    ExpectVecNear(
        Apply(scene.WorldTransform(Path("/World/Robot/Arm")),
              {0, 0, 0}),
        {100, 1, 0});
}

TEST(Xform, ResettingTheStackIgnoresAncestors) {
    Scene scene;
    ernie::Xform::Define(scene, "/World/Robot/Arm/Hand");
    scene.GetPrim(Path("/World"))
        .SetXform({.translate = glm::dvec3(100, 0, 0)});
    scene.GetPrim(Path("/World/Robot"))
        .SetXform({.translate = glm::dvec3(0, 10, 0)});
    scene.Set(Path("/World/Robot/Arm"), tokens::kTranslate,
              glm::dvec3(0, 0, 1));
    scene.Set(Path("/World/Robot/Arm"), tokens::kXformOpOrder,
              Order({tokens::kResetXformStack, tokens::kTranslate}));
    scene.GetPrim(Path("/World/Robot/Arm/Hand"))
        .SetXform({.translate = glm::dvec3(2, 0, 0)});

    EXPECT_TRUE(
        ernie::ResetsXformStack(scene, Path("/World/Robot/Arm")));
    EXPECT_FALSE(
        ernie::ResetsXformStack(scene, Path("/World/Robot")));
    ExpectVecNear(
        Apply(scene.WorldTransform(Path("/World/Robot/Arm")),
              {0, 0, 0}),
        {0, 0, 1});
    // Below the reset, descendants compose with it and not with what
    // is above it.
    ExpectVecNear(
        Apply(scene.WorldTransform(Path("/World/Robot/Arm/Hand")),
              {0, 0, 0}),
        {2, 0, 1});
    // The reset entry itself is no op.
    EXPECT_EQ(scene.LocalTransform(Path("/World/Robot/Arm")),
              glm::translate(glm::dmat4(1.0), glm::dvec3(0, 0, 1)));
}

TEST(Xform, OpEditsMarkTheTransformChanged) {
    for (const std::string_view attribute :
         {tokens::kXformOpOrder, tokens::kTranslate, tokens::kScale,
          tokens::kOrient, tokens::kRotateX, tokens::kRotateZYX,
          tokens::kTransform}) {
        EXPECT_EQ(ernie::DirtyOf(attribute), ernie::Dirty::Transform)
            << attribute;
        EXPECT_TRUE(ernie::IsXformAttribute(attribute));
    }
    EXPECT_EQ(ernie::DirtyOf("xformOp:translate:pivot"),
              ernie::Dirty::Transform);
    EXPECT_FALSE(ernie::IsXformAttribute(tokens::kPositions));
    EXPECT_FALSE(ernie::IsXformAttribute(tokens::kOrientations));

    Scene scene = OnePrim();
    ernie::ChangeCursor cursor;
    scene.Changes(cursor);  // Past the definition.
    scene.Set(Path("/World/Rock"), tokens::kTranslate,
              glm::dvec3(1, 0, 0));
    const auto changes = scene.Changes(cursor);
    ASSERT_EQ(changes.size(), 1u);
    EXPECT_EQ(changes[0].bits & ernie::Dirty::Transform,
              ernie::Dirty::Transform);
}

// Errors

TEST(Xform, BadOpsNameThePrimAndOp) {
    const Path rock("/World/Rock");
    const auto message = [&](auto author) {
        Scene scene = OnePrim();
        try {
            // Known ops' values are checked as they are set
            // (`FindXformOp`), the rest as read.
            author(scene);
            scene.LocalTransform(rock);
        } catch (const std::invalid_argument& error) {
            return std::string(error.what());
        }
        return std::string("no error");
    };
    const std::string unknown = message([&](Scene& scene) {
        scene.Set(rock, "xformOp:shear", glm::dvec3(1, 0, 0));
        scene.Set(rock, tokens::kXformOpOrder,
                  Order({"xformOp:shear"}));
    });
    EXPECT_NE(unknown.find("xformOp:shear"), std::string::npos)
        << unknown;
    EXPECT_NE(unknown.find("/World/Rock"), std::string::npos)
        << unknown;

    // An unknown op is an error even without a value (a missing value
    // is otherwise the identity).
    const std::string unknownUnset = message([&](Scene& scene) {
        scene.Set(rock, tokens::kXformOpOrder,
                  Order({"xformOp:shear"}));
    });
    EXPECT_NE(unknownUnset.find("does not know"), std::string::npos)
        << unknownUnset;

    const std::string notAnOp = message([&](Scene& scene) {
        scene.Set(rock, tokens::kXformOpOrder, Order({"translate"}));
    });
    EXPECT_NE(notAnOp.find("no xformOp"), std::string::npos)
        << notAnOp;

    const std::string fourNumbers = message([&](Scene& scene) {
        scene.Set(rock, tokens::kOrient, glm::vec4(0, 0, 0, 1));
        scene.Set(rock, tokens::kXformOpOrder,
                  Order({tokens::kOrient}));
    });
    EXPECT_NE(fourNumbers.find("quaternion"), std::string::npos)
        << fourNumbers;

    const std::string arity = message([&](Scene& scene) {
        scene.Set(rock, tokens::kTranslate, 1.0);
        scene.Set(rock, tokens::kXformOpOrder,
                  Order({tokens::kTranslate}));
    });
    EXPECT_NE(arity.find("3-vector"), std::string::npos) << arity;
    EXPECT_NE(arity.find("/World/Rock"), std::string::npos) << arity;

    const std::string order = message([&](Scene& scene) {
        scene.Set(rock, tokens::kXformOpOrder, tokens::kTranslate);
    });
    EXPECT_NE(order.find("list of op names"), std::string::npos)
        << order;

    const std::string wrongType = message([&](Scene& scene) {
        scene.Set(rock, tokens::kRotateX, "ninety");
        scene.Set(rock, tokens::kXformOpOrder,
                  Order({tokens::kRotateX}));
    });
    EXPECT_NE(wrongType.find(tokens::kRotateX), std::string::npos)
        << wrongType;
}

TEST(Xform, GpuTensorOpsAreNotReadOnTheCpu) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    // A tensor that claims CUDA memory: it must be refused before
    // anything reads it.
    static double fake[3] = {0, 0, 0};
    scene.Write(rock, tokens::kTranslate,
                Tensor(fake, DType::Float64, {3}, {},
                       ernie::Device{ernie::DeviceKind::Cuda, 0}));
    scene.Set(rock, tokens::kXformOpOrder,
              Order({tokens::kTranslate}));
    EXPECT_THROW(scene.LocalTransform(rock), std::invalid_argument);
}

TEST(Xform, GpuTensorOpsAreReadThroughAReader) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    static float fake[4] = {0, 0, 0, 0};
    const ernie::Device cuda{ernie::DeviceKind::Cuda, 0};
    scene.Write(rock, tokens::kTranslate,
                Tensor(fake, DType::Float32, {3}, {}, cuda));
    scene.Write(rock, tokens::kOrient,
                Tensor(fake, DType::Float32, {4}, {}, cuda));
    scene.Set(rock, tokens::kScale, glm::dvec3(2.0));
    scene.Set(
        rock, tokens::kXformOpOrder,
        Order({tokens::kTranslate, tokens::kOrient, tokens::kScale}));
    // The reader stands in for a copy to the host, of each GPU op it
    // is asked for in turn.
    const glm::dquat q =
        glm::angleAxis(glm::radians(30.0), glm::dvec3(1, 0, 0));
    const std::array<float, 3> t{1.0f, -2.0f, 4.0f};
    const std::array<float, 4> r{float(q.x), float(q.y), float(q.z),
                                 float(q.w)};
    std::vector<std::string> asked;
    const ernie::GpuOpReader read = [&](std::string_view attribute,
                                        const Tensor& device) {
        asked.emplace_back(attribute);
        EXPECT_EQ(device.Location().kind, ernie::DeviceKind::Cuda);
        return attribute == tokens::kTranslate
                   ? Tensor::Copy<float>(t, {3})
                   : Tensor::Copy<float>(r, {4});
    };
    const glm::dmat4 expected =
        glm::translate(glm::dmat4(1.0), glm::dvec3(1, -2, 4)) *
        glm::dmat4(glm::mat3_cast(q)) *
        glm::scale(glm::dmat4(1.0), glm::dvec3(2.0));
    EXPECT_MAT_NEAR(ernie::ComposeXformOps(scene, rock, &read),
                    expected, 1e-6);
    EXPECT_EQ(asked, Order({tokens::kTranslate, tokens::kOrient}));
    // Without one, they are refused as before.
    EXPECT_THROW(ernie::ComposeXformOps(scene, rock, nullptr),
                 std::invalid_argument);
}

// Op values as tensors

TEST(Xform, OpValuesMayBeCpuTensors) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    const std::array<float, 3> t{1.0f, 2.0f, 3.0f};
    const glm::dquat q =
        glm::angleAxis(glm::radians(60.0), glm::dvec3(0, 0, 1));
    // Quaternions as rows of x, y, z, w (USD's memory layout).
    const std::array<double, 4> xyzw{q.x, q.y, q.z, q.w};
    scene.Write(rock, tokens::kTranslate,
                Tensor::Copy<float>(t, {3}));
    scene.Write(rock, tokens::kOrient,
                Tensor::Copy<double>(xyzw, {4}));
    scene.Write(
        rock, tokens::kScale,
        Tensor::Copy<double>(std::array<double, 1>{2.0}, {1}));
    scene.Set(
        rock, tokens::kXformOpOrder,
        Order({tokens::kTranslate, tokens::kOrient, tokens::kScale}));
    EXPECT_MAT_NEAR(
        scene.LocalTransform(rock),
        Trs({1, 2, 3}, glm::mat3_cast(q), glm::dvec3(2.0)), 1e-12);
}

TEST(Xform, TransformTensorsAreRowMajorMatrices) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    // The math convention: translation in the last column of a
    // row-major 4 x 4.
    const std::array<double, 16> rows{1, 0, 0, 5,  //
                                      0, 1, 0, 6,  //
                                      0, 0, 1, 7,  //
                                      0, 0, 0, 1};
    scene.Write(rock, tokens::kTransform,
                Tensor::Copy<double>(rows, {4, 4}));
    EXPECT_EQ(glm::dvec3(scene.LocalTransform(rock)[3]),
              glm::dvec3(5, 6, 7));
}

TEST(Xform, OneTensorPosesManyPrimsRowByRow) {
    Scene scene;
    std::vector<Path> links;
    for (int world = 0; world < 8; ++world) {
        for (int link = 0; link < 3; ++link) {
            links.emplace_back("/World_" + std::to_string(world) +
                               "/Link" + std::to_string(link));
            ernie::Xform::Define(scene, links.back());
            scene.Set(links.back(), tokens::kXformOpOrder,
                      Order({tokens::kTranslate, tokens::kOrient}));
        }
    }
    std::mt19937 random(29);
    std::vector<double> positions;
    std::vector<float> orientations;
    std::vector<glm::dquat> rotations;
    for (std::size_t i = 0; i < links.size(); ++i) {
        positions.insert(positions.end(),
                         {double(i), 2.0 * double(i), -1.0});
        const glm::dquat q = RandomRotation(random);
        rotations.push_back(q);
        orientations.insert(
            orientations.end(),
            {float(q.x), float(q.y), float(q.z), float(q.w)});
    }
    const auto n = static_cast<std::int64_t>(links.size());
    scene.Write(links, tokens::kTranslate,
                Tensor::Copy<double>(positions, {n, 3}));
    scene.Write(links, tokens::kOrient,
                Tensor::Copy<float>(orientations, {n, 4}));
    for (std::size_t i = 0; i < links.size(); ++i) {
        EXPECT_MAT_NEAR(scene.LocalTransform(links[i]),
                        Trs({double(i), 2.0 * double(i), -1.0},
                            glm::mat3_cast(rotations[i]),
                            glm::dvec3(1.0)),
                        1e-6)  // float32 quaternions
            << links[i].String();
    }
}

TEST(Xform, TensorOpsMustBeFloats) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.Write(rock, tokens::kTranslate,
                Tensor::Copy<std::int32_t>(
                    std::array<std::int32_t, 3>{1, 2, 3}, {3}));
    scene.Set(rock, tokens::kXformOpOrder,
              Order({tokens::kTranslate}));
    EXPECT_THROW(scene.LocalTransform(rock), std::invalid_argument);
}

// SetXform

TEST(Xform, SetXformWritesOpsInTranslateRotateScaleOrder) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    ernie::SetXform(scene, rock,
                    {.translate = glm::dvec3(1, 2, 3),
                     .rotate = glm::dvec3(10, 20, 30),
                     .rotationOrder = RotationOrder::ZYX,
                     .scale = glm::dvec3(2, 2, 2)});
    EXPECT_EQ(*scene.Get(rock, tokens::kXformOpOrder)
                   ->Get<std::vector<std::string>>(),
              Order({tokens::kTranslate, tokens::kRotateZYX,
                     tokens::kScale}));
    EXPECT_EQ(*scene.Get(rock, tokens::kTranslate)->Get<glm::dvec3>(),
              glm::dvec3(1, 2, 3));
    EXPECT_MAT_NEAR(
        scene.LocalTransform(rock),
        glm::translate(glm::dmat4(1.0), glm::dvec3(1, 2, 3)) *
            ReferenceEuler({10, 20, 30}, RotationOrder::ZYX) *
            glm::scale(glm::dmat4(1.0), glm::dvec3(2.0)),
        1e-12);
}

TEST(Xform, SetXformWithOrientNormalizesTheQuaternion) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.GetPrim(rock).SetXform(
        {.orient = glm::dquat(2.0, 0.0, 0.0, 0.0)});
    EXPECT_EQ(*scene.Get(rock, tokens::kOrient)->Get<glm::dquat>(),
              glm::dquat(1, 0, 0, 0));
    EXPECT_EQ(*scene.Get(rock, tokens::kXformOpOrder)
                   ->Get<std::vector<std::string>>(),
              Order({tokens::kOrient}));
}

TEST(Xform, SetXformRefusesBothRotateAndOrient) {
    Scene scene = OnePrim();
    EXPECT_THROW(ernie::SetXform(scene, Path("/World/Rock"),
                                 {.rotate = glm::dvec3(0.0),
                                  .orient = glm::dquat(1, 0, 0, 0)}),
                 std::invalid_argument);
    EXPECT_THROW(ernie::SetXform(scene, Path("/Missing"),
                                 {.translate = glm::dvec3(0.0)}),
                 std::invalid_argument);
}

TEST(Xform, SetXformKeepsAResetStack) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.Set(rock, tokens::kXformOpOrder,
              Order({tokens::kResetXformStack}));
    ernie::SetXform(scene, rock, {.translate = glm::dvec3(1, 0, 0)});
    EXPECT_EQ(*scene.Get(rock, tokens::kXformOpOrder)
                   ->Get<std::vector<std::string>>(),
              Order({tokens::kResetXformStack, tokens::kTranslate}));
}

TEST(Xform, AnEmptyXformIsTheIdentity) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.Set(rock, tokens::kTransform,
              glm::translate(glm::dmat4(1.0), glm::dvec3(9, 9, 9)));
    ernie::SetXform(scene, rock, {});
    EXPECT_EQ(scene.LocalTransform(rock), glm::dmat4(1.0));
}

// Setting a matrix through the ops

TEST(Xform, APrimWithoutOpsTakesTheMatrixAsItsTransformOp) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    const glm::dmat4 m =
        glm::translate(glm::dmat4(1.0), glm::dvec3(4, 5, 6));
    const auto edits = ernie::LocalTransformEdits(scene, rock, m);
    ASSERT_EQ(edits.size(), 1u);
    EXPECT_EQ(edits[0].first, tokens::kTransform);
    scene.SetLocalTransform(rock, m);
    EXPECT_EQ(scene.LocalTransform(rock), m);
    EXPECT_EQ(scene.Get(rock, tokens::kXformOpOrder), nullptr);
}

TEST(Xform, MatricesDecomposeIntoEveryRotationOp) {
    std::mt19937 random(31);
    std::uniform_real_distribution<double> number(-5.0, 5.0);
    std::uniform_real_distribution<double> size(0.2, 3.0);
    const std::string_view rotations[] = {
        tokens::kOrient,    tokens::kRotateXYZ, tokens::kRotateXZY,
        tokens::kRotateYXZ, tokens::kRotateYZX, tokens::kRotateZXY,
        tokens::kRotateZYX};
    for (const std::string_view rotate : rotations) {
        for (int n = 0; n < 100; ++n) {
            Scene scene = OnePrim();
            const Path rock("/World/Rock");
            scene.Set(
                rock, tokens::kXformOpOrder,
                Order({tokens::kTranslate, rotate, tokens::kScale}));
            const glm::dmat4 target =
                Trs({number(random), number(random), number(random)},
                    glm::mat3_cast(RandomRotation(random)),
                    {size(random), size(random), size(random)});
            const auto edits =
                ernie::LocalTransformEdits(scene, rock, target);
            ASSERT_EQ(edits.size(), 3u) << rotate;
            EXPECT_EQ(edits[1].first, rotate);
            scene.SetLocalTransform(rock, target);
            EXPECT_MAT_NEAR(scene.LocalTransform(rock), target, 1e-9)
                << rotate;
            // The order is kept.
            EXPECT_EQ(
                *scene.Get(rock, tokens::kXformOpOrder)
                     ->Get<std::vector<std::string>>(),
                Order({tokens::kTranslate, rotate, tokens::kScale}));
        }
    }
}

TEST(Xform, MirroredMatricesDecomposeWithANegativeScale) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.GetPrim(rock).SetXform({.translate = glm::dvec3(0.0),
                                  .rotate = glm::dvec3(0.0),
                                  .scale = glm::dvec3(1.0)});
    const glm::dmat4 mirrored =
        Trs({1, 2, 3},
            ernie::EulerRotation({10, 20, 30}, RotationOrder::XYZ),
            {-2, 1, 3});
    scene.SetLocalTransform(rock, mirrored);
    EXPECT_MAT_NEAR(scene.LocalTransform(rock), mirrored, 1e-12);
    EXPECT_LT(scene.Get(rock, tokens::kScale)->Get<glm::dvec3>()->x,
              0.0);
}

TEST(Xform, PartialOpsDecomposeWhenTheyCanRepresentTheMatrix) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.Set(rock, tokens::kXformOpOrder,
              Order({tokens::kTranslate, tokens::kRotateZ}));
    // rotateZ alone cannot hold a general rotation: back to
    // xformOp:transform.
    const glm::dmat4 turned =
        glm::rotate(glm::dmat4(1.0), 0.3, glm::dvec3(0, 0, 1));
    scene.SetLocalTransform(rock, turned);
    EXPECT_EQ(*scene.Get(rock, tokens::kXformOpOrder)
                   ->Get<std::vector<std::string>>(),
              Order({tokens::kTransform}));
    EXPECT_MAT_NEAR(scene.LocalTransform(rock), turned, 1e-15);

    // Translate alone represents a pure translation.
    scene.Set(rock, tokens::kXformOpOrder,
              Order({tokens::kTranslate}));
    const glm::dmat4 moved =
        glm::translate(glm::dmat4(1.0), glm::dvec3(3, 2, 1));
    scene.SetLocalTransform(rock, moved);
    EXPECT_EQ(*scene.Get(rock, tokens::kXformOpOrder)
                   ->Get<std::vector<std::string>>(),
              Order({tokens::kTranslate}));
    EXPECT_EQ(scene.LocalTransform(rock), moved);
}

TEST(Xform, MatricesTheOpsCannotHoldBecomeTheTransformOp) {
    const Path rock("/World/Rock");
    const auto falls_back = [&](std::vector<std::string> order,
                                const glm::dmat4& target) {
        Scene scene = OnePrim();
        scene.Set(rock, tokens::kXformOpOrder, order);
        scene.SetLocalTransform(rock, target);
        EXPECT_MAT_NEAR(scene.LocalTransform(rock), target, 1e-12);
        return *scene.Get(rock, tokens::kXformOpOrder)
                    ->Get<std::vector<std::string>>();
    };
    const glm::dmat4 scaled =
        glm::scale(glm::dmat4(1.0), glm::dvec3(2.0));
    const glm::dmat4 turned =
        glm::rotate(glm::dmat4(1.0), 0.5, glm::dvec3(1, 0, 0));
    glm::dmat4 sheared(1.0);
    sheared[1][0] = 0.5;
    // A scale with no scale op; a rotation with no rotate op; shear;
    // a squashed axis.
    EXPECT_EQ(
        falls_back(Order({tokens::kTranslate, tokens::kRotateXYZ}),
                   scaled),
        Order({tokens::kTransform}));
    EXPECT_EQ(falls_back(Order({tokens::kTranslate, tokens::kScale}),
                         turned),
              Order({tokens::kTransform}));
    EXPECT_EQ(falls_back(Order({tokens::kTranslate,
                                tokens::kRotateXYZ, tokens::kScale}),
                         sheared),
              Order({tokens::kTransform}));
    EXPECT_EQ(
        falls_back(Order({tokens::kTranslate, tokens::kRotateXYZ,
                          tokens::kScale}),
                   glm::scale(glm::dmat4(1.0), glm::dvec3(1, 0, 1))),
        Order({tokens::kTransform}));
    // Orders a decomposition cannot fill: pivots, inversions, ops out
    // of order, repeats.
    EXPECT_EQ(falls_back(Order({"xformOp:translate:pivot",
                                tokens::kRotateXYZ,
                                "!invert!xformOp:translate:pivot"}),
                         turned),
              Order({tokens::kTransform}));
    EXPECT_EQ(falls_back(Order({tokens::kScale, tokens::kTranslate}),
                         scaled),
              Order({tokens::kTransform}));
    EXPECT_EQ(falls_back(Order({tokens::kRotateXYZ, tokens::kOrient}),
                         turned),
              Order({tokens::kTransform}));
}

TEST(Xform, FallingBackKeepsAResetStack) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.Set(rock, tokens::kXformOpOrder,
              Order({tokens::kResetXformStack, tokens::kRotateX}));
    const glm::dmat4 target =
        glm::translate(glm::dmat4(1.0), glm::dvec3(1, 0, 0));
    scene.SetLocalTransform(rock, target);
    EXPECT_EQ(*scene.Get(rock, tokens::kXformOpOrder)
                   ->Get<std::vector<std::string>>(),
              Order({tokens::kResetXformStack, tokens::kTransform}));
    EXPECT_TRUE(ernie::ResetsXformStack(scene, rock));
    EXPECT_EQ(scene.LocalTransform(rock), target);
}

TEST(Xform, DecomposingKeepsAResetStack) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.Set(rock, tokens::kXformOpOrder,
              Order({tokens::kResetXformStack, tokens::kTranslate}));
    const glm::dmat4 target =
        glm::translate(glm::dmat4(1.0), glm::dvec3(1, 2, 3));
    scene.SetLocalTransform(rock, target);
    EXPECT_EQ(*scene.Get(rock, tokens::kXformOpOrder)
                   ->Get<std::vector<std::string>>(),
              Order({tokens::kResetXformStack, tokens::kTranslate}));
    EXPECT_EQ(scene.LocalTransform(rock), target);
}

// Values and the edit history

TEST(Xform, ValuesHoldQuaternionsAndTokenLists) {
    const Value q(glm::dquat(1, 0, 0, 0));
    EXPECT_TRUE(q.Is<glm::dquat>());
    EXPECT_EQ(q, Value(glm::dquat(1, 0, 0, 0)));
    EXPECT_NE(q, Value(glm::dquat(0, 1, 0, 0)));
    const Value order(Order({tokens::kTranslate, tokens::kScale}));
    EXPECT_TRUE(order.Is<std::vector<std::string>>());
    EXPECT_EQ(order.Get<std::vector<std::string>>()->size(), 2u);
}

TEST(Xform, EditingOpsUndoesAsOneStep) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.GetPrim(rock).SetXform({.translate = glm::dvec3(0.0),
                                  .rotate = glm::dvec3(0.0),
                                  .scale = glm::dvec3(1.0)});
    EditHistory history;
    const glm::dmat4 target =
        Trs({1, 2, 3},
            ernie::EulerRotation({10, 20, 30}, RotationOrder::XYZ),
            {2, 2, 2});
    history.Set(scene, rock,
                ernie::LocalTransformEdits(scene, rock, target));
    EXPECT_MAT_NEAR(scene.LocalTransform(rock), target, 1e-12);
    EXPECT_EQ(history.Size(), 3u) << "three attributes in one group";
    ASSERT_TRUE(history.Undo(scene));
    EXPECT_EQ(scene.LocalTransform(rock), glm::dmat4(1.0));
    EXPECT_FALSE(history.Undo(scene))
        << "the group undid in one step";
    ASSERT_TRUE(history.Redo(scene));
    EXPECT_MAT_NEAR(scene.LocalTransform(rock), target, 1e-12);
}

TEST(Xform, ADragOfOpsMergesIntoOneStep) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.GetPrim(rock).SetXform(
        {.translate = glm::dvec3(0.0), .rotate = glm::dvec3(0.0)});
    EditHistory history;
    for (int frame = 1; frame <= 10; ++frame) {
        const glm::dmat4 target =
            Trs({frame, 0, 0},
                ernie::EulerRotation({0, 0, 3.0 * frame},
                                     RotationOrder::XYZ),
                glm::dvec3(1.0));
        history.Set(scene, rock,
                    ernie::LocalTransformEdits(scene, rock, target),
                    true);
    }
    history.Close();
    EXPECT_EQ(history.Size(), 2u)
        << "one translate and one rotate step, merged across frames";
    ASSERT_TRUE(history.Undo(scene));
    EXPECT_EQ(scene.LocalTransform(rock), glm::dmat4(1.0));
    EXPECT_FALSE(history.Undo(scene));
}

TEST(Xform, ADragDoesNotMergeIntoOtherAttributes) {
    Scene scene = OnePrim();
    const Path rock("/World/Rock");
    scene.GetPrim(rock).SetXform(
        {.translate = glm::dvec3(0.0), .rotate = glm::dvec3(0.0)});
    EditHistory history;
    history.Set(scene, rock, tokens::kVisibility,
                Value(tokens::kInvisible), true);
    history.Set(
        scene, rock,
        ernie::LocalTransformEdits(
            scene, rock,
            glm::translate(glm::dmat4(1.0), glm::dvec3(1, 0, 0))),
        true);
    history.Close();
    ASSERT_TRUE(history.Undo(scene));
    EXPECT_EQ(scene.LocalTransform(rock), glm::dmat4(1.0));
    EXPECT_NE(scene.Get(rock, tokens::kVisibility), nullptr)
        << "the visibility edit stays";
}
