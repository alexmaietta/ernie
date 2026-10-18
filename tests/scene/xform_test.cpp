// SPDX-License-Identifier: BSD-3-Clause
/// @file xform_test.cpp
/// @brief Tests of Euler rotations, transform op composition, the transform stack and SetXform.
#include "ernie/scene/xform.h"

#include "ernie/scene/scene.h"
#include "ernie/scene/schemas/geom.h"
#include "ernie/scene/tokens.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ernie {
namespace {

constexpr std::array<RotationOrder, 6> kOrders = {RotationOrder::XYZ, RotationOrder::XZY,
                                                  RotationOrder::YXZ, RotationOrder::YZX,
                                                  RotationOrder::ZXY, RotationOrder::ZYX};

constexpr std::string_view kRotateX = "xformOp:rotateX";
constexpr std::string_view kRotateY = "xformOp:rotateY";
constexpr std::string_view kRotateZ = "xformOp:rotateZ";
constexpr std::string_view kRotateXYZ = "xformOp:rotateXYZ";
constexpr std::string_view kRotateZYX = "xformOp:rotateZYX";

/// The axes of `order` in the order they apply, spelled out independently of xform.cpp.
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

/// The reference Euler rotation: glm::rotate about each axis, the first-applied rightmost.
glm::dmat4 ReferenceEuler(glm::dvec3 degrees, RotationOrder order) {
    const auto [i, j, k] = Axes(order);
    const auto rotation = [&](int axis) {
        return glm::rotate(glm::dmat4(1.0), glm::radians(degrees[axis]), Unit(axis));
    };
    return rotation(k) * rotation(j) * rotation(i);
}

double MaxError(const glm::dmat4& a, const glm::dmat4& b) {
    double error = 0.0;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            error = std::max(error, std::abs(a[col][row] - b[col][row]));
        }
    }
    return error;
}

#define EXPECT_MAT_NEAR(a, b, tolerance) EXPECT_LE(MaxError((a), (b)), (tolerance))

glm::dvec3 Apply(const glm::dmat4& m, glm::dvec3 point) {
    return glm::dvec3(m * glm::dvec4(point, 1.0));
}

void ExpectVecNear(glm::dvec3 a, glm::dvec3 b, double tolerance = 1e-12) {
    EXPECT_NEAR(a.x, b.x, tolerance);
    EXPECT_NEAR(a.y, b.y, tolerance);
    EXPECT_NEAR(a.z, b.z, tolerance);
}

/// A uniformly random rotation, from a normally distributed quaternion.
glm::dquat RandomRotation(std::mt19937& random) {
    std::normal_distribution<double> normal;
    return glm::normalize(
        glm::dquat(normal(random), normal(random), normal(random), normal(random)));
}

std::vector<std::string> Order(std::initializer_list<std::string_view> ops) {
    return {ops.begin(), ops.end()};
}

/// A scene with one Xform at `/World/Rock`.
struct Rock {
    Path path{"/World/Rock"};
    Scene scene;
    PrimId prim = scene.Define(path, tokens::kXform);

    void Set(std::string_view attribute, const Value& value) {
        scene.Set(prim, attribute, value);
    }
    std::optional<Value> Get(std::string_view attribute) const {
        return scene.Get(prim, attribute);
    }
    std::vector<std::string> OpOrder() const {
        return *Get(tokens::kXformOpOrder)->Get<std::vector<std::string>>();
    }
    glm::dmat4 Local() const {
        return scene.LocalTransform(prim);
    }
};

/// @return The world origin of the prim at `path`.
glm::dvec3 WorldOrigin(const Scene& scene, const Path& path) {
    return Apply(scene.WorldTransform(scene.Require(path)), glm::dvec3(0.0));
}

// Euler angles

TEST(Xform, EulerRotationMatchesAxisRotationsInEveryOrder) {
    std::mt19937 random(7);
    std::uniform_real_distribution<double> angle(-180.0, 180.0);
    for (const RotationOrder order : kOrders) {
        for (int n = 0; n < 200; ++n) {
            const glm::dvec3 degrees(angle(random), angle(random), angle(random));
            EXPECT_MAT_NEAR(glm::dmat4(EulerRotation(degrees, order)),
                            ReferenceEuler(degrees, order), 1e-12);
        }
    }
}

TEST(Xform, EulerRotationsAreRightHandedAboutEachAxis) {
    for (const RotationOrder order : kOrders) {
        // A quarter turn about x takes y to z; about y, z to x; about z, x to y.
        ExpectVecNear(EulerRotation({90.0, 0.0, 0.0}, order) * glm::dvec3(0, 1, 0), {0, 0, 1});
        ExpectVecNear(EulerRotation({0.0, 90.0, 0.0}, order) * glm::dvec3(0, 0, 1), {1, 0, 0});
        ExpectVecNear(EulerRotation({0.0, 0.0, 90.0}, order) * glm::dvec3(1, 0, 0), {0, 1, 0});
    }
}

TEST(Xform, EulerOrderSaysWhichAxisTurnsFirst) {
    // x then y: (0, 1, 0) goes to z under x, and z goes to x under y.
    ExpectVecNear(EulerRotation({90.0, 90.0, 0.0}, RotationOrder::XYZ) * glm::dvec3(0, 1, 0),
                  {1, 0, 0});
    // y then x: (0, 1, 0) stays under y, then goes to z under x.
    ExpectVecNear(EulerRotation({90.0, 90.0, 0.0}, RotationOrder::YXZ) * glm::dvec3(0, 1, 0),
                  {0, 0, 1});
    // Yaw, pitch, roll (ZYX) differs from XYZ for the same angles.
    EXPECT_GT(MaxError(glm::dmat4(EulerRotation({10, 20, 30}, RotationOrder::ZYX)),
                       glm::dmat4(EulerRotation({10, 20, 30}, RotationOrder::XYZ))),
              0.1);
}

TEST(Xform, EulerAnglesRecoverTheAnglesInEveryOrder) {
    std::mt19937 random(11);
    std::uniform_real_distribution<double> outer(-179.0, 179.0);
    std::uniform_real_distribution<double> middle(-89.0, 89.0);
    for (const RotationOrder order : kOrders) {
        const int j = Axes(order)[1];
        for (int n = 0; n < 200; ++n) {
            glm::dvec3 degrees(outer(random), outer(random), outer(random));
            degrees[j] = middle(random);
            ExpectVecNear(EulerAngles(EulerRotation(degrees, order), order), degrees, 1e-9);
        }
    }
}

TEST(Xform, EulerAnglesReproduceAnyRotationInEveryOrder) {
    std::mt19937 random(13);
    for (const RotationOrder order : kOrders) {
        for (int n = 0; n < 500; ++n) {
            const glm::dmat3 rotation = glm::mat3_cast(RandomRotation(random));
            const glm::dvec3 degrees = EulerAngles(rotation, order);
            EXPECT_LE(std::abs(degrees[Axes(order)[1]]), 90.0 + 1e-9);
            EXPECT_MAT_NEAR(glm::dmat4(EulerRotation(degrees, order)), glm::dmat4(rotation), 1e-9);
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
                const glm::dmat3 rotation = EulerRotation(degrees, order);
                const glm::dvec3 found = EulerAngles(rotation, order);
                EXPECT_NEAR(found[j], lock, 1e-6);
                EXPECT_EQ(found[k], 0.0) << "the last angle is 0 at gimbal lock";
                EXPECT_MAT_NEAR(glm::dmat4(EulerRotation(found, order)), glm::dmat4(rotation),
                                1e-9);
            }
        }
    }
}

TEST(Xform, RotateOpsNameTheirOrder) {
    EXPECT_EQ(RotateOp(RotationOrder::XYZ), "xformOp:rotateXYZ");
    EXPECT_EQ(RotateOp(RotationOrder::XZY), "xformOp:rotateXZY");
    EXPECT_EQ(RotateOp(RotationOrder::YXZ), "xformOp:rotateYXZ");
    EXPECT_EQ(RotateOp(RotationOrder::YZX), "xformOp:rotateYZX");
    EXPECT_EQ(RotateOp(RotationOrder::ZXY), "xformOp:rotateZXY");
    EXPECT_EQ(RotateOp(RotationOrder::ZYX), "xformOp:rotateZYX");
}

// Composing ops

TEST(Xform, APrimWithoutOpsIsAtTheIdentity) {
    const Rock rock;
    EXPECT_EQ(rock.Local(), glm::dmat4(1.0));
}

TEST(Xform, PrimsThatAreNotXformableDoNotTransform) {
    Scene scene;
    const PrimId typeless = scene.Define("/Typeless", "");
    const PrimId scope = scene.Define("/Scope", tokens::kScope);
    for (const PrimId prim : {typeless, scope}) {
        scene.Set(prim, tokens::kTranslate, glm::dvec3(1, 2, 3));
        scene.Set(prim, tokens::kXformOpOrder, Order({tokens::kTranslate}));
        EXPECT_EQ(scene.LocalTransform(prim), glm::dmat4(1.0)) << scene.PrimPath(prim).String();
    }
    // Ancestors Define adds have no type, so they leave their descendants where they are.
    const PrimId hand = scene.Define("/Arm/Hand", tokens::kXform);
    SetXform(scene, "/Arm/Hand", {.translate = glm::dvec3(1, 0, 0)});
    EXPECT_EQ(scene.TypeName(scene.Require("/Arm")), "");
    EXPECT_EQ(scene.WorldTransform(hand), scene.LocalTransform(hand));
}

TEST(Xform, WithoutAnOrderTheTransformOpAloneIsTheLocalTransform) {
    Rock rock;
    const glm::dmat4 m = glm::translate(glm::dmat4(1.0), glm::dvec3(1, 2, 3));
    rock.Set(tokens::kTransform, m);
    // Ops the missing order does not list are ignored, as in USD.
    rock.Set(tokens::kTranslate, glm::dvec3(100, 0, 0));
    EXPECT_EQ(rock.Local(), m);
}

TEST(Xform, OpsOutsideTheOrderAreIgnored) {
    Rock rock;
    rock.Set(tokens::kTranslate, glm::dvec3(1, 2, 3));
    rock.Set(tokens::kScale, glm::dvec3(5.0));
    rock.Set(tokens::kXformOpOrder, Order({tokens::kTranslate}));
    EXPECT_EQ(rock.Local(), glm::translate(glm::dmat4(1.0), glm::dvec3(1, 2, 3)));
}

TEST(Xform, AnEmptyOrderIsTheIdentityEvenWithATransformOp) {
    Rock rock;
    rock.Set(tokens::kTransform, glm::scale(glm::dmat4(1.0), glm::dvec3(2.0)));
    rock.Set(tokens::kXformOpOrder, std::vector<std::string>{});
    EXPECT_EQ(rock.Local(), glm::dmat4(1.0));
    // An empty number list, as Python passes [], counts as an empty order too.
    rock.Set(tokens::kXformOpOrder, std::vector<float>{});
    EXPECT_EQ(rock.Local(), glm::dmat4(1.0));
}

TEST(Xform, AnOpWithoutAValueIsTheIdentity) {
    Rock rock;
    rock.Set(tokens::kTranslate, glm::dvec3(1, 0, 0));
    rock.Set(tokens::kXformOpOrder, Order({tokens::kTranslate, kRotateZ}));
    EXPECT_EQ(rock.Local(), glm::translate(glm::dmat4(1.0), glm::dvec3(1, 0, 0)));
}

TEST(Xform, TranslateMovesInDoublePrecision) {
    Rock rock;
    // A point on the Moon's far side: 1737.4 km out, with a millimetre that float would lose.
    const glm::dvec3 offset(1737400.0005, -2.25, 1e7 + 0.001);
    rock.Set(tokens::kTranslate, offset);
    rock.Set(tokens::kXformOpOrder, Order({tokens::kTranslate}));
    EXPECT_EQ(glm::dvec3(rock.Local()[3]), offset);
    // A float vector works too.
    rock.Set(tokens::kTranslate, glm::vec3(1.5f, 2.5f, 3.5f));
    ExpectVecNear(glm::dvec3(rock.Local()[3]), {1.5, 2.5, 3.5});
}

TEST(Xform, ScaleTakesThreeNumbersOrOne) {
    Rock vector;
    vector.Set(tokens::kXformOpOrder, Order({tokens::kScale}));
    vector.Set(tokens::kScale, glm::dvec3(1, 2, 3));
    EXPECT_EQ(vector.Local(), glm::scale(glm::dmat4(1.0), glm::dvec3(1, 2, 3)));
    // An op's first value types it, so the scalar scale is in a scene of its own.
    Rock scalar;
    scalar.Set(tokens::kXformOpOrder, Order({tokens::kScale}));
    scalar.Set(tokens::kScale, 4.0);
    EXPECT_EQ(scalar.Local(), glm::scale(glm::dmat4(1.0), glm::dvec3(4.0)));
    scalar.Set(tokens::kScale, 2.0f);
    EXPECT_EQ(scalar.Local(), glm::scale(glm::dmat4(1.0), glm::dvec3(2.0)));
}

TEST(Xform, SingleAxisRotationsTakeDegrees) {
    const std::pair<std::string_view, int> ops[] = {{kRotateX, 0}, {kRotateY, 1}, {kRotateZ, 2}};
    for (const auto& [op, axis] : ops) {
        Rock rock;
        const glm::dmat4 expected = glm::rotate(glm::dmat4(1.0), glm::radians(30.0), Unit(axis));
        rock.Set(op, 30.0);
        rock.Set(tokens::kXformOpOrder, Order({op}));
        EXPECT_MAT_NEAR(rock.Local(), expected, 1e-15) << op;
        rock.Set(op, 30);  // An int is an angle too.
        EXPECT_MAT_NEAR(rock.Local(), expected, 1e-15) << op;
    }
}

TEST(Xform, EveryEulerOpComposesItsOrder) {
    const std::pair<std::string_view, RotationOrder> ops[] = {
        {"xformOp:rotateXYZ", RotationOrder::XYZ}, {"xformOp:rotateXZY", RotationOrder::XZY},
        {"xformOp:rotateYXZ", RotationOrder::YXZ}, {"xformOp:rotateYZX", RotationOrder::YZX},
        {"xformOp:rotateZXY", RotationOrder::ZXY}, {"xformOp:rotateZYX", RotationOrder::ZYX}};
    const glm::dvec3 degrees(12.0, -34.0, 56.0);
    for (const auto& [op, order] : ops) {
        Rock rock;
        rock.Set(op, degrees);
        rock.Set(tokens::kXformOpOrder, Order({op}));
        EXPECT_MAT_NEAR(rock.Local(), ReferenceEuler(degrees, order), 1e-12) << op;
    }
}

TEST(Xform, OrientRotatesByAQuaternionAndNormalizesIt) {
    Rock rock;
    const glm::dquat q = glm::angleAxis(glm::radians(40.0), glm::normalize(glm::dvec3(1, 2, 3)));
    rock.Set(tokens::kOrient, q);
    rock.Set(tokens::kXformOpOrder, Order({tokens::kOrient}));
    EXPECT_MAT_NEAR(rock.Local(), glm::dmat4(glm::mat3_cast(q)), 1e-14);
    // A quaternion off unit length still rotates without scaling.
    rock.Set(tokens::kOrient, q * 3.0);
    EXPECT_MAT_NEAR(rock.Local(), glm::dmat4(glm::mat3_cast(q)), 1e-14);
}

TEST(Xform, TheFirstOpInTheOrderIsOutermost) {
    Rock rock;
    rock.Set(tokens::kTranslate, glm::dvec3(10, 0, 0));
    rock.Set(kRotateZ, 90.0);
    rock.Set(tokens::kScale, glm::dvec3(2.0));
    // translate, rotate, scale: (1, 0, 0) scales to (2, 0, 0), turns to (0, 2, 0), moves.
    rock.Set(tokens::kXformOpOrder, Order({tokens::kTranslate, kRotateZ, tokens::kScale}));
    ExpectVecNear(Apply(rock.Local(), {1, 0, 0}), {10, 2, 0});
    // scale, rotate, translate: (1, 0, 0) moves to (11, 0, 0), turns, then scales.
    rock.Set(tokens::kXformOpOrder, Order({tokens::kScale, kRotateZ, tokens::kTranslate}));
    ExpectVecNear(Apply(rock.Local(), {1, 0, 0}), {0, 22, 0});
}

TEST(Xform, ManyOpsMultiplyInOrder) {
    std::mt19937 random(19);
    std::uniform_real_distribution<double> number(-3.0, 3.0);
    Rock rock;
    const glm::dvec3 t(number(random), number(random), number(random));
    const glm::dvec3 e(30 * number(random), 30 * number(random), 30 * number(random));
    const glm::dquat q = RandomRotation(random);
    const glm::dvec3 s(1.5, 0.5, 2.0);
    const glm::dmat4 m = glm::translate(glm::dmat4(1.0), glm::dvec3(0, 0, 7));
    rock.Set(tokens::kTranslate, t);
    rock.Set(kRotateZYX, e);
    rock.Set(tokens::kOrient, q);
    rock.Set(tokens::kScale, s);
    rock.Set(tokens::kTransform, m);
    rock.Set(tokens::kXformOpOrder, Order({tokens::kTranslate, kRotateZYX, tokens::kTransform,
                                           tokens::kOrient, tokens::kScale}));
    const glm::dmat4 expected = glm::translate(glm::dmat4(1.0), t) *
                                ReferenceEuler(e, RotationOrder::ZYX) * m *
                                glm::dmat4(glm::mat3_cast(q)) * glm::scale(glm::dmat4(1.0), s);
    EXPECT_MAT_NEAR(rock.Local(), expected, 1e-12);
}

TEST(Xform, AnOpMayRepeatUnderASuffix) {
    Rock rock;
    rock.Set(tokens::kTranslate, glm::dvec3(1, 0, 0));
    rock.Set("xformOp:translate:offset", glm::dvec3(0, 2, 0));
    rock.Set(tokens::kXformOpOrder, Order({tokens::kTranslate, "xformOp:translate:offset"}));
    ExpectVecNear(Apply(rock.Local(), {0, 0, 0}), {1, 2, 0});
}

TEST(Xform, InvertedPivotsRotateAboutAPoint) {
    Rock rock;
    const glm::dvec3 pivot(5, 5, 0);
    rock.Set("xformOp:translate:pivot", pivot);
    rock.Set(kRotateZ, 90.0);
    // USD's pivot pattern: translate to the pivot, rotate, and translate back.
    rock.Set(tokens::kXformOpOrder,
             Order({"xformOp:translate:pivot", kRotateZ, "!invert!xformOp:translate:pivot"}));
    const glm::dmat4 local = rock.Local();
    ExpectVecNear(Apply(local, pivot), pivot);
    ExpectVecNear(Apply(local, pivot + glm::dvec3(1, 0, 0)), pivot + glm::dvec3(0, 1, 0));
}

TEST(Xform, InvertingAnyOpUndoesIt) {
    std::mt19937 random(23);
    Rock rock;
    rock.Set(tokens::kTranslate, glm::dvec3(1, -2, 3));
    rock.Set(kRotateXYZ, glm::dvec3(10, 20, 30));
    rock.Set(tokens::kOrient, RandomRotation(random));
    rock.Set(tokens::kScale, glm::dvec3(2, 3, 4));
    for (const std::string_view op :
         {tokens::kTranslate, kRotateXYZ, tokens::kOrient, tokens::kScale}) {
        const std::string inverted = std::string(tokens::kInvertPrefix) + std::string(op);
        rock.Set(tokens::kXformOpOrder, Order({op, inverted}));
        EXPECT_MAT_NEAR(rock.Local(), glm::dmat4(1.0), 1e-12) << op;
    }
}

// The transform stack

TEST(Xform, WorldTransformsComposeOpsDownTheHierarchy) {
    Scene scene;
    for (const char* path : {"/World", "/World/Robot", "/World/Robot/Arm"}) {
        scene.Define(path, tokens::kXform);
    }
    SetXform(scene, "/World", {.translate = glm::dvec3(100, 0, 0)});
    SetXform(scene, "/World/Robot", {.rotate = glm::dvec3(0, 0, 90)});
    SetXform(scene, "/World/Robot/Arm", {.translate = glm::dvec3(1, 0, 0)});
    ExpectVecNear(WorldOrigin(scene, "/World/Robot/Arm"), {100, 1, 0});
}

TEST(Xform, ResettingTheStackIgnoresAncestors) {
    Scene scene;
    for (const char* path :
         {"/World", "/World/Robot", "/World/Robot/Arm", "/World/Robot/Arm/Hand"}) {
        scene.Define(path, tokens::kXform);
    }
    const Path arm("/World/Robot/Arm");
    SetXform(scene, "/World", {.translate = glm::dvec3(100, 0, 0)});
    SetXform(scene, "/World/Robot", {.translate = glm::dvec3(0, 10, 0)});
    scene.Set(arm, tokens::kTranslate, glm::dvec3(0, 0, 1));
    scene.Set(arm, tokens::kXformOpOrder, Order({tokens::kResetXformStack, tokens::kTranslate}));
    SetXform(scene, "/World/Robot/Arm/Hand", {.translate = glm::dvec3(2, 0, 0)});

    EXPECT_TRUE(ResetsXformStack(scene, scene.Require(arm)));
    EXPECT_FALSE(ResetsXformStack(scene, scene.Require("/World/Robot")));
    ExpectVecNear(WorldOrigin(scene, arm), {0, 0, 1});
    // Below the reset, descendants compose with it and not with what is above it.
    ExpectVecNear(WorldOrigin(scene, "/World/Robot/Arm/Hand"), {2, 0, 1});
    // The reset entry itself is no op.
    EXPECT_EQ(scene.LocalTransform(scene.Require(arm)),
              glm::translate(glm::dmat4(1.0), glm::dvec3(0, 0, 1)));
}

// Errors

TEST(Xform, BadOpsNameThePrimAndOp) {
    const auto message = [](auto author) {
        Rock rock;
        try {
            author(rock);
            rock.Local();
        } catch (const std::invalid_argument& error) {
            return std::string(error.what());
        }
        return std::string("no error");
    };
    const auto mentions = [](const std::string& text, std::string_view part) {
        return text.find(part) != std::string::npos;
    };

    const std::string unknown = message([](Rock& rock) {
        rock.Set("xformOp:shear", glm::dvec3(1, 0, 0));
        rock.Set(tokens::kXformOpOrder, Order({"xformOp:shear"}));
    });
    EXPECT_TRUE(mentions(unknown, "xformOp:shear")) << unknown;
    EXPECT_TRUE(mentions(unknown, "/World/Rock")) << unknown;

    // An unknown op is an error even without a value (a missing value is otherwise the identity).
    const std::string unknownUnset =
        message([](Rock& rock) { rock.Set(tokens::kXformOpOrder, Order({"xformOp:shear"})); });
    EXPECT_TRUE(mentions(unknownUnset, "does not know")) << unknownUnset;

    const std::string notAnOp =
        message([](Rock& rock) { rock.Set(tokens::kXformOpOrder, Order({"translate"})); });
    EXPECT_TRUE(mentions(notAnOp, "no xformOp")) << notAnOp;

    const std::string fourNumbers = message([](Rock& rock) {
        rock.Set(tokens::kOrient, glm::vec4(0, 0, 0, 1));
        rock.Set(tokens::kXformOpOrder, Order({tokens::kOrient}));
    });
    EXPECT_TRUE(mentions(fourNumbers, "quaternion")) << fourNumbers;

    const std::string arity = message([](Rock& rock) {
        rock.Set(tokens::kTranslate, 1.0);
        rock.Set(tokens::kXformOpOrder, Order({tokens::kTranslate}));
    });
    EXPECT_TRUE(mentions(arity, "three numbers")) << arity;
    EXPECT_TRUE(mentions(arity, "/World/Rock")) << arity;

    const std::string order =
        message([](Rock& rock) { rock.Set(tokens::kXformOpOrder, tokens::kTranslate); });
    EXPECT_TRUE(mentions(order, tokens::kXformOpOrder)) << order;
    EXPECT_TRUE(mentions(order, "/World/Rock")) << order;

    const std::string wrongType = message([](Rock& rock) {
        rock.Set(kRotateX, "ninety");
        rock.Set(tokens::kXformOpOrder, Order({kRotateX}));
    });
    EXPECT_TRUE(mentions(wrongType, kRotateX)) << wrongType;
}

// SetXform

TEST(Xform, SetXformWritesOpsInTranslateRotateScaleOrder) {
    Rock rock;
    SetXform(rock.scene, rock.path,
             {.translate = glm::dvec3(1, 2, 3),
              .rotate = glm::dvec3(10, 20, 30),
              .rotationOrder = RotationOrder::ZYX,
              .scale = glm::dvec3(2, 2, 2)});
    EXPECT_EQ(rock.OpOrder(), Order({tokens::kTranslate, kRotateZYX, tokens::kScale}));
    EXPECT_EQ(rock.Get(tokens::kTranslate), Value(glm::dvec3(1, 2, 3)));
    EXPECT_MAT_NEAR(rock.Local(),
                    glm::translate(glm::dmat4(1.0), glm::dvec3(1, 2, 3)) *
                        ReferenceEuler({10, 20, 30}, RotationOrder::ZYX) *
                        glm::scale(glm::dmat4(1.0), glm::dvec3(2.0)),
                    1e-12);
}

TEST(Xform, SetXformWithOrientNormalizesTheQuaternion) {
    Rock rock;
    SetXform(rock.scene, rock.path, {.orient = glm::dquat(2.0, 0.0, 0.0, 0.0)});
    EXPECT_EQ(rock.Get(tokens::kOrient), Value(glm::dquat(1, 0, 0, 0)));
    EXPECT_EQ(rock.OpOrder(), Order({tokens::kOrient}));
}

TEST(Xform, SetXformRefusesBothRotateAndOrient) {
    Rock rock;
    EXPECT_THROW(SetXform(rock.scene, rock.path,
                          {.rotate = glm::dvec3(0.0), .orient = glm::dquat(1, 0, 0, 0)}),
                 std::invalid_argument);
    EXPECT_THROW(SetXform(rock.scene, "/Missing", {.translate = glm::dvec3(0.0)}),
                 std::invalid_argument);
}

TEST(Xform, SetXformKeepsAResetStack) {
    Rock rock;
    rock.Set(tokens::kXformOpOrder, Order({tokens::kResetXformStack}));
    SetXform(rock.scene, rock.path, {.translate = glm::dvec3(1, 0, 0)});
    EXPECT_EQ(rock.OpOrder(), Order({tokens::kResetXformStack, tokens::kTranslate}));
}

TEST(Xform, AnEmptyXformIsTheIdentity) {
    Rock rock;
    rock.Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), glm::dvec3(9, 9, 9)));
    SetXform(rock.scene, rock.path, {});
    EXPECT_EQ(rock.Local(), glm::dmat4(1.0));
}

// Values

TEST(Xform, ValuesHoldQuaternionsAndTokenLists) {
    const Value q(glm::dquat(1, 0, 0, 0));
    EXPECT_TRUE(q.Is<glm::dquat>());
    EXPECT_EQ(q, Value(glm::dquat(1, 0, 0, 0)));
    EXPECT_NE(q, Value(glm::dquat(0, 1, 0, 0)));
    const Value order(Order({tokens::kTranslate, tokens::kScale}));
    EXPECT_TRUE(order.Is<std::vector<std::string>>());
    EXPECT_EQ(order.Get<std::vector<std::string>>()->size(), 2u);
}

}  // namespace
}  // namespace ernie
