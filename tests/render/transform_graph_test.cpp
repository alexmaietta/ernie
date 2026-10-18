// SPDX-License-Identifier: BSD-3-Clause
/// @file transform_graph_test.cpp
/// @brief Tests of the transform graph: world transforms composed level by level on the device
/// match the scene's, ops written on the device compose with the ops the host authors, and
/// structural changes cost what they change.
#include "ernie/render/scene_columns.h"
#include "ernie/render/transform_graph.h"
#include "ernie/scene/xform.h"

#include "tests/device/columns.h"
#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <glm/gtc/quaternion.hpp>

#include <string>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;
using test::ReadFloats;

/// Expects a node's world transform to be `expected`'s top three rows.
void ExpectWorld(const std::vector<float>& worlds, std::uint32_t node, const glm::dmat4& expected) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            EXPECT_NEAR(worlds[node * 12 + static_cast<std::uint32_t>(r * 4 + c)], expected[c][r],
                        1e-5)
                << "node " << node << ", row " << r << ", column " << c;
        }
    }
}

const glm::dquat kQuarterTurn = glm::angleAxis(glm::radians(90.0), glm::dvec3(0, 0, 1));

TEST(TransformGraph, ComposesTheScenesHierarchyLevelByLevel) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    Scene scene;
    for (const char* path : {"/World", "/World/Arm", "/World/Arm/Hand", "/World/Free"}) {
        scene.Define(path, "Xform");
    }
    scene.Define("/World/Arm/Hand/Tip", "Sphere");
    SetXform(scene, "/World", {.translate = glm::dvec3(1, 0, 0)});
    SetXform(scene, "/World/Arm", {.translate = glm::dvec3(0, 2, 0), .orient = kQuarterTurn});
    SetXform(scene, "/World/Arm/Hand", {.translate = glm::dvec3(1, 0, 0), .scale = glm::dvec3(2)});
    SetXform(scene, "/World/Arm/Hand/Tip", {.translate = glm::dvec3(0, 0, 3)});
    SetXform(scene, "/World/Free", {.translate = glm::dvec3(5, 0, 0)});
    scene.Set(*scene.Find("/World/Free"), "xformOpOrder",
              std::vector<std::string>{"!resetXformStack!", "xformOp:translate"});

    Arenas arenas(*context, MemoryUse::Device, "transforms", 1 << 16);
    TransformGraph graph(*context, arenas, scene, compiler);
    const Ticket composed = graph.Update();
    EXPECT_EQ(graph.Levels(), 4u) << "World, Arm and Free, Hand, Tip";
    const std::vector<float> worlds = ReadFloats(*context, graph.Worlds(), composed);
    for (const char* path :
         {"/World", "/World/Arm", "/World/Arm/Hand", "/World/Arm/Hand/Tip", "/World/Free"}) {
        const PrimId prim = *scene.Find(path);
        ExpectWorld(worlds, *graph.Node(prim), scene.WorldTransform(prim));
    }

    // An authored edit uploads only its node, and its descendants follow on the device.
    const Transfers before = context->Transfers();
    SetXform(scene, "/World", {.translate = glm::dvec3(0, 0, 10)});
    const std::vector<float> moved = ReadFloats(*context, graph.Worlds(), graph.Update());
    EXPECT_EQ(context->Transfers().uploaded - before.uploaded, 48u) << "one node's local";
    const PrimId tip = *scene.Find("/World/Arm/Hand/Tip");
    ExpectWorld(moved, *graph.Node(tip), scene.WorldTransform(tip));
    ExpectClean(*context);
}

TEST(TransformGraph, ComposesPosesWrittenOnTheDevice) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    Scene scene;
    const int bodies = 64;
    std::vector<PrimId> body;
    for (int i = 0; i < bodies; ++i) {
        const std::string env = "/Env_" + std::to_string(i);
        scene.Define(env, "Xform");
        SetXform(scene, env, {.translate = glm::dvec3(10.0 * i, 0, 0)});
        body.push_back(scene.Define(env + "/Body", "Xform"));
        SetXform(scene, env + "/Body",
                 {.translate = glm::dvec3(0),
                  .orient = glm::dquat(1, 0, 0, 0),
                  .scale = glm::dvec3(0.5)});
    }
    const BucketId bucket = scene.Store().PrimBucket(body.front());

    Arenas arenas(*context, MemoryUse::Device, "transforms", 1 << 20);
    SceneColumns columns(*context, arenas, scene);
    const DeviceColumn& translate = columns.Bind(bucket, "xformOp:translate");
    const DeviceColumn& orient = columns.Bind(bucket, "xformOp:orient");
    TransformGraph graph(*context, arenas, scene, compiler);
    graph.Bind(bucket, "xformOp:translate", translate);
    graph.Bind(bucket, "xformOp:orient", orient);
    graph.Update(std::array{columns.Sync()}).Wait();

    // A simulation writes the bodies' poses in place at their rows (the environments share the
    // bucket); its writes stand in for CUDA's or Warp's.
    Uploader simulation(*context, context->Main());
    for (int i = 0; i < bodies; ++i) {
        const std::uint32_t row = scene.Store().PrimRow(body[static_cast<std::size_t>(i)]);
        const std::array position{0.0f, float(i), 1.0f};
        const std::array orientation{0.0f, 0.0f, float(kQuarterTurn.z), float(kQuarterTurn.w)};
        simulation.Write(translate.Storage(), translate.Range().offset + row * 12,
                         std::span<const float>(position));
        simulation.Write(orient.Storage(), orient.Range().offset + row * 16,
                         std::span<const float>(orientation));
    }
    const Ticket simulated = simulation.Flush();

    const Transfers before = context->Transfers();
    const Ticket synced = columns.Sync(std::span(&simulated, 1));
    const std::vector<float> worlds =
        ReadFloats(*context, graph.Worlds(), graph.Update(std::span(&synced, 1)));
    EXPECT_EQ(context->Transfers().uploaded, before.uploaded) << "a step costs no uploads";
    for (const int i : {0, 1, 63}) {
        const glm::dmat4 expected = glm::translate(glm::dmat4(1.0), glm::dvec3(10.0 * i, 0, 0)) *
                                    glm::translate(glm::dmat4(1.0), glm::dvec3(0, i, 1)) *
                                    glm::mat4_cast(kQuarterTurn) *
                                    glm::scale(glm::dmat4(1.0), glm::dvec3(0.5));
        ExpectWorld(worlds, *graph.Node(body[static_cast<std::size_t>(i)]), expected);
    }
    ExpectClean(*context);
}

TEST(TransformGraph, FollowsPrimsComingAndGoingAtWhatTheyCost) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    Scene scene;
    for (int i = 0; i < 1000; ++i) {
        const std::string path = "/Ball_" + std::to_string(i);
        scene.Define(path, "Sphere");
        SetXform(scene, path, {.translate = glm::dvec3(i, 0, 0)});
    }
    Arenas arenas(*context, MemoryUse::Device, "transforms", 1 << 20);
    TransformGraph graph(*context, arenas, scene, compiler);
    graph.Update().Wait();
    const PrimId gone = *scene.Find("/Ball_1");
    scene.Remove("/Ball_1");
    const PrimId added = scene.Define("/Ball_9999", "Sphere");
    SetXform(scene, "/Ball_9999", {.translate = glm::dvec3(9, 0, 0)});

    const Transfers before = context->Transfers();
    const Ticket followed = graph.Update();
    // The added prim's parent and local, and two entries of its level's list.
    EXPECT_LE(context->Transfers().uploaded - before.uploaded, 4u + 48u + 2 * 4u)
        << "a change costs what it changes, not the scene";
    EXPECT_FALSE(graph.Node(gone));
    EXPECT_EQ(graph.Node(added), added.slot) << "a prim's node is its stable id's slot";
    const std::vector<float> worlds = ReadFloats(*context, graph.Worlds(), followed);
    for (const char* path : {"/Ball_0", "/Ball_2", "/Ball_999", "/Ball_9999"}) {
        const PrimId prim = *scene.Find(path);
        ExpectWorld(worlds, *graph.Node(prim), scene.WorldTransform(prim));
    }
    ExpectClean(*context);
}

TEST(TransformGraph, FollowsRemovedSubtreesAndNewOrders) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    Scene scene;
    scene.Define("/World", "Xform");
    SetXform(scene, "/World", {.translate = glm::dvec3(0, 0, 7)});
    for (int i = 0; i < 100; ++i) {
        const std::string env = "/World/Env_" + std::to_string(i);
        scene.Define(env, "Xform");
        SetXform(scene, env, {.translate = glm::dvec3(i, 0, 0)});
        scene.Define(env + "/Body", "Sphere");
        SetXform(scene, env + "/Body", {.translate = glm::dvec3(0, 1, 0)});
    }
    Arenas arenas(*context, MemoryUse::Device, "transforms", 1 << 20);
    TransformGraph graph(*context, arenas, scene, compiler);
    graph.Update().Wait();

    // A parent goes with its children in one update; a new order resets another's stack.
    const PrimId env = *scene.Find("/World/Env_3");
    const PrimId body = *scene.Find("/World/Env_3/Body");
    scene.Remove("/World/Env_3");
    scene.Set(*scene.Find("/World/Env_5"), "xformOpOrder",
              std::vector<std::string>{"!resetXformStack!", "xformOp:translate"});
    const std::vector<float> worlds = ReadFloats(*context, graph.Worlds(), graph.Update());
    EXPECT_FALSE(graph.Node(env));
    EXPECT_FALSE(graph.Node(body));
    for (const char* path :
         {"/World/Env_2/Body", "/World/Env_5", "/World/Env_5/Body", "/World/Env_99/Body"}) {
        const PrimId prim = *scene.Find(path);
        ExpectWorld(worlds, *graph.Node(prim), scene.WorldTransform(prim));
    }
    ExpectClean(*context);
}

TEST(TransformGraph, ARetypedAncestorKeepsItsDescendants) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    Scene scene;
    // The arm is defined as the hand's typeless ancestor, as USD does.
    scene.Define("/Robot", "Xform");
    scene.Define("/Robot/Arm/Hand", "Xform");
    SetXform(scene, "/Robot", {.translate = glm::dvec3(1, 0, 0)});
    SetXform(scene, "/Robot/Arm/Hand", {.translate = glm::dvec3(0, 0, 1)});
    Arenas arenas(*context, MemoryUse::Device, "transforms", 1 << 16);
    TransformGraph graph(*context, arenas, scene, compiler);
    graph.Update().Wait();
    EXPECT_EQ(graph.Levels(), 3u);

    // Retyped, the arm moves to another bucket: it leaves the graph and comes back, its hand
    // hanging from the robot meanwhile and from the arm again after.
    scene.Define("/Robot/Arm", "Xform");
    SetXform(scene, "/Robot/Arm", {.translate = glm::dvec3(0, 5, 0)});
    const std::vector<float> worlds = ReadFloats(*context, graph.Worlds(), graph.Update());
    EXPECT_EQ(graph.Levels(), 3u);
    for (const char* path : {"/Robot/Arm", "/Robot/Arm/Hand"}) {
        const PrimId prim = *scene.Find(path);
        ExpectWorld(worlds, *graph.Node(prim), scene.WorldTransform(prim));
    }
    ExpectClean(*context);
}

TEST(TransformGraph, AnyOpOrderComposesOnTheDevice) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    Scene scene;
    scene.Define("/Rig", "Xform");
    SetXform(scene, "/Rig", {.translate = glm::dvec3(0, 0, 2)});
    const PrimId joint = scene.Define("/Rig/Joint", "Xform");
    scene.Set(joint, "xformOp:rotateZYX", glm::dvec3(0));
    scene.Set(joint, "xformOp:translate", glm::dvec3(1, 2, 3));
    scene.Set(joint, "xformOp:scale", glm::dvec3(1));
    scene.Set(joint, "xformOpOrder",
              std::vector<std::string>{"xformOp:rotateZYX", "!invert!xformOp:translate",
                                       "xformOp:scale"});
    const BucketId bucket = scene.Store().PrimBucket(joint);

    Arenas arenas(*context, MemoryUse::Device, "transforms", 1 << 16);
    SceneColumns columns(*context, arenas, scene);
    const DeviceColumn& rotate = columns.Bind(bucket, "xformOp:rotateZYX");
    const DeviceColumn& scale = columns.Bind(bucket, "xformOp:scale");
    TransformGraph graph(*context, arenas, scene, compiler);
    graph.Bind(bucket, "xformOp:rotateZYX", rotate);
    graph.Bind(bucket, "xformOp:scale", scale);
    graph.Update(std::array{columns.Sync()}).Wait();

    // The joint's angles and scale change on the device; the inverted translate stays the host's.
    const glm::dvec3 degrees(10, 20, 30);
    const glm::dvec3 scaled(1, 2, 0.5);
    std::vector<float> angles(rotate.Rows() * 3);
    std::vector<float> scales(scale.Rows() * 3);
    const std::uint32_t row = scene.Store().PrimRow(joint);
    std::ranges::copy(std::array{10.0f, 20.0f, 30.0f}, angles.begin() + row * 3);
    std::ranges::copy(std::array{1.0f, 2.0f, 0.5f}, scales.begin() + row * 3);
    Uploader simulation(*context, context->Main());
    simulation.Write(rotate.Storage(), rotate.Range().offset, std::span<const float>(angles));
    simulation.Write(scale.Storage(), scale.Range().offset, std::span<const float>(scales));
    const Ticket simulated = simulation.Flush();
    const Ticket synced = columns.Sync(std::span(&simulated, 1));
    const std::vector<float> worlds =
        ReadFloats(*context, graph.Worlds(), graph.Update(std::span(&synced, 1)));

    const glm::dmat4 expected = glm::translate(glm::dmat4(1.0), glm::dvec3(0, 0, 2)) *
                                glm::dmat4(EulerRotation(degrees, RotationOrder::ZYX)) *
                                glm::inverse(glm::translate(glm::dmat4(1.0), glm::dvec3(1, 2, 3))) *
                                glm::scale(glm::dmat4(1.0), scaled);
    ExpectWorld(worlds, *graph.Node(joint), expected);
    ExpectClean(*context);
}

}  // namespace
}  // namespace ernie
