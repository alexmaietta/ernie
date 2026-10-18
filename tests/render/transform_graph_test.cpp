// SPDX-License-Identifier: BSD-3-Clause
/// @file transform_graph_test.cpp
/// @brief Tests of the transform graph: world transforms composed level by level on the device
/// match the scene's, poses written on the device compose with the ops the host authors, and
/// structural changes rebuild the graph and its slot table.
#include "ernie/render/scene_columns.h"
#include "ernie/render/transform_graph.h"
#include "ernie/scene/xform.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <glm/gtc/quaternion.hpp>

#include <cstring>
#include <string>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;

std::vector<float> Read(DeviceContext& context, const DeviceColumn& column, const Ticket& after) {
    Readback readback(context, context.Main(), column.Storage(), column.Range().offset,
                      column.Bytes(), std::span(&after, 1));
    std::vector<float> values(column.Bytes() / sizeof(float));
    std::memcpy(values.data(), readback.Bytes().data(), column.Bytes());
    return values;
}

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
    const std::vector<float> worlds = Read(*context, graph.Worlds(), composed);
    for (const char* path :
         {"/World", "/World/Arm", "/World/Arm/Hand", "/World/Arm/Hand/Tip", "/World/Free"}) {
        const PrimId prim = *scene.Find(path);
        ExpectWorld(worlds, *graph.Node(prim), scene.WorldTransform(prim));
    }

    // An authored edit uploads only its node, and its descendants follow on the device.
    const Transfers before = context->Transfers();
    SetXform(scene, "/World", {.translate = glm::dvec3(0, 0, 10)});
    const std::vector<float> moved = Read(*context, graph.Worlds(), graph.Update());
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
    graph.BindPoses(bucket, translate, orient);
    graph.Update(std::array{columns.Sync()}).Wait();

    // A simulation writes the bodies' poses in place at their rows (the environments share the
    // bucket); its writes stand in for CUDA's or Warp's.
    std::vector<float> positions(translate.Rows() * 3);
    std::vector<float> orientations(orient.Rows() * 4);
    for (int i = 0; i < bodies; ++i) {
        const std::uint32_t row = scene.Store().PrimRow(body[static_cast<std::size_t>(i)]);
        std::ranges::copy(std::array{0.0f, float(i), 1.0f}, positions.begin() + row * 3);
        std::ranges::copy(std::array{0.0f, 0.0f, float(kQuarterTurn.z), float(kQuarterTurn.w)},
                          orientations.begin() + row * 4);
    }
    Uploader simulation(*context, context->Main());
    simulation.Write(translate.Storage(), translate.Range().offset,
                     std::span<const float>(positions));
    simulation.Write(orient.Storage(), orient.Range().offset, std::span<const float>(orientations));
    const Ticket simulated = simulation.Flush();

    const Transfers before = context->Transfers();
    const Ticket synced = columns.Sync(std::span(&simulated, 1));
    const std::vector<float> worlds =
        Read(*context, graph.Worlds(), graph.Update(std::span(&synced, 1)));
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

TEST(TransformGraph, RebuildsWhenPrimsComeAndGo) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    Scene scene;
    for (int i = 0; i < 4; ++i) {
        const std::string path = "/Ball_" + std::to_string(i);
        scene.Define(path, "Sphere");
        SetXform(scene, path, {.translate = glm::dvec3(i, 0, 0)});
    }
    Arenas arenas(*context, MemoryUse::Device, "transforms", 1 << 16);
    TransformGraph graph(*context, arenas, scene, compiler);
    graph.Update().Wait();
    const PrimId gone = *scene.Find("/Ball_1");
    scene.Remove("/Ball_1");
    const PrimId added = scene.Define("/Ball_9", "Sphere");
    SetXform(scene, "/Ball_9", {.translate = glm::dvec3(9, 0, 0)});

    const Ticket rebuilt = graph.Update();
    EXPECT_FALSE(graph.Node(gone));
    const std::vector<float> worlds = Read(*context, graph.Worlds(), rebuilt);
    for (const char* path : {"/Ball_0", "/Ball_2", "/Ball_3", "/Ball_9"}) {
        const PrimId prim = *scene.Find(path);
        ExpectWorld(worlds, *graph.Node(prim), scene.WorldTransform(prim));
    }
    // The slot table finds a prim's node from its stable id, on the device.
    Readback slots(*context, context->Main(), graph.SlotNodes().Storage(),
                   graph.SlotNodes().Range().offset, graph.SlotNodes().Bytes(),
                   std::span(&rebuilt, 1));
    std::vector<std::uint32_t> table(graph.SlotNodes().Rows());
    std::memcpy(table.data(), slots.Bytes().data(), graph.SlotNodes().Bytes());
    EXPECT_EQ(table[added.slot], *graph.Node(added));
    ExpectClean(*context);
}

}  // namespace
}  // namespace ernie
