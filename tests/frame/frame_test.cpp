// SPDX-License-Identifier: BSD-3-Clause
/// @file frame_test.cpp
/// @brief Tests of a frame: the scene's columns, its transforms and the frame's own passes run as
/// one graph in one submission, under synchronization validation, as the device and the host
/// change the scene between frames.
#include "ernie/frame/frame.h"
#include "ernie/scene/xform.h"

#include "tests/device/columns.h"
#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;
using test::ReadFloats;

constexpr int kBodies = 32;

/// Environments along x, each with a body whose translate is on the device.
struct Bodies {
    Scene scene;
    std::vector<PrimId> body;
    BucketId bucket = 0;

    Bodies() {
        for (int i = 0; i < kBodies; ++i) {
            const std::string env = "/Env_" + std::to_string(i);
            scene.Define(env, "Xform");
            SetXform(scene, env, {.translate = glm::dvec3(10.0 * i, 0, 0)});
            body.push_back(scene.Define(env + "/Body", "Xform"));
            SetXform(scene, env + "/Body", {.translate = glm::dvec3(0, 0, 1)});
        }
        bucket = scene.Store().PrimBucket(body.front());
    }
};

/// Expects a prim's world translation, column 3 of its three rows.
void ExpectTranslation(const std::vector<float>& worlds, std::uint32_t node, glm::dvec3 expected) {
    const glm::dmat4 world = TransformGraph::ReadWorld(worlds, node);
    for (int r = 0; r < 3; ++r) {
        EXPECT_NEAR(world[3][r], expected[r], 1e-5) << "node " << node << ", row " << r;
    }
}

TEST(Frame, RunsTheScenesPassesBeforeItsOwnInOneGraph) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    Bodies bodies;
    Arenas arenas(*context, MemoryUse::Device, "frame", 1 << 20);
    SceneColumns columns(*context, arenas, bodies.scene);
    TransformGraph transforms(*context, arenas, bodies.scene, compiler);
    transforms.Bind(bodies.bucket, "xformOp:translate",
                    columns.Bind(bodies.bucket, "xformOp:translate"));

    graph::TransientPool pool(*context);
    const auto run = [&] {
        graph::Graph graph(*context, pool);
        columns.AddSync(graph);
        transforms.AddUpdate(graph);
        graph.Add(
            "Read the worlds", graph::Kind::Compute,
            [&](graph::PassBuilder& pass) {
                pass.ReadAddress(graph.Import(transforms.Worlds(), "worlds"));
                pass.NeverCull();
            },
            [](graph::PassContext&) {});
        const Ticket done = graph.Execute(context->Main());
        return std::pair(graph.ExecutedPasses(), done);
    };
    // The first frame syncs nothing: binding uploaded the columns' values.
    EXPECT_EQ(run().first,
              (std::vector<std::string>{"Upload transforms", "Local transforms", "World transforms",
                                        "World transforms", "Read the worlds"}));

    // A removed environment moves the bucket's last rows into its own.
    bodies.scene.Remove("/Env_4");
    const auto [passes, done] = run();
    EXPECT_EQ(passes, (std::vector<std::string>{"Sync scene columns", "Upload transforms",
                                                "Local transforms", "World transforms",
                                                "World transforms", "Read the worlds"}));
    const std::vector<float> worlds = ReadFloats(*context, transforms.Worlds(), done);
    ExpectTranslation(worlds, *transforms.Node(bodies.body[3]), {30, 0, 1});
    ExpectTranslation(worlds, *transforms.Node(bodies.body[kBodies - 1]),
                      {10.0 * (kBodies - 1), 0, 1});
    ExpectClean(*context);
}

TEST(Frame, FollowsTheDeviceAndTheHostFrameByFrame) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    Bodies bodies;
    Arenas arenas(*context, MemoryUse::Device, "frame", 1 << 20);
    Frame frame(*context, arenas, bodies.scene, compiler);
    const DeviceColumn& translate = frame.Columns().Bind(bodies.bucket, "xformOp:translate");
    frame.Transforms().Bind(bodies.bucket, "xformOp:translate", translate);
    frame.Execute({}).Wait();

    // A simulation writes a body's translate in place; the host moves an environment and removes
    // another, whose body's row the bucket fills from its last.
    Uploader simulation(*context, context->Main());
    const std::array height{0.0f, 0.0f, 5.0f};
    simulation.Write(translate.Storage(),
                     translate.Range().offset +
                         bodies.scene.Store().PrimRow(bodies.body[7]) * translate.RowBytes(),
                     std::span<const float>(height));
    const Ticket simulated = simulation.Flush();
    SetXform(bodies.scene, "/Env_2", {.translate = glm::dvec3(0, 100, 0)});
    bodies.scene.Remove("/Env_4");

    const Ticket done = frame.Execute({}, std::span(&simulated, 1));
    const std::vector<float> worlds = ReadFloats(*context, frame.Transforms().Worlds(), done);
    ExpectTranslation(worlds, *frame.Transforms().Node(bodies.body[7]), {70, 0, 5});
    ExpectTranslation(worlds, *frame.Transforms().Node(bodies.body[2]), {0, 100, 1});
    ExpectTranslation(worlds, *frame.Transforms().Node(bodies.body[kBodies - 1]),
                      {10.0 * (kBodies - 1), 0, 1});
    EXPECT_FALSE(frame.Transforms().Node(bodies.body[4]));
    ExpectClean(*context);
}

}  // namespace
}  // namespace ernie
