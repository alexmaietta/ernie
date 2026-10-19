// SPDX-License-Identifier: BSD-3-Clause
/// @file device_scene_test.cpp
/// @brief Tests of the device scene: the scene's columns, its transforms and a frame's own passes
/// run as one graph in one submission, under synchronization validation, as the device and the
/// host change the scene between frames.
#include "ernie/render/device_scene.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/xform.h"

#include "tests/device/columns.h"
#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <algorithm>
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

TEST(DeviceScene, RunsTheScenesPassesBeforeAFramesOwnInOneGraph) {
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

TEST(DeviceScene, FollowsTheDeviceAndTheHostFrameByFrame) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    Bodies bodies;
    Arenas arenas(*context, MemoryUse::Device, "frame", 1 << 20);
    DeviceScene frame(*context, arenas, bodies.scene, compiler);
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

TEST(DeviceScene, RemakesWorldsAndTopLevelsOnlyWhenSomethingCanMove) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    Scene scene;
    const PrimId env = scene.Define("/Env", "Xform");
    SetXform(scene, "/Env", {.translate = glm::dvec3(1, 0, 0)});
    Sphere::Define(scene, "/Env/Ball");
    scene.Update();
    Arenas arenas(*context, MemoryUse::Device, "frame", 1 << 20);
    DeviceScene frame(*context, arenas, scene, compiler);
    const auto run = [&] {
        std::vector<std::string> names;
        frame.Execute([&](graph::Graph& graph) { names = graph.PassNames(); }).Wait();
        return names;
    };
    // Whether a frame composed the worlds and, with a ray tracer, built the top levels.
    const auto remade = [&](const std::vector<std::string>& names) {
        const auto has = [&](std::string_view name) {
            return std::ranges::find(names, name) != names.end();
        };
        return has("World transforms") &&
               (frame.Tracer() == nullptr || has("Build the top levels"));
    };
    const auto skipped = [&](const std::vector<std::string>& names) {
        return std::ranges::find(names, "World transforms") == names.end() &&
               std::ranges::find(names, "Build the top levels") == names.end();
    };
    EXPECT_TRUE(remade(run())) << "the first frame";
    // Compacting the bottom levels moves them, so top levels build until they settle.
    for (int frames = 0; frame.Tracer() != nullptr && !frame.Tracer()->Bottoms().Idle(); ++frames) {
        ASSERT_LT(frames, 4) << "the bottom levels settle";
        EXPECT_TRUE(remade(run())) << "while the bottom levels compact";
    }
    EXPECT_TRUE(skipped(run())) << "nothing changed";
    SetXform(scene, "/Env", {.translate = glm::dvec3(2, 0, 0)});
    scene.Update();
    EXPECT_TRUE(remade(run())) << "a host edit";
    EXPECT_TRUE(skipped(run())) << "still again";

    // Bound columns move worlds on the device unseen, so every frame remakes them.
    const BucketId bucket = scene.Store().PrimBucket(env);
    frame.Transforms().Bind(bucket, "xformOp:translate",
                            frame.Columns().Bind(bucket, "xformOp:translate"));
    EXPECT_TRUE(remade(run()));
    EXPECT_TRUE(remade(run())) << "bound: every frame";
    ExpectClean(*context);
}

/// Expects each world's records to hold the shared rows, then its own, then none, in any order
/// within the shared rows and within its own.
void ExpectSources(DeviceContext& context, DeviceScene& frame, const Scene& scene,
                   const std::vector<std::vector<std::string>>& own,
                   const std::vector<std::string>& shared, const Ticket& done) {
    const InstanceTable& instances = frame.Instances();
    const std::vector<std::uint32_t> sources =
        test::ReadRows<std::uint32_t>(context, instances.Sources(), done);
    ASSERT_EQ(sources.size(), instances.Records());
    const auto rows = [&](const std::vector<std::string>& paths) {
        std::vector<std::uint32_t> found;
        for (const std::string& path : paths) {
            found.push_back(*instances.Row(*scene.Find(path)));
        }
        std::ranges::sort(found);
        return found;
    };
    const auto sorted = [&](std::size_t from, std::size_t to) {
        std::vector<std::uint32_t> part(sources.begin() + static_cast<std::ptrdiff_t>(from),
                                        sources.begin() + static_cast<std::ptrdiff_t>(to));
        std::ranges::sort(part);
        return part;
    };
    ASSERT_EQ(instances.Worlds().size(), own.size());
    // The shared rows lead each world's records, or have their own after every world's.
    const std::optional<InstanceTable::WorldRecords> sharedTop = instances.Shared();
    const std::size_t ahead = sharedTop ? 0 : shared.size();
    const std::size_t stride =
        (instances.Records() - (sharedTop ? sharedTop->count : 0)) / own.size();
    if (sharedTop) {
        EXPECT_EQ(sharedTop->first, own.size() * stride);
        EXPECT_EQ(sharedTop->count, shared.size());
        EXPECT_EQ(sorted(sharedTop->first, sharedTop->first + sharedTop->count), rows(shared));
    }
    for (std::size_t world = 0; world < own.size(); ++world) {
        const InstanceTable::WorldRecords records = instances.Worlds()[world];
        EXPECT_EQ(records.first, world * stride);
        EXPECT_EQ(records.count, ahead + own[world].size()) << "world " << world;
        EXPECT_EQ(sorted(records.first, records.first + ahead),
                  sharedTop ? std::vector<std::uint32_t>{} : rows(shared));
        EXPECT_EQ(sorted(records.first + ahead, records.first + records.count), rows(own[world]))
            << "world " << world;
        for (std::size_t spare = records.first + records.count; spare < (world + 1) * stride;
             ++spare) {
            EXPECT_EQ(sources[spare], InstanceTable::kNoRow) << "record " << spare;
        }
    }
}

TEST(DeviceScene, SortsInstancesByWorldOnTheDevice) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    for (const WorldCapacities capacities :
         {WorldCapacities{.sharedPrims = SharedPrims::Duplicated},
          WorldCapacities{4, 6, SharedPrims::Duplicated},
          WorldCapacities{.sharedPrims = SharedPrims::SharedTopLevel},
          WorldCapacities{4, 6, SharedPrims::SharedTopLevel}}) {
        SCOPED_TRACE(
            std::to_string(capacities.worlds) +
            (capacities.sharedPrims == SharedPrims::Duplicated ? " duplicated" : " own top level"));
        // Two walls in every world; world 0 has a ball, world 1 none and world 2 three.
        Scene scene;
        Cube::Define(scene, "/Wall0");
        Cube::Define(scene, "/Wall1");
        for (int world = 0; world < 3; ++world) {
            Xform::Define(scene, "/World" + std::to_string(world)).SetWorld(world);
        }
        Sphere::Define(scene, "/World0/Ball");
        for (int ball = 0; ball < 3; ++ball) {
            Sphere::Define(scene, "/World2/Ball" + std::to_string(ball));
        }
        Arenas arenas(*context, MemoryUse::Device, "frame", 1 << 20);
        DeviceScene frame(*context, arenas, scene, compiler, capacities);
        const std::vector<std::string> walls{"/Wall0", "/Wall1"};
        const auto expect = [&](std::vector<std::vector<std::string>> own) {
            own.resize(std::max<std::size_t>(own.size(), capacities.worlds));
            scene.Update();
            ExpectSources(*context, frame, scene, own, walls, frame.Execute({}));
        };
        expect({{"/World0/Ball"}, {}, {"/World2/Ball0", "/World2/Ball1", "/World2/Ball2"}});

        // A ball moves to world 1, and world 0's goes, moving the last row into its own.
        scene.Set("/World2/Ball1", Imageable::kWorld, 1);
        scene.Remove("/World0/Ball");
        expect({{}, {"/World2/Ball1"}, {"/World2/Ball0", "/World2/Ball2"}});
        ExpectClean(*context);
    }
}

}  // namespace
}  // namespace ernie
