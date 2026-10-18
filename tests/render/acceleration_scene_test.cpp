// SPDX-License-Identifier: BSD-3-Clause
/// @file acceleration_scene_test.cpp
/// @brief Tests of the device scene's acceleration structures under validation: a mesh and
/// analytic spheres traced through the top level each frame builds, following the scene as prims
/// move, come and go, with bottom levels compacted in later frames.
#include "ernie/render/device_scene.h"
#include "ernie/scene/xform.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;

/// Traces a ray down -z from (x, y, 100) for each (x, y), writing the hit distance or -1.
/// Procedural candidates are unit spheres, as the scenes here hold no other shape.
const ShaderModule kTrace{"trace", R"(
struct Parameters { uint64_t scene; float2* rays; float* hits; uint count; };
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= parameters.count) {
        return;
    }
    RayDesc ray;
    ray.Origin = float3(parameters.rays[id.x], 100);
    ray.Direction = float3(0, 0, -1);
    ray.TMin = 0;
    ray.TMax = 1e30;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(RaytracingAccelerationStructure(parameters.scene), RAY_FLAG_NONE, 0xff,
                         ray);
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            float3x4 toObject = query.CandidateWorldToObject3x4();
            float3 origin = mul(toObject, float4(query.WorldRayOrigin(), 1));
            float3 direction = mul(toObject, float4(query.WorldRayDirection(), 0));
            // Object-space distance along an unnormalised direction is world distance.
            float a = dot(direction, direction);
            float b = dot(origin, direction);
            float c = dot(origin, origin) - 1;
            float discriminant = b * b - a * c;
            if (discriminant >= 0) {
                float t = (-b - sqrt(discriminant)) / a;
                if (t >= 0 && t < query.CommittedRayT()) {
                    query.CommitProceduralPrimitiveHit(t);
                }
            }
        }
    }
    parameters.hits[id.x] =
        query.CommittedStatus() == COMMITTED_NOTHING ? -1 : query.CommittedRayT();
}
)"};

struct TraceParameters {
    VkDeviceAddress scene, rays, hits;
    std::uint32_t count, pad;
};

class AccelerationScene : public ::testing::Test {
  protected:
    void SetUp() override {
        context_ = test::MakeContext();
        if (!context_) {
            GTEST_SKIP() << "No Vulkan device";
        }
        if (!context_->Capabilities().rayQuery) {
            GTEST_SKIP() << context_->Info().name << " has no ray queries";
        }
        const ShaderCompiler compiler;
        trace_ = std::make_unique<ComputeKernel>(*context_,
                                                 compiler.Spirv(std::span(&kTrace, 1), "main"),
                                                 "trace", std::uint32_t{sizeof(TraceParameters)});
        arenas_ = std::make_unique<Arenas>(*context_, MemoryUse::Device, "scene", 1 << 20);
        rays_ = std::make_unique<DeviceColumn>(*arenas_, "rays", DType::Float32,
                                               std::vector<std::int64_t>{2}, 8);
        hits_ = std::make_unique<DeviceColumn>(*arenas_, "hits", DType::Float32,
                                               std::vector<std::int64_t>{}, 8);

        // A 20 m ground at z = 0, and a sphere of radius 2 at (3, 0, 4) under a moved parent.
        const PrimId ground = scene_.Define("/Ground", "Mesh");
        const std::vector<float> points{-10, -10, 0, 10, -10, 0, 10, 10, 0, -10, 10, 0};
        scene_.Write(ground, "points", Tensor::Copy(std::span<const float>(points), {4, 3}));
        scene_.Write(
            ground, "faceVertexCounts",
            Tensor::Copy(std::span<const std::int32_t>(std::vector<std::int32_t>{4}), {1}));
        scene_.Write(
            ground, "faceVertexIndices",
            Tensor::Copy(std::span<const std::int32_t>(std::vector<std::int32_t>{0, 1, 2, 3}),
                         {4}));
        scene_.Define("/Props", "Xform");
        SetXform(scene_, "/Props", {.translate = glm::dvec3(1, 0, 0)});
        const PrimId ball = scene_.Define("/Props/Ball", "Sphere");
        scene_.Set(ball, "radius", 2.0);
        SetXform(scene_, "/Props/Ball", {.translate = glm::dvec3(2, 0, 4)});
        device_ = std::make_unique<DeviceScene>(*context_, *arenas_, scene_, compiler);
    }
    void TearDown() override {
        if (context_) {
            context_->WaitIdle();
            ExpectClean(*context_);
        }
    }

    /// @return Each ray's hit distance after a frame of the device scene.
    std::vector<float> Frame(const std::vector<glm::vec2>& rays) {
        Uploader uploader(*context_, context_->Main());
        uploader.Write(rays_->Storage(), rays_->Range().offset, std::span<const glm::vec2>(rays));
        const Ticket uploaded = uploader.Flush();
        const auto count = static_cast<std::uint32_t>(rays.size());
        const Ticket done = device_->Execute(
            [&](graph::Graph& graph) {
                const SceneStructure& structure = graph.Board().Get<SceneStructure>();
                const graph::Buffer from = graph.Import(*rays_, "rays");
                graph::Buffer to = graph.Import(*hits_, "hits");
                graph.Add(
                    "Trace", graph::Kind::Compute,
                    [&](graph::PassBuilder& pass) {
                        pass.Read(structure.top);
                        pass.ReadAddress(from);
                        to = pass.WriteAddress(to, graph::Use::StorageWrite);
                    },
                    [this, from, to, count, address = structure.address](graph::PassContext& pass) {
                        trace_->Dispatch(pass.commands, {1, 1, 1},
                                         TraceParameters{address, pass.Get(from).Address(),
                                                         pass.Get(to).Address(), count, 0});
                    });
            },
            std::span(&uploaded, 1));
        Readback readback(*context_, context_->Main(), hits_->Storage(), hits_->Range().offset,
                          rays.size() * sizeof(float), std::span(&done, 1));
        std::vector<float> hits(rays.size());
        std::memcpy(hits.data(), readback.Bytes().data(), hits.size() * sizeof(float));
        return hits;
    }

    std::unique_ptr<DeviceContext> context_;
    std::unique_ptr<ComputeKernel> trace_;
    std::unique_ptr<Arenas> arenas_;
    std::unique_ptr<DeviceColumn> rays_;
    std::unique_ptr<DeviceColumn> hits_;
    Scene scene_;
    std::unique_ptr<DeviceScene> device_;
};

TEST_F(AccelerationScene, TracesMeshesAndSpheresWhereTheSceneHasThem) {
    const std::vector<float> hits = Frame({{0, 0}, {3, 0}, {50, 0}});
    EXPECT_NEAR(hits[0], 100.0f, 1e-4f) << "the ground";
    EXPECT_NEAR(hits[1], 94.0f, 1e-4f) << "the ball's top, exactly round";
    EXPECT_EQ(hits[2], -1.0f) << "beyond the ground";
    EXPECT_EQ(device_->Instances().Count(), 2u);
    EXPECT_EQ(device_->Bottoms()->Pending(), 0u);
}

TEST_F(AccelerationScene, FollowsPrimsMovingComingAndGoing) {
    Frame({{0, 0}});
    // Frames pass while the ground's compacted size comes back and it moves into its compact copy.
    for (int frame = 0; frame < 3; ++frame) {
        EXPECT_NEAR(Frame({{0, 0}})[0], 100.0f, 1e-4f) << "frame " << frame;
    }
    const BottomLevels::Statistics& totals = device_->Bottoms()->Totals();
    EXPECT_EQ(totals.built, 2u) << "the ground and the unit sphere, once each";
    EXPECT_EQ(totals.compacted, 2u) << "both, a frame or more after their builds";
    SetXform(scene_, "/Props/Ball", {.translate = glm::dvec3(-4, 0, 4)});
    const PrimId pebble = scene_.Define("/Pebble", "Sphere");
    SetXform(scene_, "/Pebble", {.translate = glm::dvec3(5, 5, 1)});
    scene_.Remove("/Ground");
    const std::vector<float> hits = Frame({{-3, 0}, {3, 0}, {5, 5}, {0, -8}});
    EXPECT_NEAR(hits[0], 94.0f, 1e-4f) << "the ball, moved";
    EXPECT_EQ(hits[1], -1.0f) << "where the ball was";
    EXPECT_NEAR(hits[2], 98.0f, 1e-4f) << "the pebble, sharing the ball's unit sphere";
    EXPECT_EQ(hits[3], -1.0f) << "the ground, removed";
    EXPECT_TRUE(device_->Instances().Row(pebble));
}

}  // namespace
}  // namespace ernie
