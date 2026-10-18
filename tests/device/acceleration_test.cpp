// SPDX-License-Identifier: BSD-3-Clause
/// @file acceleration_test.cpp
/// @brief Tests of acceleration structures under validation: triangles and procedural spheres
/// traced through instanced top levels, batched builds in a pool, compaction read back without
/// waiting, and refits of deforming meshes.
#include "ernie/device/acceleration.h"
#include "ernie/device/kernel.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/device/staging.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;

/// Traces a ray along +z from (x, 0, 0) for each x, writing the hit distance or -1. Procedural
/// candidates are spheres filling their boxes, intersected exactly in object space.
const ShaderModule kTrace{"trace", R"(
struct Parameters { uint64_t scene; float* xs; float* hits; uint count; };
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

float Sphere(float3 origin, float3 direction, float3 centre, float radius) {
    float3 f = origin - centre;
    float b = dot(f, direction);
    float3 l = f - b * direction;
    float discriminant = radius * radius - dot(l, l);
    return discriminant < 0 ? -1 : -b - sqrt(discriminant);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= parameters.count) {
        return;
    }
    RayDesc ray;
    ray.Origin = float3(parameters.xs[id.x], 0, 0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0;
    ray.TMax = 1e30;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(RaytracingAccelerationStructure(parameters.scene), RAY_FLAG_NONE, 0xff,
                         ray);
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            // The object-space ray from the candidate's matrix: NVIDIA's 610 compiler crashes on
            // CandidateObjectRayOrigin and CandidateObjectRayDirection.
            float3x4 toObject = query.CandidateWorldToObject3x4();
            float3 origin = mul(toObject, float4(query.WorldRayOrigin(), 1));
            float3 direction = mul(toObject, float4(query.WorldRayDirection(), 0));
            float t = Sphere(origin, direction, float3(0), 1);
            if (t >= 0) {
                query.CommitProceduralPrimitiveHit(t);
            }
        }
    }
    uint status = query.CommittedStatus();
    parameters.hits[id.x] = status == COMMITTED_NOTHING ? -1 : query.CommittedRayT();
}
)"};

struct TraceParameters {
    VkDeviceAddress scene, xs, hits;
    std::uint32_t count, pad;
};

/// Orders the next step after everything recorded before it.
void Then(VkCommandBuffer commands) {
    Barrier(commands, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, kAnyAccess);
}

/// A device with a pool, a tracing kernel, and buffers for geometry, instances and rays.
class Acceleration : public ::testing::Test {
  protected:
    void SetUp() override {
        context_ = test::MakeContext();
        if (!context_) {
            GTEST_SKIP() << "No Vulkan device";
        }
        if (!context_->Capabilities().rayQuery) {
            GTEST_SKIP() << context_->Info().name << " has no ray queries";
        }
        trace_ = std::make_unique<ComputeKernel>(
            *context_, ShaderCompiler().Spirv(std::span(&kTrace, 1), "main"), "trace",
            std::uint32_t{sizeof(TraceParameters)});
        pool_ = std::make_unique<AccelerationPool>(*context_, "test structures");
        geometry_ = std::make_unique<Buffer>(*context_, 4096, MemoryUse::Device, "geometry");
        instances_ = std::make_unique<Buffer>(*context_, 4096, MemoryUse::Device, "instances");
        rays_ = std::make_unique<Buffer>(*context_, 1024, MemoryUse::Device, "rays");
        scratch_ = std::make_unique<Buffer>(*context_, 4 << 20, MemoryUse::Device, "scratch");
        uploader_ = std::make_unique<Uploader>(*context_, context_->Main());
    }
    void TearDown() override {
        if (context_) {
            context_->WaitIdle();
            ExpectClean(*context_);
        }
    }

    /// @return A triangle around the z axis at z = 0, its vertices at byte `at` of the geometry.
    TriangleGeometry Triangle(std::uint64_t at, float scale = 1.0f) {
        const std::array<float, 9> vertices{-scale, -scale, 0, scale, -scale, 0, 0, scale, 0};
        const std::array<std::uint32_t, 3> indices{0, 1, 2};
        uploader_->Write(*geometry_, at, std::span<const float>(vertices));
        uploader_->Write(*geometry_, at + 64, std::span<const std::uint32_t>(indices));
        return {.vertices = {geometry_.get(), at},
                .vertexCount = 3,
                .indices = {geometry_.get(), at + 64},
                .triangleCount = 1};
    }

    /// @return An instance of `bottom` moved by `offset`.
    static AccelerationInstance Instance(const AccelerationStructure& bottom, glm::vec3 offset) {
        AccelerationInstance instance;
        instance.transform[0][3] = offset.x;
        instance.transform[1][3] = offset.y;
        instance.transform[2][3] = offset.z;
        instance.bottom = bottom.Address();
        return instance;
    }

    /// Records the bottom builds, then the top level over `instances`, in their own calls.
    void Build(VkCommandBuffer commands, std::span<const BottomBuild> bottoms,
               const AccelerationStructure& top, std::uint32_t instances) {
        RecordBuilds(commands, bottoms, {}, {scratch_.get(), 0});
        Then(commands);
        const TopBuild build{
            .target = &top, .instances = {instances_.get(), 0}, .count = instances};
        RecordBuilds(commands, {}, std::span(&build, 1), {scratch_.get(), 0});
        Then(commands);
    }

    /// @return Each ray's hit distance in the scene `top`, after `commands` and the uploads.
    std::vector<float> Trace(const std::function<void(VkCommandBuffer)>& commands,
                             const AccelerationStructure& top, std::span<const float> xs) {
        uploader_->Write(*rays_, 0, xs);
        const Ticket uploaded = uploader_->Flush();
        const Ticket traced = context_->Main().Submit(
            [&](VkCommandBuffer recording) {
                Then(recording);
                commands(recording);
                trace_->Dispatch(recording, {1, 1, 1},
                                 TraceParameters{top.Address(), rays_->Address(),
                                                 rays_->Address() + 512,
                                                 static_cast<std::uint32_t>(xs.size()), 0});
            },
            std::span(&uploaded, 1));
        Readback readback(*context_, context_->Main(), *rays_, 512, xs.size() * sizeof(float),
                          std::span(&traced, 1));
        std::vector<float> hits(xs.size());
        std::memcpy(hits.data(), readback.Bytes().data(), xs.size() * sizeof(float));
        return hits;
    }

    std::unique_ptr<DeviceContext> context_;
    std::unique_ptr<ComputeKernel> trace_;
    std::unique_ptr<AccelerationPool> pool_;
    std::unique_ptr<Buffer> geometry_;
    std::unique_ptr<Buffer> instances_;
    std::unique_ptr<Buffer> rays_;
    std::unique_ptr<Buffer> scratch_;
    std::unique_ptr<Uploader> uploader_;
};

TEST_F(Acceleration, TracesInstancesOfATriangle) {
    const TriangleGeometry triangle = Triangle(0);
    const AccelerationStructure bottom(*pool_, Sizes(*context_, std::span(&triangle, 1), kStatic),
                                       "triangle");
    const AccelerationStructure top(*pool_, Sizes(*context_, 2, kRebuilt), "scene");
    const std::array instances{Instance(bottom, {0, 0, 5}), Instance(bottom, {10, 0, 2})};
    uploader_->Write(*instances_, 0, std::span<const AccelerationInstance>(instances));
    const BottomBuild build{.target = &bottom, .triangles = std::span(&triangle, 1)};
    const std::array xs{0.0f, 10.0f, 5.0f};
    const std::vector<float> hits = Trace(
        [&](VkCommandBuffer commands) { Build(commands, std::span(&build, 1), top, 2); }, top, xs);
    EXPECT_NEAR(hits[0], 5.0f, 1e-5f) << "the first instance";
    EXPECT_NEAR(hits[1], 2.0f, 1e-5f) << "the second, moved aside and nearer";
    EXPECT_EQ(hits[2], -1.0f) << "between them";
}

TEST_F(Acceleration, BuildsManyBottomLevelsInOneBatchFromOnePool) {
    constexpr std::uint32_t kMeshes = 16;
    std::vector<TriangleGeometry> triangles;
    std::vector<std::unique_ptr<AccelerationStructure>> bottoms;
    std::vector<BottomBuild> builds;
    std::vector<AccelerationInstance> instances;
    for (std::uint32_t i = 0; i < kMeshes; ++i) {
        triangles.push_back(Triangle(128 * i, 0.4f));
    }
    for (std::uint32_t i = 0; i < kMeshes; ++i) {
        bottoms.push_back(std::make_unique<AccelerationStructure>(
            *pool_, Sizes(*context_, std::span(&triangles[i], 1), kStatic),
            "mesh " + std::to_string(i)));
        builds.push_back(
            {.target = bottoms.back().get(), .triangles = std::span(&triangles[i], 1)});
        const auto at = static_cast<float>(i);
        instances.push_back(Instance(*bottoms.back(), {at, 0, 1.0f + at}));
    }
    EXPECT_EQ(pool_->Usage().bottoms, kMeshes);
    EXPECT_GT(pool_->Usage().bottomBytes, 0u);
    EXPECT_LE(ScratchBytes(builds, {}), scratch_->Size()) << "one scratch for the whole batch";
    const AccelerationStructure top(*pool_, Sizes(*context_, kMeshes, kRebuilt), "scene");
    uploader_->Write(*instances_, 0, std::span<const AccelerationInstance>(instances));
    std::vector<float> xs;
    for (std::uint32_t i = 0; i < kMeshes; ++i) {
        xs.push_back(static_cast<float>(i));
    }
    const std::vector<float> hits =
        Trace([&](VkCommandBuffer commands) { Build(commands, builds, top, kMeshes); }, top, xs);
    for (std::uint32_t i = 0; i < kMeshes; ++i) {
        EXPECT_NEAR(hits[i], 1.0f + static_cast<float>(i), 1e-5f) << "mesh " << i;
    }
    bottoms.clear();
    EXPECT_EQ(pool_->Usage().bottoms, 0u) << "retired structures leave the pool's statistics";
}

TEST_F(Acceleration, CompactsWithoutWaitingOnTheHost) {
    const TriangleGeometry triangle = Triangle(0);
    const AccelerationStructure built(*pool_, Sizes(*context_, std::span(&triangle, 1), kStatic),
                                      "triangle");
    const BottomBuild build{.target = &built, .triangles = std::span(&triangle, 1)};
    CompactionQuery query(*context_, 1);
    const Ticket uploaded = uploader_->Flush();
    const AccelerationStructure* const structures[] = {&built};
    const Ticket recorded = context_->Main().Submit(
        [&](VkCommandBuffer commands) {
            Then(commands);
            RecordBuilds(commands, std::span(&build, 1), {}, {scratch_.get(), 0});
            Then(commands);
            query.Record(commands, structures);
        },
        std::span(&uploaded, 1));
    // The sizes arrive with the build's ticket; nothing blocks for them.
    recorded.Wait();
    const std::optional<std::vector<std::uint64_t>> sizes = query.Sizes();
    ASSERT_TRUE(sizes);
    EXPECT_GT((*sizes)[0], 0u);
    EXPECT_LE((*sizes)[0], built.Sizes().storage);

    AccelerationSizes compactedSizes = built.Sizes();
    compactedSizes.storage = (*sizes)[0];
    const AccelerationStructure compacted(*pool_, compactedSizes, "compacted triangle");
    const AccelerationStructure top(*pool_, Sizes(*context_, 1, kRebuilt), "scene");
    const AccelerationInstance instance = Instance(compacted, {0, 0, 3});
    uploader_->Write(*instances_, 0, std::span(&instance, 1));
    const std::array xs{0.0f};
    const std::vector<float> hits = Trace(
        [&](VkCommandBuffer commands) {
            RecordCompaction(commands, built, compacted);
            Then(commands);
            const TopBuild topBuild{.target = &top, .instances = {instances_.get(), 0}, .count = 1};
            RecordBuilds(commands, {}, std::span(&topBuild, 1), {scratch_.get(), 0});
            Then(commands);
        },
        top, xs);
    EXPECT_NEAR(hits[0], 3.0f, 1e-5f) << "the compacted copy traces as the original";
}

TEST_F(Acceleration, RefitsADeformingMeshInPlace) {
    const TriangleGeometry triangle = Triangle(0);
    const AccelerationStructure mesh(*pool_, Sizes(*context_, std::span(&triangle, 1), kDeforming),
                                     "deforming");
    const AccelerationStructure top(*pool_, Sizes(*context_, 1, kRebuilt), "scene");
    const AccelerationInstance instance = Instance(mesh, {0, 0, 4});
    uploader_->Write(*instances_, 0, std::span(&instance, 1));
    const BottomBuild build{.target = &mesh, .triangles = std::span(&triangle, 1)};
    // At y = 0 the triangle spans |x| <= 0.5, and the grown one |x| <= 1.5.
    const std::array xs{0.0f, 1.2f};
    std::vector<float> hits = Trace(
        [&](VkCommandBuffer commands) { Build(commands, std::span(&build, 1), top, 1); }, top, xs);
    EXPECT_EQ(hits[1], -1.0f) << "beyond the triangle";

    // The vertices grow threefold in place, as a skinning kernel would write them; a refit follows.
    Triangle(0, 3.0f);
    const BottomBuild refit{.target = &mesh, .triangles = std::span(&triangle, 1), .update = true};
    hits = Trace([&](VkCommandBuffer commands) { Build(commands, std::span(&refit, 1), top, 1); },
                 top, xs);
    EXPECT_NEAR(hits[0], 4.0f, 1e-5f);
    EXPECT_NEAR(hits[1], 4.0f, 1e-5f) << "inside the grown triangle";
}

TEST_F(Acceleration, IntersectsAnalyticSpheresInProceduralBoxes) {
    // A unit sphere's box, instanced at a planetary radius: the shader solves the sphere exactly.
    const std::array<float, 6> box{-1, -1, -1, 1, 1, 1};
    uploader_->Write(*geometry_, 0, std::span<const float>(box));
    const BoxGeometry boxes{.boxes = {geometry_.get(), 0}, .count = 1};
    const AccelerationStructure sphere(*pool_, Sizes(*context_, std::span(&boxes, 1), kStatic),
                                       "sphere");
    const AccelerationStructure top(*pool_, Sizes(*context_, 1, kRebuilt), "scene");
    const AccelerationInstance instance = Instance(sphere, {0, 0, 10});
    uploader_->Write(*instances_, 0, std::span(&instance, 1));
    const BottomBuild build{.target = &sphere, .boxes = std::span(&boxes, 1)};
    const std::array xs{0.0f, 0.6f, 0.99f, 1.01f};
    const std::vector<float> hits = Trace(
        [&](VkCommandBuffer commands) { Build(commands, std::span(&build, 1), top, 1); }, top, xs);
    for (std::size_t i = 0; i < 3; ++i) {
        const float expected = 10.0f - std::sqrt(1.0f - xs[i] * xs[i]);
        EXPECT_NEAR(hits[i], expected, 1e-4f) << "x = " << xs[i] << ", on the round surface";
    }
    EXPECT_EQ(hits[3], -1.0f) << "inside the box's corner but outside the sphere";
}

TEST_F(Acceleration, RejectsBuildsTheirTargetsCannotHold) {
    const TriangleGeometry triangle = Triangle(0);
    const AccelerationStructure fixed(*pool_, Sizes(*context_, std::span(&triangle, 1), kStatic),
                                      "fixed");
    const BottomBuild refit{.target = &fixed, .triangles = std::span(&triangle, 1), .update = true};
    EXPECT_THROW(context_->Main().Submit([&](VkCommandBuffer commands) {
        RecordBuilds(commands, std::span(&refit, 1), {}, {scratch_.get(), 0});
    }),
                 std::invalid_argument)
        << "a refit of a structure not built updatable";
    // Twice the triangles the structure was sized for.
    const std::array<std::uint32_t, 6> twice{0, 1, 2, 0, 1, 2};
    uploader_->Write(*geometry_, 64, std::span<const std::uint32_t>(twice));
    TriangleGeometry larger = triangle;
    larger.triangleCount = 2;
    const BottomBuild grown{.target = &fixed, .triangles = std::span(&larger, 1)};
    EXPECT_THROW(context_->Main().Submit([&](VkCommandBuffer commands) {
        RecordBuilds(commands, std::span(&grown, 1), {}, {scratch_.get(), 0});
    }),
                 std::invalid_argument)
        << "more triangles than it was sized for";
    EXPECT_THROW(Sizes(*context_, std::span<const TriangleGeometry>(), kStatic),
                 std::invalid_argument)
        << "no geometry";
}

}  // namespace
}  // namespace ernie
