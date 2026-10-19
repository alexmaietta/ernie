// SPDX-License-Identifier: BSD-3-Clause
/// @file graph_test.cpp
/// @brief Tests of the frame graph on the device, under synchronization validation: passes chained
/// through transients into imports, culled when nothing needs them, images cleared and copied,
/// transients pooled across graphs, and the debugging modes.
#include "ernie/device/kernel.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/device/staging.h"
#include "ernie/graph/graph.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <vector>

namespace ernie::graph {
namespace {

using test::ExpectClean;

const ShaderModule kAffine{"affine", R"(
struct Parameters { uint* input; uint* output; uint count; uint scale; uint add; uint indices; };
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i < parameters.count) {
        uint x = parameters.indices != 0 ? i : parameters.input[i];
        parameters.output[i] = x * parameters.scale + parameters.add;
    }
}
)"};

struct AffineParameters {
    VkDeviceAddress input, output;
    std::uint32_t count, scale, add, indices;
};

constexpr std::uint32_t kCount = 1000;

std::vector<std::uint32_t> Read(DeviceContext& context, const DeviceColumn& column,
                                const Ticket& after) {
    Readback readback(context, context.Main(), column.Storage(), column.Range().offset,
                      column.Bytes(), std::span(&after, 1));
    std::vector<std::uint32_t> values(column.Rows());
    std::memcpy(values.data(), readback.Bytes().data(), column.Bytes());
    return values;
}

class FrameGraph : public ::testing::Test {
  protected:
    void SetUp() override {
        context_ = test::MakeContext();
        if (!context_) {
            GTEST_SKIP() << "No Vulkan device";
        }
        affine_ = std::make_unique<ComputeKernel>(
            *context_, compiler_.Spirv(std::span(&kAffine, 1), "main"), "affine",
            std::uint32_t{sizeof(AffineParameters)});
        arenas_ = std::make_unique<Arenas>(*context_, MemoryUse::Device, "columns", 1 << 20);
        pool_ = std::make_unique<TransientPool>(*context_, 1 << 20);
    }
    void TearDown() override {
        if (context_) {
            context_->WaitIdle();
            ExpectClean(*context_);
        }
    }

    /// Adds `output = (input or the index) * scale + add` as a compute pass.
    void AddAffine(Graph& graph, const char* name, Buffer input, Buffer& output,
                   std::uint32_t scale, std::uint32_t add) {
        graph.Add(
            name, Kind::Compute,
            [&](PassBuilder& pass) {
                if (input.Valid()) {
                    pass.ReadAddress(input);
                }
                output = pass.WriteAddress(output, Use::StorageWrite);
            },
            [this, input, output, scale, add](PassContext& pass) {
                affine_->Dispatch(pass.commands, {(kCount + 63) / 64, 1, 1},
                                  AffineParameters{input.Valid() ? pass.Get(input).Address() : 0,
                                                   pass.Get(output).Address(), kCount, scale, add,
                                                   input.Valid() ? 0u : 1u});
            });
    }

    std::unique_ptr<DeviceContext> context_;
    ShaderCompiler compiler_;
    std::unique_ptr<ComputeKernel> affine_;
    std::unique_ptr<Arenas> arenas_;
    std::unique_ptr<TransientPool> pool_;
};

TEST_F(FrameGraph, ChainsPassesThroughTransientsAndCullsWhatNothingNeeds) {
    const DeviceColumn values(*arenas_, "values", DType::UInt32, {}, kCount);
    Graph graph(*context_, *pool_);
    const Buffer output = graph.Import(values, "values");
    Buffer doubled = graph.Create(kCount * 4, "doubled");
    Buffer unused = graph.Create(kCount * 4, "unused");
    Buffer result = output;
    AddAffine(graph, "Double", {}, doubled, 2, 0);
    AddAffine(graph, "Unused", {}, unused, 5, 0);
    AddAffine(graph, "Add one", doubled, result, 1, 1);
    const Ticket done = graph.Execute(context_->Main());
    EXPECT_EQ(graph.ExecutedPasses(), (std::vector<std::string>{"Double", "Add one"}))
        << "nothing reads what Unused writes";
    const std::vector<std::uint32_t> read = Read(*context_, values, done);
    for (std::uint32_t i = 0; i < kCount; ++i) {
        ASSERT_EQ(read[i], 2 * i + 1) << i;
    }
    EXPECT_NE(graph.Dot().find("label=\"Unused\", style=dashed"), std::string::npos);
}

TEST_F(FrameGraph, ClearsAndCopiesImagesInTheirLayouts) {
    const DeviceColumn texels(*arenas_, "texels", DType::Float32, {4}, 64);
    Graph graph(*context_, *pool_);
    Image image = graph.Create({.format = ImageFormat::Rgba32Float, .extent = {8, 8, 1}}, "image");
    const Buffer out = graph.Import(texels, "texels");
    graph.Add(
        "Clear", Kind::Copy, [&](PassBuilder& pass) { image = pass.Write(image, Use::CopyDst); },
        [image](PassContext& pass) { pass.Clear(image, {0.25f, 0.5f, 0.75f, 1.0f}); });
    Buffer copied = out;
    graph.Add(
        "Copy", Kind::Copy,
        [&](PassBuilder& pass) {
            pass.Read(image, Use::CopySrc);
            copied = pass.Write(copied, Use::CopyDst);
        },
        [image, out](PassContext& pass) {
            const BufferSlice slice = pass.Get(out);
            const VkBufferImageCopy region{
                .bufferOffset = slice.offset,
                .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
                .imageExtent = {8, 8, 1}};
            vkCmdCopyImageToBuffer(pass.commands.Handle(), pass.Get(image).Handle(),
                                   pass.Layout(image), slice.buffer->Handle(), 1, &region);
        });
    const Ticket done = graph.Execute(context_->Main());
    Readback readback(*context_, context_->Main(), texels.Storage(), texels.Range().offset,
                      texels.Bytes(), std::span(&done, 1));
    std::vector<float> values(64 * 4);
    std::memcpy(values.data(), readback.Bytes().data(), texels.Bytes());
    for (std::size_t i = 0; i < values.size(); i += 4) {
        ASSERT_EQ(values[i + 1], 0.5f) << i;
    }
}

TEST_F(FrameGraph, PoolsTransientImagesAcrossGraphs) {
    const DeviceColumn values(*arenas_, "values", DType::Float32, {4}, 64);
    for (int frame = 0; frame < 3; ++frame) {
        Graph graph(*context_, *pool_);
        Image image =
            graph.Create({.format = ImageFormat::Rgba32Float, .extent = {8, 8, 1}}, "image");
        Buffer out = graph.Import(values, "values");
        graph.Add(
            "Clear", Kind::Copy,
            [&](PassBuilder& pass) { image = pass.Write(image, Use::CopyDst); },
            [image](PassContext& pass) { pass.Clear(image, glm::vec4(0.0f)); });
        graph.Add(
            "Copy", Kind::Copy,
            [&](PassBuilder& pass) {
                pass.Read(image, Use::CopySrc);
                out = pass.Write(out, Use::CopyDst);
            },
            [image, out](PassContext& pass) {
                const BufferSlice slice = pass.Get(out);
                const VkBufferImageCopy region{
                    .bufferOffset = slice.offset,
                    .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
                    .imageExtent = {8, 8, 1}};
                vkCmdCopyImageToBuffer(pass.commands.Handle(), pass.Get(image).Handle(),
                                       pass.Layout(image), slice.buffer->Handle(), 1, &region);
            });
        graph.Execute(context_->Main());
    }
    EXPECT_EQ(pool_->Images(), 1u) << "one image serves every frame";
    pool_->Trim();
    EXPECT_EQ(pool_->Images(), 0u);
}

TEST_F(FrameGraph, RecordsPassesInNestedScopes) {
    const DeviceColumn values(*arenas_, "values", DType::UInt32, {}, kCount);
    Graph graph(*context_, *pool_);
    Buffer doubled = graph.Create(kCount * 4, "doubled");
    Buffer result = graph.Import(values, "values");
    {
        const Graph::ScopeGuard frame = graph.Scope("Frame");
        {
            const Graph::ScopeGuard shade = graph.Scope("Shade");
            AddAffine(graph, "Double", {}, doubled, 2, 0);
        }
        AddAffine(graph, "Add one", doubled, result, 1, 1);
    }
    const std::vector<std::uint32_t> read =
        Read(*context_, values, graph.Execute(context_->Main()));
    EXPECT_EQ(read[10], 21u);
}

/// Traces a ray along +z from (x, 0, 0) for each x, writing the hit distance or -1.
const ShaderModule kTrace{"trace", R"(
struct Parameters { uint64_t scene; float* xs; float* hits; uint count; };
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

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
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(RaytracingAccelerationStructure(parameters.scene), RAY_FLAG_NONE, 0xff,
                         ray);
    query.Proceed();
    parameters.hits[id.x] =
        query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? query.CommittedRayT() : -1;
}
)"};

struct TraceParameters {
    VkDeviceAddress scene, xs, hits;
    std::uint32_t count, pad;
};

TEST_F(FrameGraph, OrdersBuildsBeforeTheBuildsAndTracesThatReadThem) {
    ERNIE_REQUIRE_RAY_QUERIES(*context_);
    const ComputeKernel trace(*context_, compiler_.Spirv(std::span(&kTrace, 1), "main"), "trace",
                              std::uint32_t{sizeof(TraceParameters)});
    AccelerationPool structures(*context_, "test structures");
    // A triangle at z = 0 around the z axis, an instance of it at z = 5, and two rays.
    const ernie::Buffer geometry(*context_, 256, MemoryUse::Device, "geometry");
    const ernie::Buffer inputs(*context_, 256, MemoryUse::Device, "instances and rays");
    const DeviceColumn hits(*arenas_, "hits", DType::Float32, {}, 2);
    const TriangleGeometry triangle{.vertices = {&geometry, 0},
                                    .vertexCount = 3,
                                    .indices = {&geometry, 64},
                                    .triangleCount = 1};
    const AccelerationStructure bottom(
        structures, Sizes(*context_, std::span(&triangle, 1), kStatic), "triangle");
    const AccelerationStructure top(structures, Sizes(*context_, 1, kRebuilt), "scene");
    AccelerationInstance instance;
    instance.transform[2][3] = 5.0f;
    instance.bottom = bottom.Address();
    Uploader uploader(*context_, context_->Main());
    uploader.Write(geometry, 0,
                   std::span<const float>(std::array<float, 9>{-1, -1, 0, 1, -1, 0, 0, 1, 0}));
    uploader.Write(geometry, 64,
                   std::span<const std::uint32_t>(std::array<std::uint32_t, 3>{0, 1, 2}));
    uploader.Write(inputs, 0, std::span<const AccelerationInstance>(&instance, 1));
    uploader.Write(inputs, 128, std::span<const float>(std::array<float, 2>{0.0f, 4.0f}));
    const Ticket uploaded = uploader.Flush();

    Graph graph(*context_, *pool_);
    graph::Acceleration triangles = graph.Import(bottom);
    graph::Acceleration scene = graph.Import(top);
    const graph::Buffer vertices = graph.Import(geometry, 0, geometry.Size(), "geometry");
    const graph::Buffer instances = graph.Import(inputs, 0, 128, "instances");
    const graph::Buffer rays = graph.Import(inputs, 128, 8, "rays");
    graph::Buffer distances = graph.Import(hits, "hits");
    const BottomBuild bottomBuild{.target = &bottom, .triangles = std::span(&triangle, 1)};
    const TopBuild topBuild{.target = &top, .count = 1};
    graph::Buffer bottomScratch =
        graph.Create(ScratchBytes(std::span(&bottomBuild, 1), {}), "scratch");
    graph::Buffer topScratch = graph.Create(ScratchBytes({}, std::span(&topBuild, 1)), "scratch");
    graph.Add(
        "Build bottom levels", Kind::Build,
        [&](PassBuilder& pass) {
            pass.Read(vertices, Use::BuildInput);
            bottomScratch = pass.Write(bottomScratch, Use::BuildScratch);
            triangles = pass.Build(triangles);
        },
        [&](PassContext& pass) {
            const BufferSlice scratch = pass.Get(bottomScratch);
            RecordBuilds(pass.commands, std::span(&bottomBuild, 1), {},
                         {scratch.buffer, scratch.offset});
        });
    graph.Add(
        "Build the top level", Kind::Build,
        [&](PassBuilder& pass) {
            pass.Read(triangles, Use::BuildInput);
            pass.Read(instances, Use::BuildInput);
            topScratch = pass.Write(topScratch, Use::BuildScratch);
            scene = pass.Build(scene);
        },
        [&](PassContext& pass) {
            const BufferSlice scratch = pass.Get(topScratch);
            TopBuild build = topBuild;
            build.instances = {pass.Get(instances).buffer, pass.Get(instances).offset};
            RecordBuilds(pass.commands, {}, std::span(&build, 1), {scratch.buffer, scratch.offset});
        });
    graph.Add(
        "Trace", Kind::Compute,
        [&](PassBuilder& pass) {
            pass.Read(scene);
            pass.ReadAddress(rays);
            distances = pass.WriteAddress(distances, Use::StorageWrite);
        },
        [&](PassContext& pass) {
            trace.Dispatch(pass.commands, {1, 1, 1},
                           TraceParameters{pass.Get(scene).Address(), pass.Get(rays).Address(),
                                           pass.Get(distances).Address(), 2, 0});
        });
    EXPECT_NE(graph.Dot().find("shape=hexagon"), std::string::npos) << "structures drawn as such";
    const Ticket done = graph.Execute(context_->Main(), std::span(&uploaded, 1));
    Readback readback(*context_, context_->Main(), hits.Storage(), hits.Range().offset, 8,
                      std::span(&done, 1));
    std::array<float, 2> found{};
    std::memcpy(found.data(), readback.Bytes().data(), sizeof(found));
    EXPECT_NEAR(found[0], 5.0f, 1e-5f) << "through the triangle";
    EXPECT_EQ(found[1], -1.0f) << "beside it";
}

TEST_F(FrameGraph, KeepsStructuresAndBuffersToTheirOwnUses) {
    ERNIE_REQUIRE_RAY_QUERIES(*context_);
    AccelerationPool structures(*context_, "test structures");
    const AccelerationStructure top(structures, Sizes(*context_, 1, kRebuilt), "scene");
    const DeviceColumn values(*arenas_, "values", DType::UInt32, {}, kCount);
    Graph graph(*context_, *pool_);
    const graph::Acceleration scene = graph.Import(top);
    const graph::Buffer buffer = graph.Import(values, "values");
    EXPECT_THROW(graph.Add(
                     "Trace a buffer", Kind::Compute,
                     [&](PassBuilder& pass) { pass.Read(buffer, Use::Trace); },
                     [](PassContext&) {}),
                 std::invalid_argument);
    EXPECT_THROW(graph.Add(
                     "Sample a structure", Kind::Compute,
                     [&](PassBuilder& pass) { pass.Read(scene, Use::StorageRead); },
                     [](PassContext&) {}),
                 std::invalid_argument);
    EXPECT_EQ(graph.Import(top).id, scene.id) << "importing it again finds the same resource";
    // A pool stands for all its structures: builds write it and the top level's build reads it.
    graph::Acceleration bottoms = graph.Import(structures);
    EXPECT_EQ(graph.Import(structures).id, bottoms.id);
    EXPECT_NO_THROW(graph.Add(
        "Build bottom levels", Kind::Build,
        [&](PassBuilder& pass) { bottoms = pass.Build(bottoms); }, [](PassContext&) {}));
    EXPECT_NO_THROW(graph.Add(
        "Build the top level", Kind::Build,
        [&](PassBuilder& pass) { pass.Read(bottoms, Use::BuildInput); }, [](PassContext&) {}));
}

TEST_F(FrameGraph, ReportsReadsOfWhatNothingWrote) {
    const DeviceColumn values(*arenas_, "values", DType::UInt32, {}, kCount);
    Graph graph(*context_, *pool_);
    const Buffer never = graph.Create(kCount * 4, "never");
    Buffer result = graph.Import(values, "values");
    AddAffine(graph, "Read", never, result, 1, 0);
    EXPECT_THROW(graph.Compile(), GraphError);
}

TEST_F(FrameGraph, DebuggingModesGiveTheSameResult) {
    for (const bool flush : {false, true}) {
        const DeviceColumn values(*arenas_, "values", DType::UInt32, {}, kCount);
        Graph graph(*context_, *pool_);
        graph.SetFullBarriers(true);
        graph.SetFlushEachPass(flush);
        Buffer tripled = graph.Create(kCount * 4, "tripled");
        Buffer result = graph.Import(values, "values");
        AddAffine(graph, "Triple", {}, tripled, 3, 0);
        AddAffine(graph, "Add two", tripled, result, 1, 2);
        const std::vector<std::uint32_t> read =
            Read(*context_, values, graph.Execute(context_->Main()));
        EXPECT_EQ(read[10], 32u) << (flush ? "flushing each pass" : "in one submission");
    }
}

}  // namespace
}  // namespace ernie::graph
