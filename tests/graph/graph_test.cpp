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
        [image](PassContext& pass) {
            const VkClearColorValue color{.float32 = {0.25f, 0.5f, 0.75f, 1.0f}};
            const VkImageSubresourceRange all{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};
            vkCmdClearColorImage(pass.commands, pass.Get(image).Handle(), pass.Layout(image),
                                 &color, 1, &all);
        });
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
            vkCmdCopyImageToBuffer(pass.commands, pass.Get(image).Handle(), pass.Layout(image),
                                   slice.buffer->Handle(), 1, &region);
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
            [image](PassContext& pass) {
                const VkClearColorValue color{};
                const VkImageSubresourceRange all{
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};
                vkCmdClearColorImage(pass.commands, pass.Get(image).Handle(), pass.Layout(image),
                                     &color, 1, &all);
            });
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
                vkCmdCopyImageToBuffer(pass.commands, pass.Get(image).Handle(), pass.Layout(image),
                                       slice.buffer->Handle(), 1, &region);
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
