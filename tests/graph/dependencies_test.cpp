// SPDX-License-Identifier: BSD-3-Clause
/// @file dependencies_test.cpp
/// @brief Tests of the frame graph's dependency analysis and blackboard, without a device.
#include "ernie/graph/blackboard.h"
#include "ernie/graph/dependencies.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace ernie::graph {
namespace {

constexpr VkPipelineStageFlags2 kCompute = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
constexpr VkPipelineStageFlags2 kRasterShaders =
    VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
constexpr VkPipelineStageFlags2 kAll = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
constexpr VkPipelineStageFlags2 kBuild = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;

DependencyResource ImageNamed(const char* name, Origin origin = Origin::Fresh) {
    return {name, true, origin};
}

DependencyResource BufferNamed(const char* name, Origin origin = Origin::Fresh) {
    return {name, false, origin};
}

PassDependencies Analyze(const std::vector<DependencyResource>& resources,
                         const std::vector<DependencyPass>& passes) {
    PassDependencies dependencies = ComputeDependencies(resources, passes);
    EXPECT_EQ(dependencies.passes.size(), passes.size());
    return dependencies;
}

TEST(Dependencies, AWriteBecomesVisibleToItsReaders) {
    const PassDependencies d =
        Analyze({ImageNamed("radiance")}, {{"Shade", Kind::Compute, {{0, Use::StorageWrite}}},
                                           {"Present", Kind::Raster, {{0, Use::Sampled}}}});
    ASSERT_TRUE(d.errors.empty());
    EXPECT_TRUE(d.passes[0].Empty()) << "a fresh image is in the general layout already";
    ASSERT_EQ(d.passes[1].images.size(), 1u);
    const Barrier& read = d.passes[1].images[0];
    EXPECT_EQ(read.srcStages, kCompute);
    EXPECT_EQ(read.srcAccess, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    EXPECT_EQ(read.dstStages, kRasterShaders);
    EXPECT_EQ(read.dstAccess, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    EXPECT_EQ(read.oldLayout, VK_IMAGE_LAYOUT_GENERAL) << "sampled in the general layout";
    EXPECT_EQ(read.newLayout, VK_IMAGE_LAYOUT_GENERAL);
    EXPECT_TRUE(d.after.Empty()) << "transients need nothing after the graph";
}

TEST(Dependencies, AWriteWaitsForEarlierReaders) {
    const PassDependencies d =
        Analyze({BufferNamed("rows")}, {{"Fill", Kind::Compute, {{0, Use::StorageWrite}}},
                                        {"Draw", Kind::Raster, {{0, Use::StorageRead}}},
                                        {"Refill", Kind::Compute, {{0, Use::StorageWrite}}}});
    ASSERT_TRUE(d.errors.empty());
    EXPECT_TRUE(d.passes[0].Empty());
    ASSERT_EQ(d.passes[2].buffers.size(), 1u);
    const Barrier& write = d.passes[2].buffers[0];
    EXPECT_EQ(write.srcStages, kCompute | kRasterShaders);
    EXPECT_EQ(write.dstStages, kCompute);
    EXPECT_EQ(write.dstAccess, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
}

TEST(Dependencies, ReadsAfterReadsMerge) {
    const PassDependencies d =
        Analyze({BufferNamed("sums")},
                {{"Sum", Kind::Compute, {{0, Use::StorageWrite}}},
                 {"Mean", Kind::Compute, {{0, Use::StorageRead}, {0, Use::StorageRead}}},
                 {"Spread", Kind::Compute, {{0, Use::StorageRead}}},
                 {"Draw", Kind::Raster, {{0, Use::StorageRead}}}});
    ASSERT_TRUE(d.errors.empty());
    EXPECT_EQ(d.passes[1].buffers.size(), 1u) << "two reads in one pass share a barrier";
    EXPECT_TRUE(d.passes[2].Empty()) << "the same read again is visible already";
    ASSERT_EQ(d.passes[3].buffers.size(), 1u) << "other stages still wait for the write";
    EXPECT_EQ(d.passes[3].buffers[0].srcStages, kCompute);
    EXPECT_EQ(d.passes[3].buffers[0].dstStages, kRasterShaders);
}

TEST(Dependencies, AttachmentsTakeTheirLayoutsAndComeBackToGeneral) {
    const PassDependencies d =
        Analyze({ImageNamed("color"), ImageNamed("depth")},
                {{"Draw", Kind::Raster, {{0, Use::ColorWrite}, {1, Use::DepthWrite}}},
                 {"Overlay", Kind::Raster, {{0, Use::ColorReadWrite}, {1, Use::DepthRead}}},
                 {"Filter", Kind::Compute, {{0, Use::Sampled}, {1, Use::Sampled}}},
                 {"Copy", Kind::Copy, {{0, Use::CopySrc}}}});
    ASSERT_TRUE(d.errors.empty());
    ASSERT_EQ(d.passes[0].images.size(), 2u);
    EXPECT_EQ(d.passes[0].images[0].newLayout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    EXPECT_EQ(d.passes[0].images[1].newLayout, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    ASSERT_EQ(d.passes[1].images.size(), 2u);
    const Barrier& color = d.passes[1].images[0];
    EXPECT_EQ(color.oldLayout, color.newLayout) << "colour stays put";
    EXPECT_EQ(color.srcStages, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
    const Barrier& depth = d.passes[1].images[1];
    EXPECT_EQ(depth.newLayout, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
    EXPECT_EQ(depth.srcAccess, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    const Barrier& sampled = d.passes[2].images[1];
    EXPECT_EQ(sampled.oldLayout, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
    EXPECT_EQ(sampled.newLayout, VK_IMAGE_LAYOUT_GENERAL) << "back to general to be sampled";
    EXPECT_EQ(sampled.dstStages, kCompute);

    ASSERT_EQ(d.passes[3].images.size(), 1u);
    const Barrier& copy = d.passes[3].images[0];
    EXPECT_EQ(copy.oldLayout, VK_IMAGE_LAYOUT_GENERAL);
    EXPECT_EQ(copy.newLayout, VK_IMAGE_LAYOUT_GENERAL);
    EXPECT_EQ(copy.srcStages, kCompute) << "the transition ordered the attachment writes already";
    EXPECT_EQ(copy.dstStages, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);
}

TEST(Dependencies, LayerRangesNarrowMemoryBarriers) {
    const PassDependencies d =
        Analyze({ImageNamed("faces")},
                {{"Faces", Kind::Compute, {{0, Use::StorageWrite}}},
                 {"Edges",
                  Kind::Compute,
                  {{0, Use::StorageReadWrite, {2, 1}}, {0, Use::StorageRead, {4, 2}}}}});
    ASSERT_TRUE(d.errors.empty());
    ASSERT_EQ(d.passes[1].images.size(), 1u);
    const Barrier& barrier = d.passes[1].images[0];
    EXPECT_EQ(barrier.oldLayout, barrier.newLayout);
    EXPECT_EQ(barrier.layers.first, 2u);
    EXPECT_EQ(barrier.layers.count, 4u);
}

TEST(Dependencies, IndirectAndUniformBuffers) {
    const PassDependencies d =
        Analyze({BufferNamed("args"), BufferNamed("view", Origin::Imported)},
                {{"Cull", Kind::Compute, {{0, Use::StorageWrite}, {1, Use::Uniform}}},
                 {"Draw", Kind::Raster, {{0, Use::Indirect}, {1, Use::Uniform}}}});
    ASSERT_TRUE(d.errors.empty());
    ASSERT_EQ(d.passes[0].buffers.size(), 1u);
    EXPECT_EQ(d.passes[0].buffers[0].resource, 1u);
    ASSERT_EQ(d.passes[1].buffers.size(), 2u);
    const Barrier& args = d.passes[1].buffers[0];
    EXPECT_EQ(args.dstStages, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
    EXPECT_EQ(args.dstAccess, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    const Barrier& view = d.passes[1].buffers[1];
    EXPECT_EQ(view.srcStages, kAll) << "graphics waits for the import's earlier writes too";
    EXPECT_EQ(view.dstStages, kRasterShaders);
}

TEST(Dependencies, ImportsWaitForEverythingBeforeAndShowTheirWritesAfter) {
    const PassDependencies d =
        Analyze({ImageNamed("lut", Origin::Imported), ImageNamed("history", Origin::Imported),
                 BufferNamed("readback", Origin::Imported)},
                {{"Blend",
                  Kind::Compute,
                  {{0, Use::Sampled}, {1, Use::StorageReadWrite}, {2, Use::StorageWrite}}},
                 {"Copy", Kind::Copy, {{1, Use::CopySrc}}}});
    ASSERT_TRUE(d.errors.empty());
    ASSERT_EQ(d.passes[0].images.size(), 2u);
    for (const Barrier& barrier : d.passes[0].images) {
        EXPECT_EQ(barrier.srcStages, kAll);
        EXPECT_EQ(barrier.srcAccess, VK_ACCESS_2_MEMORY_WRITE_BIT);
        EXPECT_EQ(barrier.oldLayout, barrier.newLayout);
    }
    ASSERT_EQ(d.passes[0].buffers.size(), 1u);
    EXPECT_EQ(d.passes[0].buffers[0].srcStages, kAll);

    // The LUT was only read; history's and the readback's writes become visible after.
    ASSERT_EQ(d.after.images.size(), 1u);
    EXPECT_EQ(d.after.images[0].resource, 1u);
    EXPECT_EQ(d.after.images[0].newLayout, VK_IMAGE_LAYOUT_GENERAL);
    ASSERT_EQ(d.after.buffers.size(), 1u);
    EXPECT_EQ(d.after.buffers[0].srcStages, kCompute);
    EXPECT_NE(d.after.buffers[0].dstAccess & VK_ACCESS_2_HOST_READ_BIT, 0u) << "for readbacks";
}

TEST(Dependencies, RecycledTransientsWaitForEarlierGraphs) {
    const PassDependencies d = Analyze({BufferNamed("scratch", Origin::Recycled)},
                                       {{"Clear", Kind::Copy, {{0, Use::CopyDst}}}});
    ASSERT_TRUE(d.errors.empty());
    ASSERT_EQ(d.passes[0].buffers.size(), 1u);
    EXPECT_EQ(d.passes[0].buffers[0].srcStages, kAll);
    EXPECT_EQ(d.passes[0].buffers[0].srcAccess, VK_ACCESS_2_NONE);
}

TEST(Dependencies, BuildsOrderAccelerationStructuresBeforeTheirTraces) {
    const PassDependencies d =
        Analyze({BufferNamed("vertices", Origin::Imported), BufferNamed("scratch"),
                 BufferNamed("blas"), BufferNamed("tlas")},
                {{"Build bottom",
                  Kind::Build,
                  {{0, Use::BuildInput}, {1, Use::BuildScratch}, {2, Use::BuildOutput}}},
                 {"Build top",
                  Kind::Build,
                  {{2, Use::BuildInput}, {1, Use::BuildScratch}, {3, Use::BuildOutput}}},
                 {"Trace", Kind::Compute, {{3, Use::Trace}, {2, Use::Trace}}}});
    ASSERT_TRUE(d.errors.empty());
    // The top build reads the bottom one's structure and reuses its scratch.
    ASSERT_EQ(d.passes[1].buffers.size(), 2u);
    const Barrier& blas = d.passes[1].buffers[0];
    EXPECT_EQ(blas.srcStages, kBuild);
    EXPECT_EQ(blas.srcAccess, VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
    EXPECT_EQ(blas.dstStages, kBuild);
    // Traces wait for both builds.
    ASSERT_EQ(d.passes[2].buffers.size(), 2u);
    for (const Barrier& trace : d.passes[2].buffers) {
        EXPECT_EQ(trace.srcStages, kBuild);
        EXPECT_EQ(trace.dstStages, kCompute);
        EXPECT_EQ(trace.dstAccess, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    }
}

TEST(Dependencies, ReportsReadsOfUnwrittenTransients) {
    const PassDependencies d =
        Analyze({ImageNamed("ids"), BufferNamed("sums")},
                {{"Shade", Kind::Compute, {{0, Use::Sampled}, {1, Use::StorageReadWrite}}}});
    ASSERT_EQ(d.errors.size(), 2u);
    EXPECT_NE(d.errors[0].find("'Shade' reads 'ids'"), std::string::npos);
    EXPECT_NE(d.errors[1].find("'sums'"), std::string::npos);
}

TEST(Dependencies, ReportsUsesAResourceCannotHave) {
    const PassDependencies d =
        Analyze({ImageNamed("color"), BufferNamed("rows")},
                {{"Dispatch", Kind::Compute, {{0, Use::ColorWrite}}},
                 {"Sample", Kind::Compute, {{1, Use::Sampled}}},
                 {"Copy", Kind::Copy, {{1, Use::StorageWrite}}},
                 {"Mixed", Kind::Raster, {{0, Use::ColorWrite}, {0, Use::StorageWrite}}},
                 {"Build", Kind::Compute, {{1, Use::BuildOutput}}}});
    ASSERT_EQ(d.errors.size(), 5u);
    EXPECT_NE(d.errors[0].find("as ColorWrite"), std::string::npos);
    EXPECT_NE(d.errors[1].find("buffer 'rows' as Sampled"), std::string::npos);
    EXPECT_NE(d.errors[2].find("'Copy'"), std::string::npos);
    EXPECT_NE(d.errors[3].find("conflicting layouts"), std::string::npos);
    EXPECT_NE(d.errors[4].find("as BuildOutput"), std::string::npos) << "builds are Build passes";
}

TEST(Blackboard, FindsValuesByType) {
    struct Hits {
        int ids = 0;
        int depth = 0;
    };
    struct Sky {
        int luminance = 0;
    };
    Blackboard board;
    EXPECT_EQ(board.Find<Hits>(), nullptr);
    EXPECT_THROW(board.Get<Hits>(), std::out_of_range);
    board.Put(Hits{1, 2});
    board.Put(Sky{3});
    EXPECT_EQ(board.Get<Hits>().depth, 2);
    EXPECT_EQ(board.Get<Sky>().luminance, 3);
    board.Put(Hits{4, 5});
    EXPECT_EQ(board.Get<Hits>().ids, 4) << "a later pass replaces the value of its type";
    board.Clear();
    EXPECT_FALSE(board.Has<Sky>());
}

}  // namespace
}  // namespace ernie::graph
