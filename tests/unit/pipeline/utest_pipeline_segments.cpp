// SPDX-License-Identifier: BSD-3-Clause
#include <vector>

#include <gtest/gtest.h>

#include "ernie/pipeline/definition.h"
#include "unit/pipeline/registry.h"

namespace ernie::pipeline {
namespace {

using test::FakeRegistry;
using test::Node;
using test::Pipeline;

const StoredValue* Find(const SegmentPlan& plan, uint32_t value) {
    for (const StoredValue& stored : plan.stored) {
        if (stored.value == value) {
            return &stored;
        }
    }
    return nullptr;
}

TEST(PipelineSegmentsTest, PointNodesFuseIntoOnePass) {
    const NodeRegistry registry = FakeRegistry();
    const ResolvedPipeline pipeline =
        ResolvePipeline(Pipeline({Node(1, "Signal"), Node(2, "Integrate", {{"rate", 1, "rate"}}),
                                  Node(3, "Noise", {{"electrons", 2, "electrons"}}),
                                  Node(4, "Adc", {{"electrons", 3, "electrons"}})},
                                 {{"raw", 4, "digital"}}),
                        registry);
    const PipelineDefinition definition = BuildDefinition(7, pipeline, registry);
    const SegmentPlan& plan = definition.segments;
    ASSERT_EQ(plan.segments.size(), 1u);
    EXPECT_EQ(plan.segments[0].nodes, (std::vector<uint32_t>{1, 2, 3}));
    ASSERT_EQ(plan.stored.size(), 1u);
    EXPECT_EQ(plan.stored[0].value, pipeline.nodes[0].outputs[0]);
    EXPECT_EQ(plan.stored[0].first, kBeforeSegments);
    EXPECT_EQ(definition.parameterOffsets, (std::vector<uint32_t>{0, 0, 1, 3}));
    EXPECT_EQ(definition.parameterWords, 6u);
}

TEST(PipelineSegmentsTest, StencilsAndReductionsSplitPassesAndShareTheArena) {
    const NodeRegistry registry = FakeRegistry();
    const ResolvedPipeline pipeline = ResolvePipeline(
        Pipeline({Node(1, "Signal"), Node(2, "Integrate", {{"rate", 1, "rate"}}),
                  Node(3, "Blur", {{"electrons", 2, "electrons"}}),
                  Node(4, "Adc", {{"electrons", 3, "electrons"}}),
                  Node(5, "Meter", {{"digital", 4, "digital"}}),
                  Node(6, "Expose", {{"digital", 4, "digital"}, {"stat", 5, "stat"}})},
                 {{"image", 6, "digital"}}),
        registry);
    const SegmentPlan plan = SplitSegments(pipeline, registry);
    ASSERT_EQ(plan.segments.size(), 4u);
    EXPECT_EQ(plan.segments[0].kind, NodeKind::Point);
    EXPECT_EQ(plan.segments[1].kind, NodeKind::Stencil);
    EXPECT_EQ(plan.segments[1].nodes, (std::vector<uint32_t>{2, 3}));
    EXPECT_EQ(plan.segments[2].kind, NodeKind::Reduce);
    EXPECT_EQ(plan.segments[3].nodes, (std::vector<uint32_t>{5}));

    const uint32_t electrons = pipeline.nodes[1].outputs[0];
    const uint32_t digital = pipeline.nodes[3].outputs[0];
    const uint32_t stat = pipeline.nodes[4].outputs[0];
    ASSERT_NE(Find(plan, electrons), nullptr);
    EXPECT_EQ(Find(plan, electrons)->last, 1u);
    ASSERT_NE(Find(plan, digital), nullptr);
    EXPECT_EQ(Find(plan, digital)->first, 1u);
    EXPECT_EQ(Find(plan, digital)->last, 3u);
    EXPECT_EQ(Find(plan, pipeline.nodes[2].outputs[0]), nullptr);
    EXPECT_EQ(Find(plan, pipeline.nodes[5].outputs[0]), nullptr);

    const std::vector<SignalLayout> layouts = PropagateLayouts(pipeline, registry, {.channels = 3});
    const ArenaLayout arena = LayoutArena(plan, layouts, 1000);
    const uint64_t slot = 1000 * 3 * 4 / kArenaAlignment * kArenaAlignment + kArenaAlignment;
    EXPECT_EQ(arena.offsets[stat], arena.offsets[electrons]);
    EXPECT_NE(arena.offsets[digital], arena.offsets[electrons]);
    EXPECT_NE(arena.offsets[pipeline.nodes[0].outputs[0]], arena.offsets[electrons]);
    EXPECT_EQ(arena.offsets[digital], arena.offsets[pipeline.nodes[0].outputs[0]]);
    EXPECT_EQ(arena.bytes, 2 * slot);
    EXPECT_EQ(arena.offsets[pipeline.nodes[5].outputs[0]], kNotStored);

    std::vector<SignalLayout> halves = layouts;
    for (SignalLayout& layout : halves) {
        layout.element = ElementType::F16;
    }
    EXPECT_EQ(LayoutArena(plan, halves, 1000).bytes, 2 * 8192u);
}

}
}
