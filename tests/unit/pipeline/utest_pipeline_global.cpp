// SPDX-License-Identifier: BSD-3-Clause
#include <bit>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/pipeline/definition.h"
#include "ernie/pipeline/node.h"
#include "support/devices.h"
#include "unit/pipeline/registry.h"
#include "unit/pipeline/runner.h"

namespace ernie::pipeline {
namespace {

using test::Frame;
using test::Node;
using test::Pipeline;
using test::RunPipeline;

NodeRegistry Builtins() {
    NodeRegistry registry;
    RegisterBuiltinNodes(registry);
    return registry;
}

DetectorRecord Detector(float fullWell) {
    DetectorRecord detector{};
    detector.channels = 1;
    detector.pattern = uint32_t(DetectorPattern::Mono);
    detector.fullWell = fullWell;
    detector.systemGain = 1;
    detector.bitDepth = 16;
    return detector;
}

CameraRecord Camera() {
    CameraRecord camera{};
    camera.exposureTime = 1;
    camera.iso = 100;
    return camera;
}

TEST(PipelineGlobalTest, GlobalNodesGetTheirOwnPassesAndScratch) {
    const NodeRegistry registry = Builtins();
    const PipelineDesc desc = Pipeline({Node(1, "CameraSignal"),
                                        Node(2, "Vignette", {{"electronRate", 1, "electronRate"}}),
                                        Node(3, "Glare", {{"electronRate", 2, "electronRate"}}),
                                        Node(4, "Vignette", {{"electronRate", 3, "electronRate"}})},
                                       {{"out", 4, "electronRate"}});
    const ResolvedPipeline pipeline = ResolvePipeline(desc, registry);
    ASSERT_EQ(pipeline.nodes[2].scratch.size(), 2u);
    const SegmentPlan plan = SplitSegments(pipeline, registry);
    ASSERT_EQ(plan.segments.size(), 3u);
    EXPECT_EQ(plan.segments[1].kind, NodeKind::Global);
    for (const uint32_t scratch : pipeline.nodes[2].scratch) {
        EXPECT_NE(std::ranges::find(plan.stored, scratch, &StoredValue::value), plan.stored.end());
    }
}

TEST(PipelineGlobalTest, GlareSpreadsFarAndKeepsTheTotal) {
    ernie::test::ForEachDevice([](device::Device& device) {
        const NodeRegistry registry = Builtins();
        constexpr uint32_t kSide = 64;
        const PipelineDesc desc = Pipeline(
            {Node(1, "CameraSignal"), Node(2, "Glare", {{"electronRate", 1, "electronRate"}})},
            {{"glared", 2, "electronRate"}});
        Frame frame{.width = kSide, .height = kSide, .channels = 1, .parameters = {8, 0.5f}};
        frame.rates.assign(kSide * kSide, 0);
        frame.rates[32 * kSide + 32] = 1000;
        const std::vector<float> glared =
            RunPipeline(device, registry, desc, Camera(), Detector(1e9f), frame).at("glared");
        EXPECT_NEAR(std::accumulate(glared.begin(), glared.end(), 0.0), 1000, 0.5);
        EXPECT_GT(glared[32 * kSide + 32], 500);
        EXPECT_LT(glared[32 * kSide + 32], 510);
        EXPECT_GT(glared[32 * kSide + 52], 0);
        EXPECT_NEAR(glared[32 * kSide + 40], glared[40 * kSide + 32], 1e-4f);

        frame.parameters = {8, 0};
        EXPECT_EQ(RunPipeline(device, registry, desc, Camera(), Detector(1e9f), frame).at("glared"),
                  frame.rates);
    });
}

TEST(PipelineGlobalTest, BloomingSpillsAlongColumnsAndKeepsTheCharge) {
    ernie::test::ForEachDevice([](device::Device& device) {
        const NodeRegistry registry = Builtins();
        constexpr uint32_t kWidth = 3;
        constexpr uint32_t kHeight = 32;
        constexpr float kWell = 100;
        const PipelineDesc desc = Pipeline(
            {Node(1, "CameraSignal"), Node(2, "Integrate", {{"electronRate", 1, "electronRate"}}),
             Node(3, "Blooming", {{"electrons", 2, "electrons"}})},
            {{"bloomed", 3, "electrons"}});
        Frame frame{.width = kWidth,
                    .height = kHeight,
                    .channels = 1,
                    .parameters = {0, std::bit_cast<float>(int32_t(32))}};
        frame.rates.assign(kWidth * kHeight, 50);
        frame.rates[16 * kWidth + 1] = 5 * kWell;
        const std::vector<float> bloomed =
            RunPipeline(device, registry, desc, Camera(), Detector(kWell), frame).at("bloomed");
        EXPECT_FLOAT_EQ(std::accumulate(bloomed.begin(), bloomed.end(), 0.0f),
                        std::accumulate(frame.rates.begin(), frame.rates.end(), 0.0f));
        for (uint32_t y = 0; y < kHeight; ++y) {
            EXPECT_FLOAT_EQ(bloomed[y * kWidth], 50) << y;
            EXPECT_FLOAT_EQ(bloomed[y * kWidth + 2], 50) << y;
            EXPECT_FLOAT_EQ(bloomed[y * kWidth + 1], y >= 12 && y <= 20 ? kWell : 50) << y;
        }

        frame.parameters = {0, std::bit_cast<float>(int32_t(2))};
        const std::vector<float> near =
            RunPipeline(device, registry, desc, Camera(), Detector(kWell), frame).at("bloomed");
        EXPECT_FLOAT_EQ(near[13 * kWidth + 1], 50);
        EXPECT_FLOAT_EQ(near[14 * kWidth + 1], kWell);
    });
}

}
}
