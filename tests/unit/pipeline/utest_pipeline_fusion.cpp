// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

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

TEST(PipelineFusionTest, AFusedGraphMatchesItsNodesRunOnePassEach) {
    ernie::test::ForEachDevice([](device::Device& device) {
        NodeRegistry registry;
        RegisterBuiltinNodes(registry);
        const PipelineDesc desc = Pipeline(
            {Node(1, "CameraSignal"), Node(2, "Integrate", {{"electronRate", 1, "electronRate"}}),
             Node(3, "Emva1288", {{"electrons", 2, "electrons"}}),
             Node(4, "Adc", {{"electrons", 3, "electrons"}}),
             Node(5, "Gain", {{"digital", 4, "digital"}}),
             Node(6, "ToColor", {{"digital", 5, "digital"}}),
             Node(7, "Encode", {{"color", 6, "color"}})},
            {{"raw", 4, "digital"}, {"display", 7, "display"}});
        CameraRecord camera{};
        camera.exposureTime = 0.02f;
        camera.iso = 100;
        DetectorRecord detector{};
        detector.fullWell = 2000;
        detector.readNoise = 2;
        detector.darkCurrent = 50;
        detector.darkCurrentDoubling = 7;
        detector.temperature = 300;
        detector.prnu = 0.02f;
        detector.dsnu = 0.5f;
        detector.systemGain = 0.25f;
        detector.bitDepth = 10;
        detector.blackLevel = 16;
        Frame frame{.width = 32, .height = 8};
        for (uint32_t i = 0; i < frame.width * frame.height * frame.channels; ++i) {
            frame.rates.push_back(float(i % 97) * 900);
        }
        frame.parameters = {0, 0, 1.5f};
        const auto fused = RunPipeline(device, registry, desc, camera, detector, frame);
        frame.fusion = Fusion::PerNode;
        const auto split = RunPipeline(device, registry, desc, camera, detector, frame);
        EXPECT_EQ(fused.at("raw"), split.at("raw"));
        EXPECT_EQ(fused.at("display"), split.at("display"));
    });
}

}
}
