// SPDX-License-Identifier: BSD-3-Clause
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/util/error.h"
#include "unit/sensor_pipeline/registry.h"

namespace ernie::sensor_pipeline {
namespace {

using test::FakeRegistry;
using test::Node;
using test::Pipeline;

SensorPipelineDesc RawChain(uint32_t first = 10, std::string shot = "poisson") {
    return Pipeline(
        {Node(first + 3, "Adc", {{"electrons", first + 2, "electrons"}}),
         Node(first + 2, "Noise", {{"electrons", first + 1, "electrons"}}, {{"shot", shot}}),
         Node(first + 1, "Integrate", {{"rate", first, "rate"}}), Node(first, "Signal"),
         Node(first + 9, "Integrate", {{"rate", first, "rate"}})},
        {{"raw", first + 3, "digital"}});
}

std::string Message(const SensorPipelineDesc& desc) {
    const NodeRegistry registry = FakeRegistry();
    try {
        ResolveSensorPipeline(desc, registry);
    } catch (const Error& error) {
        return error.what();
    }
    return "";
}

TEST(SensorPipelineResolveTest, OrdersNodesPrunesUnusedOnesAndMapsOutputs) {
    const NodeRegistry registry = FakeRegistry();
    const ResolvedSensorPipeline pipeline = ResolveSensorPipeline(RawChain(), registry);
    ASSERT_EQ(pipeline.nodes.size(), 4u);
    std::vector<std::string> ids;
    for (const ResolvedNode& node : pipeline.nodes) {
        ids.push_back(registry[node.def].id);
    }
    EXPECT_EQ(ids, (std::vector<std::string>{"Signal", "Integrate", "Noise", "Adc"}));
    EXPECT_EQ(pipeline.nodes[0].uniforms, (std::vector<uint32_t>{0, 0}));
    EXPECT_EQ(pipeline.nodes[2].uniforms, (std::vector<uint32_t>{0}));
    EXPECT_EQ(pipeline.nodes[1].inputs[0], pipeline.nodes[0].outputs[0]);
    ASSERT_EQ(pipeline.outputs.size(), 1u);
    EXPECT_EQ(pipeline.outputs[0].name, "raw");
    EXPECT_EQ(pipeline.outputs[0].value, pipeline.nodes[3].outputs[0]);
    EXPECT_EQ(pipeline.values[pipeline.outputs[0].value].type, SignalType::Digital);
}

TEST(SensorPipelineResolveTest, KeysFollowShapeAndUniformsButNotPrims) {
    const NodeRegistry registry = FakeRegistry();
    const graph::ShapeKey key = ResolveSensorPipeline(RawChain(), registry).key;
    EXPECT_EQ(ResolveSensorPipeline(RawChain(50), registry).key, key);
    EXPECT_NE(ResolveSensorPipeline(RawChain(10, "gaussian"), registry).key, key);
    SensorPipelineDesc renamed = RawChain();
    renamed.outputs[0].input = "digital";
    EXPECT_NE(ResolveSensorPipeline(renamed, registry).key, key);
}

TEST(SensorPipelineResolveTest, ErrorsNameThePrimsInvolved) {
    EXPECT_EQ(Message(Pipeline({Node(2, "Adc", {{"electrons", 1, "rate"}}), Node(1, "Signal")},
                               {{"raw", 2, "digital"}})),
              "/Pipe/Adc2's electrons takes Electrons, but /Pipe/node1's rate is ElectronRate");
    EXPECT_EQ(Message(Pipeline({Node(2, "Mystery")}, {{"raw", 2, "out"}})),
              "/Pipe/Mystery2 runs Mystery, which no pipeline node is");
    EXPECT_EQ(Message(Pipeline({Node(2, "Adc")}, {{"raw", 2, "digital"}})),
              "/Pipe/Adc2's electrons isn't connected, and Adc needs it");
    EXPECT_EQ(Message(RawChain(10, "loud")),
              "/Pipe/Noise12's shot is loud, and it takes poisson, gaussian, off");
    EXPECT_EQ(Message(Pipeline({Node(2, "Adc", {{"electrons", 7, "electrons"}})},
                               {{"raw", 2, "digital"}})),
              "/Pipe/Adc2's electrons connects to /Pipe/node7, which isn't a node of /Pipe");
    EXPECT_EQ(Message(Pipeline({Node(2, "Noise", {{"electrons", 3, "electrons"}}),
                                Node(3, "Noise", {{"electrons", 2, "electrons"}})},
                               {{"raw", 2, "electrons"}})),
              "/Pipe loops through /Pipe/Noise2, /Pipe/Noise3");
    EXPECT_EQ(Message(Pipeline({Node(1, "Signal")}, {})), "/Pipe has no outputs");
}

TEST(SensorPipelineResolveTest, LayoutsFollowTheDetectorAndEachNodesRule) {
    const NodeRegistry registry = FakeRegistry();
    SensorPipelineDesc desc =
        Pipeline({Node(1, "Signal", {}, {{"layout", "mosaic"}, {"precision", "half"}}),
                  Node(2, "Integrate", {{"rate", 1, "rate"}}),
                  Node(3, "Adc", {{"electrons", 2, "electrons"}}),
                  Node(4, "ToColor", {{"digital", 3, "digital"}})},
                 {{"color", 4, "color"}});
    const ResolvedSensorPipeline pipeline = ResolveSensorPipeline(desc, registry);
    const std::vector<SignalLayout> mosaic =
        PropagateLayouts(pipeline, registry, {.channels = 3, .pattern = 2, .mosaics = true});
    EXPECT_EQ(mosaic[0], (SignalLayout{1, 2, true, ElementType::F16}));
    EXPECT_EQ(mosaic[2], mosaic[0]);
    EXPECT_EQ(mosaic[3], (SignalLayout{.channels = 3}));
    const std::vector<SignalLayout> full =
        PropagateLayouts(pipeline, registry, {.channels = 3, .pattern = 1, .mosaics = false});
    EXPECT_EQ(full[0], (SignalLayout{3, 1, false, ElementType::F16}));
}

}
}
