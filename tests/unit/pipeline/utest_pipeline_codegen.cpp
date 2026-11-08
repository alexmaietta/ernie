// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/pipeline/codegen.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/device/program.h"
#include "ernie/device/queue.h"
#include "ernie/render/kernels.h"
#include "ernie/render/kernels/pipeline/pipeline.h"
#include "support/buffers.h"
#include "support/devices.h"
#include "unit/pipeline/registry.h"

namespace ernie::pipeline {
namespace {

using test::Node;
using test::Pipeline;

const char* const kTestNodes = R"(
import ernie.pipeline;

using namespace ernie;

struct TestScale {
    static void Evaluate(PointContext context, ElectronRate rate, out Electrons electrons) {
        Channels channels = rate.Get();
        [ForceUnroll]
        for (uint i = 0; i < kMostSignalChannels; ++i) {
            channels[i] *= context.Parameter(0);
        }
        electrons = Electrons.Make(channels);
    }
}

struct TestOffset<let sign : uint> {
    static void Evaluate(PointContext context, Electrons electrons, out Digital digital) {
        Channels channels = electrons.Get();
        const float offset = context.Parameter(0) * (sign == 1 ? -1.0f : 1.0f);
        [ForceUnroll]
        for (uint i = 0; i < kMostSignalChannels; ++i) {
            channels[i] += offset + float(context.element.coord.x);
        }
        digital = Digital.Make(channels);
    }
}
)";

float HalfBits(uint16_t bits) {
    const int exponent = (bits >> 10) & 0x1f;
    const float mantissa = float(bits & 0x3ff);
    const float magnitude =
        exponent == 0 ? mantissa * 5.9604645e-8f : std::ldexp(1 + mantissa / 1024, exponent - 15);
    return (bits & 0x8000) != 0 ? -magnitude : magnitude;
}

NodeRegistry CodegenRegistry() {
    NodeRegistry registry;
    registry.Add({.id = "Signal",
                  .kind = NodeKind::Source,
                  .outputs = {{"rate", SignalType::ElectronRate}},
                  .layout = LayoutRule::Detector});
    registry.Add({.id = "Scale",
                  .inputs = {{"rate", SignalType::ElectronRate}},
                  .outputs = {{"electrons", SignalType::Electrons}},
                  .parameters = {{"factor"}},
                  .module = "pipeline_test.nodes",
                  .type = "TestScale"});
    registry.Add({.id = "Offset",
                  .inputs = {{"electrons", SignalType::Electrons}},
                  .outputs = {{"digital", SignalType::Digital}},
                  .parameters = {{"offset"}},
                  .uniforms = {{"sign", {"plus", "minus"}, "plus"}},
                  .module = "pipeline_test.nodes",
                  .type = "TestOffset"});
    return registry;
}

TEST(PipelineCodegenTest, AGeneratedDefinitionCompilesAndRuns) {
    ernie::test::ForEachDevice([](device::Device& device) {
        device::RegisterFile("pipeline_test/nodes.slang", kTestNodes);
        const NodeRegistry registry = CodegenRegistry();
        const ResolvedPipeline pipeline = ResolvePipeline(
            Pipeline({Node(1, "Signal"), Node(2, "Scale", {{"rate", 1, "rate"}}),
                      Node(3, "Offset", {{"electrons", 2, "electrons"}}, {{"sign", "minus"}})},
                     {{"raw", 3, "digital"}, {"scaled", 2, "electrons"}}),
            registry);
        const PipelineDefinition definition = BuildDefinition(4, pipeline, registry);
        const GeneratedPipeline generated = GeneratePipeline(definition, registry);
        ASSERT_EQ(generated.entries, (std::vector<std::string>{"Segment0"}));
        ASSERT_EQ(generated.slots.size(), 3u);

        const Ref<device::Program> program =
            device.CreateProgram({.modules = {generated.module},
                                  .library = &kRenderKernels,
                                  .sources = {{generated.path, generated.source}},
                                  .entries = generated.entries,
                                  .label = "generated pipeline"});
        device::Kernel& kernel = program->Entry("Segment0");
        kernel.Wait();

        constexpr uint32_t kWidth = 4;
        constexpr uint32_t kHeight = 2;
        constexpr uint32_t kChannels = 3;
        constexpr uint32_t kElements = kWidth * kHeight;
        std::vector<float> rates(kElements * kChannels);
        for (uint32_t i = 0; i < rates.size(); ++i) {
            rates[i] = float(i);
        }
        const Ref<device::Buffer> input = device.CreateBuffer({.label = "rates"}, rates);
        const Ref<device::Buffer> raw =
            device.CreateBuffer({.label = "raw"}, std::vector<float>(rates.size()));
        const Ref<device::Buffer> scaled = device.CreateBuffer(
            {.label = "scaled"}, std::vector<uint16_t>(kElements * (kChannels + 1)));
        const std::vector<uint32_t> words = {std::bit_cast<uint32_t>(2.0f),
                                             std::bit_cast<uint32_t>(0.5f)};
        const Ref<device::Buffer> wordBuffer = device.CreateBuffer({.label = "words"}, words);
        std::vector<PipelineValue> values(3);
        for (uint32_t slot = 0; slot < 3; ++slot) {
            const uint32_t value = generated.slots[slot];
            const device::Buffer& buffer = value == pipeline.nodes[0].outputs[0]   ? *input
                                           : value == pipeline.nodes[2].outputs[0] ? *raw
                                                                                   : *scaled;
            values[slot] = {buffer.Address(), kChannels, &buffer == scaled.Get() ? kSignalHalf : 0};
        }
        const Ref<device::Buffer> valueBuffer = device.CreateBuffer({.label = "values"}, values);
        const std::vector<PipelineCapture> captures = {{.firstElement = 0,
                                                        .words = {wordBuffer->Address()},
                                                        .width = kWidth,
                                                        .height = kHeight,
                                                        .firstValue = 0}};
        const Ref<device::Buffer> captureBuffer =
            device.CreateBuffer({.label = "captures"}, captures);
        const PipelineParams params{.captures = {captureBuffer->Address()},
                                    .values = {valueBuffer->Address()},
                                    .elementCount = kElements,
                                    .captureCount = 1};
        device::CommandEncoder encoder = device.GetQueue().CreateEncoder();
        encoder.Dispatch(kernel, params, {.x = device::GroupCount(kElements, kPipelineGroup)});
        device.GetQueue().Submit(encoder.Finish());

        const std::vector<float> rawOut = ernie::test::Download<float>(device, *raw, rates.size());
        const std::vector<uint16_t> scaledOut =
            ernie::test::Download<uint16_t>(device, *scaled, kElements * (kChannels + 1));
        for (uint32_t i = 0; i < rates.size(); ++i) {
            const float x = float(i / kChannels % kWidth);
            const uint32_t element = i / kChannels;
            EXPECT_EQ(HalfBits(scaledOut[element * (kChannels + 1) + i % kChannels]), rates[i] * 2)
                << i;
            EXPECT_FLOAT_EQ(rawOut[i], rates[i] * 2 - 0.5f + x) << i;
        }
    });
}

}
}
