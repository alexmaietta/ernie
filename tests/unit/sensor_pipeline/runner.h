// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <bit>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ernie/device/device.h"
#include "ernie/device/program.h"
#include "ernie/device/queue.h"
#include "ernie/render/kernels.h"
#include "ernie/render/kernels/sensor_pipeline/pipeline.h"
#include "ernie/sensor_pipeline/codegen.h"
#include "support/buffers.h"

namespace ernie::sensor_pipeline::test {

struct Frame {
    uint32_t width = 4;
    uint32_t height = 1;
    uint32_t channels = 3;
    std::vector<float> rates;
    std::vector<float> parameters;
    uint32_t capture = 0;
    uint32_t identity = 7;
    uint32_t captures = 1;
    Fusion fusion = Fusion::Fused;
    uint32_t pattern = 0;
    bool mosaic = false;
};

inline std::map<std::string, std::vector<float>>
RunPipeline(device::Device& device, const NodeRegistry& registry, const SensorPipelineDesc& desc,
            const CameraRecord& camera, const DetectorRecord& detector, const Frame& frame) {
    const ResolvedSensorPipeline pipeline = ResolveSensorPipeline(desc, registry);
    const uint32_t id = uint32_t(pipeline.key.Hash() % 100000) * 2 + uint32_t(frame.fusion);
    const SensorPipelineDefinition definition =
        BuildDefinition(id, pipeline, registry, frame.fusion);
    const GeneratedSensorPipeline generated = GenerateSensorPipeline(definition, registry);
    const Ref<device::Program> program =
        device.CreateProgram({.modules = {generated.module},
                              .library = &kRenderKernels,
                              .sources = {{generated.path, generated.source}},
                              .entries = generated.Entries(),
                              .label = "test pipeline"});
    const std::vector<SignalLayout> layouts = PropagateLayouts(
        pipeline, registry,
        {.channels = frame.channels, .pattern = frame.pattern, .mosaics = frame.mosaic});
    const uint32_t perCapture = frame.width * frame.height;
    const uint32_t elements = perCapture * frame.captures;
    std::vector<float> rates;
    for (uint32_t capture = 0; capture < frame.captures; ++capture) {
        rates.insert(rates.end(), frame.rates.begin(), frame.rates.end());
    }

    std::vector<Ref<device::Buffer>> held;
    std::vector<SensorPipelineValue> values;
    std::map<std::string, device::Buffer*> outputs;
    const auto storedCount = uint32_t(definition.segments.stored.size());
    for (uint32_t slot = 0; slot < generated.slots.size(); ++slot) {
        const uint32_t value = generated.slots[slot];
        const uint32_t channels = layouts[value].channels;
        const bool source =
            registry[pipeline.nodes[pipeline.values[value].node].def].kind == NodeKind::Source;
        const Ref<device::Buffer> buffer =
            source ? device.CreateBuffer({.label = "rates"}, rates)
                   : device.CreateBuffer(
                         {.label = "value"},
                         std::vector<float>(
                             size_t(layouts[value].perCapture ? frame.captures : elements) *
                             channels));
        if (slot >= storedCount) {
            outputs[pipeline.outputs[slot - storedCount].name] = buffer.Get();
        }
        values.push_back({buffer->Address(), channels, layouts[value].mosaic ? kSignalMosaic : 0u});
        held.push_back(buffer);
    }
    std::vector<SensorPipelineValue> slots;
    std::vector<SensorPipelineCapture> captures;
    std::vector<uint32_t> words;
    for (const float parameter : frame.parameters) {
        words.push_back(std::bit_cast<uint32_t>(parameter));
    }
    words.push_back(0);
    const Ref<device::Buffer> wordBuffer = device.CreateBuffer({.label = "words"}, words);
    for (uint32_t capture = 0; capture < frame.captures; ++capture) {
        for (uint32_t slot = 0; slot < values.size(); ++slot) {
            SensorPipelineValue value = values[slot];
            const bool stat = layouts[generated.slots[slot]].perCapture;
            value.address +=
                uint64_t(capture) * (stat ? 1 : perCapture) * value.channels * sizeof(float);
            slots.push_back(value);
        }
        captures.push_back({.firstElement = uint64_t(capture) * perCapture,
                            .words = {wordBuffer->Address()},
                            .width = frame.width,
                            .height = frame.height,
                            .capture = frame.capture + capture,
                            .identity = frame.identity,
                            .firstValue = uint32_t(capture * values.size())});
    }
    const Ref<device::Buffer> cameras =
        device.CreateBuffer({.label = "cameras"}, std::vector<CameraRecord>{camera});
    const Ref<device::Buffer> detectors =
        device.CreateBuffer({.label = "detectors"}, std::vector<DetectorRecord>{detector});
    RenderTables tables{};
    tables.cameras.address = cameras->Address();
    tables.detectors.address = detectors->Address();
    const Ref<device::Buffer> tableBuffer =
        device.CreateBuffer({.label = "tables"}, std::vector<RenderTables>{tables});
    const Ref<device::Buffer> valueBuffer = device.CreateBuffer({.label = "values"}, slots);
    const Ref<device::Buffer> captureBuffer = device.CreateBuffer({.label = "captures"}, captures);
    const SensorPipelineParams params{.tables = {tableBuffer->Address()},
                                      .captures = {captureBuffer->Address()},
                                      .values = {valueBuffer->Address()},
                                      .elementCount = elements,
                                      .captureCount = frame.captures};
    device::CommandEncoder encoder = device.GetQueue().CreateEncoder();
    for (const GeneratedKernel& step : generated.kernels) {
        device::Kernel& kernel = program->Entry(step.entry);
        kernel.Wait();
        const uint32_t groups = step.shape == DispatchShape::Elements
                                    ? device::GroupCount(elements, kPipelineGroup)
                                : step.shape == DispatchShape::ReduceGroups
                                    ? frame.captures * kReduceGroups
                                    : device::GroupCount(frame.captures, kPipelineGroup);
        encoder.Dispatch(kernel, params, {.x = groups});
        encoder.Barrier();
    }
    device.GetQueue().Submit(encoder.Finish());
    std::map<std::string, std::vector<float>> results;
    for (const auto& [name, buffer] : outputs) {
        results[name] =
            ernie::test::Download<float>(device, *buffer, buffer->Bytes() / sizeof(float));
    }
    return results;
}

}
