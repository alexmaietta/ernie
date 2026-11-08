// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/render/standalone.h"

#include <algorithm>
#include <atomic>

#include "ernie/device/queue.h"
#include "ernie/pipeline/builtins.h"
#include "ernie/render/pipelines.h"
#include "ernie/util/error.h"

namespace ernie {

namespace {

uint32_t PatternChannels(DetectorPattern pattern) {
    switch (pattern) {
    case DetectorPattern::Mono:
        return 1;
    case DetectorPattern::Rccc:
        return 2;
    case DetectorPattern::Bands:
        return kDetectorBands;
    default:
        return 3;
    }
}

bool PatternMosaics(DetectorPattern pattern) {
    switch (pattern) {
    case DetectorPattern::Rggb:
    case DetectorPattern::Bggr:
    case DetectorPattern::Grbg:
    case DetectorPattern::Gbrg:
    case DetectorPattern::Rccb:
    case DetectorPattern::Rccc:
        return true;
    default:
        return false;
    }
}

DetectorRecord Record(const DetectorDesc& desc) {
    DetectorRecord record{};
    record.pattern = uint32_t(desc.pattern);
    record.channels = PatternChannels(desc.pattern);
    record.fullWell = desc.fullWell;
    record.readNoise = desc.readNoise;
    record.darkCurrent = desc.darkCurrent;
    record.darkCurrentDoubling = desc.darkCurrentDoubling;
    record.temperature = desc.temperature;
    record.prnu = desc.prnu;
    record.dsnu = desc.dsnu;
    record.systemGain = desc.systemGain;
    record.bitDepth = desc.bitDepth;
    record.blackLevel = desc.blackLevel;
    record.seed = desc.seed;
    return record;
}

std::atomic<uint32_t> g_standaloneDefinitions{1u << 24};

}

Pipeline::Pipeline(device::Device& device, pipeline::PipelineDesc desc)
    : Pipeline(device, std::move(desc), pipeline::BuiltinNodes()) {}

Pipeline::Pipeline(device::Device& device, pipeline::PipelineDesc desc,
                   const pipeline::NodeRegistry& registry)
    : m_device(&device), m_registry(&registry),
      m_instance(pipeline::ResolvePipeline(desc, registry)) {
    auto definition = std::make_shared<pipeline::PipelineDefinition>(
        pipeline::BuildDefinition(g_standaloneDefinitions++, m_instance, registry));
    m_compiled = CompilePipeline(device, std::move(definition), registry);
}

PipelineOutputs Pipeline::Run(const PipelineRunDesc& desc) const {
    Ref<device::Buffer> state;
    return Run(desc, state);
}

PipelineOutputs Pipeline::Run(const PipelineRunDesc& desc, Ref<device::Buffer>& state) const {
    const pipeline::PipelineDefinition& definition = *m_compiled->definition;
    const pipeline::GeneratedPipeline& generated = m_compiled->generated;
    const DetectorRecord detector = Record(desc.detector);
    const std::vector<pipeline::SignalLayout> layouts =
        pipeline::PropagateLayouts(m_instance, *m_registry,
                                   {.channels = detector.channels,
                                    .pattern = detector.pattern,
                                    .mosaics = PatternMosaics(desc.detector.pattern)});

    int64_t captures = -1;
    int64_t height = 0;
    int64_t width = 0;
    std::vector<const device::Tensor*> sources(generated.slots.size(), nullptr);
    for (uint32_t slot = 0; slot < generated.slots.size(); ++slot) {
        const pipeline::Value& value = m_instance.values[generated.slots[slot]];
        const pipeline::ResolvedNode& node = m_instance.nodes[value.node];
        if ((*m_registry)[node.def].kind != pipeline::NodeKind::Source) {
            continue;
        }
        const auto found = desc.inputs.find(node.path);
        if (found == desc.inputs.end()) {
            throw Error("The run has no input for {}", node.path);
        }
        const device::Tensor& tensor = found->second;
        const Span<const int64_t> shape = tensor.Shape();
        const uint32_t channels = layouts[generated.slots[slot]].channels;
        const int64_t given = shape.size() == 4 ? shape[3] : 1;
        if (tensor.Type() != device::DType::Float32 || (shape.size() != 3 && shape.size() != 4) ||
            given != channels) {
            throw Error("{}'s input must be float32 shaped [sensors, height, width{}]", node.path,
                        channels == 1 ? std::string() : ", " + std::to_string(channels));
        }
        if (captures >= 0 && (shape[0] != captures || shape[1] != height || shape[2] != width)) {
            throw Error("{}'s input is shaped unlike the run's other inputs", node.path);
        }
        captures = shape[0];
        height = shape[1];
        width = shape[2];
        sources[slot] = &tensor;
    }
    if (captures <= 0 || height <= 0 || width <= 0) {
        throw Error("A standalone run needs at least one nonempty input");
    }

    device::Device& device = *m_device;
    const uint64_t perCapture = uint64_t(height) * width;
    const uint64_t elements = perCapture * captures;
    std::vector<device::Tensor> tensors(generated.slots.size());
    std::vector<PipelineValue> values;
    const auto storedCount = uint32_t(definition.segments.stored.size());
    for (uint32_t slot = 0; slot < generated.slots.size(); ++slot) {
        const pipeline::SignalLayout& layout = layouts[generated.slots[slot]];
        if (sources[slot] != nullptr) {
            tensors[slot] = *sources[slot];
        } else {
            std::vector<int64_t> shape = {captures};
            if (!layout.perCapture) {
                shape.insert(shape.end(), {height, width});
            }
            if (layout.channels > 1) {
                shape.push_back(layout.channels);
            }
            const uint64_t count =
                uint64_t(captures) * (layout.perCapture ? 1 : perCapture) * layout.channels;
            tensors[slot] = device::Tensor(
                device.CreateBuffer({.bytes = count * sizeof(float), .label = "pipeline value"}), 0,
                device::DType::Float32, std::move(shape));
        }
        values.push_back(
            {tensors[slot].Address(), layout.channels, layout.mosaic ? kSignalMosaic : 0u});
    }

    const std::vector<uint32_t> packed = pipeline::PackWords(definition, m_instance);
    std::vector<uint32_t> words = packed;
    words.push_back(0);
    CameraRecord camera{};
    camera.exposureTime = desc.exposureTime;
    camera.iso = desc.iso;
    std::vector<PipelineValue> slots;
    std::vector<PipelineCapture> records;
    const Ref<device::Buffer> wordBuffer = device.CreateBuffer({.label = "pipeline words"}, words);
    const uint64_t stateBytes = std::max<uint64_t>(
        uint64_t(definition.stateWords) * captures * sizeof(float), sizeof(float));
    if (!state || state->Bytes() != stateBytes) {
        const std::vector<float> initial = pipeline::InitialState(definition, *m_registry);
        std::vector<float> states;
        for (int64_t capture = 0; capture < captures; ++capture) {
            states.insert(states.end(), initial.begin(), initial.end());
        }
        states.resize(stateBytes / sizeof(float));
        state = device.CreateBuffer({.label = "pipeline state"}, states);
    }
    for (int64_t capture = 0; capture < captures; ++capture) {
        for (uint32_t slot = 0; slot < values.size(); ++slot) {
            PipelineValue value = values[slot];
            const bool stat = layouts[generated.slots[slot]].perCapture;
            value.address +=
                uint64_t(capture) * (stat ? 1 : perCapture) * value.channels * sizeof(float);
            slots.push_back(value);
        }
        records.push_back({.firstElement = uint64_t(capture) * perCapture,
                           .words = {wordBuffer->Address()},
                           .width = uint32_t(width),
                           .height = uint32_t(height),
                           .capture = desc.capture,
                           .identity = desc.identity + uint32_t(capture),
                           .firstValue = uint32_t(capture * values.size()),
                           .state = {state->Address() +
                                     uint64_t(capture) * definition.stateWords * sizeof(float)}});
    }
    const Ref<device::Buffer> cameras =
        device.CreateBuffer({.label = "pipeline camera"}, std::vector<CameraRecord>{camera});
    const Ref<device::Buffer> detectors =
        device.CreateBuffer({.label = "pipeline detector"}, std::vector<DetectorRecord>{detector});
    RenderTables tables{};
    tables.cameras.address = cameras->Address();
    tables.detectors.address = detectors->Address();
    const Ref<device::Buffer> tableBuffer =
        device.CreateBuffer({.label = "pipeline tables"}, std::vector<RenderTables>{tables});
    const Ref<device::Buffer> valueBuffer =
        device.CreateBuffer({.label = "pipeline values"}, slots);
    const Ref<device::Buffer> captureBuffer =
        device.CreateBuffer({.label = "pipeline captures"}, records);
    const PipelineParams params{.tables = {tableBuffer->Address()},
                                .captures = {captureBuffer->Address()},
                                .values = {valueBuffer->Address()},
                                .elementCount = elements,
                                .captureCount = uint32_t(captures)};
    device::CommandEncoder encoder = device.GetQueue().CreateEncoder();
    RecordPipeline(encoder, *m_compiled, params, uint32_t(captures), elements);
    device.GetQueue().Submit(encoder.Finish()).Wait();

    PipelineOutputs outputs;
    for (uint32_t output = 0; output < m_instance.outputs.size(); ++output) {
        outputs.emplace(m_instance.outputs[output].name, tensors[storedCount + output]);
    }
    return outputs;
}

PipelineOutputs PipelineSession::Run(PipelineRunDesc desc) {
    desc.capture = m_captures++;
    return m_pipeline.Run(desc, m_state);
}

}
