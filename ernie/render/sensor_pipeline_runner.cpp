// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/render/sensor_pipeline_runner.h"

#include <algorithm>
#include <atomic>

#include "ernie/device/queue.h"
#include "ernie/render/sensor_pipelines.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/sensor_pipeline/node.h"
#include "ernie/sensor_pipeline/resolve.h"
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

struct SensorPipelineRunner::State {
    device::Device* device = nullptr;
    std::shared_ptr<const sensor_pipeline::NodeRegistry> registry;
    std::string path;
    sensor_pipeline::ResolvedSensorPipeline instance;
    std::shared_ptr<const CompiledSensorPipeline> compiled;
    std::optional<std::vector<uint32_t>> words;

    State(device::Device& on, std::shared_ptr<const sensor_pipeline::NodeRegistry> nodes,
          std::string name, sensor_pipeline::ResolvedSensorPipeline resolved)
        : device(&on), registry(std::move(nodes)), path(std::move(name)),
          instance(std::move(resolved)) {
        auto definition = std::make_shared<sensor_pipeline::SensorPipelineDefinition>(
            sensor_pipeline::BuildDefinition(g_standaloneDefinitions++, instance, *registry));
        compiled = CompileSensorPipeline(on, std::move(definition), *registry);
    }
};

std::optional<SensorPipelineNodeInfo> FindSensorPipelineNode(std::string_view id) {
    const std::shared_ptr<const sensor_pipeline::NodeRegistry> nodes =
        sensor_pipeline::SensorPipelineNodes();
    const sensor_pipeline::NodeRegistry& registry = *nodes;
    const uint32_t found = registry.Find(id);
    if (found == sensor_pipeline::kNoDef) {
        return std::nullopt;
    }
    const sensor_pipeline::NodeDef& def = registry[found];
    SensorPipelineNodeInfo info{
        .id = def.id, .kind = std::string(sensor_pipeline::kNodeKindNames[uint32_t(def.kind)])};
    for (const sensor_pipeline::PortDef& port : def.inputs) {
        info.inputs.push_back(
            {port.name, std::string(sensor_pipeline::Signal(port.type).name), port.optional});
    }
    for (const sensor_pipeline::PortDef& port : def.outputs) {
        info.outputs.push_back(
            {port.name, std::string(sensor_pipeline::Signal(port.type).name), port.optional});
    }
    return info;
}

SensorPipelineRunner::SensorPipelineRunner(const Ref<Scene>& scene, std::string_view path) {
    if (scene->Pending()) {
        scene->Publish().Wait();
    }
    const std::optional<Prim> prim = scene->Find(path);
    if (!prim || !prim->IsA<SensorPipeline>()) {
        throw Error("{} isn't a SensorPipeline", path);
    }
    const SensorPipelineTable& table = SceneImpl::Get(*scene).GetSensorPipelines();
    const std::shared_ptr<const SensorPipelineInstance> instance = table.Find(prim->Index());
    if (!instance) {
        throw Error("{} isn't resolved yet", path);
    }
    if (!instance->error.empty()) {
        throw Error("{}", instance->error);
    }
    m_state = std::make_unique<State>(scene->GetDevice(), table.Registry(), std::string(path),
                                      instance->resolved);
    m_state->words = table.ReadWords(*instance);
}

SensorPipelineRunner::SensorPipelineRunner(device::Device& device,
                                           const sensor_pipeline::SensorPipelineDesc& desc)
    : SensorPipelineRunner(device, desc, *sensor_pipeline::SensorPipelineNodes()) {}

SensorPipelineRunner::SensorPipelineRunner(device::Device& device,
                                           const sensor_pipeline::SensorPipelineDesc& desc,
                                           const sensor_pipeline::NodeRegistry& registry)
    : m_state(std::make_unique<State>(
          device, std::make_shared<sensor_pipeline::NodeRegistry>(registry), desc.path,
          sensor_pipeline::ResolveSensorPipeline(desc, registry))) {}

SensorPipelineRunner::~SensorPipelineRunner() = default;
SensorPipelineRunner::SensorPipelineRunner(SensorPipelineRunner&&) noexcept = default;
SensorPipelineRunner& SensorPipelineRunner::operator=(SensorPipelineRunner&&) noexcept = default;

device::Device& SensorPipelineRunner::GetDevice() const {
    return *m_state->device;
}

std::vector<std::string> SensorPipelineRunner::Sources() const {
    std::vector<std::string> sources;
    for (const sensor_pipeline::ResolvedNode& node : m_state->instance.nodes) {
        if ((*m_state->registry)[node.def].kind == sensor_pipeline::NodeKind::Source) {
            sources.push_back(node.path);
        }
    }
    return sources;
}

std::vector<std::string> SensorPipelineRunner::Outputs() const {
    std::vector<std::string> outputs;
    for (const sensor_pipeline::ResolvedOutput& output : m_state->instance.outputs) {
        outputs.push_back(output.name);
    }
    return outputs;
}

SensorPipelineOutputs SensorPipelineRunner::Run(const SensorPipelineRunDesc& desc) const {
    device::Tensor state;
    return Run(desc, state);
}

SensorPipelineOutputs SensorPipelineRunner::Run(const SensorPipelineRunDesc& desc,
                                                device::Tensor& state) const {
    const sensor_pipeline::NodeRegistry& registry = *m_state->registry;
    const sensor_pipeline::ResolvedSensorPipeline& instance = m_state->instance;
    const sensor_pipeline::SensorPipelineDefinition& definition = *m_state->compiled->definition;
    const sensor_pipeline::GeneratedSensorPipeline& generated = m_state->compiled->generated;
    const DetectorRecord detector = Record(desc.detector);
    const std::vector<sensor_pipeline::SignalLayout> layouts =
        sensor_pipeline::PropagateLayouts(instance, registry,
                                          {.channels = detector.channels,
                                           .pattern = detector.pattern,
                                           .mosaics = PatternMosaics(desc.detector.pattern)});

    int64_t captures = -1;
    int64_t height = 0;
    int64_t width = 0;
    std::vector<const device::Tensor*> sources(generated.slots.size(), nullptr);
    for (uint32_t slot = 0; slot < definition.segments.stored.size(); ++slot) {
        const sensor_pipeline::Value& value = instance.values[generated.slots[slot]];
        const sensor_pipeline::ResolvedNode& node = instance.nodes[value.node];
        if (registry[node.def].kind != sensor_pipeline::NodeKind::Source) {
            continue;
        }
        auto found = desc.inputs.find(node.path);
        if (found == desc.inputs.end() && node.path.starts_with(m_state->path + "/")) {
            found = desc.inputs.find(std::string_view(node.path).substr(m_state->path.size() + 1));
        }
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

    device::Device& device = *m_state->device;
    const uint64_t perCapture = uint64_t(height) * width;
    const uint64_t elements = perCapture * captures;
    std::vector<device::Tensor> tensors(generated.slots.size());
    std::vector<SensorPipelineValue> values;
    const auto storedCount = uint32_t(definition.segments.stored.size());
    for (uint32_t slot = 0; slot < generated.slots.size(); ++slot) {
        const sensor_pipeline::SignalLayout& layout = layouts[generated.slots[slot]];
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

    std::vector<uint32_t> words = m_state->words
                                      ? *m_state->words
                                      : sensor_pipeline::PackWords(definition, instance, registry);
    words.push_back(0);
    CameraRecord camera{};
    camera.exposureTime = desc.exposureTime;
    camera.iso = desc.iso;
    std::vector<SensorPipelineValue> slots;
    std::vector<SensorPipelineCapture> records;
    const Ref<device::Buffer> wordBuffer = device.CreateBuffer({.label = "pipeline words"}, words);
    const uint64_t stateBytes = std::max<uint64_t>(
        uint64_t(definition.stateWords) * captures * sizeof(float), sizeof(float));
    if (state.Empty() || state.GetBuffer().Bytes() != stateBytes) {
        const std::vector<float> initial = sensor_pipeline::InitialState(definition, registry);
        std::vector<float> states;
        for (int64_t capture = 0; capture < captures; ++capture) {
            states.insert(states.end(), initial.begin(), initial.end());
        }
        states.resize(stateBytes / sizeof(float));
        state = device::Tensor(device.CreateBuffer({.label = "pipeline state"}, states), 0,
                               device::DType::Float32, {int64_t(states.size())});
    }
    for (int64_t capture = 0; capture < captures; ++capture) {
        for (uint32_t slot = 0; slot < values.size(); ++slot) {
            SensorPipelineValue value = values[slot];
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
                           .state = {state.Address() +
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
    const SensorPipelineParams params{.tables = {tableBuffer->Address()},
                                      .captures = {captureBuffer->Address()},
                                      .values = {valueBuffer->Address()},
                                      .elementCount = elements,
                                      .captureCount = uint32_t(captures)};
    device::CommandEncoder encoder = device.GetQueue().CreateEncoder();
    RecordSensorPipeline(encoder, *m_state->compiled, params, uint32_t(captures), elements);
    device.GetQueue().Submit(encoder.Finish()).Wait();

    SensorPipelineOutputs outputs;
    for (uint32_t output = 0; output < instance.outputs.size(); ++output) {
        outputs.emplace(instance.outputs[output].name, tensors[storedCount + output]);
    }
    return outputs;
}

SensorPipelineOutputs SensorPipelineSession::Run(SensorPipelineRunDesc desc) {
    desc.capture = m_captures++;
    return m_pipeline.Run(desc, m_state);
}

}
