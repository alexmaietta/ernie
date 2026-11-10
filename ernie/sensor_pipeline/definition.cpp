// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/sensor_pipeline/definition.h"

#include <algorithm>
#include <bit>
#include <format>

namespace ernie::sensor_pipeline {

namespace {

uint64_t StoredBytes(const SignalLayout& layout, uint64_t elements) {
    const uint64_t channels =
        layout.element == ElementType::F16 ? (layout.channels + 1) / 2 * 2 : layout.channels;
    const uint64_t raw =
        (layout.perCapture ? 1 : elements) * channels * kElementBytes[uint32_t(layout.element)];
    return (raw + kArenaAlignment - 1) / kArenaAlignment * kArenaAlignment;
}

}

SegmentPlan SplitSegments(const ResolvedSensorPipeline& pipeline, const NodeRegistry& registry,
                          Fusion fusion) {
    SegmentPlan plan;
    std::vector<int32_t> segmentOfNode(pipeline.nodes.size(), kBeforeSegments);
    bool open = false;
    for (uint32_t node = 0; node < pipeline.nodes.size(); ++node) {
        const NodeKind kind = registry[pipeline.nodes[node].def].kind;
        if (kind == NodeKind::Source) {
            continue;
        }
        if (!(open && kind == NodeKind::Point && fusion == Fusion::Fused)) {
            plan.segments.push_back({kind, {}});
            open = kind == NodeKind::Point || kind == NodeKind::Stencil;
        }
        plan.segments.back().nodes.push_back(node);
        segmentOfNode[node] = int32_t(plan.segments.size() - 1);
    }

    std::vector<std::vector<uint32_t>> readers(pipeline.values.size());
    for (uint32_t node = 0; node < pipeline.nodes.size(); ++node) {
        for (const uint32_t value : pipeline.nodes[node].inputs) {
            if (value != kNoValue) {
                readers[value].push_back(node);
            }
        }
    }

    const auto end = int32_t(plan.segments.size());
    for (uint32_t value = 0; value < pipeline.values.size(); ++value) {
        const ResolvedNode& maker = pipeline.nodes[pipeline.values[value].node];
        const int32_t made = segmentOfNode[pipeline.values[value].node];
        bool stored = false;
        int32_t last = 0;
        for (const uint32_t reader : readers[value]) {
            if (segmentOfNode[reader] != made || made == kBeforeSegments) {
                stored = true;
                last = std::max(last, segmentOfNode[reader]);
            }
        }
        const bool output = std::ranges::find(pipeline.outputs, value, &ResolvedOutput::value) !=
                            pipeline.outputs.end();
        if (output && made == kBeforeSegments) {
            stored = true;
            last = end;
        }
        if (made != kBeforeSegments &&
            std::ranges::find(maker.scratch, value) != maker.scratch.end()) {
            stored = true;
            last = made;
        }
        if (stored) {
            plan.stored.push_back({value, made, last});
        }
    }
    return plan;
}

ArenaLayout LayoutArena(const SegmentPlan& plan, std::span<const SignalLayout> layouts,
                        uint64_t elements) {
    struct Slot {
        uint64_t offset = 0;
        uint64_t bytes = 0;
        int32_t last = 0;
    };

    ArenaLayout arena;
    arena.offsets.assign(layouts.size(), kNotStored);
    std::vector<StoredValue> order = plan.stored;
    std::ranges::stable_sort(order, {}, &StoredValue::first);
    std::vector<Slot> slots;
    for (const StoredValue& stored : order) {
        const uint64_t bytes = StoredBytes(layouts[stored.value], elements);
        Slot* best = nullptr;
        for (Slot& slot : slots) {
            if (slot.last < stored.first && slot.bytes >= bytes &&
                (best == nullptr || slot.bytes < best->bytes)) {
                best = &slot;
            }
        }
        if (best == nullptr) {
            best = &slots.emplace_back(Slot{arena.bytes, bytes, 0});
            arena.bytes += bytes;
        }
        best->last = stored.last;
        arena.offsets[stored.value] = best->offset;
    }
    return arena;
}

SensorPipelineDefinition BuildDefinition(uint32_t id, const ResolvedSensorPipeline& pipeline,
                                         const NodeRegistry& registry, Fusion fusion) {
    SensorPipelineDefinition definition{.id = id, .pipeline = pipeline};
    definition.segments = SplitSegments(pipeline, registry, fusion);
    for (const ResolvedNode& node : pipeline.nodes) {
        definition.parameterOffsets.push_back(definition.parameterWords);
        for (const ParameterDef& parameter : registry[node.def].parameters) {
            definition.parameterWords += parameter.words;
        }
        const std::vector<float>& initial = registry[node.def].state;
        definition.stateOffsets.push_back(uint32_t(definition.initialState.size()));
        definition.initialState.insert(definition.initialState.end(), initial.begin(),
                                       initial.end());
    }
    return definition;
}

std::vector<uint32_t> PackWords(const SensorPipelineDefinition& definition,
                                const ResolvedSensorPipeline& instance,
                                const NodeRegistry& registry) {
    std::vector<uint32_t> words(definition.parameterWords, 0);
    for (uint32_t node = 0; node < instance.nodes.size(); ++node) {
        const std::vector<float>& values = instance.nodes[node].parameters;
        uint32_t index = 0;
        for (const ParameterDef& parameter : registry[instance.nodes[node].def].parameters) {
            for (uint32_t word = 0; word < parameter.words; ++word, ++index) {
                const float value = values[index];
                words[definition.parameterOffsets[node] + index] =
                    parameter.type == ParameterType::Int ? std::bit_cast<uint32_t>(int32_t(value))
                    : parameter.type == ParameterType::Bool ? uint32_t(value != 0)
                                                            : std::bit_cast<uint32_t>(value);
            }
        }
    }
    return words;
}

std::string ReadParameter(const ParameterDef& parameter, std::string_view context, uint32_t word) {
    switch (parameter.type) {
    case ParameterType::Int:
        return std::format("asint({}.ParameterBits({}))", context, word);
    case ParameterType::Bool:
        return std::format("{}.ParameterBits({}) != 0", context, word);
    case ParameterType::Float:
        break;
    }
    if (parameter.words == 1) {
        return std::format("{}.Parameter({})", context, word);
    }
    std::string components;
    for (uint32_t component = 0; component < parameter.words; ++component) {
        components += std::format("{}{}.Parameter({})", component == 0 ? "" : ", ", context,
                                  word + component);
    }
    return std::format("float{}({})", parameter.words, components);
}

}
