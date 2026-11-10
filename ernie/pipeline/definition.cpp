// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/pipeline/definition.h"

#include <algorithm>
#include <bit>
#include <format>

namespace ernie::pipeline {

namespace {

int64_t Start(uint32_t first) {
    return first == kBeforeSegments ? -1 : int64_t(first);
}

}

SegmentPlan SplitSegments(const ResolvedPipeline& pipeline, const NodeRegistry& registry,
                          Fusion fusion) {
    SegmentPlan plan;
    std::vector<uint32_t> segmentOfNode(pipeline.nodes.size(), kBeforeSegments);
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
        segmentOfNode[node] = uint32_t(plan.segments.size() - 1);
    }

    const auto end = uint32_t(plan.segments.size());
    for (uint32_t value = 0; value < pipeline.values.size(); ++value) {
        const uint32_t made = segmentOfNode[pipeline.values[value].node];
        bool stored = false;
        uint32_t last = 0;
        for (uint32_t node = 0; node < pipeline.nodes.size(); ++node) {
            if (std::ranges::find(pipeline.nodes[node].inputs, value) ==
                pipeline.nodes[node].inputs.end()) {
                continue;
            }
            const uint32_t used = segmentOfNode[node];
            if (used != made || made == kBeforeSegments) {
                stored = true;
                last = std::max(last, used);
            }
        }
        const bool output = std::ranges::find(pipeline.outputs, value, &ResolvedOutput::value) !=
                            pipeline.outputs.end();
        if (output && made == kBeforeSegments) {
            stored = true;
            last = end;
        }
        const auto scratch = [&](const ResolvedNode& node) {
            return std::ranges::find(node.scratch, value) != node.scratch.end();
        };
        if (made != kBeforeSegments && scratch(pipeline.nodes[pipeline.values[value].node])) {
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
        int64_t last = 0;
    };

    ArenaLayout arena;
    arena.offsets.assign(layouts.size(), kNotStored);
    std::vector<const StoredValue*> order;
    for (const StoredValue& stored : plan.stored) {
        order.push_back(&stored);
    }
    std::ranges::stable_sort(order, {},
                             [](const StoredValue* stored) { return Start(stored->first); });
    std::vector<Slot> slots;
    for (const StoredValue* stored : order) {
        const SignalLayout& layout = layouts[stored->value];
        const uint64_t channels =
            layout.element == ElementType::F16 ? (layout.channels + 1) / 2 * 2 : layout.channels;
        const uint64_t raw =
            (layout.perCapture ? 1 : elements) * channels * kElementBytes[uint32_t(layout.element)];
        const uint64_t bytes = (raw + kArenaAlignment - 1) / kArenaAlignment * kArenaAlignment;
        Slot* best = nullptr;
        for (Slot& slot : slots) {
            if (slot.last < Start(stored->first) && slot.bytes >= bytes &&
                (best == nullptr || slot.bytes < best->bytes)) {
                best = &slot;
            }
        }
        if (best == nullptr) {
            best = &slots.emplace_back(Slot{arena.bytes, bytes, 0});
            arena.bytes += bytes;
        }
        best->last = int64_t(stored->last);
        arena.offsets[stored->value] = best->offset;
    }
    return arena;
}

PipelineDefinition BuildDefinition(uint32_t id, const ResolvedPipeline& pipeline,
                                   const NodeRegistry& registry, Fusion fusion) {
    PipelineDefinition definition{.id = id, .pipeline = pipeline};
    definition.segments = SplitSegments(pipeline, registry, fusion);
    for (const ResolvedNode& node : pipeline.nodes) {
        definition.parameterOffsets.push_back(definition.parameterWords);
        for (const ParameterDef& parameter : registry[node.def].parameters) {
            definition.parameterWords += parameter.words;
        }
        definition.stateOffsets.push_back(definition.stateWords);
        definition.stateWords += uint32_t(registry[node.def].state.size());
    }
    return definition;
}

std::vector<float> InitialState(const PipelineDefinition& definition,
                                const NodeRegistry& registry) {
    std::vector<float> state;
    for (const ResolvedNode& node : definition.pipeline.nodes) {
        const std::vector<float>& initial = registry[node.def].state;
        state.insert(state.end(), initial.begin(), initial.end());
    }
    return state;
}

std::vector<uint32_t> PackWords(const PipelineDefinition& definition,
                                const ResolvedPipeline& instance, const NodeRegistry& registry) {
    std::vector<uint32_t> words(definition.parameterWords, 0);
    for (uint32_t node = 0; node < instance.nodes.size(); ++node) {
        const std::vector<float>& values = instance.nodes[node].parameters;
        uint32_t index = 0;
        for (const ParameterDef& parameter : registry[instance.nodes[node].def].parameters) {
            for (uint32_t word = 0; word < parameter.words && index < values.size();
                 ++word, ++index) {
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
