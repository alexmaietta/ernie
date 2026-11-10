// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/graph/library.h"
#include "ernie/pipeline/resolve.h"

namespace ernie::pipeline {

inline constexpr uint32_t kBeforeSegments = ~0u;
inline constexpr uint64_t kNotStored = ~uint64_t(0);
inline constexpr uint64_t kArenaAlignment = 256;

struct Segment {
    NodeKind kind = NodeKind::Point;
    std::vector<uint32_t> nodes;
};

struct StoredValue {
    uint32_t value = kNoValue;
    uint32_t first = kBeforeSegments;
    uint32_t last = 0;
};

struct SegmentPlan {
    std::vector<Segment> segments;
    std::vector<StoredValue> stored;
};

struct ArenaLayout {
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
};

enum class Fusion : uint8_t {
    Fused,
    PerNode,
};

SegmentPlan SplitSegments(const ResolvedPipeline& pipeline, const NodeRegistry& registry,
                          Fusion fusion = Fusion::Fused);

ArenaLayout LayoutArena(const SegmentPlan& plan, std::span<const SignalLayout> layouts,
                        uint64_t elements);

struct PipelineDefinition {
    uint32_t id = 0;
    ResolvedPipeline pipeline;
    SegmentPlan segments;
    std::vector<uint32_t> parameterOffsets;
    uint32_t parameterWords = 0;
    std::vector<uint32_t> stateOffsets;
    uint32_t stateWords = 0;
};

using PipelineLibrary = graph::DefinitionLibrary<PipelineDefinition>;
using PipelineSet = graph::DefinitionSet<PipelineDefinition>;

PipelineDefinition BuildDefinition(uint32_t id, const ResolvedPipeline& pipeline,
                                   const NodeRegistry& registry, Fusion fusion = Fusion::Fused);

std::vector<float> InitialState(const PipelineDefinition& definition, const NodeRegistry& registry);

std::vector<uint32_t> PackWords(const PipelineDefinition& definition,
                                const ResolvedPipeline& instance, const NodeRegistry& registry);

std::string ReadParameter(const ParameterDef& parameter, std::string_view context, uint32_t word);

}
