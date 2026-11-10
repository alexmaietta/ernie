// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/graph/library.h"
#include "ernie/sensor_pipeline/resolve.h"

namespace ernie::sensor_pipeline {

inline constexpr int32_t kBeforeSegments = -1;
inline constexpr uint64_t kNotStored = ~uint64_t(0);
inline constexpr uint64_t kArenaAlignment = 256;

struct Segment {
    NodeKind kind = NodeKind::Point;
    std::vector<uint32_t> nodes;
};

struct StoredValue {
    uint32_t value = kNoValue;
    int32_t first = kBeforeSegments;
    int32_t last = 0;
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

SegmentPlan SplitSegments(const ResolvedSensorPipeline& pipeline, const NodeRegistry& registry,
                          Fusion fusion = Fusion::Fused);

ArenaLayout LayoutArena(const SegmentPlan& plan, std::span<const SignalLayout> layouts,
                        uint64_t elements);

struct SensorPipelineDefinition {
    uint32_t id = 0;
    ResolvedSensorPipeline pipeline;
    SegmentPlan segments;
    std::vector<uint32_t> parameterOffsets;
    uint32_t parameterWords = 0;
    std::vector<uint32_t> stateOffsets;
    std::vector<float> initialState;
};

using SensorPipelineLibrary = graph::DefinitionLibrary<SensorPipelineDefinition>;
using SensorPipelineSet = graph::DefinitionSet<SensorPipelineDefinition>;

SensorPipelineDefinition BuildDefinition(uint32_t id, const ResolvedSensorPipeline& pipeline,
                                         const NodeRegistry& registry,
                                         Fusion fusion = Fusion::Fused);

std::vector<uint32_t> PackWords(const SensorPipelineDefinition& definition,
                                const ResolvedSensorPipeline& instance,
                                const NodeRegistry& registry);

std::string ReadParameter(const ParameterDef& parameter, std::string_view context, uint32_t word);

}
