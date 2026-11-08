// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <vector>

#include "ernie/graph/library.h"
#include "ernie/pipeline/segments.h"

namespace ernie::pipeline {

struct PipelineDefinition {
    uint32_t id = 0;
    ResolvedPipeline pipeline;
    SegmentPlan segments;
    std::vector<uint32_t> parameterOffsets;
    uint32_t parameterWords = 0;
};

using PipelineLibrary = graph::DefinitionLibrary<PipelineDefinition>;
using PipelineSet = graph::DefinitionSet<PipelineDefinition>;

PipelineDefinition BuildDefinition(uint32_t id, const ResolvedPipeline& pipeline,
                                   const NodeRegistry& registry);

}
