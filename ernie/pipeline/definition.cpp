// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/pipeline/definition.h"

namespace ernie::pipeline {

PipelineDefinition BuildDefinition(uint32_t id, const ResolvedPipeline& pipeline,
                                   const NodeRegistry& registry) {
    PipelineDefinition definition{.id = id, .pipeline = pipeline};
    definition.segments = SplitSegments(pipeline, registry);
    for (const ResolvedNode& node : pipeline.nodes) {
        definition.parameterOffsets.push_back(definition.parameterWords);
        for (const ParameterDef& parameter : registry[node.def].parameters) {
            definition.parameterWords += parameter.words;
        }
    }
    return definition;
}

}
