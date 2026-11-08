// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/pipeline/definition.h"

#include <bit>

namespace ernie::pipeline {

std::vector<uint32_t> PackWords(const PipelineDefinition& definition,
                                const ResolvedPipeline& instance) {
    std::vector<uint32_t> words(definition.parameterWords, 0);
    for (uint32_t node = 0; node < instance.nodes.size(); ++node) {
        const std::vector<float>& values = instance.nodes[node].parameters;
        for (uint32_t index = 0; index < values.size(); ++index) {
            words[definition.parameterOffsets[node] + index] =
                std::bit_cast<uint32_t>(values[index]);
        }
    }
    return words;
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

}
