// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ernie/pipeline/definition.h"

namespace ernie::pipeline {

enum class DispatchShape : uint8_t {
    Elements,
    ReduceGroups,
    Captures,
};

struct GeneratedPipeline {
    std::string module;
    std::string path;
    std::string source;
    std::vector<std::string> entries;
    std::vector<DispatchShape> shapes;
    std::vector<uint32_t> slots;
};

GeneratedPipeline GeneratePipeline(const PipelineDefinition& definition,
                                   const NodeRegistry& registry);

}
