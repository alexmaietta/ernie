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

struct GeneratedKernel {
    std::string entry;
    DispatchShape shape = DispatchShape::Elements;
};

struct GeneratedPipeline {
    std::string module;
    std::string path;
    std::string source;
    std::vector<GeneratedKernel> kernels;
    std::vector<uint32_t> slots;

    std::vector<std::string> Entries() const;
};

GeneratedPipeline GeneratePipeline(const PipelineDefinition& definition,
                                   const NodeRegistry& registry);

}
