// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ernie/sensor_pipeline/definition.h"

namespace ernie::sensor_pipeline {

enum class DispatchShape : uint8_t {
    Elements,
    ReduceGroups,
    Captures,
};

struct GeneratedKernel {
    std::string entry;
    DispatchShape shape = DispatchShape::Elements;
};

struct GeneratedSensorPipeline {
    std::string module;
    std::string path;
    std::string source;
    std::vector<GeneratedKernel> kernels;
    std::vector<uint32_t> slots;

    std::vector<std::string> Entries() const;
};

GeneratedSensorPipeline GenerateSensorPipeline(const SensorPipelineDefinition& definition,
                                               const NodeRegistry& registry);

}
