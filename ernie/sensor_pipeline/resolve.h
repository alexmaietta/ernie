// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "ernie/graph/graph.h"
#include "ernie/sensor_pipeline/node.h"

namespace ernie::sensor_pipeline {

inline constexpr uint32_t kNoValue = ~0u;

struct ConnectionDesc {
    std::string input;
    uint32_t source = ~0u;
    std::string sourcePath;
    std::string output;
};

struct NodeDesc {
    uint32_t prim = 0;
    std::string path;
    std::string id;
    std::vector<std::pair<std::string, std::string>> uniforms;
    std::vector<std::pair<std::string, std::vector<float>>> values;
    std::vector<ConnectionDesc> inputs;
};

struct SensorPipelineDesc {
    uint32_t prim = 0;
    std::string path;
    std::vector<NodeDesc> nodes;
    std::vector<ConnectionDesc> outputs;
};

struct ResolvedNode {
    uint32_t def = kNoDef;
    uint32_t prim = 0;
    std::string path;
    std::vector<uint32_t> uniforms;
    std::vector<float> parameters;
    std::vector<uint32_t> inputs;
    std::vector<uint32_t> outputs;
    std::vector<uint32_t> scratch;
};

struct Value {
    uint32_t node = 0;
    uint32_t port = 0;
    SignalType type = SignalType::ElectronRate;
};

struct ResolvedOutput {
    std::string name;
    uint32_t value = kNoValue;
};

struct ResolvedSensorPipeline {
    std::vector<ResolvedNode> nodes;
    std::vector<Value> values;
    std::vector<ResolvedOutput> outputs;
    graph::ShapeKey key;
};

struct DetectorLayout {
    uint32_t channels = 3;
    uint32_t pattern = 0;
    bool mosaics = false;
};

ResolvedSensorPipeline ResolveSensorPipeline(const SensorPipelineDesc& desc,
                                             const NodeRegistry& registry);

std::vector<SignalLayout> PropagateLayouts(const ResolvedSensorPipeline& pipeline,
                                           const NodeRegistry& registry, DetectorLayout detector);

}
