// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "ernie/sensor_pipeline/resolve.h"

namespace ernie::sensor_pipeline::test {

inline NodeRegistry FakeRegistry() {
    NodeRegistry registry;
    registry.Add({.id = "Signal",
                  .kind = NodeKind::Source,
                  .outputs = {{"rate", SignalType::ElectronRate}},
                  .uniforms = {{"layout", {"full", "mosaic"}, "full"},
                               {"precision", {"float", "half"}, "float"}},
                  .layout = LayoutRule::Detector});
    registry.Add({.id = "Integrate",
                  .inputs = {{"rate", SignalType::ElectronRate}},
                  .outputs = {{"electrons", SignalType::Electrons}},
                  .parameters = {{"time"}}});
    registry.Add({.id = "Noise",
                  .inputs = {{"electrons", SignalType::Electrons}},
                  .outputs = {{"electrons", SignalType::Electrons}},
                  .parameters = {{"readNoise"}, {"fullWell"}},
                  .uniforms = {{"shot", {"poisson", "gaussian", "off"}, "poisson"}}});
    registry.Add({.id = "Blur",
                  .kind = NodeKind::Stencil,
                  .inputs = {{"electrons", SignalType::Electrons}},
                  .outputs = {{"electrons", SignalType::Electrons}},
                  .radius = 2});
    registry.Add({.id = "Adc",
                  .inputs = {{"electrons", SignalType::Electrons}},
                  .outputs = {{"digital", SignalType::Digital}},
                  .parameters = {{"gain"}, {"blackLevel"}, {"bits"}}});
    registry.Add({.id = "Meter",
                  .kind = NodeKind::Reduce,
                  .inputs = {{"digital", SignalType::Digital}},
                  .outputs = {{"stat", SignalType::CaptureStat}},
                  .layout = LayoutRule::Stat});
    registry.Add(
        {.id = "Expose",
         .inputs = {{"digital", SignalType::Digital}, {"stat", SignalType::CaptureStat, true}},
         .outputs = {{"digital", SignalType::Digital}}});
    registry.Add({.id = "ToColor",
                  .inputs = {{"digital", SignalType::Digital}},
                  .outputs = {{"color", SignalType::LinearColor}},
                  .layout = LayoutRule::Color});
    return registry;
}

struct Link {
    std::string input;
    uint32_t source;
    std::string output;
};

inline NodeDesc Node(uint32_t prim, std::string id, std::initializer_list<Link> links = {},
                     std::vector<std::pair<std::string, std::string>> uniforms = {}) {
    NodeDesc node{.prim = prim, .path = "/Pipe/" + id + std::to_string(prim), .id = id};
    node.uniforms = std::move(uniforms);
    for (const Link& link : links) {
        node.inputs.push_back(
            {link.input, link.source, "/Pipe/node" + std::to_string(link.source), link.output});
    }
    return node;
}

inline SensorPipelineDesc Pipeline(std::vector<NodeDesc> nodes,
                                   std::initializer_list<Link> outputs) {
    SensorPipelineDesc desc{.prim = 1, .path = "/Pipe", .nodes = std::move(nodes)};
    for (const Link& link : outputs) {
        desc.outputs.push_back(
            {link.input, link.source, "/Pipe/node" + std::to_string(link.source), link.output});
    }
    return desc;
}

}
