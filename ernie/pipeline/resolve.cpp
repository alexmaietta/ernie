// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/pipeline/resolve.h"

#include <algorithm>
#include <unordered_map>

#include "ernie/graph/graph.h"
#include "ernie/util/error.h"

namespace ernie::pipeline {

namespace {

uint32_t Port(std::span<const PortDef> ports, std::string_view name) {
    const auto found = std::ranges::find(ports, name, &PortDef::name);
    return found == ports.end() ? kNoValue : uint32_t(found - ports.begin());
}

std::string Uniform(const NodeDesc& node, const UniformDef& uniform) {
    const auto found =
        std::ranges::find(node.uniforms, uniform.name, &std::pair<std::string, std::string>::first);
    std::string value = found == node.uniforms.end() ? uniform.fallback : found->second;
    if (!uniform.allowed.empty() &&
        std::ranges::find(uniform.allowed, value) == uniform.allowed.end()) {
        std::string allowed;
        for (const std::string& each : uniform.allowed) {
            allowed += (allowed.empty() ? "" : ", ") + each;
        }
        throw Error("{}'s {} is {}, and it takes {}", node.path, uniform.name, value, allowed);
    }
    return value;
}

}

ResolvedPipeline ResolvePipeline(const PipelineDesc& desc, const NodeRegistry& registry) {
    std::unordered_map<uint32_t, uint32_t> byPrim;
    std::vector<uint32_t> defs;
    for (uint32_t index = 0; index < desc.nodes.size(); ++index) {
        const NodeDesc& node = desc.nodes[index];
        const uint32_t def = registry.Find(node.id);
        if (def == kNoDef) {
            throw Error("{} runs {}, which no pipeline node is", node.path, node.id);
        }
        byPrim.emplace(node.prim, index);
        defs.push_back(def);
    }
    const auto source = [&](const ConnectionDesc& connection, std::string_view reader,
                            std::string_view input) {
        const auto found = byPrim.find(connection.source);
        if (found == byPrim.end()) {
            throw Error("{}'s {} connects to {}, which isn't a node of {}", reader, input,
                        connection.sourcePath, desc.path);
        }
        const NodeDef& def = registry[defs[found->second]];
        const uint32_t port = Port(def.outputs, connection.output);
        if (port == kNoValue) {
            throw Error("{}'s {} reads {}'s {}, which {} doesn't output", reader, input,
                        connection.sourcePath, connection.output, def.id);
        }
        return std::pair{found->second, port};
    };

    if (desc.outputs.empty()) {
        throw Error("{} has no outputs", desc.path);
    }
    std::vector<bool> reached(desc.nodes.size(), false);
    std::vector<uint32_t> pending;
    for (const ConnectionDesc& output : desc.outputs) {
        const uint32_t node = source(output, desc.path, output.input).first;
        if (!reached[node]) {
            reached[node] = true;
            pending.push_back(node);
        }
    }
    while (!pending.empty()) {
        const uint32_t node = pending.back();
        pending.pop_back();
        for (const ConnectionDesc& connection : desc.nodes[node].inputs) {
            const auto found = byPrim.find(connection.source);
            if (found != byPrim.end() && !reached[found->second]) {
                reached[found->second] = true;
                pending.push_back(found->second);
            }
        }
    }

    graph::Graph graph;
    std::vector<uint32_t> graphNode(desc.nodes.size(), graph::kNoNode);
    std::vector<uint32_t> descNode;
    for (uint32_t index = 0; index < desc.nodes.size(); ++index) {
        if (reached[index]) {
            graphNode[index] = graph.Add(desc.nodes[index].prim, defs[index]);
            descNode.push_back(index);
        }
    }
    for (const uint32_t index : descNode) {
        const NodeDesc& node = desc.nodes[index];
        const NodeDef& def = registry[defs[index]];
        for (const ConnectionDesc& connection : node.inputs) {
            const uint32_t input = Port(def.inputs, connection.input);
            if (input == kNoValue) {
                throw Error("{}'s {} is connected, but {} has no such signal input", node.path,
                            connection.input, def.id);
            }
            const auto [from, port] = source(connection, node.path, connection.input);
            const SignalType made = registry[defs[from]].outputs[port].type;
            const SignalType taken = def.inputs[input].type;
            if (made != taken) {
                throw Error("{}'s {} takes {}, but {}'s {} is {}", node.path, connection.input,
                            kSignalNames[uint32_t(taken)], connection.sourcePath, connection.output,
                            kSignalNames[uint32_t(made)]);
            }
            graph.Connect(graphNode[index], connection.input, graphNode[from], connection.output);
        }
        for (const PortDef& input : def.inputs) {
            const bool connected = std::ranges::any_of(
                node.inputs, [&](const ConnectionDesc& each) { return each.input == input.name; });
            if (!connected && !input.optional) {
                throw Error("{}'s {} isn't connected, and {} needs it", node.path, input.name,
                            def.id);
            }
        }
    }

    const graph::Ordering ordering = graph.Order();
    if (!ordering.cycle.empty()) {
        std::string cycle;
        for (const uint32_t node : ordering.cycle) {
            cycle += (cycle.empty() ? "" : ", ") + desc.nodes[descNode[node]].path;
        }
        throw Error("{} loops through {}", desc.path, cycle);
    }

    ResolvedPipeline pipeline;
    std::vector<uint32_t> position(graph.Nodes().size());
    for (const uint32_t node : ordering.order) {
        const NodeDesc& each = desc.nodes[descNode[node]];
        const NodeDef& def = registry[defs[descNode[node]]];
        position[node] = uint32_t(pipeline.nodes.size());
        ResolvedNode& resolved = pipeline.nodes.emplace_back();
        resolved.def = defs[descNode[node]];
        resolved.prim = each.prim;
        resolved.path = each.path;
        for (const UniformDef& uniform : def.uniforms) {
            resolved.uniforms.push_back(Uniform(each, uniform));
        }
        for (uint32_t port = 0; port < def.outputs.size(); ++port) {
            resolved.outputs.push_back(uint32_t(pipeline.values.size()));
            pipeline.values.push_back({position[node], port, def.outputs[port].type});
        }
        resolved.inputs.assign(def.inputs.size(), kNoValue);
        for (const graph::Edge& edge : graph[node].inputs) {
            const uint32_t from = position[edge.source];
            const uint32_t port = Port(registry[pipeline.nodes[from].def].outputs, edge.output);
            resolved.inputs[Port(def.inputs, edge.input)] = pipeline.nodes[from].outputs[port];
        }
    }
    for (const ConnectionDesc& output : desc.outputs) {
        const auto [node, port] = source(output, desc.path, output.input);
        pipeline.outputs.push_back(
            {output.input, pipeline.nodes[position[graphNode[node]]].outputs[port]});
    }
    std::ranges::sort(pipeline.outputs, {}, &ResolvedOutput::name);

    pipeline.key =
        graph::GraphShape(graph, ordering.order, [&](uint32_t node, graph::ShapeWriter& writer) {
            for (const std::string& uniform : pipeline.nodes[position[node]].uniforms) {
                writer.Add(uniform);
            }
            if (node == ordering.order.back()) {
                writer.Add(uint32_t(pipeline.outputs.size()));
                for (const ResolvedOutput& output : pipeline.outputs) {
                    writer.Add(output.name);
                    writer.Add(output.value);
                }
            }
        });
    return pipeline;
}

std::vector<SignalLayout> PropagateLayouts(const ResolvedPipeline& pipeline,
                                           const NodeRegistry& registry, DetectorLayout detector) {
    std::vector<SignalLayout> layouts(pipeline.values.size());
    for (const ResolvedNode& node : pipeline.nodes) {
        const NodeDef& def = registry[node.def];
        SignalLayout layout;
        const auto uniform = [&](std::string_view name) -> std::string_view {
            for (uint32_t index = 0; index < def.uniforms.size(); ++index) {
                if (def.uniforms[index].name == name) {
                    return node.uniforms[index];
                }
            }
            return {};
        };
        switch (def.layout) {
        case LayoutRule::FirstInput: {
            const auto first =
                std::ranges::find_if(node.inputs, [](uint32_t value) { return value != kNoValue; });
            if (first != node.inputs.end()) {
                layout = layouts[*first];
            }
            break;
        }
        case LayoutRule::Detector:
            layout.channels = detector.channels;
            layout.pattern = detector.pattern;
            layout.mosaic = detector.mosaics && uniform("layout") == "mosaic";
            if (layout.mosaic) {
                layout.channels = 1;
            }
            layout.element = uniform("precision") == "half" ? ElementType::F16 : ElementType::F32;
            break;
        case LayoutRule::FullChannels: {
            const auto first =
                std::ranges::find_if(node.inputs, [](uint32_t value) { return value != kNoValue; });
            if (first != node.inputs.end()) {
                layout = layouts[*first];
            }
            layout.channels = detector.channels;
            layout.pattern = detector.pattern;
            layout.mosaic = false;
            break;
        }
        case LayoutRule::Color:
            layout.channels = 3;
            break;
        case LayoutRule::Scalar:
            break;
        case LayoutRule::Stat:
            layout.channels = kStatStorageWords;
            layout.perCapture = true;
            break;
        }
        for (const uint32_t value : node.outputs) {
            layouts[value] = layout;
        }
    }
    return layouts;
}

}
