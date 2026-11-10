// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/sensor_pipeline/resolve.h"

#include <algorithm>
#include <unordered_map>

#include "ernie/util/error.h"
#include "ernie/util/text.h"

namespace ernie::sensor_pipeline {

namespace {

uint32_t Port(std::span<const PortDef> ports, std::string_view name) {
    const auto found = std::ranges::find(ports, name, &PortDef::name);
    return found == ports.end() ? kNoValue : uint32_t(found - ports.begin());
}

uint32_t ChooseUniform(const NodeDesc& node, const UniformDef& uniform) {
    const auto set =
        std::ranges::find(node.uniforms, uniform.name, &std::pair<std::string, std::string>::first);
    const std::string& value = set == node.uniforms.end() ? uniform.fallback : set->second;
    const auto found = std::ranges::find(uniform.allowed, value);
    if (found == uniform.allowed.end()) {
        throw Error("{}'s {} is {}, and it takes {}", node.path, uniform.name, value,
                    Join(uniform.allowed));
    }
    return uint32_t(found - uniform.allowed.begin());
}

std::vector<float> ParameterValues(const NodeDesc& node, const ParameterDef& parameter) {
    const auto set = std::ranges::find(node.values, parameter.name,
                                       &std::pair<std::string, std::vector<float>>::first);
    std::vector<float> values = set != node.values.end() ? set->second
                                : parameter.fallback.empty()
                                    ? std::vector<float>(parameter.words, 0)
                                    : parameter.fallback;
    if (values.size() != parameter.words) {
        throw Error("{}'s {} has {} values, and it takes {}", node.path, parameter.name,
                    values.size(), parameter.words);
    }
    return values;
}

std::string_view UniformValue(const NodeDef& def, const ResolvedNode& node, std::string_view name) {
    for (uint32_t index = 0; index < def.uniforms.size(); ++index) {
        if (def.uniforms[index].name == name) {
            return def.uniforms[index].allowed[node.uniforms[index]];
        }
    }
    return {};
}

class Resolver {
  public:
    Resolver(const SensorPipelineDesc& desc, const NodeRegistry& registry)
        : m_desc(desc), m_registry(registry) {
        for (uint32_t index = 0; index < desc.nodes.size(); ++index) {
            const NodeDesc& node = desc.nodes[index];
            const uint32_t def = registry.Find(node.id);
            if (def == kNoDef) {
                throw Error("{} runs {}, which no pipeline node is", node.path, node.id);
            }
            m_byPrim.emplace(node.prim, index);
            m_defs.push_back(def);
        }
    }

    ResolvedSensorPipeline Resolve() {
        if (m_desc.outputs.empty()) {
            throw Error("{} has no outputs", m_desc.path);
        }
        std::vector<Source> outputs;
        for (const ConnectionDesc& output : m_desc.outputs) {
            outputs.push_back(FindSource(output, m_desc.path, output.input));
        }
        const std::vector<uint32_t> nodes = ReachedFrom(outputs);
        const graph::Graph graph = Connect(nodes);
        const std::vector<uint32_t> order = Order(graph, nodes);

        ResolvedSensorPipeline pipeline;
        std::vector<uint32_t> position(graph.Nodes().size());
        for (const uint32_t node : order) {
            position[node] = uint32_t(pipeline.nodes.size());
            AddNode(pipeline, nodes[node], graph[node], position);
        }
        for (uint32_t index = 0; index < outputs.size(); ++index) {
            const ResolvedNode& made = pipeline.nodes[position[m_graphNode[outputs[index].node]]];
            pipeline.outputs.push_back(
                {m_desc.outputs[index].input, made.outputs[outputs[index].port]});
        }
        std::ranges::sort(pipeline.outputs, {}, &ResolvedOutput::name);
        pipeline.key = Key(graph, order, pipeline, position);
        return pipeline;
    }

  private:
    struct Source {
        uint32_t node;
        uint32_t port;
    };

    const NodeDef& Def(uint32_t node) const {
        return m_registry[m_defs[node]];
    }

    Source FindSource(const ConnectionDesc& connection, std::string_view reader,
                      std::string_view input) const {
        const auto found = m_byPrim.find(connection.source);
        if (found == m_byPrim.end()) {
            throw Error("{}'s {} connects to {}, which isn't a node of {}", reader, input,
                        connection.sourcePath, m_desc.path);
        }
        const NodeDef& def = Def(found->second);
        const uint32_t port = Port(def.outputs, connection.output);
        if (port == kNoValue) {
            throw Error("{}'s {} reads {}'s {}, which {} doesn't output", reader, input,
                        connection.sourcePath, connection.output, def.id);
        }
        return {found->second, port};
    }

    std::vector<uint32_t> ReachedFrom(std::span<const Source> outputs) const {
        std::vector<bool> reached(m_desc.nodes.size(), false);
        std::vector<uint32_t> pending;
        const auto reach = [&](uint32_t node) {
            if (!reached[node]) {
                reached[node] = true;
                pending.push_back(node);
            }
        };
        for (const Source& output : outputs) {
            reach(output.node);
        }
        while (!pending.empty()) {
            const uint32_t node = pending.back();
            pending.pop_back();
            for (const ConnectionDesc& connection : m_desc.nodes[node].inputs) {
                if (const auto found = m_byPrim.find(connection.source); found != m_byPrim.end()) {
                    reach(found->second);
                }
            }
        }
        std::vector<uint32_t> nodes;
        for (uint32_t node = 0; node < reached.size(); ++node) {
            if (reached[node]) {
                nodes.push_back(node);
            }
        }
        return nodes;
    }

    graph::Graph Connect(std::span<const uint32_t> nodes) {
        graph::Graph graph;
        m_graphNode.assign(m_desc.nodes.size(), graph::kNoNode);
        for (const uint32_t node : nodes) {
            m_graphNode[node] = graph.Add(m_defs[node]);
        }
        for (const uint32_t node : nodes) {
            const NodeDesc& desc = m_desc.nodes[node];
            const NodeDef& def = Def(node);
            std::vector<bool> connected(def.inputs.size(), false);
            for (const ConnectionDesc& connection : desc.inputs) {
                const uint32_t input = Port(def.inputs, connection.input);
                if (input == kNoValue) {
                    throw Error("{}'s {} is connected, but {} has no such signal input", desc.path,
                                connection.input, def.id);
                }
                const Source source = FindSource(connection, desc.path, connection.input);
                const SignalType made = Def(source.node).outputs[source.port].type;
                const SignalType taken = def.inputs[input].type;
                if (made != taken) {
                    throw Error("{}'s {} takes {}, but {}'s {} is {}", desc.path, connection.input,
                                Signal(taken).name, connection.sourcePath, connection.output,
                                Signal(made).name);
                }
                graph.Connect(m_graphNode[node], input, m_graphNode[source.node], source.port);
                connected[input] = true;
            }
            for (uint32_t input = 0; input < def.inputs.size(); ++input) {
                if (!connected[input] && !def.inputs[input].optional) {
                    throw Error("{}'s {} isn't connected, and {} needs it", desc.path,
                                def.inputs[input].name, def.id);
                }
            }
        }
        return graph;
    }

    std::vector<uint32_t> Order(const graph::Graph& graph, std::span<const uint32_t> nodes) const {
        graph::Ordering ordering = graph.Order();
        if (!ordering.cycle.empty()) {
            std::vector<std::string> paths;
            for (const uint32_t node : ordering.cycle) {
                paths.push_back(m_desc.nodes[nodes[node]].path);
            }
            throw Error("{} loops through {}", m_desc.path, Join(paths));
        }
        return std::move(ordering.order);
    }

    void AddNode(ResolvedSensorPipeline& pipeline, uint32_t node, const graph::Node& connected,
                 std::span<const uint32_t> position) const {
        const NodeDesc& desc = m_desc.nodes[node];
        const NodeDef& def = Def(node);
        const auto at = uint32_t(pipeline.nodes.size());
        ResolvedNode& resolved = pipeline.nodes.emplace_back();
        resolved.def = m_defs[node];
        resolved.prim = desc.prim;
        resolved.path = desc.path;
        for (const UniformDef& uniform : def.uniforms) {
            resolved.uniforms.push_back(ChooseUniform(desc, uniform));
        }
        for (const ParameterDef& parameter : def.parameters) {
            const std::vector<float> values = ParameterValues(desc, parameter);
            resolved.parameters.insert(resolved.parameters.end(), values.begin(), values.end());
        }
        for (uint32_t port = 0; port < def.outputs.size(); ++port) {
            resolved.outputs.push_back(uint32_t(pipeline.values.size()));
            pipeline.values.push_back({at, port, def.outputs[port].type});
        }
        if (def.kind == NodeKind::Global) {
            for (uint32_t pass = 0; pass + 1 < def.passes; ++pass) {
                resolved.scratch.push_back(uint32_t(pipeline.values.size()));
                pipeline.values.push_back(
                    {at, uint32_t(def.outputs.size()) + pass, def.outputs[0].type});
            }
        }
        resolved.inputs.assign(def.inputs.size(), kNoValue);
        for (const graph::Edge& edge : connected.inputs) {
            resolved.inputs[edge.input] =
                pipeline.nodes[position[edge.source]].outputs[edge.output];
        }
    }

    graph::ShapeKey Key(const graph::Graph& graph, std::span<const uint32_t> order,
                        const ResolvedSensorPipeline& pipeline,
                        std::span<const uint32_t> position) const {
        return graph::GraphShape(graph, order, [&](uint32_t node, graph::ShapeWriter& writer) {
            const ResolvedNode& resolved = pipeline.nodes[position[node]];
            writer.Add(m_registry[resolved.def].code.version);
            for (const uint32_t uniform : resolved.uniforms) {
                writer.Add(uniform);
            }
            if (node == order.back()) {
                writer.Add(uint32_t(pipeline.outputs.size()));
                for (const ResolvedOutput& output : pipeline.outputs) {
                    writer.Add(output.name);
                    writer.Add(output.value);
                }
            }
        });
    }

    const SensorPipelineDesc& m_desc;
    const NodeRegistry& m_registry;
    std::unordered_map<uint32_t, uint32_t> m_byPrim;
    std::vector<uint32_t> m_defs;
    std::vector<uint32_t> m_graphNode;
};

}

ResolvedSensorPipeline ResolveSensorPipeline(const SensorPipelineDesc& desc,
                                             const NodeRegistry& registry) {
    return Resolver(desc, registry).Resolve();
}

std::vector<SignalLayout> PropagateLayouts(const ResolvedSensorPipeline& pipeline,
                                           const NodeRegistry& registry, DetectorLayout detector) {
    std::vector<SignalLayout> layouts(pipeline.values.size());
    for (const ResolvedNode& node : pipeline.nodes) {
        const NodeDef& def = registry[node.def];
        const auto first =
            std::ranges::find_if(node.inputs, [](uint32_t value) { return value != kNoValue; });
        SignalLayout layout = first != node.inputs.end() ? layouts[*first] : SignalLayout{};
        switch (def.layout) {
        case LayoutRule::FirstInput:
            break;
        case LayoutRule::Detector:
            layout = SignalLayout{.channels = detector.channels, .pattern = detector.pattern};
            layout.mosaic = detector.mosaics && UniformValue(def, node, "layout") == "mosaic";
            if (layout.mosaic) {
                layout.channels = 1;
            }
            layout.element = UniformValue(def, node, "precision") == "half" ? ElementType::F16
                                                                            : ElementType::F32;
            break;
        case LayoutRule::FullChannels:
            layout.channels = detector.channels;
            layout.pattern = detector.pattern;
            layout.mosaic = false;
            break;
        case LayoutRule::Color:
            layout = SignalLayout{.channels = 3};
            break;
        case LayoutRule::Stat:
            layout = SignalLayout{.channels = kStatStorageWords, .perCapture = true};
            break;
        case LayoutRule::Returns:
            layout = SignalLayout{.channels = kReturnsChannels};
            break;
        }
        for (const uint32_t value : node.outputs) {
            layouts[value] = layout;
        }
        for (const uint32_t value : node.scratch) {
            layouts[value] = layout;
        }
    }
    return layouts;
}

}
