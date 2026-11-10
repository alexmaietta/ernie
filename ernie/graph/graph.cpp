// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/graph/graph.h"

#include <algorithm>
#include <cstring>

#include "ernie/util/error.h"

namespace ernie::graph {

uint32_t Graph::Add(uint32_t prim, uint32_t kind) {
    m_nodes.push_back({.prim = prim, .kind = kind});
    return uint32_t(m_nodes.size() - 1);
}

void Graph::Connect(uint32_t node, uint32_t input, uint32_t source, uint32_t output) {
    if (node >= m_nodes.size() || source >= m_nodes.size()) {
        throw Error("Connecting node {} to node {} of a graph of {}", node, source, m_nodes.size());
    }
    std::vector<Edge>& inputs = m_nodes[node].inputs;
    if (std::ranges::find(inputs, input, &Edge::input) != inputs.end()) {
        throw Error("Node {}'s input {} is already connected", node, input);
    }
    inputs.push_back({input, source, output});
}

uint32_t Graph::Find(uint32_t prim) const {
    const auto found = std::ranges::find(m_nodes, prim, &Node::prim);
    return found == m_nodes.end() ? kNoNode : uint32_t(found - m_nodes.begin());
}

Ordering Graph::Order() const {
    enum class Mark : uint8_t { None, Open, Done };
    std::vector<Mark> marks(m_nodes.size(), Mark::None);
    std::vector<uint32_t> path;
    Ordering ordering;

    struct Frame {
        uint32_t node;
        uint32_t next;
    };

    std::vector<Frame> stack;
    for (uint32_t root = 0; root < m_nodes.size(); ++root) {
        if (marks[root] != Mark::None) {
            continue;
        }
        stack.push_back({root, 0});
        marks[root] = Mark::Open;
        path.push_back(root);
        while (!stack.empty()) {
            Frame& frame = stack.back();
            const std::vector<Edge>& inputs = m_nodes[frame.node].inputs;
            if (frame.next == inputs.size()) {
                marks[frame.node] = Mark::Done;
                ordering.order.push_back(frame.node);
                path.pop_back();
                stack.pop_back();
                continue;
            }
            const uint32_t source = inputs[frame.next++].source;
            if (marks[source] == Mark::Open) {
                ordering.cycle.assign(std::ranges::find(path, source), path.end());
                ordering.order.clear();
                return ordering;
            }
            if (marks[source] == Mark::None) {
                marks[source] = Mark::Open;
                path.push_back(source);
                stack.push_back({source, 0});
            }
        }
    }
    return ordering;
}

uint64_t ShapeKey::Hash() const {
    uint64_t hash = 0xcbf29ce484222325ull;
    for (const char byte : m_bytes) {
        hash = (hash ^ uint8_t(byte)) * 0x100000001b3ull;
    }
    return hash;
}

void ShapeWriter::Add(uint32_t value) {
    char bytes[sizeof(value)];
    std::memcpy(bytes, &value, sizeof(value));
    m_key.m_bytes.append(bytes, sizeof(bytes));
}

void ShapeWriter::Add(std::string_view text) {
    Add(uint32_t(text.size()));
    m_key.m_bytes.append(text);
}

void ShapeWriter::Add(std::span<const std::byte> bytes) {
    Add(uint32_t(bytes.size()));
    m_key.m_bytes.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

ShapeKey ShapeWriter::Finish() && {
    return std::move(m_key);
}

ShapeKey GraphShape(const Graph& graph, std::span<const uint32_t> order,
                    const NodeUniforms& uniforms) {
    std::vector<uint32_t> positions(graph.Nodes().size(), kNoNode);
    for (uint32_t position = 0; position < order.size(); ++position) {
        positions[order[position]] = position;
    }
    ShapeWriter writer;
    writer.Add(uint32_t(order.size()));
    for (const uint32_t node : order) {
        const Node& each = graph[node];
        writer.Add(each.kind);
        std::vector<Edge> inputs = each.inputs;
        std::ranges::sort(inputs, {}, &Edge::input);
        writer.Add(uint32_t(inputs.size()));
        for (const Edge& edge : inputs) {
            if (positions[edge.source] == kNoNode) {
                throw Error("Node {} reads node {}, which the order leaves out", node, edge.source);
            }
            writer.Add(edge.input);
            writer.Add(positions[edge.source]);
            writer.Add(edge.output);
        }
        if (uniforms) {
            uniforms(node, writer);
        }
    }
    return std::move(writer).Finish();
}

}
