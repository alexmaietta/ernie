// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/graph/shape.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "ernie/util/error.h"

namespace ernie::graph {

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
        std::vector<const Edge*> inputs;
        for (const Edge& edge : each.inputs) {
            inputs.push_back(&edge);
        }
        std::ranges::sort(inputs, {}, &Edge::input);
        writer.Add(uint32_t(inputs.size()));
        for (const Edge* edge : inputs) {
            if (positions[edge->source] == kNoNode) {
                throw Error("Node {} reads node {}, which the order leaves out", node,
                            edge->source);
            }
            writer.Add(edge->input);
            writer.Add(positions[edge->source]);
            writer.Add(edge->output);
        }
        if (uniforms) {
            uniforms(node, writer);
        }
    }
    return std::move(writer).Finish();
}

}
