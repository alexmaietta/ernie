// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/graph/graph.h"

#include <algorithm>

#include "ernie/util/error.h"

namespace ernie::graph {

uint32_t Graph::Add(uint32_t prim, uint32_t kind) {
    m_nodes.push_back({.prim = prim, .kind = kind});
    return uint32_t(m_nodes.size() - 1);
}

void Graph::Connect(uint32_t node, std::string input, uint32_t source, std::string output) {
    if (node >= m_nodes.size() || source >= m_nodes.size()) {
        throw Error("Connecting node {} to node {} of a graph of {}", node, source, m_nodes.size());
    }
    std::vector<Edge>& inputs = m_nodes[node].inputs;
    const auto found = std::ranges::find(inputs, input, &Edge::input);
    if (found != inputs.end()) {
        throw Error("Node {}'s input {} is already connected", node, input);
    }
    inputs.push_back({std::move(input), source, std::move(output)});
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

}
