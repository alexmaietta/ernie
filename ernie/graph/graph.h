// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ernie::graph {

inline constexpr uint32_t kNoNode = ~0u;

struct Edge {
    std::string input;
    uint32_t source = kNoNode;
    std::string output;
};

struct Node {
    uint32_t prim = 0;
    uint32_t kind = 0;
    std::vector<Edge> inputs;
};

struct Ordering {
    std::vector<uint32_t> order;
    std::vector<uint32_t> cycle;
};

class Graph {
  public:
    uint32_t Add(uint32_t prim, uint32_t kind);
    void Connect(uint32_t node, std::string input, uint32_t source, std::string output);

    std::span<const Node> Nodes() const {
        return m_nodes;
    }

    const Node& operator[](uint32_t node) const {
        return m_nodes[node];
    }

    uint32_t Find(uint32_t prim) const;
    Ordering Order() const;

  private:
    std::vector<Node> m_nodes;
};

}
