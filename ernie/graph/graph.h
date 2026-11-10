// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <compare>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie::graph {

inline constexpr uint32_t kNoNode = ~0u;

struct Edge {
    uint32_t input = 0;
    uint32_t source = kNoNode;
    uint32_t output = 0;
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
    void Connect(uint32_t node, uint32_t input, uint32_t source, uint32_t output);

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

class ShapeKey {
  public:
    std::string_view Bytes() const {
        return m_bytes;
    }

    uint64_t Hash() const;

    bool operator==(const ShapeKey&) const = default;
    auto operator<=>(const ShapeKey&) const = default;

  private:
    friend class ShapeWriter;

    std::string m_bytes;
};

class ShapeWriter {
  public:
    void Add(uint32_t value);
    void Add(std::string_view text);
    void Add(std::span<const std::byte> bytes);

    ShapeKey Finish() &&;

  private:
    ShapeKey m_key;
};

using NodeUniforms = std::function<void(uint32_t node, ShapeWriter& writer)>;

ShapeKey GraphShape(const Graph& graph, std::span<const uint32_t> order,
                    const NodeUniforms& uniforms = {});

}

template <> struct std::hash<ernie::graph::ShapeKey> {
    size_t operator()(const ernie::graph::ShapeKey& key) const {
        return size_t(key.Hash());
    }
};
