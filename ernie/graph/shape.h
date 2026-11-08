// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <compare>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>

#include "ernie/graph/graph.h"

namespace ernie::graph {

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
