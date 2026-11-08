// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ernie/pipeline/signal.h"

namespace ernie::pipeline {

enum class NodeKind : uint8_t {
    Source,
    Point,
    Stencil,
    Reduce,
    Global,
    Compact,
    Host,
};

enum class LayoutRule : uint8_t {
    FirstInput,
    Detector,
    Color,
    Scalar,
};

struct PortDef {
    std::string name;
    SignalType type = SignalType::ElectronRate;
    bool optional = false;
};

struct ParameterDef {
    std::string name;
    uint32_t words = 1;
};

struct UniformDef {
    std::string name;
    std::vector<std::string> allowed;
    std::string fallback;
};

struct NodeDef {
    std::string id;
    NodeKind kind = NodeKind::Point;
    std::vector<PortDef> inputs;
    std::vector<PortDef> outputs;
    std::vector<ParameterDef> parameters;
    std::vector<UniformDef> uniforms;
    uint32_t radius = 0;
    LayoutRule layout = LayoutRule::FirstInput;
    std::string module;
    std::string type;
};

class NodeRegistry {
  public:
    uint32_t Add(NodeDef def);

    const NodeDef& operator[](uint32_t index) const {
        return m_defs[index];
    }

    uint32_t Find(std::string_view id) const;

    uint32_t Size() const {
        return uint32_t(m_defs.size());
    }

  private:
    std::deque<NodeDef> m_defs;
    std::unordered_map<std::string, uint32_t> m_ids;
};

inline constexpr uint32_t kNoDef = ~0u;

}
