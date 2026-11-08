// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

enum class NodeInputType { Bool, Int, Float, Float2, Float3, Float4 };

struct NodeInput {
    std::string name;
    NodeInputType type = NodeInputType::Float;
    bool color = false;
};

struct NodeDefinition {
    std::string name;
    uint32_t schema = 0;
    std::string file;
    std::string module;
    std::string source;
    std::vector<NodeInput> inputs;
};

void RegisterNode(NodeDefinition node);

std::shared_ptr<const NodeDefinition> FindNode(std::string_view name);

uint32_t NodeInputBytes(NodeInputType type);

}
