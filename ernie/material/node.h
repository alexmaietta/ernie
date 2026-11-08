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

enum class NodeRole { Material, Pattern };

struct NodeParameter {
    enum class Source { Input, Context, Point };

    Source source = Source::Input;
    uint32_t input = 0;
};

struct NodeDefinition {
    std::string name;
    NodeRole role = NodeRole::Material;
    uint32_t schema = 0;
    std::string file;
    std::string module;
    std::string source;
    std::vector<NodeInput> inputs;
    NodeInputType result = NodeInputType::Float;
    std::vector<NodeParameter> parameters;
};

void RegisterNode(NodeDefinition node);

std::shared_ptr<const NodeDefinition> FindNode(std::string_view name);

uint32_t NodeInputBytes(NodeInputType type);

}
