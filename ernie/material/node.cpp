// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/material/node.h"

#include <map>
#include <string>
#include <vector>

#include "ernie/graph/published.h"
#include "ernie/util/error.h"

namespace ernie {

namespace {

struct Nodes {
    std::map<std::string, std::shared_ptr<const NodeDefinition>, std::less<>> byName;
    std::vector<std::shared_ptr<const PrimitiveDefinition>> primitives;
    uint64_t version = 0;
};

graph::Published<Nodes>& Registry() {
    static graph::Published<Nodes> nodes;
    return nodes;
}

}

void RegisterNode(NodeDefinition node) {
    Registry().Update([&](Nodes& nodes) {
        const auto found = nodes.byName.find(node.name);
        if (found != nodes.byName.end() && found->second->file != node.file) {
            throw Error("A material node called {} is already loaded from {}", node.name,
                        found->second->file);
        }
        if (found != nodes.byName.end() && found->second->source == node.source &&
            found->second->schema == node.schema) {
            return;
        }
        nodes.version += found != nodes.byName.end() ? 1 : 0;
        const std::string name = node.name;
        nodes.byName[name] = std::make_shared<const NodeDefinition>(std::move(node));
    });
}

std::shared_ptr<const NodeDefinition> FindNode(std::string_view name) {
    const std::shared_ptr<const Nodes> nodes = Registry().Get();
    const auto found = nodes->byName.find(name);
    return found == nodes->byName.end() ? nullptr : found->second;
}

uint32_t PrimitiveKind(std::string_view name) {
    const std::shared_ptr<const Nodes> nodes = Registry().Get();
    for (const std::shared_ptr<const PrimitiveDefinition>& primitive : nodes->primitives) {
        if (primitive->name == name) {
            return primitive->kind;
        }
    }
    return kFirstPrimitiveLobe + uint32_t(nodes->primitives.size());
}

void RegisterPrimitive(PrimitiveDefinition primitive) {
    Registry().Update([&](Nodes& nodes) {
        for (std::shared_ptr<const PrimitiveDefinition>& known : nodes.primitives) {
            if (known->name != primitive.name) {
                continue;
            }
            if (known->file != primitive.file) {
                throw Error("A closure primitive called {} is already loaded from {}",
                            primitive.name, known->file);
            }
            if (known->source != primitive.source) {
                primitive.kind = known->kind;
                known = std::make_shared<const PrimitiveDefinition>(std::move(primitive));
                ++nodes.version;
            }
            return;
        }
        primitive.kind = kFirstPrimitiveLobe + uint32_t(nodes.primitives.size());
        nodes.primitives.push_back(
            std::make_shared<const PrimitiveDefinition>(std::move(primitive)));
        ++nodes.version;
    });
}

std::vector<std::shared_ptr<const PrimitiveDefinition>> Primitives() {
    return Registry().Get()->primitives;
}

uint64_t NodeVersion() {
    return Registry().Get()->version;
}

uint32_t NodeInputBytes(NodeInputType type) {
    switch (type) {
    case NodeInputType::Float2:
        return 8;
    case NodeInputType::Float3:
        return 12;
    case NodeInputType::Float4:
    case NodeInputType::Spectral:
        return 16;
    default:
        return 4;
    }
}

}
