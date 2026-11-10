// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/material/node.h"

#include <map>
#include <string>
#include <vector>

#include "ernie/graph/library.h"
#include "ernie/util/error.h"

namespace ernie::material {

namespace {

struct Nodes {
    std::map<std::string, std::shared_ptr<const MaterialNodeDefinition>, std::less<>> byName;
    std::vector<std::shared_ptr<const ClosurePrimitive>> primitives;
    uint64_t version = 0;
};

graph::Published<Nodes>& Registry() {
    static graph::Published<Nodes> nodes;
    return nodes;
}

}

void RegisterMaterialNode(MaterialNodeDefinition node) {
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
        nodes.byName[name] = std::make_shared<const MaterialNodeDefinition>(std::move(node));
    });
}

std::shared_ptr<const MaterialNodeDefinition> FindMaterialNode(std::string_view name) {
    const std::shared_ptr<const Nodes> nodes = Registry().Get();
    const auto found = nodes->byName.find(name);
    return found == nodes->byName.end() ? nullptr : found->second;
}

uint32_t ClosurePrimitiveKind(std::string_view name) {
    const std::shared_ptr<const Nodes> nodes = Registry().Get();
    for (const std::shared_ptr<const ClosurePrimitive>& primitive : nodes->primitives) {
        if (primitive->name == name) {
            return primitive->kind;
        }
    }
    return kFirstPrimitiveLobe + uint32_t(nodes->primitives.size());
}

void RegisterClosurePrimitive(ClosurePrimitive primitive) {
    Registry().Update([&](Nodes& nodes) {
        for (std::shared_ptr<const ClosurePrimitive>& known : nodes.primitives) {
            if (known->name != primitive.name) {
                continue;
            }
            if (known->file != primitive.file) {
                throw Error("A closure primitive called {} is already loaded from {}",
                            primitive.name, known->file);
            }
            if (known->source != primitive.source) {
                primitive.kind = known->kind;
                known = std::make_shared<const ClosurePrimitive>(std::move(primitive));
                ++nodes.version;
            }
            return;
        }
        primitive.kind = kFirstPrimitiveLobe + uint32_t(nodes.primitives.size());
        nodes.primitives.push_back(std::make_shared<const ClosurePrimitive>(std::move(primitive)));
        ++nodes.version;
    });
}

std::vector<std::shared_ptr<const ClosurePrimitive>> ClosurePrimitives() {
    return Registry().Get()->primitives;
}

uint64_t MaterialNodeVersion() {
    return Registry().Get()->version;
}

uint32_t MaterialInputBytes(MaterialInputType type) {
    switch (type) {
    case MaterialInputType::Float2:
        return 8;
    case MaterialInputType::Float3:
        return 12;
    case MaterialInputType::Float4:
    case MaterialInputType::Spectral:
        return 16;
    default:
        return 4;
    }
}

}
