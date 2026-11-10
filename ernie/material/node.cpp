// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/material/node.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "ernie/graph/library.h"
#include "ernie/util/error.h"

namespace ernie::material {

namespace {

struct Nodes {
    std::map<std::string, std::shared_ptr<const MaterialNodeDefinition>, std::less<>> byName;
    ClosurePrimitiveList primitives;
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

ClosurePrimitiveList WithClosurePrimitive(ClosurePrimitiveList primitives,
                                          ClosurePrimitive primitive) {
    const auto known = std::ranges::find(primitives, primitive.name,
                                         [](const auto& entry) { return entry->name; });
    if (known == primitives.end()) {
        primitive.kind = kFirstPrimitiveLobe + uint32_t(primitives.size());
        primitives.push_back(std::make_shared<const ClosurePrimitive>(std::move(primitive)));
        return primitives;
    }
    if ((*known)->file != primitive.file) {
        throw Error("A closure primitive called {} is already loaded from {}", primitive.name,
                    (*known)->file);
    }
    if ((*known)->source != primitive.source) {
        primitive.kind = (*known)->kind;
        *known = std::make_shared<const ClosurePrimitive>(std::move(primitive));
    }
    return primitives;
}

void RegisterClosurePrimitive(ClosurePrimitive primitive) {
    Registry().Update([&](Nodes& nodes) {
        ClosurePrimitiveList primitives =
            WithClosurePrimitive(nodes.primitives, std::move(primitive));
        if (primitives != nodes.primitives) {
            nodes.primitives = std::move(primitives);
            ++nodes.version;
        }
    });
}

ClosurePrimitiveList ClosurePrimitives() {
    return Registry().Get()->primitives;
}

uint64_t MaterialNodeVersion() {
    return Registry().Get()->version;
}

}
