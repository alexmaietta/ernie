// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/material/node.h"

#include <atomic>
#include <map>
#include <mutex>

#include "ernie/util/error.h"

namespace ernie {

namespace {

struct Nodes {
    std::mutex mutex;
    std::map<std::string, std::shared_ptr<const NodeDefinition>, std::less<>> byName;
    std::atomic<uint64_t> version{0};
};

Nodes& Registry() {
    static Nodes nodes;
    return nodes;
}

}

void RegisterNode(NodeDefinition node) {
    Nodes& nodes = Registry();
    const std::lock_guard lock(nodes.mutex);
    const auto found = nodes.byName.find(node.name);
    if (found != nodes.byName.end() && found->second->file != node.file) {
        throw Error("A material node called {} is already loaded from {}", node.name,
                    found->second->file);
    }
    if (found != nodes.byName.end() && found->second->source == node.source &&
        found->second->schema == node.schema) {
        return;
    }
    const std::string name = node.name;
    nodes.byName[name] = std::make_shared<const NodeDefinition>(std::move(node));
    if (found != nodes.byName.end()) {
        ++nodes.version;
    }
}

std::shared_ptr<const NodeDefinition> FindNode(std::string_view name) {
    Nodes& nodes = Registry();
    const std::lock_guard lock(nodes.mutex);
    const auto found = nodes.byName.find(name);
    return found == nodes.byName.end() ? nullptr : found->second;
}

uint64_t NodeVersion() {
    return Registry().version.load();
}

uint32_t NodeInputBytes(NodeInputType type) {
    switch (type) {
    case NodeInputType::Float2:
        return 8;
    case NodeInputType::Float3:
        return 12;
    case NodeInputType::Float4:
        return 16;
    default:
        return 4;
    }
}

}
