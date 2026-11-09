// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>

#include <span>
#include <vector>

#include "ernie/graph/shape.h"
#include "ernie/material/definition.h"
#include "ernie/material/node.h"

namespace ernie {

inline constexpr const char* kDispatchFile = "ernie/render/kernels/material/dispatch.slang";

inline constexpr uint32_t kTextureInputBytes = 48;
inline constexpr uint32_t kMappingBytes = 64;
inline constexpr uint32_t kNoTexture = 0xFFFFFFFF;

enum class BindingKind : uint8_t { Value, Texture, Pattern };

struct InputBinding {
    BindingKind kind = BindingKind::Value;
    uint32_t source = 0;
};

struct ShadingNode {
    std::shared_ptr<const NodeDefinition> node;
    std::vector<InputBinding> inputs;
};

struct ShadingGraph {
    std::vector<ShadingNode> nodes;
};

struct GraphLayout {
    std::vector<std::vector<uint32_t>> cells;
    std::vector<std::vector<uint32_t>> textures;
    uint32_t textureCount = 0;
    uint32_t mapping = 0;
    uint32_t bytes = 0;
};

bool Texturable(NodeInputType type);

GraphLayout LayOut(const ShadingGraph& graph);

graph::ShapeKey DefinitionKey(const ShadingGraph& graph);

DefinitionModule GenerateDefinition(uint32_t id, const ShadingGraph& graph);

inline constexpr const char* kRegisteredFile = "ernie/render/kernels/material/registered.slang";

std::map<std::string, std::string, std::less<>>
PrimitiveSources(std::span<const std::shared_ptr<const PrimitiveDefinition>> primitives);

std::map<std::string, std::string, std::less<>> DispatchSources(const DefinitionSet& set);

}
