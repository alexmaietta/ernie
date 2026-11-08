// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <functional>
#include <map>
#include <string>

#include <span>
#include <vector>

#include "ernie/material/definition.h"
#include "ernie/material/node.h"

namespace ernie {

inline constexpr const char* kDispatchFile = "ernie/render/kernels/material/dispatch.slang";

inline constexpr uint32_t kTextureInputBytes = 48;
inline constexpr uint32_t kMappingBytes = 64;
inline constexpr uint32_t kNoTexture = 0xFFFFFFFF;

struct NodeLayout {
    std::vector<uint32_t> textures;
    std::vector<uint32_t> cells;
    uint32_t textureCount = 0;
    uint32_t mapping = 0;
    uint32_t bytes = 0;
};

bool Texturable(NodeInputType type);

NodeLayout LayOut(const NodeDefinition& node, std::span<const uint8_t> textured);

DefinitionModule GenerateNodeDefinition(uint32_t id, const NodeDefinition& node,
                                        std::span<const uint8_t> textured);

std::map<std::string, std::string, std::less<>> DispatchSources(const DefinitionSet& set);

}
