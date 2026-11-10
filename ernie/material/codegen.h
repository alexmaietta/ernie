// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "ernie/graph/graph.h"
#include "ernie/material/node.h"

namespace ernie::material {

inline constexpr uint32_t kFirstGeneratedDefinition = 4;

struct MaterialDefinition {
    uint32_t id = 0;
    std::string type;
    std::string module;
    std::string source;
    std::vector<std::pair<std::string, std::string>> files;
    std::vector<std::pair<std::string, std::string>> origins;
    uint32_t words = 0;
    bool surface = false;
};

class MaterialDefinitionSet {
  public:
    explicit MaterialDefinitionSet(std::vector<MaterialDefinition> definitions = {},
                                   std::map<std::string, std::string, std::less<>> files = {});

    std::span<const MaterialDefinition> Definitions() const {
        return m_definitions;
    }

    const std::map<std::string, std::string, std::less<>>& Files() const {
        return m_files;
    }

    uint64_t Hash() const {
        return m_hash;
    }

    bool Contains(uint32_t id) const;

  private:
    std::vector<MaterialDefinition> m_definitions;
    std::map<std::string, std::string, std::less<>> m_files;
    uint64_t m_hash = 0;
};

using MaterialDefinitionSetRef = std::shared_ptr<const MaterialDefinitionSet>;

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
    std::shared_ptr<const MaterialNodeDefinition> node;
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

bool Texturable(MaterialInputType type);

GraphLayout LayOut(const ShadingGraph& graph);

graph::ShapeKey MaterialDefinitionKey(const ShadingGraph& graph);

MaterialDefinition GenerateMaterialDefinition(uint32_t id, const ShadingGraph& graph);

inline constexpr const char* kRegisteredFile = "ernie/render/kernels/material/registered.slang";

std::map<std::string, std::string, std::less<>>
ClosurePrimitiveSources(std::span<const std::shared_ptr<const ClosurePrimitive>> primitives);

std::map<std::string, std::string, std::less<>>
MaterialDispatchSources(const MaterialDefinitionSet& set);

}
