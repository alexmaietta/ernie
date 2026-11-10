// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ernie::material {

enum class MaterialInputType { Bool, Int, Float, Float2, Float3, Float4, Spectral };

struct MaterialNodeInput {
    std::string name;
    MaterialInputType type = MaterialInputType::Float;
    bool color = false;
};

enum class MaterialNodeRole { Material, Surface, Pattern };

struct MaterialNodeParameter {
    enum class Source { Input, Context, Point };

    Source source = Source::Input;
    uint32_t input = 0;
};

struct MaterialNodeDefinition {
    std::string name;
    MaterialNodeRole role = MaterialNodeRole::Material;
    uint32_t schema = 0;
    std::string file;
    std::string module;
    std::string source;
    std::vector<MaterialNodeInput> inputs;
    MaterialInputType result = MaterialInputType::Float;
    std::vector<MaterialNodeParameter> parameters;
};

void RegisterMaterialNode(MaterialNodeDefinition node);

std::shared_ptr<const MaterialNodeDefinition> FindMaterialNode(std::string_view name);

uint64_t MaterialNodeVersion();

inline constexpr uint32_t kFirstPrimitiveLobe = 3;

struct ClosurePrimitive {
    std::string name;
    uint32_t kind = 0;
    std::string file;
    std::string module;
    std::string source;
};

uint32_t ClosurePrimitiveKind(std::string_view name);

void RegisterClosurePrimitive(ClosurePrimitive primitive);

std::vector<std::shared_ptr<const ClosurePrimitive>> ClosurePrimitives();

uint32_t MaterialInputBytes(MaterialInputType type);

}
