// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/device/adapter.h"
#include "ernie/device/slang/compiler.h"

namespace ernie::device {

enum class FieldType { Bool, Int, Float, Float2, Float3, Float4, Spectral };

struct ReflectedField {
    std::string name;
    FieldType type = FieldType::Float;
    std::vector<std::byte> fallback;
    bool color = false;
    float minimum = 0;
    float maximum = 0;
    std::string displayName;
};

enum class NodeRole { Material, Surface, Pattern, Primitive };

enum class ParameterSource { Input, Context, Point };

struct ReflectedParameter {
    ParameterSource source = ParameterSource::Input;
    uint32_t input = 0;
};

struct ReflectedNode {
    std::string name;
    NodeRole role = NodeRole::Material;
    std::vector<ReflectedField> fields;
    FieldType result = FieldType::Float;
    std::vector<ReflectedParameter> parameters;
};

enum class SensorPipelineNodeKind { Point, Stencil };

struct ReflectedPort {
    std::string name;
    std::string type;
};

inline constexpr uint32_t kNoContext = ~0u;

struct ReflectedSensorPipelineNode {
    std::string name;
    std::string displayName;
    SensorPipelineNodeKind kind = SensorPipelineNodeKind::Point;
    uint32_t radius = 0;
    std::vector<ReflectedField> fields;
    std::vector<ReflectedPort> inputs;
    uint32_t context = kNoContext;
    std::string result;
};

struct ReflectedNodes {
    std::vector<ReflectedNode> material;
    std::vector<ReflectedSensorPipelineNode> pipeline;
};

ERNIE_API ReflectedNodes ReflectNodes(Backend backend, const KernelLibrary* library,
                                      const SourceFiles& sources, std::string_view module,
                                      std::string_view label);

ERNIE_API std::vector<ReflectedSensorPipelineNode>
ReflectSensorPipelineNodes(Backend backend, const KernelLibrary* library,
                           const SourceFiles& sources, std::string_view module,
                           std::string_view label);

ERNIE_API std::vector<ReflectedNode>
ReflectMaterialNodes(Backend backend, const KernelLibrary* library, const SourceFiles& sources,
                     std::string_view module, std::string_view label);

}
