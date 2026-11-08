// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/render/shaders.h"

#include <cstring>
#include <fstream>
#include <iterator>

#include "ernie/device/slang/reflection.h"
#include "ernie/material/node.h"
#include "ernie/render/kernels.h"
#include "ernie/scene/schema/runtime.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/text.h"

namespace ernie {

namespace {

template <class T> T Read(const std::vector<std::byte>& bytes) {
    T value{};
    std::memcpy(&value, bytes.data(), std::min(bytes.size(), sizeof(T)));
    return value;
}

RuntimeColumnDesc Column(const device::ReflectedField& field) {
    RuntimeColumnDesc column{.field = field.name,
                             .role = field.color ? ColumnRole::Color : ColumnRole::None,
                             .displayName = field.displayName,
                             .group = "Inputs",
                             .minimum = field.minimum,
                             .maximum = field.maximum};
    switch (field.type) {
    case device::FieldType::Bool:
        column.type = ColumnType::Bool;
        column.fallback = Read<uint32_t>(field.fallback) != 0;
        break;
    case device::FieldType::Int:
        column.type = ColumnType::Int;
        column.fallback = Read<int32_t>(field.fallback);
        break;
    case device::FieldType::Float:
        column.type = ColumnType::Float;
        column.fallback = Read<float>(field.fallback);
        break;
    case device::FieldType::Float2:
        column.type = ColumnType::Float2;
        column.fallback = Read<Float2>(field.fallback);
        break;
    case device::FieldType::Float3:
        column.type = ColumnType::Float3;
        column.fallback = Read<Float3>(field.fallback);
        break;
    case device::FieldType::Float4:
        column.type = ColumnType::Float4;
        column.fallback = Read<Float4>(field.fallback);
        break;
    }
    return column;
}

NodeInputType InputType(device::FieldType type) {
    switch (type) {
    case device::FieldType::Bool:
        return NodeInputType::Bool;
    case device::FieldType::Int:
        return NodeInputType::Int;
    case device::FieldType::Float2:
        return NodeInputType::Float2;
    case device::FieldType::Float3:
        return NodeInputType::Float3;
    case device::FieldType::Float4:
        return NodeInputType::Float4;
    default:
        return NodeInputType::Float;
    }
}

std::string Replaced(std::string text, std::string_view from, std::string_view to) {
    for (size_t at = text.find(from); at != std::string::npos; at = text.find(from, at)) {
        text.replace(at, from.size(), to);
        at += to.size();
    }
    return text;
}

const bool kLoaderRegistered = [] {
    SetShaderLoader(
        [](Scene& scene, const std::filesystem::path& file) { return LoadShaders(scene, file); });
    return true;
}();

}

std::vector<std::string> LoadShaders(Scene& scene, const std::filesystem::path& requested) {
    const std::filesystem::path file = std::filesystem::weakly_canonical(requested);
    std::ifstream stream(file, std::ios::binary);
    if (!stream) {
        throw Error("Can't read shaders from {}", requested.string());
    }
    const std::string source((std::istreambuf_iterator<char>(stream)), {});
    const std::string module = "ernie.material.nodes.n" + Digest(source);
    std::string path = module;
    std::replace(path.begin(), path.end(), '.', '/');
    path += ".slang";
    std::vector<device::ReflectedNode> nodes;
    try {
        nodes = device::ReflectMaterialNodes(scene.GetDevice().Adapter().backend, &kRenderKernels,
                                             {{path, source}}, module, file.string());
    } catch (const Error& error) {
        std::string message = Replaced(error.what(), "./" + path, file.string());
        message =
            Replaced(Replaced(message, path, file.string()), module + ".slang", file.string());
        throw Error("{}", message);
    }
    std::vector<std::string> names;
    for (const device::ReflectedNode& reflected : nodes) {
        RuntimeSchemaDesc desc{
            .name = reflected.name,
            .base = SchemaId::Shader,
            .doc = "A material node from " + file.string() + ".",
            .inherited = {{"implementationSource", int32_t(ImplementationSource::SourceAsset)},
                          {"slangSourceAsset", file.string()},
                          {"slangSourceAssetSubIdentifier", reflected.name}}};
        NodeDefinition node{.name = reflected.name,
                            .role = reflected.role == device::NodeRole::Pattern
                                        ? NodeRole::Pattern
                                        : NodeRole::Material,
                            .file = file.string(),
                            .module = module,
                            .source = source,
                            .result = InputType(reflected.result)};
        for (const device::ReflectedParameter& parameter : reflected.parameters) {
            node.parameters.push_back(
                {.source = parameter.source == device::ParameterSource::Context
                               ? NodeParameter::Source::Context
                           : parameter.source == device::ParameterSource::Point
                               ? NodeParameter::Source::Point
                               : NodeParameter::Source::Input,
                 .input = parameter.input});
        }
        std::vector<std::string> inputs;
        for (const device::ReflectedField& field : reflected.fields) {
            desc.columns.push_back(Column(field));
            node.inputs.push_back(
                {.name = field.name, .type = InputType(field.type), .color = field.color});
            inputs.push_back(field.name);
        }
        node.schema = uint32_t(RegisterSchema(desc));
        RegisterNode(std::move(node));
        Log(LogLevel::Debug, "loaded {} {} from {} with inputs {}",
            reflected.role == device::NodeRole::Pattern ? "pattern" : "material node",
            reflected.name, file.string(), inputs.empty() ? std::string("(none)") : Join(inputs));
        names.push_back(reflected.name);
    }
    if (names.empty()) {
        Log(LogLevel::Warning,
            "{} has no material nodes; a node is a struct with a "
            "Material(inout SurfaceInputs, ShadeContext) method or a function marked [Pattern]",
            file.string());
    }
    return names;
}

}
