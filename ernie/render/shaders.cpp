// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/render/shaders.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>

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

struct LoadedFile {
    std::string digest;
    std::vector<std::string> names;
};

struct LoadedFiles {
    std::mutex mutex;
    std::map<std::filesystem::path, LoadedFile> files;
};

LoadedFiles& Loaded() {
    static LoadedFiles loaded;
    return loaded;
}

std::string ReadSource(const std::filesystem::path& file, const std::filesystem::path& requested) {
    std::ifstream stream(file, std::ios::binary);
    if (!stream) {
        throw Error("Can't read shaders from {}", requested.string());
    }
    return std::string((std::istreambuf_iterator<char>(stream)), {});
}

const bool kLoaderRegistered = [] {
    SetShaderLoader(
        [](Scene& scene, const std::filesystem::path& file) { return LoadShaders(scene, file); });
    return true;
}();

}

std::vector<std::string> LoadShaders(Scene& scene, const std::filesystem::path& requested) {
    const std::filesystem::path file = std::filesystem::weakly_canonical(requested);
    const std::string source = ReadSource(file, requested);
    const std::string digest = Digest(source);
    {
        LoadedFiles& loaded = Loaded();
        const std::lock_guard lock(loaded.mutex);
        if (const auto found = loaded.files.find(file);
            found != loaded.files.end() && found->second.digest == digest) {
            return found->second.names;
        }
    }
    const std::string module = "ernie.material.nodes.n" + digest;
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
                            .role = reflected.role == device::NodeRole::Pattern ? NodeRole::Pattern
                                    : reflected.role == device::NodeRole::Surface
                                        ? NodeRole::Surface
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
        const std::shared_ptr<const NodeDefinition> previous = FindNode(reflected.name);
        const bool reloading = previous && previous->file == file.string();
        const SchemaId schema = RegisterSchema(desc, reloading);
        if (reloading && SchemaId(previous->schema) != schema) {
            Log(LogLevel::Info, "{} in {} changed its inputs, so its prims move to the new ones",
                reflected.name, file.string());
            RetypePrims(scene, SchemaId(previous->schema), schema);
        }
        node.schema = uint32_t(schema);
        RegisterNode(std::move(node));
        Log(LogLevel::Debug, "loaded {} {} from {} with inputs {}",
            reflected.role == device::NodeRole::Pattern   ? "pattern"
            : reflected.role == device::NodeRole::Surface ? "surface"
                                                          : "material node",
            reflected.name, file.string(), inputs.empty() ? std::string("(none)") : Join(inputs));
        names.push_back(reflected.name);
    }
    if (names.empty()) {
        Log(LogLevel::Warning,
            "{} has no material nodes; a node is a struct with a "
            "Material(inout SurfaceInputs, ShadeContext) method or a function marked [Pattern]",
            file.string());
    }
    {
        LoadedFiles& loaded = Loaded();
        const std::lock_guard lock(loaded.mutex);
        loaded.files[file] = {digest, names};
    }
    return names;
}

std::vector<std::string> ReloadShaders(Scene& scene) {
    std::vector<std::string> reloaded;
    for (const std::filesystem::path& file : LoadedShaderFiles()) {
        std::string digest;
        try {
            digest = Digest(ReadSource(file, file));
        } catch (const Error& error) {
            Log(LogLevel::Warning, "{}; its nodes keep their last version", error.what());
            continue;
        }
        {
            LoadedFiles& loaded = Loaded();
            const std::lock_guard lock(loaded.mutex);
            if (loaded.files.at(file).digest == digest) {
                continue;
            }
        }
        try {
            const std::vector<std::string> names = LoadShaders(scene, file);
            reloaded.insert(reloaded.end(), names.begin(), names.end());
            Log(LogLevel::Info, "reloaded {} from {}", Join(names), file.string());
        } catch (const Error& error) {
            Log(LogLevel::Error, "reloading {} failed, so its nodes keep their last version: {}",
                file.string(), error.what());
        }
    }
    return reloaded;
}

std::vector<std::filesystem::path> LoadedShaderFiles() {
    LoadedFiles& loaded = Loaded();
    const std::lock_guard lock(loaded.mutex);
    std::vector<std::filesystem::path> files;
    for (const auto& [file, entry] : loaded.files) {
        files.push_back(file);
    }
    return files;
}

}
