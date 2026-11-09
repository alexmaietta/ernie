// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/render/shaders.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>

#include "ernie/device/slang/reflection.h"
#include "ernie/material/codegen.h"
#include "ernie/material/node.h"
#include "ernie/pipeline/builtins.h"
#include "ernie/render/kernels.h"
#include "ernie/stage/schema/runtime.h"
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
    case device::FieldType::Spectral:
        column.type = ColumnType::Float3;
        column.fallback = Read<Float3>(field.fallback);
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
    case device::FieldType::Spectral:
        return NodeInputType::Spectral;
    default:
        return NodeInputType::Float;
    }
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
    const std::string module = "ernie.nodes.n" + digest;
    const std::string path = ModulePath(module);
    const std::pair<std::string, std::string> origins[] = {{path, file.string()},
                                                           {module + ".slang", file.string()}};
    const auto mapped = [&](const Error& error) {
        return Error("{}", MapOrigins(error.what(), origins));
    };
    const device::Backend backend = scene.GetDevice().Adapter().backend;
    auto sources = PrimitiveSources(Primitives());
    sources[path] = source;
    device::ReflectedNodes reflection;
    try {
        reflection = device::ReflectNodes(backend, &kRenderKernels,
                                          {sources.begin(), sources.end()}, module, file.string());
    } catch (const Error& error) {
        throw mapped(error);
    }
    if (!reflection.pipeline.empty() && !reflection.material.empty()) {
        throw Error("{} defines both pipeline nodes ({}) and material nodes ({}); put them in "
                    "separate files",
                    file.string(), reflection.pipeline.front().name,
                    reflection.material.front().name);
    }
    if (!reflection.pipeline.empty()) {
        const std::vector<device::ReflectedPipelineNode>& reflected = reflection.pipeline;
        const auto signal = [&](const std::string& node, const std::string& type) {
            const auto found = std::ranges::find(pipeline::kSignalTypes, type);
            if (found == std::end(pipeline::kSignalTypes)) {
                throw Error(
                    "{} in {} uses {}, which isn't a pipeline signal; signals are {}", node,
                    file.string(), type,
                    Join({std::begin(pipeline::kSignalTypes), std::end(pipeline::kSignalTypes)}));
            }
            return pipeline::SignalType(found - std::begin(pipeline::kSignalTypes));
        };
        for (const device::ReflectedPipelineNode& node : reflected) {
            signal(node.name, node.result);
            for (const device::ReflectedPort& port : node.inputs) {
                signal(node.name, port.type);
            }
        }
        std::vector<std::string> names;
        for (const device::ReflectedPipelineNode& node : reflected) {
            const pipeline::SignalType result = signal(node.name, node.result);
            pipeline::NodeDef def{
                .id = node.name,
                .kind = node.kind == device::PipelineNodeKind::Stencil ? pipeline::NodeKind::Stencil
                                                                       : pipeline::NodeKind::Point,
                .outputs = {{std::string(pipeline::kSignalPortNames[uint32_t(result)]), result}},
                .radius = node.radius,
                .module = module,
                .type = node.name,
                .file = file.string(),
                .source = source,
                .version = digest,
                .applies = true,
                .context = node.context};
            for (const device::ReflectedPort& port : node.inputs) {
                def.inputs.push_back({port.name, signal(node.name, port.type)});
            }
            RuntimeSchemaDesc desc{.name = node.name,
                                   .base = SchemaId::SensorNode,
                                   .doc = "A sensor pipeline node from " + file.string() + ".",
                                   .inherited = {{"id", node.name}}};
            std::vector<std::string> inputs;
            for (const device::ReflectedField& field : node.fields) {
                RuntimeColumnDesc column = Column(field);
                column.group = node.displayName.empty() ? node.name : node.displayName;
                desc.columns.push_back(std::move(column));
                pipeline::ParameterDef parameter{.name = field.name};
                switch (field.type) {
                case device::FieldType::Bool:
                    parameter.type = pipeline::ParameterType::Bool;
                    parameter.fallback = {Read<uint32_t>(field.fallback) != 0 ? 1.0f : 0.0f};
                    break;
                case device::FieldType::Int:
                    parameter.type = pipeline::ParameterType::Int;
                    parameter.fallback = {float(Read<int32_t>(field.fallback))};
                    break;
                default:
                    parameter.words = uint32_t(field.fallback.size() / sizeof(float));
                    parameter.fallback.resize(parameter.words);
                    std::memcpy(parameter.fallback.data(), field.fallback.data(),
                                field.fallback.size());
                    break;
                }
                def.parameters.push_back(std::move(parameter));
                inputs.push_back(field.name);
            }
            const std::shared_ptr<const pipeline::NodeRegistry> known = pipeline::PipelineNodes();
            const uint32_t found = known->Find(node.name);
            if (found != pipeline::kNoDef && (*known)[found].file.empty()) {
                throw Error("{} in {} has the name of a built-in pipeline node; rename it",
                            node.name, file.string());
            }
            const SchemaId previous = FindSchemaByName(node.name);
            const bool reloading = found != pipeline::kNoDef && previous != SchemaId::None;
            const SchemaId schema = RegisterSchema(desc, reloading);
            if (reloading && previous != schema) {
                Log(LogLevel::Info,
                    "{} in {} changed its inputs, so its prims move to the new ones", node.name,
                    file.string());
                RetypePrims(scene, previous, schema);
            }
            pipeline::RegisterPipelineNode(std::move(def));
            Log(LogLevel::Debug, "loaded {} pipeline node {} from {} with inputs {}",
                node.kind == device::PipelineNodeKind::Stencil ? "stencil" : "point", node.name,
                file.string(), inputs.empty() ? std::string("(none)") : Join(inputs));
            names.push_back(node.name);
        }
        LoadedFiles& loaded = Loaded();
        const std::lock_guard lock(loaded.mutex);
        loaded.files[file] = {digest, names};
        return names;
    }
    std::vector<device::ReflectedNode> nodes = std::move(reflection.material);
    std::vector<std::string> names;
    const auto primitive = [](const device::ReflectedNode& node) {
        return node.role == device::NodeRole::Primitive;
    };
    if (std::ranges::any_of(nodes, primitive)) {
        if (!std::ranges::all_of(nodes, primitive) ||
            source.find("ernie.material") != std::string::npos) {
            throw Error("{} defines a closure primitive, so it may only import ernie.primitive "
                        "and define primitives",
                        file.string());
        }
        std::vector<std::shared_ptr<const PrimitiveDefinition>> candidates = Primitives();
        for (const device::ReflectedNode& reflected : nodes) {
            auto definition = std::make_shared<const PrimitiveDefinition>(
                PrimitiveDefinition{.name = reflected.name,
                                    .kind = kFirstPrimitiveLobe + uint32_t(candidates.size()),
                                    .file = file.string(),
                                    .module = module,
                                    .source = source});
            const auto known = std::ranges::find(candidates, reflected.name,
                                                 [](const auto& entry) { return entry->name; });
            if (known != candidates.end()) {
                *known = definition;
            } else {
                candidates.push_back(definition);
            }
        }
        const auto checked = PrimitiveSources(candidates);
        try {
            device::ReflectMaterialNodes(backend, &kRenderKernels, {checked.begin(), checked.end()},
                                         "ernie.render.kernels.material.registered", file.string());
        } catch (const Error& error) {
            throw mapped(error);
        }
        for (const device::ReflectedNode& reflected : nodes) {
            RegisterPrimitive({.name = reflected.name,
                               .file = file.string(),
                               .module = module,
                               .source = source});
            Log(LogLevel::Debug, "loaded closure primitive {} from {} as lobe kind {}",
                reflected.name, file.string(), PrimitiveKind(reflected.name));
            names.push_back(reflected.name);
        }
        nodes.clear();
    }
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
            "{} has no nodes; a material node is a struct with a "
            "Material(inout SurfaceInputs, ShadeContext) method or a function marked [Pattern], "
            "and a pipeline node is a struct conforming to IPointNode or IStencilNode",
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
