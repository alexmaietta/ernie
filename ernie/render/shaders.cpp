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
#include "ernie/render/kernels.h"
#include "ernie/sensor_pipeline/node.h"
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

material::MaterialInputType InputType(device::FieldType type) {
    switch (type) {
    case device::FieldType::Bool:
        return material::MaterialInputType::Bool;
    case device::FieldType::Int:
        return material::MaterialInputType::Int;
    case device::FieldType::Float2:
        return material::MaterialInputType::Float2;
    case device::FieldType::Float3:
        return material::MaterialInputType::Float3;
    case device::FieldType::Float4:
        return material::MaterialInputType::Float4;
    case device::FieldType::Spectral:
        return material::MaterialInputType::Spectral;
    default:
        return material::MaterialInputType::Float;
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
    auto sources = material::ClosurePrimitiveSources(material::ClosurePrimitives());
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
        const std::vector<device::ReflectedSensorPipelineNode>& reflected = reflection.pipeline;
        const auto signal = [&](const std::string& node, const std::string& type) {
            const auto found = std::ranges::find(sensor_pipeline::kSignals, type,
                                                 &sensor_pipeline::SignalInfo::slangType);
            if (found == std::end(sensor_pipeline::kSignals)) {
                std::vector<std::string> types;
                for (const sensor_pipeline::SignalInfo& info : sensor_pipeline::kSignals) {
                    types.emplace_back(info.slangType);
                }
                throw Error("{} in {} uses {}, which isn't a pipeline signal; signals are {}", node,
                            file.string(), type, Join(types));
            }
            return sensor_pipeline::SignalType(found - std::begin(sensor_pipeline::kSignals));
        };
        for (const device::ReflectedSensorPipelineNode& node : reflected) {
            signal(node.name, node.result);
            for (const device::ReflectedPort& port : node.inputs) {
                signal(node.name, port.type);
            }
        }
        std::vector<std::string> names;
        for (const device::ReflectedSensorPipelineNode& node : reflected) {
            const sensor_pipeline::SignalType result = signal(node.name, node.result);
            sensor_pipeline::NodeDef def{
                .id = node.name,
                .kind = node.kind == device::SensorPipelineNodeKind::Stencil
                            ? sensor_pipeline::NodeKind::Stencil
                            : sensor_pipeline::NodeKind::Point,
                .outputs = {{std::string(sensor_pipeline::Signal(result).port), result}},
                .radius = node.radius,
                .code = {.module = module,
                         .type = node.name,
                         .file = file.string(),
                         .source = source,
                         .version = digest,
                         .applies = true,
                         .context = node.context}};
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
                sensor_pipeline::ParameterDef parameter{.name = field.name};
                switch (field.type) {
                case device::FieldType::Bool:
                    parameter.type = sensor_pipeline::ParameterType::Bool;
                    parameter.fallback = {Read<uint32_t>(field.fallback) != 0 ? 1.0f : 0.0f};
                    break;
                case device::FieldType::Int:
                    parameter.type = sensor_pipeline::ParameterType::Int;
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
            const std::shared_ptr<const sensor_pipeline::NodeRegistry> known =
                sensor_pipeline::SensorPipelineNodes();
            const uint32_t found = known->Find(node.name);
            if (found != sensor_pipeline::kNoDef && (*known)[found].code.file.empty()) {
                throw Error("{} in {} has the name of a built-in pipeline node; rename it",
                            node.name, file.string());
            }
            const SchemaId previous = FindSchemaByName(node.name);
            const bool reloading = found != sensor_pipeline::kNoDef && previous != SchemaId::None;
            const SchemaId schema = RegisterSchema(desc, reloading);
            if (reloading && previous != schema) {
                Log(LogLevel::Info,
                    "{} in {} changed its inputs, so its prims move to the new ones", node.name,
                    file.string());
                RetypePrims(scene, previous, schema);
            }
            sensor_pipeline::RegisterSensorPipelineNode(std::move(def));
            Log(LogLevel::Debug, "loaded {} pipeline node {} from {} with inputs {}",
                node.kind == device::SensorPipelineNodeKind::Stencil ? "stencil" : "point",
                node.name, file.string(), inputs.empty() ? std::string("(none)") : Join(inputs));
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
        material::ClosurePrimitiveList candidates = material::ClosurePrimitives();
        for (const device::ReflectedNode& reflected : nodes) {
            candidates =
                material::WithClosurePrimitive(std::move(candidates), {.name = reflected.name,
                                                                       .file = file.string(),
                                                                       .module = module,
                                                                       .source = source});
        }
        const auto checked = material::ClosurePrimitiveSources(candidates);
        try {
            device::ReflectMaterialNodes(backend, &kRenderKernels, {checked.begin(), checked.end()},
                                         "ernie.render.kernels.material.registered", file.string());
        } catch (const Error& error) {
            throw mapped(error);
        }
        for (const device::ReflectedNode& reflected : nodes) {
            material::RegisterClosurePrimitive({.name = reflected.name,
                                                .file = file.string(),
                                                .module = module,
                                                .source = source});
            const auto loaded = std::ranges::find(candidates, reflected.name,
                                                  [](const auto& entry) { return entry->name; });
            Log(LogLevel::Debug, "loaded closure primitive {} from {} as lobe kind {}",
                reflected.name, file.string(), (*loaded)->kind);
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
        material::MaterialNodeDefinition node{.name = reflected.name,
                                              .role = reflected.role == device::NodeRole::Pattern
                                                          ? material::MaterialNodeRole::Pattern
                                                      : reflected.role == device::NodeRole::Surface
                                                          ? material::MaterialNodeRole::Surface
                                                          : material::MaterialNodeRole::Material,
                                              .file = file.string(),
                                              .module = module,
                                              .source = source,
                                              .result = InputType(reflected.result)};
        for (const device::ReflectedParameter& parameter : reflected.parameters) {
            node.parameters.push_back(
                {.source = parameter.source == device::ParameterSource::Context
                               ? material::MaterialNodeParameter::Source::Context
                           : parameter.source == device::ParameterSource::Point
                               ? material::MaterialNodeParameter::Source::Point
                               : material::MaterialNodeParameter::Source::Input,
                 .input = parameter.input});
        }
        std::vector<std::string> inputs;
        for (const device::ReflectedField& field : reflected.fields) {
            desc.columns.push_back(Column(field));
            node.inputs.push_back({.name = field.name, .type = InputType(field.type)});
            inputs.push_back(field.name);
        }
        const std::shared_ptr<const material::MaterialNodeDefinition> previous =
            material::FindMaterialNode(reflected.name);
        const bool reloading = previous && previous->file == file.string();
        const SchemaId schema = RegisterSchema(desc, reloading);
        if (reloading && SchemaId(previous->schema) != schema) {
            Log(LogLevel::Info, "{} in {} changed its inputs, so its prims move to the new ones",
                reflected.name, file.string());
            RetypePrims(scene, SchemaId(previous->schema), schema);
        }
        node.schema = uint32_t(schema);
        material::RegisterMaterialNode(std::move(node));
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
