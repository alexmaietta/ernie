// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/pipelines.h"

#include <algorithm>
#include <cstring>
#include <format>

#include "ernie/device/queue.h"
#include "ernie/pipeline/node.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/stage/publish/change_set.h"
#include "ernie/stage/views/attribute.h"
#include "ernie/stage/views/view.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"

namespace ernie {

namespace {

enum class AnyEnum : uint32_t {};

std::vector<ColumnRef> PipelineInputs() {
    std::vector<ColumnRef> inputs;
    for (uint32_t id = 0; id < SchemaCount(); ++id) {
        const auto schema = SchemaId(id);
        if (schema != SchemaId::Connectable && schema != SchemaId::Shader &&
            !IsA(schema, SchemaId::SensorNode) && schema != SchemaId::SensorPipeline &&
            schema != SchemaId::RenderVar) {
            continue;
        }
        const Span<const ColumnInfo> columns = Columns(schema);
        for (uint32_t index = 0; index < columns.size(); ++index) {
            inputs.push_back({schema, &columns[index], index});
        }
    }
    return inputs;
}

bool SameInstance(const PipelineInstance& a, const PipelineInstance& b) {
    return a.definition == b.definition && a.words == b.words && a.error == b.error &&
           a.resolved.key == b.resolved.key;
}

}

PipelineTable::PipelineTable(Structure& structure)
    : m_structure(structure), m_library(1), m_registry(pipeline::PipelineNodes()) {}

std::span<const ColumnRef> PipelineTable::Inputs() const {
    if (m_inputSchemas != SchemaCount()) {
        m_inputs = PipelineInputs();
        m_inputSchemas = SchemaCount();
    }
    return m_inputs;
}

std::shared_ptr<const pipeline::NodeRegistry> PipelineTable::Registry() const {
    const std::lock_guard lock(m_mutex);
    return m_registry;
}

std::shared_ptr<const PipelineInstance> PipelineTable::Find(uint32_t prim) const {
    const std::lock_guard lock(m_mutex);
    const auto found = m_instances.find(prim);
    return found == m_instances.end() ? nullptr : found->second;
}

std::shared_ptr<const pipeline::PipelineSet> PipelineTable::Definitions() const {
    return m_library.Snapshot();
}

std::vector<pipeline::ConnectionDesc> PipelineTable::Connections(uint32_t prim) const {
    std::vector<pipeline::ConnectionDesc> connections;
    for (const ConnectionCache::Connection& connection : m_connections.Get(prim)) {
        connections.push_back({.input = std::string(connection.input),
                               .source = connection.source,
                               .sourcePath = connection.source != kNoPrim
                                                 ? m_structure.GetPaths().Path(connection.source)
                                                 : std::string("nothing"),
                               .output = std::string(connection.output)});
    }
    return connections;
}

pipeline::NodeDesc PipelineTable::GatherNode(uint32_t prim) const {
    const Prims& prims = m_structure.GetPrims();
    const SchemaId schema = prims.Record(prim).schema;
    const uint64_t apis = prims.Record(prim).apis;
    const View view(m_structure, schema, std::vector<uint32_t>{prim});
    const Tokens& tokens = m_structure.GetTokens();
    pipeline::NodeDesc node{.prim = prim, .path = m_structure.GetPaths().Path(prim)};
    node.id = std::string(tokens.Text(view.Get<Token>(kShaderId).Get().front()));
    node.inputs = Connections(prim);
    const pipeline::NodeRegistry& registry = *m_registry;
    const uint32_t def = registry.Find(node.id);
    if (def == pipeline::kNoDef) {
        return node;
    }
    for (const pipeline::UniformDef& uniform : registry[def].uniforms) {
        const ColumnRef column = FindColumn(schema, apis, uniform.name);
        if (!column) {
            continue;
        }
        std::string value;
        switch (column.column->type) {
        case ColumnType::Enum: {
            const uint32_t index = uint32_t(view.Get<AnyEnum>(column).Get().front());
            value = index < column.column->tokens.size() ? std::string(column.column->tokens[index])
                                                         : uniform.fallback;
            break;
        }
        case ColumnType::Token:
            value = std::string(tokens.Text(view.Get<Token>(column).Get().front()));
            break;
        case ColumnType::Bool:
            value = view.Get<bool>(column).Get().front() ? "true" : "false";
            break;
        default:
            throw Error("{}'s {} is a uniform, so it must be a token or a bool", node.path,
                        uniform.name);
        }
        node.uniforms.emplace_back(uniform.name, std::move(value));
    }
    const std::vector<pipeline::ParameterDef>& parameters = registry[def].parameters;
    for (uint32_t index = 0; index < parameters.size(); ++index) {
        const pipeline::ParameterDef& parameter = parameters[index];
        const ColumnRef column = FindColumn(schema, apis, parameter.name);
        if (!column) {
            continue;
        }
        const ColumnType type = column.column->type;
        const bool copied = parameter.type == pipeline::ParameterType::Int ? type == ColumnType::Int
                            : parameter.type == pipeline::ParameterType::Bool
                                ? type == ColumnType::Bool
                                : (type == ColumnType::Float || type == ColumnType::Float2 ||
                                   type == ColumnType::Float3 || type == ColumnType::Float4) &&
                                      ElementBytes(type) == parameter.words * sizeof(uint32_t);
        if (copied) {
            m_gathered[prim].push_back(
                {index, column, m_structure.GetPrims().Row(prim, column.schema)});
            continue;
        }
        std::vector<float> values;
        switch (type) {
        case ColumnType::Float:
            values = {view.Get<float>(column).Get().front()};
            break;
        case ColumnType::Int:
            values = {float(view.Get<int32_t>(column).Get().front())};
            break;
        case ColumnType::Bool:
            values = {view.Get<bool>(column).Get().front() ? 1.0f : 0.0f};
            break;
        case ColumnType::Float2: {
            const Float2 value = view.Get<Float2>(column).Get().front();
            values = {value.x, value.y};
            break;
        }
        case ColumnType::Float3: {
            const Float3 value = view.Get<Float3>(column).Get().front();
            values = {value.x, value.y, value.z};
            break;
        }
        case ColumnType::Float4: {
            const Float4 value = view.Get<Float4>(column).Get().front();
            values = {value.x, value.y, value.z, value.w};
            break;
        }
        default:
            throw Error("{}'s {} is a value, so it must be numeric", node.path, parameter.name);
        }
        node.values.emplace_back(parameter.name, std::move(values));
    }
    return node;
}

std::vector<PipelineTable::Tap> PipelineTable::GatherTaps() const {
    const Prims& prims = m_structure.GetPrims();
    const Tokens& tokens = m_structure.GetTokens();
    std::vector<Tap> taps;
    for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
        if (!prims.Alive(prim) || prims.Record(prim).schema != SchemaId::RenderVar) {
            continue;
        }
        const View view(m_structure, SchemaId::RenderVar, std::vector<uint32_t>{prim});
        if (tokens.Text(view.Get<Token>(kRenderVarSourceName).Get().front()) != kTapSource) {
            continue;
        }
        const std::vector<uint32_t> targets = view.GetArray<uint32_t>(kRenderVarTap).Get(0);
        if (targets.empty() || !prims.Alive(targets.front())) {
            continue;
        }
        taps.push_back(
            {m_structure.GetPaths().Path(prim), targets.front(),
             std::string(tokens.Text(view.Get<Token>(kRenderVarTapOutput).Get().front()))});
    }
    return taps;
}

pipeline::PipelineDesc PipelineTable::Gather(uint32_t pipeline, std::span<const Tap> taps) const {
    const Prims& prims = m_structure.GetPrims();
    pipeline::PipelineDesc desc{.prim = pipeline, .path = m_structure.GetPaths().Path(pipeline)};
    desc.outputs = Connections(pipeline);
    for (const Tap& tap : taps) {
        const std::string node = m_structure.GetPaths().Path(tap.node);
        if (!node.starts_with(desc.path + "/") ||
            !IsA(prims.Record(tap.node).schema, SchemaId::SensorNode)) {
            continue;
        }
        std::string output = tap.output;
        if (output.empty()) {
            const pipeline::NodeDesc described = GatherNode(tap.node);
            const uint32_t def = m_registry->Find(described.id);
            if (def == pipeline::kNoDef || (*m_registry)[def].outputs.size() != 1) {
                throw Error("{} taps {}, so its tapOutput must name one of the node's outputs",
                            tap.name, node);
            }
            output = (*m_registry)[def].outputs.front().name;
        }
        desc.outputs.push_back(
            {.input = tap.name, .source = tap.node, .sourcePath = node, .output = output});
    }
    std::vector<uint32_t> pending;
    std::vector<uint32_t> seen;
    for (const pipeline::ConnectionDesc& output : desc.outputs) {
        pending.push_back(output.source);
    }
    while (!pending.empty()) {
        const uint32_t prim = pending.back();
        pending.pop_back();
        if (prim == kNoPrim || std::ranges::find(seen, prim) != seen.end() ||
            !IsA(prims.Record(prim).schema, SchemaId::SensorNode)) {
            continue;
        }
        seen.push_back(prim);
        pipeline::NodeDesc node = GatherNode(prim);
        for (const pipeline::ConnectionDesc& input : node.inputs) {
            pending.push_back(input.source);
        }
        desc.nodes.push_back(std::move(node));
    }
    std::ranges::sort(desc.nodes, {}, &pipeline::NodeDesc::prim);
    return desc;
}

void PipelineTable::Run(const ChangeSet& changes, TableVersions&) {
    ERNIE_ZONE("Pipelines/Run");
    ColumnStore& store = m_structure.GetColumnStore();
    std::shared_ptr<const pipeline::NodeRegistry> nodes = pipeline::PipelineNodes();
    const bool reloaded = nodes != m_registry;
    if (reloaded) {
        const std::lock_guard lock(m_mutex);
        m_registry = std::move(nodes);
    }
    std::vector<const ColumnInfo*> written;
    for (const ColumnRef input : Inputs()) {
        if (store.LastWrite(input) > m_seen) {
            written.push_back(input.column);
        }
    }
    m_seen = store.WriteSequence();
    const bool restructured =
        reloaded || changes.resync || changes.structure || !changes.apis.empty();
    if (!restructured && std::ranges::all_of(written, [&](const ColumnInfo* column) {
            return m_pack.Reads(column);
        })) {
        if (!written.empty()) {
            Pack(false);
        }
        return;
    }
    const Prims& prims = m_structure.GetPrims();
    const pipeline::NodeRegistry& registry = *m_registry;
    std::map<uint32_t, std::shared_ptr<const PipelineInstance>> resolved;
    m_connections.Update(changes);
    const std::vector<Tap> taps = GatherTaps();
    m_gathered.clear();
    m_pack.Clear();
    uint32_t words = 0;
    bool changed = false;
    for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
        if (!prims.Alive(prim) || !IsA(prims.Record(prim).schema, SchemaId::SensorPipeline)) {
            continue;
        }
        auto instance = std::make_shared<PipelineInstance>();
        instance->prim = prim;
        instance->path = m_structure.GetPaths().Path(prim);
        try {
            instance->resolved = pipeline::ResolvePipeline(Gather(prim, taps), registry);
            instance->definition = m_library.Intern(instance->resolved.key, [&](uint32_t id) {
                return pipeline::BuildDefinition(id, instance->resolved, registry);
            });
            const auto definitions = m_library.Snapshot();
            const pipeline::PipelineDefinition& definition =
                *definitions->Find(instance->definition);
            instance->words = pipeline::PackWords(definition, instance->resolved, registry);
            instance->firstWord = words;
            words += uint32_t(instance->words.size());
            for (uint32_t node = 0; node < instance->resolved.nodes.size(); ++node) {
                const pipeline::ResolvedNode& each = instance->resolved.nodes[node];
                const auto found = m_gathered.find(each.prim);
                if (found == m_gathered.end()) {
                    continue;
                }
                const std::vector<pipeline::ParameterDef>& parameters =
                    registry[each.def].parameters;
                for (const Gathered& gathered : found->second) {
                    uint32_t offset = definition.parameterOffsets[node];
                    for (uint32_t index = 0; index < gathered.parameter; ++index) {
                        offset += parameters[index].words;
                    }
                    m_pack.Copy(gathered.column, gathered.row, instance->firstWord + offset);
                }
            }
        } catch (const Error& error) {
            instance->error = error.what();
            Log(LogLevel::Warning, "{}", instance->error);
        }
        const std::shared_ptr<const PipelineInstance> previous = Find(prim);
        changed |= !previous || !SameInstance(*previous, *instance);
        resolved.emplace(prim, std::move(instance));
    }
    {
        const std::lock_guard lock(m_mutex);
        changed |= resolved.size() != m_instances.size();
        m_instances = std::move(resolved);
        if (changed) {
            ++m_version;
        }
    }
    m_words.Reserve(m_structure.GetColumnStore().GetDevice(), std::max(words, 1u));
    m_pack.Build(store, false);
    Pack(true);
}

void PipelineTable::Pack(bool rewrite) {
    ERNIE_ZONE("Pipelines/Pack");
    device::Device& device = m_structure.GetColumnStore().GetDevice();
    device::Queue& queue = device.GetQueue();
    if (rewrite) {
        for (const auto& [prim, instance] : m_instances) {
            if (!instance->words.empty()) {
                queue.Write(
                    m_words.Buffer()->Slice(uint64_t(instance->firstWord) * sizeof(uint32_t),
                                            instance->words.size() * sizeof(uint32_t)),
                    instance->words.data());
            }
        }
    }
    m_pack.Pack(m_structure.GetColumnStore(), m_words.Address());
}

std::vector<uint32_t> PipelineTable::ReadWords(const PipelineInstance& instance) const {
    if (instance.words.empty() || !m_words.Buffer()) {
        return instance.words;
    }
    device::Queue& queue = m_structure.GetColumnStore().GetDevice().GetQueue();
    const device::Download download = queue.ReadAsync(m_words.Buffer()->Slice(
        uint64_t(instance.firstWord) * sizeof(uint32_t), instance.words.size() * sizeof(uint32_t)));
    download.Wait();
    std::vector<uint32_t> words(instance.words.size());
    std::memcpy(words.data(), download.Bytes().data(), words.size() * sizeof(uint32_t));
    return words;
}

}
