// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/prim.h"

#include <algorithm>

#include "ernie/scene/scene.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"

namespace ernie {

Scene& Prim::GetScene() const {
    if (m_scene == nullptr) {
        throw Error("An empty prim belongs to no scene");
    }
    return *m_scene;
}

bool Prim::Valid() const {
    if (m_scene == nullptr) {
        return false;
    }
    const Prims& prims = SceneImpl::Get(*m_scene).GetStructure().GetPrims();
    return prims.Alive(m_index) && prims.Generation(m_index) == m_generation;
}

void Prim::Check() const {
    if (!Valid()) {
        throw Error("Prim {} was removed", m_index);
    }
}

SchemaId Prim::Schema() const {
    Check();
    return SceneImpl::Get(*m_scene).GetStructure().GetPrims().Record(m_index).schema;
}

bool Prim::IsSession() const {
    Check();
    return SceneImpl::Get(*m_scene).GetStructure().GetPrims().Session(m_index);
}

uint32_t Prim::World() const {
    Check();
    return SceneImpl::Get(*m_scene).GetStructure().GetPrims().Record(m_index).world;
}

std::string Prim::Path() const {
    Check();
    return SceneImpl::Get(*m_scene).GetStructure().GetPaths().Path(m_index);
}

std::string_view Prim::Name() const {
    Check();
    return SceneImpl::Get(*m_scene).GetStructure().GetPaths().Name(m_index);
}

std::optional<Prim> Prim::Parent() const {
    Check();
    const uint32_t parent =
        SceneImpl::Get(*m_scene).GetStructure().GetPrims().Record(m_index).parent;
    if (parent == kNoPrim) {
        return std::nullopt;
    }
    return SceneImpl::Get(*m_scene).Handle(parent);
}

std::vector<Prim> Prim::Children() const {
    Check();
    std::vector<Prim> children;
    for (const uint32_t child :
         SceneImpl::Get(*m_scene).GetStructure().GetPaths().Children(m_index)) {
        children.push_back(SceneImpl::Get(*m_scene).Handle(child));
    }
    return children;
}

std::vector<Connection> Prim::Connections() const {
    CheckSchema(SchemaId::Connectable);
    const std::vector<std::string> sources = Get(Connectable::kConnectionsSources);
    const std::vector<std::string> inputs = Get(Connectable::kConnectionsInputs);
    const std::vector<std::string> outputs = Get(Connectable::kConnectionsOutputs);
    const size_t count = std::min({sources.size(), inputs.size(), outputs.size()});
    std::vector<Connection> connections;
    connections.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        connections.push_back({inputs[i], sources[i], outputs[i]});
    }
    return connections;
}

namespace {

void WriteConnections(const Prim& prim, const std::vector<Connection>& connections) {
    std::vector<std::string> sources;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    for (const Connection& connection : connections) {
        sources.push_back(connection.source);
        inputs.push_back(connection.input);
        outputs.push_back(connection.output);
    }
    prim.Set(Connectable::kConnectionsSources, sources);
    prim.Set(Connectable::kConnectionsInputs, inputs);
    prim.Set(Connectable::kConnectionsOutputs, outputs);
}

}

void Prim::Connect(std::string_view input, const Prim& source, std::string_view output) const {
    CheckSchema(SchemaId::Connectable);
    source.CheckSchema(SchemaId::Connectable);
    if (&source.GetScene() != &GetScene()) {
        throw Error("{} can't connect {} to {}, which is in another scene", Path(), input,
                    source.Path());
    }
    if (input.empty() || output.empty()) {
        throw Error("Connecting {} to {} needs both an input and an output name", Path(),
                    source.Path());
    }
    std::vector<Connection> connections = Connections();
    const auto found = std::ranges::find(connections, input, &Connection::input);
    if (found != connections.end()) {
        Log(LogLevel::Debug, "{}'s {} reconnects from {}.{} to {}.{}", Path(), input, found->source,
            found->output, source.Path(), output);
        found->source = source.Path();
        found->output = std::string(output);
    } else {
        Log(LogLevel::Debug, "{}'s {} connects to {}.{}", Path(), input, source.Path(), output);
        connections.push_back({std::string(input), source.Path(), std::string(output)});
    }
    WriteConnections(*this, connections);
}

void Prim::Disconnect(std::string_view input) const {
    std::vector<Connection> connections = Connections();
    const auto found = std::ranges::find(connections, input, &Connection::input);
    if (found == connections.end()) {
        return;
    }
    Log(LogLevel::Debug, "{}'s {} disconnects from {}.{}", Path(), input, found->source,
        found->output);
    connections.erase(found);
    WriteConnections(*this, connections);
}

View Prim::Single() const {
    return GetScene().Select(*this);
}

View Prim::Single(SchemaId api, std::string_view instance) const {
    return GetScene().Select(*this, api, instance);
}

std::vector<std::string> Prim::Instances(SchemaId api) const {
    Check();
    const Structure& structure = SceneImpl::Get(*m_scene).GetStructure();
    std::vector<std::string> names;
    for (const ApiInstance& entry : structure.GetPrims().Instances(m_index, api)) {
        names.emplace_back(structure.GetTokens().Text(Token{entry.instance}));
    }
    std::ranges::sort(names);
    return names;
}

bool Prim::IsA(SchemaId schema) const {
    return ernie::IsA(Schema(), schema);
}

bool Prim::HasApi(SchemaId api) const {
    Check();
    return SceneImpl::Get(*m_scene).GetStructure().GetPrims().HasApi(m_index, api);
}

void Prim::CheckSchema(SchemaId schema) const {
    if (!ernie::IsA(Schema(), schema) && !HasApi(schema)) {
        throw Error("{} is a {}, not a {}", Path(), Info(Schema()).name, Info(schema).name);
    }
}

Token Prim::Intern(std::string_view text) const {
    return GetScene().Token(text);
}

std::string_view Prim::Text(Token token) const {
    return GetScene().Text(token);
}

}
