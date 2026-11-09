// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/graph/library.h"
#include "ernie/material/definition.h"
#include "ernie/material/node.h"
#include "ernie/scene/records/materials.h"
#include "ernie/scene/tables/connections.h"
#include "ernie/scene/tables/pack.h"
#include "ernie/scene/tables/storage.h"
#include "ernie/scene/tables/table.h"
#include "ernie/stage/structure/structure.h"

namespace ernie {

class TextureTable;

class MaterialTable : public RenderTable {
  public:
    MaterialTable(Structure& structure, const TextureTable& textures)
        : m_structure(structure), m_textures(textures) {}

    std::string_view Name() const override {
        return "materials";
    }

    std::span<const ColumnRef> Inputs() const override;

    std::span<const TableId> TableInputs() const override {
        static constexpr TableId kTables[] = {TableId::Textures};
        return kTables;
    }

    void Prepare() override;

    bool Pending() const override {
        return m_nodeVersion != NodeVersion();
    }

    void Run(const ChangeSet& changes, TableVersions& versions) override;
    void Describe(RenderTables& tables) const override;

    void Hold(std::vector<Ref<const RefCounted>>& held) const override {
        HoldBuffers(held, {m_records.Buffer(), m_words.Buffer(), m_checkers.Buffer(),
                           m_temperatures.Buffer(), m_radios.Buffer()});
    }

    const MaterialRecord& Record(uint32_t prim) const {
        return m_records.Row(prim);
    }

    uint32_t Resolves() const {
        return m_resolves;
    }

    const DefinitionSetRef& Definitions() const {
        return m_definitions;
    }

  private:
    using Connection = ConnectionCache::Connection;

    struct Binding {
        uint32_t texture = kNoPrim;
        uint32_t low = 0;
        uint32_t high = 0;
    };

    struct Carry {
        uint32_t from = 0;
        uint32_t to = 0;
        uint32_t words = 0;
    };

    struct RadarLink {
        uint32_t material = kNoPrim;
        uint32_t row = 0;
    };

    void Resolve(const ChangeSet& changes);
    void ResolveMaterial(uint32_t material);
    void ResolvePreview(uint32_t material, uint32_t shader);
    void ResolveNode(uint32_t material, uint32_t shader,
                     std::shared_ptr<const NodeDefinition> surface);
    bool BindTexture(uint32_t shader, const Connection& connection, std::string_view input,
                     bool color, uint32_t slot, TextureChannel& channel);
    void CarryPrevious(const std::vector<MaterialRecord>& before);
    void Define(uint32_t material, uint32_t definition, uint32_t words);
    void WarnOutsideVisible();
    void ResolveRadios();
    void WriteRadios();
    void ResolvePlacement(uint32_t placement, uint32_t mapping);
    bool Restructured(const ChangeSet& changes) const;
    bool Changed(SchemaId schema) const;
    std::span<const Connection> Connections(uint32_t prim);
    void CopyCell(ColumnRef column, uint32_t prim, uint32_t destination);
    void BuildEntries();
    device::Program& Kernels();

    Structure& m_structure;
    const TextureTable& m_textures;
    Ref<device::Program> m_program;
    RecordTable<MaterialRecord> m_records{"material records"};
    DeviceTable<uint32_t> m_words{"material words"};
    PackPlan m_pack{"material"};
    std::vector<Binding> m_bindings;
    std::vector<Carry> m_carries;
    ConnectionCache m_connections{m_structure, {SchemaId::Connectable}};
    std::vector<bool> m_touched;
    uint64_t m_writes = 0;
    std::vector<RadarLink> m_radarLinks;
    Ref<device::Buffer> m_carried;
    graph::DefinitionLibrary<DefinitionModule> m_library{kFirstGeneratedDefinition};
    std::vector<uint32_t> m_generations;
    std::set<uint32_t> m_spectral;
    std::set<std::pair<uint64_t, uint64_t>> m_warned;
    mutable std::vector<ColumnRef> m_inputs;
    mutable uint32_t m_inputSchemas = 0;
    uint64_t m_nodeVersion = NodeVersion();
    uint32_t m_wordCount = 0;
    uint32_t m_resolves = 0;
    DefinitionSetRef m_definitions = std::make_shared<const DefinitionSet>();
    DeviceTable<CheckerRecord> m_checkers{"checkerboard records"};
    DeviceTable<float> m_temperatures{"temperatures"};
    uint32_t m_temperaturePrimvar = 0;
    RecordTable<RadioRecord> m_radios{"radio materials"};
};

}
