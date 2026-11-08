// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/material/definition.h"
#include "ernie/scene/records/materials.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/tables/storage.h"
#include "ernie/scene/tables/table.h"

namespace ernie {

class TextureTable;

class MaterialTable : public Table {
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
    void Run(const ChangeSet& changes, TableVersions& versions) override;
    void Describe(RenderTables& tables) const override;

    void Hold(std::vector<Ref<const RefCounted>>& held) const override {
        HoldBuffers(held, {m_records.Buffer(), m_words.Buffer(), m_nonVisuals.Buffer(),
                           m_checkers.Buffer()});
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
    struct Connection {
        std::string input;
        uint32_t source = kNoPrim;
        std::string output;
    };

    struct Copy {
        ColumnRef column;
        uint32_t row = 0;
        uint32_t destination = 0;
    };

    struct Literal {
        uint32_t destination = 0;
        uint32_t value = 0;
    };

    struct Binding {
        uint32_t texture = kNoPrim;
        uint32_t destination = 0;
    };

    void Resolve();
    void ResolveMaterial(uint32_t material);
    void ResolvePreview(uint32_t material, uint32_t shader);
    void ResolvePlacement(uint32_t placement, uint32_t mapping);
    std::vector<Connection> Connections(uint32_t prim);
    void CopyCell(ColumnRef column, uint32_t prim, uint32_t destination);
    void Pack();
    device::Program& Kernels();

    Structure& m_structure;
    const TextureTable& m_textures;
    Ref<device::Program> m_program;
    RecordTable<MaterialRecord> m_records{"material records"};
    DeviceTable<uint32_t> m_words{"material words"};
    Ref<device::Buffer> m_entries;
    std::vector<Copy> m_copies;
    std::vector<Literal> m_literals;
    std::vector<Binding> m_bindings;
    uint32_t m_wordCount = 0;
    uint32_t m_resolves = 0;
    DefinitionSetRef m_definitions = std::make_shared<const DefinitionSet>();
    DeviceTable<NonVisualRecord> m_nonVisuals{"nonvisual material records"};
    DeviceTable<CheckerRecord> m_checkers{"checkerboard records"};
};

}
