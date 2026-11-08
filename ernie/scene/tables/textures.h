// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ernie/scene/records/textures.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/tables/storage.h"
#include "ernie/scene/tables/table.h"
#include "ernie/texture/cache.h"

namespace ernie {

struct TextureStats {
    uint32_t images = 0;
    uint32_t bound = 0;
    uint32_t loaded = 0;
    uint32_t failed = 0;
    uint32_t loading = 0;
};

struct TextureSlot {
    SchemaId table = SchemaId::None;
    uint32_t texture = 0;
    bool color = false;
};

class TextureTable : public RenderTable {
  public:
    explicit TextureTable(Structure& structure, unsigned threads = 4);
    ~TextureTable() override;
    TextureTable(const TextureTable&) = delete;
    TextureTable& operator=(const TextureTable&) = delete;

    std::string_view Name() const override {
        return "textures";
    }

    std::span<const ColumnRef> Inputs() const override {
        return m_inputs;
    }

    std::span<const ColumnRef> RowInputs() const override {
        return m_inputs;
    }

    std::span<const TableId> TableOutputs() const override {
        static constexpr TableId kTables[] = {TableId::Textures};
        return kTables;
    }

    void Describe(RenderTables& tables) const override;
    void Hold(std::vector<Ref<const RefCounted>>& held) const override;

    const TextureRecord& Record(uint32_t prim) const {
        return m_records.Row(prim);
    }

    bool Pending() const override;
    void Run(const ChangeSet& changes, TableVersions& versions) override;

    void Finish() override;
    uint64_t Handle(uint32_t prim, std::string_view input) const;

    uint64_t Checker() const {
        return m_cache.Binding(kCheckerTextureBinding);
    }

    const TextureStats& Stats() const {
        return m_stats;
    }

  private:
    using RowKey = std::pair<SchemaId, uint32_t>;

    struct Bound {
        TextureSlot slot;
        TextureBinding binding;
    };

    struct Material {
        std::vector<Bound> slots;
        std::vector<RowKey> rows;
        uint32_t bindings = kNoTextures;
        bool checkered = false;
    };

    struct Cells {
        std::map<RowKey, std::vector<uint32_t>> columns;
    };

    const std::vector<uint32_t>& Read(Cells& cells, SchemaId table, uint32_t column);
    std::set<uint32_t> Collect();
    void Refresh(uint32_t prim);
    void Release(uint32_t prim);
    void Publish(uint32_t prim, const Material& material);
    void Rebound(uint32_t world);
    void Rebind(uint32_t prim, Cells& cells);
    void UploadBindings();

    Structure& m_structure;
    TextureCache m_cache;
    std::map<SchemaId, std::vector<TextureSlot>> m_slots;
    std::vector<ColumnRef> m_inputs;
    std::unordered_map<uint32_t, Material> m_materials;
    std::map<RowKey, uint32_t> m_byRow;
    uint64_t m_bindingAddress = 0;
    TextureBinding m_checker;
    uint32_t m_checkerUsers = 0;
    TextureStats m_stats;
    RecordTable<TextureRecord> m_records{"texture records"};
    std::vector<uint32_t> m_rebound;
};

}
