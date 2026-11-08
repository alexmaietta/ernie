// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/ref.h"
#include "ernie/device/tensor.h"
#include "ernie/scene/kernels/authored.h"
#include "ernie/scene/records/semantics.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/tables/table.h"
#include "ernie/scene/types.h"

namespace ernie {

struct IdEntry {
    StableId id;
    std::string text;
    std::string path;
    std::vector<std::pair<std::string, std::vector<std::string>>> labels;
};

using IdEntries = std::vector<IdEntry>;

StableId HashPath(std::string_view path);
std::vector<std::byte> PackIdMap(std::span<const std::pair<StableId, std::string>> entries);

struct SemanticMap {
    std::shared_ptr<const IdMap> map;
    device::Tensor packed;
};

struct SemanticStats {
    uint32_t resolved = 0;
    bool reallocated = false;
};

class SemanticTable : public Table {
  public:
    explicit SemanticTable(Structure& structure) : m_structure(structure) {}

    std::string_view Name() const override {
        return "semantics";
    }

    std::span<const ColumnRef> Inputs() const override;
    bool Pending() const override;
    void Run(const ChangeSet& changes, TableVersions& versions) override;

    void Require(std::span<const Token> taxonomies);

    const SemanticsData& Data() const {
        return m_data;
    }

    const SemanticStats& Stats() const {
        return m_stats;
    }

    const Ref<device::Buffer>& StableIds() const {
        return m_stableBuffer;
    }

    void Describe(RenderTables& tables) const override;

    void Hold(std::vector<Ref<const RefCounted>>& held) const override {
        HoldBuffers(held, {m_stableBuffer, m_idBuffer, m_primBuffer});
    }

    uint32_t SemanticId(Token taxonomy, uint32_t prim) const;
    uint32_t LabeledPrim(Token taxonomy, uint32_t prim) const;

    SemanticMap Map(RenderVarSource source, Token taxonomy);

  private:
    IdEntries Entries(RenderVarSource source, Token taxonomy) const;
    IdEntries SemanticIdEntries(Token taxonomy) const;
    IdEntries StableIdEntries() const;
    using Labels = std::vector<uint32_t>;

    uint32_t TaxonomyIndex(Token taxonomy) const;
    std::set<uint32_t> LabeledPrims(Token taxonomy) const;
    std::vector<std::vector<std::string>> SetTexts(uint32_t taxonomy) const;
    IdEntries PrimEntries(Token taxonomy, bool stable) const;
    const Labels* FindLabels(uint32_t prim, Token taxonomy,
                             std::map<uint64_t, Labels>& reads) const;
    void TrackRows(uint32_t prim);
    void Resolve(uint32_t prim, std::map<uint64_t, Labels>& reads);
    void Grow(uint32_t prims);
    void Upload(std::span<const uint32_t> prims);

    Structure& m_structure;
    std::vector<Token> m_wanted;
    std::vector<Token> m_taxonomies;
    uint32_t m_stride = 0;
    std::vector<StableId> m_stableIds;
    std::vector<uint32_t> m_semanticIds;
    std::vector<uint32_t> m_semanticPrims;
    std::vector<std::map<Labels, uint32_t>> m_sets;
    std::unordered_map<uint32_t, uint32_t> m_byLabelRow;
    std::unordered_map<uint32_t, std::vector<uint32_t>> m_labelRows;
    Ref<device::Buffer> m_stableBuffer;
    Ref<device::Buffer> m_idBuffer;
    Ref<device::Buffer> m_primBuffer;
    SemanticsData m_data{};
    SemanticStats m_stats;
    bool m_built = false;
    std::mutex m_mapMutex;
    std::map<std::pair<RenderVarSource, uint32_t>, SemanticMap> m_maps;
};

}
