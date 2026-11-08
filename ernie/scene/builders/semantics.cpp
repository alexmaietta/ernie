// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/builders/semantics.h"

#include <algorithm>
#include <cstring>

#include "ernie/device/queue.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/publish/change_set.h"
#include "ernie/scene/tables/render_tables.h"
#include "ernie/util/error.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"

namespace ernie {

namespace {

uint64_t Mix(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

}

StableId HashPath(std::string_view path) {
    uint64_t a = 0xcbf29ce484222325ull;
    uint64_t b = 0x6a09e667f3bcc909ull;
    for (const char c : path) {
        a = (a ^ uint8_t(c)) * 0x100000001b3ull;
        b = Mix(b ^ uint8_t(c));
    }
    a = Mix(a ^ path.size());
    b = Mix(b + a);
    StableId id{{uint32_t(a), uint32_t(a >> 32), uint32_t(b), uint32_t(b >> 32)}};
    if (id.words[0] == 0 && id.words[1] == 0 && id.words[2] == 0) {
        id.words[0] = 1;
    }
    return id;
}

std::vector<std::byte> PackIdMap(std::span<const std::pair<StableId, std::string>> entries) {
    constexpr size_t kRecordBytes = 6 * sizeof(uint32_t);
    size_t textBytes = 0;
    for (const auto& [id, text] : entries) {
        textBytes += text.size();
    }
    const size_t first = entries.size() * kRecordBytes;
    std::vector<std::byte> bytes(first + textBytes + sizeof(uint32_t));
    size_t offset = first;
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& [id, text] = entries[i];
        const uint32_t record[6] = {id.words[0], id.words[1],           id.words[2],
                                    id.words[3], uint32_t(text.size()), uint32_t(offset)};
        std::memcpy(bytes.data() + i * kRecordBytes, record, kRecordBytes);
        std::memcpy(bytes.data() + offset, text.data(), text.size());
        offset += text.size();
    }
    const uint32_t count = uint32_t(entries.size());
    std::memcpy(bytes.data() + offset, &count, sizeof(count));
    return bytes;
}

void SemanticBuilder::Describe(RenderTables& tables) const {
    tables.semantics = Data();
}

std::span<const ColumnRef> SemanticBuilder::Inputs() const {
    static constexpr ColumnRef kInputs[] = {kSemanticsLabelsAPISemanticsLabels};
    return kInputs;
}

bool SemanticBuilder::Pending() const {
    return m_wanted != m_taxonomies;
}

void SemanticBuilder::Require(std::span<const Token> taxonomies) {
    m_wanted.assign(taxonomies.begin(), taxonomies.end());
}

uint32_t SemanticBuilder::TaxonomyIndex(Token taxonomy) const {
    const auto found = std::ranges::find(m_taxonomies, taxonomy);
    if (found == m_taxonomies.end()) {
        throw Error("Semantics has no taxonomy {}", m_structure.GetTokens().Text(taxonomy));
    }
    return uint32_t(found - m_taxonomies.begin());
}

uint32_t SemanticBuilder::SemanticId(Token taxonomy, uint32_t prim) const {
    return m_semanticIds[size_t(TaxonomyIndex(taxonomy)) * m_stride + prim];
}

uint32_t SemanticBuilder::LabeledPrim(Token taxonomy, uint32_t prim) const {
    return m_semanticPrims[size_t(TaxonomyIndex(taxonomy)) * m_stride + prim];
}

const SemanticBuilder::Labels*
SemanticBuilder::FindLabels(uint32_t prim, Token taxonomy,
                            std::map<uint64_t, Labels>& reads) const {
    const uint32_t row =
        m_structure.GetPrims().InstanceRow(prim, SchemaId::SemanticsLabelsAPI, taxonomy.id);
    if (row == kNoRow) {
        return nullptr;
    }
    ColumnStore& store = m_structure.GetColumnStore();
    const ArrayRef array = store.Array(kSemanticsLabelsAPISemanticsLabels, row);
    if (array.address == 0) {
        return nullptr;
    }
    auto [found, added] = reads.try_emplace(array.address);
    if (added) {
        found->second = Items<uint32_t>(store.Arrays(ColumnType::TokenArray).Read(array));
    }
    return &found->second;
}

void SemanticBuilder::TrackRows(uint32_t prim) {
    if (const auto rows = m_labelRows.find(prim); rows != m_labelRows.end()) {
        for (const uint32_t row : rows->second) {
            m_byLabelRow.erase(row);
        }
        m_labelRows.erase(rows);
    }
    const Prims& prims = m_structure.GetPrims();
    if (!prims.Alive(prim)) {
        return;
    }
    for (const ApiInstance& entry : prims.Instances(prim, SchemaId::SemanticsLabelsAPI)) {
        m_byLabelRow[entry.row] = prim;
        m_labelRows[prim].push_back(entry.row);
    }
}

void SemanticBuilder::Resolve(uint32_t prim, std::map<uint64_t, Labels>& reads) {
    const Prims& prims = m_structure.GetPrims();
    for (uint32_t taxonomy = 0; taxonomy < m_taxonomies.size(); ++taxonomy) {
        uint32_t at = prim;
        std::span<const uint32_t> tokens;
        while (at != kNoPrim) {
            if (const Labels* labels = FindLabels(at, m_taxonomies[taxonomy], reads)) {
                tokens = *labels;
                if (!tokens.empty()) {
                    break;
                }
            }
            at = prims.At(at).parent;
        }
        uint32_t id = 1;
        if (at != kNoPrim) {
            std::map<Labels, uint32_t>& sets = m_sets[taxonomy];
            id = sets.try_emplace(Labels(tokens.begin(), tokens.end()), uint32_t(sets.size() + 2))
                     .first->second;
        }
        m_semanticIds[size_t(taxonomy) * m_stride + prim] = id;
        m_semanticPrims[size_t(taxonomy) * m_stride + prim] = at;
    }
    ++m_stats.resolved;
}

void SemanticBuilder::Grow(uint32_t prims) {
    if (prims <= m_stride) {
        return;
    }
    const uint32_t stride = std::max({prims, m_stride * 2, 64u});
    const size_t taxonomies = m_taxonomies.size();
    std::vector<uint32_t> ids(taxonomies * stride, 0);
    std::vector<uint32_t> owners(taxonomies * stride, kNoPrim);
    for (size_t taxonomy = 0; taxonomy < taxonomies; ++taxonomy) {
        std::copy_n(m_semanticIds.begin() + ptrdiff_t(taxonomy * m_stride), m_stride,
                    ids.begin() + ptrdiff_t(taxonomy * stride));
        std::copy_n(m_semanticPrims.begin() + ptrdiff_t(taxonomy * m_stride), m_stride,
                    owners.begin() + ptrdiff_t(taxonomy * stride));
    }
    m_semanticIds = std::move(ids);
    m_semanticPrims = std::move(owners);
    m_stableIds.resize(stride, StableId{});
    m_stride = stride;
    m_stats.reallocated = true;
}

void SemanticBuilder::Upload(std::span<const uint32_t> prims) {
    device::Device& device = m_structure.GetColumnStore().GetDevice();
    device::Queue& queue = device.GetQueue();
    if (m_stats.reallocated || !m_stableBuffer || prims.size() > 256) {
        m_stableBuffer = device.CreateBuffer({.label = "stable ids"}, m_stableIds);
        m_idBuffer = device.CreateBuffer(
            {.bytes = std::max<uint64_t>(m_semanticIds.size() * sizeof(uint32_t), 4),
             .label = "semantic ids"},
            std::span<const uint32_t>(m_semanticIds));
        m_primBuffer = device.CreateBuffer(
            {.bytes = std::max<uint64_t>(m_semanticPrims.size() * sizeof(uint32_t), 4),
             .label = "semantic prims"},
            std::span<const uint32_t>(m_semanticPrims));
    } else {
        for (const uint32_t prim : prims) {
            queue.Write(m_stableBuffer->Slice(prim * sizeof(StableId), sizeof(StableId)),
                        &m_stableIds[prim]);
            for (size_t taxonomy = 0; taxonomy < m_taxonomies.size(); ++taxonomy) {
                const size_t at = taxonomy * m_stride + prim;
                queue.Write(m_idBuffer->Slice(at * sizeof(uint32_t), sizeof(uint32_t)),
                            &m_semanticIds[at]);
                queue.Write(m_primBuffer->Slice(at * sizeof(uint32_t), sizeof(uint32_t)),
                            &m_semanticPrims[at]);
            }
        }
    }
    m_data.stableIds.address = m_stableBuffer->Address();
    m_data.semanticIds.address = m_idBuffer->Address();
    m_data.semanticPrims.address = m_primBuffer->Address();
    m_data.stride = m_stride;
    m_data.taxonomyCount = uint32_t(m_taxonomies.size());
}

void SemanticBuilder::Run(const ChangeSet& changes, TableWrites&) {
    ERNIE_ZONE("Semantics/Run");
    m_stats = {};
    const Prims& prims = m_structure.GetPrims();
    const Paths& paths = m_structure.GetPaths();
    const bool full = !m_built || changes.resync || m_wanted != m_taxonomies;
    std::map<uint64_t, Labels> reads;
    std::vector<uint32_t> changed;

    if (full) {
        m_taxonomies = m_wanted;
        m_sets.assign(m_taxonomies.size(), {});
        m_stride = 0;
        m_semanticIds.clear();
        m_semanticPrims.clear();
        m_stableIds.clear();
        m_byLabelRow.clear();
        m_labelRows.clear();
        Grow(prims.Size());
        for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
            if (!prims.Alive(prim)) {
                continue;
            }
            m_stableIds[prim] = HashPath(paths.Path(prim));
            TrackRows(prim);
            Resolve(prim, reads);
            changed.push_back(prim);
        }
        m_stats.reallocated = true;
    } else {
        Grow(prims.Size());
        for (const StructureEntry& entry : changes.removed) {
            if (entry.prim >= m_stride) {
                continue;
            }
            m_stableIds[entry.prim] = {};
            for (size_t taxonomy = 0; taxonomy < m_taxonomies.size(); ++taxonomy) {
                m_semanticIds[taxonomy * m_stride + entry.prim] = 0;
                m_semanticPrims[taxonomy * m_stride + entry.prim] = kNoPrim;
            }
            TrackRows(entry.prim);
            changed.push_back(entry.prim);
        }
        std::set<uint32_t> resolve;
        for (const StructureEntry& entry : changes.added) {
            if (!prims.Alive(entry.prim) || prims.Generation(entry.prim) != entry.generation) {
                continue;
            }
            m_stableIds[entry.prim] = HashPath(paths.Path(entry.prim));
            TrackRows(entry.prim);
            resolve.insert(entry.prim);
        }
        for (const StructureEntry& entry : changes.apis) {
            if (entry.api != SchemaId::SemanticsLabelsAPI || !prims.Alive(entry.prim) ||
                prims.Generation(entry.prim) != entry.generation) {
                continue;
            }
            TrackRows(entry.prim);
            for (const uint32_t at : paths.Subtree(entry.prim)) {
                resolve.insert(at);
            }
        }
        for (const uint32_t row : changes.ArrayRows(kSemanticsLabelsAPISemanticsLabels)) {
            const auto prim = m_byLabelRow.find(row);
            if (prim == m_byLabelRow.end() || !prims.Alive(prim->second)) {
                continue;
            }
            for (const uint32_t at : paths.Subtree(prim->second)) {
                resolve.insert(at);
            }
        }
        for (const uint32_t prim : resolve) {
            Resolve(prim, reads);
            changed.push_back(prim);
        }
    }
    if (!changed.empty() || !m_built) {
        Upload(changed);
    }
    m_built = true;
}

std::vector<std::vector<std::string>> SemanticBuilder::SetTexts(uint32_t taxonomy) const {
    const Tokens& tokens = m_structure.GetTokens();
    std::vector<std::vector<std::string>> texts(m_sets[taxonomy].size());
    for (const auto& [labels, id] : m_sets[taxonomy]) {
        for (const uint32_t label : labels) {
            texts[id - 2].emplace_back(tokens.Text(Token{label}));
        }
    }
    return texts;
}

IdEntries SemanticBuilder::SemanticIdEntries(Token taxonomy) const {
    const std::string name(m_structure.GetTokens().Text(taxonomy));
    IdEntries entries = {
        {StableId{{0, 0, 0, 0}}, "BACKGROUND", "", {{name, {"BACKGROUND"}}}},
        {StableId{{1, 0, 0, 0}}, "UNLABELLED", "", {{name, {"UNLABELLED"}}}},
    };
    const std::vector<std::vector<std::string>> texts = SetTexts(TaxonomyIndex(taxonomy));
    for (uint32_t i = 0; i < texts.size(); ++i) {
        std::string text = name + ": ";
        for (size_t j = 0; j < texts[i].size(); ++j) {
            text += (j > 0 ? "," : "") + texts[i][j];
        }
        entries.push_back({StableId{{i + 2, 0, 0, 0}}, text + ";", "", {{name, texts[i]}}});
    }
    return entries;
}

std::set<uint32_t> SemanticBuilder::LabeledPrims(Token taxonomy) const {
    const Prims& prims = m_structure.GetPrims();
    const size_t first = size_t(TaxonomyIndex(taxonomy)) * m_stride;
    std::set<uint32_t> labeled;
    for (uint32_t prim = 0; prim < std::min(prims.Size(), m_stride); ++prim) {
        const uint32_t at = m_semanticPrims[first + prim];
        if (prims.Alive(prim) && at != kNoPrim) {
            labeled.insert(at);
        }
    }
    return labeled;
}

IdEntries SemanticBuilder::PrimEntries(Token taxonomy, bool stable) const {
    const std::string name(m_structure.GetTokens().Text(taxonomy));
    const uint32_t index = TaxonomyIndex(taxonomy);
    IdEntries entries = {
        {StableId{{0, 0, 0, 0}}, "BACKGROUND", "", {{name, {"BACKGROUND"}}}},
        {stable ? StableId{{0, 0, 0, 1}} : StableId{{1, 0, 0, 0}},
         "UNLABELLED",
         "",
         {{name, {"UNLABELLED"}}}},
    };
    const std::vector<std::vector<std::string>> texts = SetTexts(index);
    for (const uint32_t prim : LabeledPrims(taxonomy)) {
        const uint32_t id = m_semanticIds[size_t(index) * m_stride + prim];
        std::string path = m_structure.GetPaths().Path(prim);
        entries.push_back({stable ? m_stableIds[prim] : StableId{{prim + 2, 0, 0, 0}},
                           path,
                           path,
                           {{name, texts[id - 2]}}});
    }
    return entries;
}

IdEntries SemanticBuilder::InstanceIdEntries(Token taxonomy) const {
    return PrimEntries(taxonomy, false);
}

IdEntries SemanticBuilder::SemanticInstanceIdEntries(Token taxonomy) const {
    return PrimEntries(taxonomy, true);
}

IdEntries SemanticBuilder::StableIdEntries() const {
    const Prims& prims = m_structure.GetPrims();
    std::vector<std::vector<std::vector<std::string>>> texts;
    for (uint32_t taxonomy = 0; taxonomy < m_taxonomies.size(); ++taxonomy) {
        texts.push_back(SetTexts(taxonomy));
    }
    IdEntries entries;
    for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
        if (!prims.Alive(prim) ||
            (!IsA(prims.At(prim).schema, SchemaId::Gprim) && !m_labelRows.contains(prim))) {
            continue;
        }
        IdEntry& entry = entries.emplace_back();
        entry.id = m_stableIds[prim];
        entry.path = entry.text = m_structure.GetPaths().Path(prim);
        for (uint32_t taxonomy = 0; prim < m_stride && taxonomy < m_taxonomies.size(); ++taxonomy) {
            const uint32_t id = m_semanticIds[size_t(taxonomy) * m_stride + prim];
            if (id >= 2) {
                entry.labels.emplace_back(m_structure.GetTokens().Text(m_taxonomies[taxonomy]),
                                          texts[taxonomy][id - 2]);
            }
        }
    }
    return entries;
}

}
