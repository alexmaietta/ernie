// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/textures.h"

#include <algorithm>
#include <span>
#include <string>

#include "ernie/io/image.h"
#include "ernie/scene/publish/change_set.h"
#include "ernie/scene/records/render_tables.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/util/profiler.h"

namespace ernie {
namespace {

TextureWrap Wrap(TextureWrapMode mode) {
    switch (mode) {
    case TextureWrapMode::Black:
        return TextureWrap::Border;
    case TextureWrapMode::Clamp:
        return TextureWrap::Clamp;
    case TextureWrapMode::Mirror:
        return TextureWrap::Mirror;
    default:
        return TextureWrap::Repeat;
    }
}

}

namespace {

std::map<SchemaId, std::vector<TextureSlot>> MaterialSlots() {
    std::map<SchemaId, std::vector<TextureSlot>> materials;
    materials[SchemaId::UsdUVTexture] = {
        {.table = SchemaId::UsdUVTexture, .texture = kUsdUVTextureFile.index, .color = true}};
    materials[SchemaId::DomeLight] = {
        {.table = SchemaId::DomeLight, .texture = kDomeLightTextureFile.index, .color = true}};
    for (const SchemaId light : {SchemaId::SphereLight, SchemaId::RectLight, SchemaId::DiskLight,
                                 SchemaId::CylinderLight}) {
        materials[light] = {{.table = SchemaId::ShapingAPI,
                             .texture = kShapingAPIShapingIesFile.index,
                             .color = false}};
    }
    materials[SchemaId::RectLight].push_back(
        {.table = SchemaId::RectLight, .texture = kRectLightTextureFile.index, .color = true});
    return materials;
}

std::vector<ColumnRef> InputColumns(const std::map<SchemaId, std::vector<TextureSlot>>& materials) {
    std::vector<ColumnRef> inputs = {kUsdUVTextureWrapS, kUsdUVTextureWrapT,
                                     kUsdUVTextureSourceColorSpace, kCheckerboardAPICheckerboard};
    for (const auto& [schema, slots] : materials) {
        for (const TextureSlot& slot : slots) {
            const ColumnRef column = {slot.table, &Columns(slot.table)[slot.texture], slot.texture};
            if (std::ranges::none_of(inputs,
                                     [&](ColumnRef c) { return c.column == column.column; })) {
                inputs.push_back(column);
            }
        }
    }
    return inputs;
}

}

TextureTable::TextureTable(Structure& structure, unsigned threads)
    : m_structure(structure),
      m_cache(
          structure.GetColumnStore().GetDevice(),
          [&structure](const std::string& path) { return structure.FindFile(path); }, threads),
      m_slots(MaterialSlots()), m_inputs(InputColumns(m_slots)) {
    UploadBindings();
}

TextureTable::~TextureTable() = default;

void TextureTable::Describe(RenderTables& tables) const {
    tables.textures.address = m_records.Buffer() ? m_records.Buffer()->Address() : 0;
    tables.textureBindings.address = m_bindingAddress;
}

void TextureTable::Hold(std::vector<Ref<const RefCounted>>& held) const {
    HoldBuffers(held, {m_records.Buffer()});
}

bool TextureTable::Pending() const {
    return m_cache.Pending();
}

void TextureTable::Finish() {
    m_cache.Finish();
}

uint64_t TextureTable::Handle(uint32_t prim, std::string_view input) const {
    const auto found = m_materials.find(prim);
    if (found == m_materials.end()) {
        return 0;
    }
    for (const Bound& bound : found->second.slots) {
        if (Columns(bound.slot.table)[bound.slot.texture].field == input) {
            return bound.binding.handle;
        }
    }
    return 0;
}

void TextureTable::UploadBindings() {
    const uint64_t address = m_cache.UploadBindings();
    m_bindingAddress = address;
}

std::set<uint32_t> TextureTable::Collect() {
    const std::set<TextureKey> finished = m_cache.Collect();
    std::set<uint32_t> dirty;
    for (const TextureKey& key : finished) {
        for (const auto& [prim, material] : m_materials) {
            if (std::ranges::any_of(material.slots,
                                    [&](const Bound& bound) { return bound.binding.key == key; }) ||
                (material.checkered && key.first == kCheckerboardTexture)) {
                dirty.insert(prim);
            }
        }
    }
    return dirty;
}

void TextureTable::Refresh(uint32_t prim) {
    Material& material = m_materials.at(prim);
    bool resolved = false;
    for (Bound& bound : material.slots) {
        if (bound.binding.handle == 0) {
            m_cache.Resolve(bound.binding);
            resolved = resolved || bound.binding.handle != 0;
        }
    }
    if (!resolved) {
        return;
    }
    if (material.bindings == kNoTextures) {
        material.bindings = m_cache.AllocateBindings(uint32_t(material.slots.size()));
    }
    for (uint32_t index = 0; index < material.slots.size(); ++index) {
        m_cache.SetBinding(material.bindings + index, material.slots[index].binding.handle);
    }
    Publish(prim, material);
}

void TextureTable::Publish(uint32_t prim, const Material& material) {
    const Prims& prims = m_structure.GetPrims();
    const SchemaId schema = prims.Record(prim).schema;
    TextureRecord record{};
    if (schema == SchemaId::UsdUVTexture || schema == SchemaId::DomeLight) {
        record.textures = material.bindings;
    }
    for (uint32_t index = 0; index < material.slots.size(); ++index) {
        const SchemaId table = material.slots[index].slot.table;
        const uint32_t binding =
            material.bindings == kNoTextures ? kNoTextures : material.bindings + index;
        if (table == SchemaId::ShapingAPI && prims.HasApi(prim, SchemaId::ShapingAPI)) {
            record.shaping = binding;
        } else if (table == SchemaId::RectLight) {
            record.textures = binding;
        }
    }
    if (schema == SchemaId::DomeLight && material.bindings != kNoTextures) {
        if (const auto found = m_cache.Size(material.slots[0].binding.key)) {
            record.width = found->first;
            record.height = found->second;
        }
    }
    m_records.Set(prim, record);
    Rebound(prims.Record(prim).world);
}

void TextureTable::Rebound(uint32_t world) {
    if (std::ranges::find(m_rebound, world) == m_rebound.end()) {
        m_rebound.push_back(world);
    }
}

void TextureTable::Release(uint32_t prim) {
    const auto found = m_materials.find(prim);
    if (found == m_materials.end()) {
        return;
    }
    m_records.Set(prim, TextureRecord{});
    const Prims& prims = m_structure.GetPrims();
    if (prims.Alive(prim)) {
        Rebound(prims.Record(prim).world);
    } else {
        Rebound(kGlobalWorld);
    }
    const Material& material = found->second;
    for (const Bound& bound : material.slots) {
        m_cache.Unbind(bound.binding);
    }
    if (material.bindings != kNoTextures) {
        m_cache.FreeBindings(material.bindings, uint32_t(material.slots.size()));
    }
    if (material.checkered && --m_checkerUsers == 0) {
        m_cache.Unbind(m_checker);
        m_checker = {};
        m_cache.SetBinding(kCheckerTextureBinding, 0);
    }
    for (const RowKey& row : material.rows) {
        m_byRow.erase(row);
    }
    m_materials.erase(found);
}

const std::vector<uint32_t>& TextureTable::Read(Cells& cells, SchemaId table, uint32_t column) {
    auto [found, added] = cells.columns.try_emplace({table, column});
    if (added) {
        ColumnTable& rows = m_structure.GetColumnStore().Schema(table);
        found->second = Items<uint32_t>(rows.Read(column, 0, rows.Rows()));
    }
    return found->second;
}

void TextureTable::Rebind(uint32_t prim, Cells& cells) {
    const Prims& prims = m_structure.GetPrims();
    const SchemaId schema = prims.Record(prim).schema;
    Material material;
    TextureWrap wrapS = TextureWrap::Clamp;
    TextureWrap wrapT = TextureWrap::Clamp;
    TextureColorSpace space = TextureColorSpace::Auto;
    if (schema == SchemaId::UsdUVTexture) {
        const uint32_t row = prims.Row(prim, SchemaId::UsdUVTexture);
        wrapS = Wrap(
            TextureWrapMode(Read(cells, SchemaId::UsdUVTexture, kUsdUVTextureWrapS.index)[row]));
        wrapT = Wrap(
            TextureWrapMode(Read(cells, SchemaId::UsdUVTexture, kUsdUVTextureWrapT.index)[row]));
        space = TextureColorSpace(
            Read(cells, SchemaId::UsdUVTexture, kUsdUVTextureSourceColorSpace.index)[row]);
    }
    TextureBinding checker;
    if (IsA(schema, SchemaId::Plane)) {
        const uint32_t row = prims.Row(prim, SchemaId::CheckerboardAPI);
        material.checkered =
            Read(cells, SchemaId::CheckerboardAPI, kCheckerboardAPICheckerboard.index)[row] != 0;
        if (material.checkered) {
            checker = m_cache.Bind({std::string(kCheckerboardTexture), false}, TextureWrap::Repeat,
                                   TextureWrap::Repeat);
        }
        material.rows.push_back({SchemaId::CheckerboardAPI, row});
    }
    const auto slots = m_slots.find(schema);
    if (slots != m_slots.end()) {
        for (const TextureSlot& slot : slots->second) {
            if (IsApi(slot.table) && !prims.HasApi(prim, slot.table)) {
                material.slots.push_back({slot, {}});
                continue;
            }
            const uint32_t row = prims.Row(prim, slot.table);
            std::string path(
                m_structure.GetTokens().Text(Token{Read(cells, slot.table, slot.texture)[row]}));
            const bool srgb = space == TextureColorSpace::SRGB ||
                              (space == TextureColorSpace::Auto && slot.color && !path.empty() &&
                               IsPngOrJpeg(path));
            if (schema == SchemaId::DomeLight) {
                wrapS = TextureWrap::Repeat;
                wrapT = TextureWrap::Repeat;
            }
            material.slots.push_back({slot, m_cache.Bind({std::move(path), srgb}, wrapS, wrapT)});
            if (std::ranges::find(material.rows, RowKey{slot.table, row}) == material.rows.end()) {
                material.rows.push_back({slot.table, row});
            }
        }
    }

    Release(prim);
    if (material.checkered) {
        if (m_checkerUsers++ == 0 || m_checker.handle == 0) {
            m_cache.Unbind(m_checker);
            m_checker = checker;
            m_cache.SetBinding(kCheckerTextureBinding, m_checker.handle);
        } else {
            m_cache.Unbind(checker);
        }
    }
    if (std::ranges::any_of(material.slots,
                            [](const Bound& bound) { return bound.binding.handle != 0; })) {
        material.bindings = m_cache.AllocateBindings(uint32_t(material.slots.size()));
        for (uint32_t index = 0; index < material.slots.size(); ++index) {
            m_cache.SetBinding(material.bindings + index, material.slots[index].binding.handle);
        }
    }
    Publish(prim, material);
    for (const RowKey& row : material.rows) {
        m_byRow[row] = prim;
    }
    m_structure.GetColumnStore().StampWorld(prims.Record(prim).world);
    m_materials[prim] = std::move(material);
}

void TextureTable::Run(const ChangeSet& changes, TableVersions& versions) {
    ERNIE_ZONE("Textures/Run");
    const Prims& prims = m_structure.GetPrims();
    const auto textured = [&](uint32_t prim) {
        return prims.Alive(prim) && (m_slots.contains(prims.Record(prim).schema) ||
                                     IsA(prims.Record(prim).schema, SchemaId::Plane));
    };

    const std::set<uint32_t> loaded = Collect();
    std::set<uint32_t> dirty;
    if (changes.resync) {
        for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
            if (textured(prim)) {
                dirty.insert(prim);
            }
        }
    } else {
        for (const StructureEntry& entry : changes.removed) {
            Release(entry.prim);
            dirty.erase(entry.prim);
        }
        for (const StructureEntry& entry : changes.added) {
            if (textured(entry.prim) && prims.Generation(entry.prim) == entry.generation) {
                dirty.insert(entry.prim);
            }
        }
        for (const StructureEntry& entry : changes.apis) {
            if (entry.api == SchemaId::ShapingAPI && textured(entry.prim) &&
                prims.Generation(entry.prim) == entry.generation) {
                dirty.insert(entry.prim);
            }
        }
        for (const ColumnRef column : m_inputs) {
            const ChangedRows* rows = changes.Rows(column);
            if (rows == nullptr) {
                continue;
            }
            for (const uint32_t row : rows->ToHost()) {
                if (const auto prim = m_byRow.find({column.schema, row}); prim != m_byRow.end()) {
                    dirty.insert(prim->second);
                }
            }
        }
    }

    for (const uint32_t prim : loaded) {
        const auto material = m_materials.find(prim);
        if (dirty.contains(prim) || material == m_materials.end()) {
            continue;
        }
        if (material->second.checkered) {
            dirty.insert(prim);
        } else {
            Refresh(prim);
        }
    }

    Cells cells;
    for (const uint32_t prim : dirty) {
        if (textured(prim)) {
            Rebind(prim, cells);
        } else {
            Release(prim);
        }
    }
    UploadBindings();
    m_records.Reserve(prims.Size());
    if (m_records.Upload(m_structure.GetColumnStore().GetDevice())) {
        Rebound(kGlobalWorld);
    }
    if (!m_rebound.empty()) {
        versions.Record(TableId::Textures, m_rebound);
    }
    m_rebound.clear();

    m_stats.images = m_cache.Images();
    m_stats.loaded = m_cache.Loaded();
    m_stats.failed = m_cache.Failed();
    m_stats.loading = m_cache.Loading();
    m_stats.bound = 0;
    for (const auto& [prim, material] : m_materials) {
        for (const Bound& bound : material.slots) {
            m_stats.bound += bound.binding.handle != 0 ? 1 : 0;
        }
    }
}

}
