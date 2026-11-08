// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/columns/column_store.h"

#include <algorithm>
#include <cstring>
#include <span>

#include <fmt/format.h>

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/columns/kernels/cells.h"
#include "ernie/scene/kernels.h"
#include "ernie/util/error.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"

namespace ernie {

namespace {

constexpr uint32_t kNoPrimFill = 0xFFFFFFFF;
constexpr uint64_t kLargestStage = 64 * 1024;

std::vector<std::byte> Repeat(std::span<const std::byte> cell, uint32_t count) {
    std::vector<std::byte> bytes;
    bytes.reserve(cell.size() * count);
    for (uint32_t i = 0; i < count; ++i) {
        bytes.insert(bytes.end(), cell.begin(), cell.end());
    }
    return bytes;
}

std::vector<TableColumn> SchemaColumns(SchemaId schema, std::vector<uint32_t>& stamps) {
    std::vector<TableColumn> columns;
    const uint32_t count = Info(schema).columnCount;
    for (uint32_t column = 0; column < count; ++column) {
        columns.push_back({ElementBytes(Columns(schema)[column].type), 0});
    }
    columns.push_back({sizeof(uint32_t), kNoPrimFill});
    for (uint32_t column = 0; column < count; ++column) {
        if (Has(Columns(schema)[column].flags, ColumnFlags::Tracked)) {
            stamps.push_back(uint32_t(columns.size()));
            columns.push_back({sizeof(uint64_t), 0});
        } else {
            stamps.push_back(kNoStamp);
        }
    }
    return columns;
}

}

ColumnStore::ColumnStore(device::Device& device)
    : m_device(device), m_heap(device, "scene"),
      m_prims(m_heap, "prims",
              {{4, 0}, {4, kNoPrimFill}, {4, kNoPrimFill}, {4, 0}, {4, kNoPrimFill}, {8, 0}}),
      m_slots(m_heap, "prim slots", {{4, kNoPrimFill}}),
      m_worlds(m_heap, "world stamps", {{sizeof(uint64_t), 0}}) {
    m_entries.reserve(uint32_t(SchemaId::Count));
    for (uint32_t id = 0; id < uint32_t(SchemaId::Count); ++id) {
        const SchemaId schema = SchemaId(id);
        Entry entry;
        std::vector<TableColumn> columns = SchemaColumns(schema, entry.stamps);
        entry.table = std::make_unique<ColumnTable>(
            m_heap, fmt::format("{} data", Info(schema).name), columns);
        entry.writes.assign(Info(schema).columnCount, 0);
        m_entries.push_back(std::move(entry));
    }
    m_worlds.Allocate(1);
}

const ColumnStore::Entry& ColumnStore::Lookup(SchemaId schema) const {
    if (uint32_t(schema) >= m_entries.size()) {
        throw Error("No data for schema {}", uint32_t(schema));
    }
    return m_entries[uint32_t(schema)];
}

ColumnTable& ColumnStore::Schema(SchemaId schema) {
    return *Lookup(schema).table;
}

const ColumnTable& ColumnStore::Schema(SchemaId schema) const {
    return *Lookup(schema).table;
}

void ColumnStore::Pin(SchemaId schema) {
    Lookup(schema);
    ++m_entries[uint32_t(schema)].pins;
}

void ColumnStore::Unpin(SchemaId schema) {
    if (!Pinned(schema)) {
        throw Error("{} data isn't pinned", Info(schema).name);
    }
    --m_entries[uint32_t(schema)].pins;
}

ColumnStore::~ColumnStore() {
    if (!m_attachments.empty()) {
        try {
            m_device.GetQueue().Mark().Wait();
        } catch (...) {
        }
    }
}

namespace {

bool SameColumn(ColumnRef a, ColumnRef b) {
    return a.schema == b.schema && a.index == b.index;
}

}

void ColumnStore::Attach(Attachment attachment) {
    if (Attached(attachment.column, attachment.rows)) {
        throw Error("{}'s {} is already attached for some of these prims",
                    Info(attachment.column.schema).name, attachment.column.column->field);
    }
    attachment.sorted = attachment.rows;
    std::ranges::sort(attachment.sorted);
    Pin(attachment.column.schema);
    m_attachments.push_back(std::move(attachment));
}

const Attachment* ColumnStore::FindAttachment(ColumnRef column,
                                              std::span<const uint32_t> rows) const {
    for (const Attachment& attachment : m_attachments) {
        if (SameColumn(attachment.column, column) && std::ranges::equal(attachment.rows, rows)) {
            return &attachment;
        }
    }
    return nullptr;
}

bool ColumnStore::Attached(ColumnRef column, std::span<const uint32_t> rows) const {
    for (const Attachment& attachment : m_attachments) {
        if (!SameColumn(attachment.column, column)) {
            continue;
        }
        for (const uint32_t row : rows) {
            if (row != kNoRow && std::ranges::binary_search(attachment.sorted, row)) {
                return true;
            }
        }
    }
    return false;
}

Attachment ColumnStore::Detach(ColumnRef column, std::span<const uint32_t> rows) {
    const auto found = std::ranges::find_if(m_attachments, [&](const Attachment& attachment) {
        return SameColumn(attachment.column, column) && std::ranges::equal(attachment.rows, rows);
    });
    if (found == m_attachments.end()) {
        throw Error("{}'s {} isn't attached for these prims", Info(column.schema).name,
                    column.column->field);
    }
    Attachment attachment = std::move(*found);
    m_attachments.erase(found);
    Unpin(column.schema);
    return attachment;
}

bool ColumnStore::Pinned(SchemaId schema) const {
    return Lookup(schema).pins > 0;
}

ArrayPool& ColumnStore::Arrays(ColumnType type) {
    const uint32_t item = ItemBytes(type);
    if (item == 0) {
        throw Error("Column type {} holds no arrays", uint32_t(type));
    }
    std::unique_ptr<ArrayPool>& arrays = m_arrayHeaps[item];
    if (!arrays) {
        arrays = std::make_unique<ArrayPool>(m_heap, fmt::format("{}-byte arrays", item), item);
    }
    return *arrays;
}

uint32_t ColumnStore::PrimsColumn(SchemaId schema) const {
    Lookup(schema);
    return Info(schema).columnCount;
}

uint32_t ColumnStore::StampColumn(ColumnRef column) const {
    const std::vector<uint32_t>& stamps = Lookup(column.schema).stamps;
    if (column.index >= stamps.size()) {
        throw Error("{} has no column {}", Info(column.schema).name, column.index);
    }
    return stamps[column.index];
}

bool ColumnStore::Tracked(ColumnRef column) const {
    return StampColumn(column) != kNoStamp;
}

void ColumnStore::Touch(ColumnRef column) {
    StampColumn(column);
    m_entries[uint32_t(column.schema)].writes[column.index] = ++m_writes;
}

uint64_t ColumnStore::LastWrite(ColumnRef column) const {
    StampColumn(column);
    return Lookup(column.schema).writes[column.index];
}

StampTarget ColumnStore::Stamps(ColumnRef column) {
    const uint32_t stamps = StampColumn(column);
    const ColumnTable& table = Schema(column.schema);
    StampTarget target{};
    target.stamps.address = stamps == kNoStamp ? 0 : table.Pages(stamps);
    target.tablePrims.address = table.Pages(PrimsColumn(column.schema));
    target.scene.address = DeviceData();
    target.stamp = m_stamp;
    return target;
}

void ColumnStore::StampWorld(uint32_t world) {
    const uint32_t slot = WorldSlot(world);
    if (slot >= m_worlds.Rows()) {
        throw Error("World {} of a scene of {}", world, m_worldCount);
    }
    m_worlds.Stage(0, slot, std::as_bytes(std::span(&m_stamp, 1)));
    m_worldStamps[slot] = m_stamp;
}

Ordinal ColumnStore::WorldStamp(uint32_t world) const {
    const uint32_t slot = WorldSlot(world);
    if (slot >= m_worldStamps.size()) {
        throw Error("World {} of a scene of {}", world, m_worldCount);
    }
    return m_worldStamps[slot];
}

device::Program& ColumnStore::Kernels() {
    if (!m_program) {
        m_program = m_device.CreateProgram(
            {.modules = {"ernie.scene.columns.kernels.cells",
                         "ernie.scene.columns.kernels.changes"},
             .library = &kSceneKernels,
             .entries = {"CopyWords", "ScatterCells", "StampRows", "GatherCells", "FillCells",
                         "CloneRows", "WritePoses", "ReadPoses", "CompactChanged", "GatherRows"},
             .label = "scene store"});
    }
    return *m_program;
}

device::Download ColumnStore::ReadRowsAsync(const Ref<device::Buffer>& source, uint32_t stride,
                                            std::span<const uint32_t> rows) {
    if (rows.empty() || !source) {
        return {};
    }
    const uint64_t bytes = uint64_t(rows.size()) * stride;
    const Ref<device::Buffer> out = m_device.CreateBuffer({.bytes = bytes, .label = "read rows"});
    const Ref<device::Buffer> list = m_device.CreateBuffer({.label = "read row list"}, rows);
    GatherRowsParams params{};
    params.source.address = source->Address();
    params.rows.address = list->Address();
    params.out.address = out->Address();
    params.words = stride / uint32_t(sizeof(uint32_t));
    params.count = uint32_t(rows.size());
    Dispatch("GatherRows", params, params.count);
    return m_device.GetQueue().ReadAsync(out->Slice(0, bytes));
}

void ColumnStore::FillWords(ColumnTable& table, uint32_t column, uint32_t first, uint32_t count,
                            std::span<const std::byte> cell) {
    if (cell.size() != table.CellBytes(column)) {
        throw Error("{} bytes for a cell of {} bytes in {}", cell.size(), table.CellBytes(column),
                    table.Label());
    }
    if (count == 0) {
        return;
    }
    if (cell.size() * count <= kLargestStage) {
        table.Stage(column, first, Repeat(cell, count));
        return;
    }
    if (cell.size() > kMaxFillWords * sizeof(uint32_t)) {
        throw Error("Cells of {} bytes are too wide to fill", cell.size());
    }
    table.Flush();
    FillCellsParams params{};
    params.column.address = table.Pages(column);
    params.first = first;
    params.count = count;
    params.words = uint32_t(cell.size() / sizeof(uint32_t));
    std::memcpy(params.cell, cell.data(), cell.size());
    Dispatch("FillCells", params, uint64_t(count) * params.words);
}

void ColumnStore::StampRows(ColumnRef column, uint32_t first, uint32_t count) {
    const uint32_t stamps = StampColumn(column);
    if (stamps != kNoStamp) {
        FillWords(Schema(column.schema), stamps, first, count,
                  std::as_bytes(std::span(&m_stamp, 1)));
    }
}

void ColumnStore::StageLive(ColumnRef column, uint32_t first, std::span<const std::byte> cells) {
    ColumnTable& table = Schema(column.schema);
    table.Stage(column.index, first, cells);
    Touch(column);
    StampRows(column, first, uint32_t(cells.size() / table.CellBytes(column.index)));
}

bool ColumnStore::OverridesBetween(ColumnRef column, uint32_t first, uint32_t count) const {
    if (m_overrides.empty() || count == 0) {
        return false;
    }
    const auto found = m_overrides.lower_bound(ArrayKey(column, first));
    return found != m_overrides.end() && found->first <= ArrayKey(column, first + count - 1);
}

void ColumnStore::Stage(ColumnRef column, uint32_t first, std::span<const std::byte> cells) {
    const uint32_t cell = Schema(column.schema).CellBytes(column.index);
    const auto count = uint32_t(cells.size() / cell);
    if (!OverridesBetween(column, first, count)) {
        StageLive(column, first, cells);
        return;
    }
    std::vector<std::byte> live(cells.begin(), cells.end());
    for (auto at = m_overrides.lower_bound(ArrayKey(column, first));
         at != m_overrides.end() && at->first <= ArrayKey(column, first + count - 1); ++at) {
        const size_t offset = size_t(uint32_t(at->first) - first) * cell;
        at->second.authored.assign(cells.begin() + offset, cells.begin() + offset + cell);
        std::ranges::copy(at->second.layers.back().value, live.begin() + offset);
    }
    StageLive(column, first, live);
}

void ColumnStore::Fill(ColumnRef column, uint32_t first, uint32_t count,
                       std::span<const std::byte> cell) {
    if (OverridesBetween(column, first, count)) {
        Stage(column, first, Repeat(cell, count));
        return;
    }
    FillWords(Schema(column.schema), column.index, first, count, cell);
    Touch(column);
    StampRows(column, first, count);
}

void ColumnStore::LandAuthored(bool wait) const {
    std::erase_if(m_authoredReads, [&](AuthoredRead& pending) {
        if (!wait && !pending.read.Ready()) {
            return false;
        }
        pending.read.Wait();
        const auto found = m_overrides.find(pending.key);
        const Span<const std::byte> bytes = pending.read.Bytes();
        if (found != m_overrides.end() && bytes.size() == found->second.authored.size() &&
            !std::ranges::equal(bytes, found->second.layers.back().value)) {
            found->second.authored.assign(bytes.begin(), bytes.end());
        }
        return true;
    });
}

void ColumnStore::Override(ColumnRef column, uint32_t row, uint64_t owner,
                           std::span<const std::byte> authored, std::span<const std::byte> value) {
    LandAuthored(true);
    const uint32_t cell = Schema(column.schema).CellBytes(column.index);
    if (value.size() != cell) {
        throw Error("{} bytes override {}, whose cells are {}", value.size(), column.column->field,
                    cell);
    }
    OverrideCell& entry = m_overrides[ArrayKey(column, row)];
    if (entry.layers.empty()) {
        entry.authored.assign(authored.begin(), authored.end());
    }
    const auto layer = std::ranges::find(entry.layers, owner, &OverrideLayer::owner);
    if (layer != entry.layers.end()) {
        layer->value.assign(value.begin(), value.end());
    } else {
        entry.layers.push_back({owner, {value.begin(), value.end()}});
        std::ranges::sort(entry.layers, {}, &OverrideLayer::owner);
    }
    StageLive(column, row, entry.layers.back().value);
}

void ColumnStore::ClearOverride(ColumnRef column, uint32_t row, uint64_t owner) {
    LandAuthored(true);
    const auto found = m_overrides.find(ArrayKey(column, row));
    if (found == m_overrides.end()) {
        return;
    }
    std::erase_if(found->second.layers,
                  [&](const OverrideLayer& layer) { return layer.owner == owner; });
    if (found->second.layers.empty()) {
        const std::vector<std::byte> authored = std::move(found->second.authored);
        m_overrides.erase(found);
        StageLive(column, row, authored);
    } else {
        StageLive(column, row, found->second.layers.back().value);
    }
}

bool ColumnStore::Overridden(ColumnRef column, uint32_t row) const {
    return m_overrides.contains(ArrayKey(column, row));
}

std::vector<std::byte> ColumnStore::Authored(ColumnRef column, uint32_t row) const {
    LandAuthored(true);
    const auto found = m_overrides.find(ArrayKey(column, row));
    return found == m_overrides.end() ? std::vector<std::byte>() : found->second.authored;
}

void ColumnStore::Reassert(ColumnRef column, std::span<const uint32_t> rows) {
    if (m_overrides.empty()) {
        return;
    }
    ColumnTable& table = Schema(column.schema);
    for (const uint32_t row : rows) {
        if (const auto found = m_overrides.find(ArrayKey(column, row));
            found != m_overrides.end()) {
            if (const std::optional<device::BufferRange> cell =
                    table.Contiguous(column.index, row, 1)) {
                m_authoredReads.push_back(
                    {ArrayKey(column, row), m_device.GetQueue().ReadAsync(*cell)});
            }
            StageLive(column, row, found->second.layers.back().value);
        }
    }
}

void ColumnStore::DropOverrides(SchemaId table, uint32_t row) {
    if (m_overrides.empty()) {
        return;
    }
    LandAuthored(true);
    for (uint32_t index = 0; index < Info(table).columnCount; ++index) {
        m_overrides.erase(ArrayKey({table, &Columns(table)[index], index}, row));
    }
}

void ColumnStore::CloneRows(SchemaId table, std::span<const uint32_t> sources, uint32_t first,
                            uint32_t copies) {
    if (sources.empty() || copies == 0) {
        return;
    }
    Flush();
    ColumnTable& rows = Schema(table);
    const Ref<device::Buffer> sourceRows = m_device.CreateBuffer({.label = "cloned rows"}, sources);
    const uint32_t count = uint32_t(sources.size());
    const Span<const ColumnInfo> columns = Columns(table);
    for (uint32_t index = 0; index < Info(table).columnCount; ++index) {
        const ColumnInfo& info = columns[index];
        const ColumnRef column{table, &info, index};
        CloneRowsParams params{};
        params.column.address = rows.Pages(index);
        params.sources.address = sourceRows->Address();
        params.first = first;
        params.count = count;
        params.copies = copies;
        params.words = rows.CellBytes(index) / sizeof(uint32_t);
        Dispatch("CloneRows", params, uint64_t(copies) * count * params.words);
        Touch(column);
        StampRows(column, first, count * copies);
        for (uint32_t source = 0; source < count && !m_overrides.empty(); ++source) {
            const auto found = m_overrides.find(ArrayKey(column, sources[source]));
            if (found == m_overrides.end()) {
                continue;
            }
            for (uint32_t copy = 0; copy < copies; ++copy) {
                StageLive(column, first + copy * count + source, found->second.authored);
            }
        }
    }
}

uint64_t ColumnStore::ArrayKey(ColumnRef column, uint32_t row) {
    return (uint64_t(column.schema) << 48) | (uint64_t(column.index) << 32) | row;
}

ArrayRef ColumnStore::Array(ColumnRef column, uint32_t row) const {
    const auto found = m_arrays.find(ArrayKey(column, row));
    return found == m_arrays.end() ? ArrayRef{} : found->second;
}

bool ColumnStore::Exclusive(ColumnRef column, uint32_t row) const {
    const ArrayRef array = Array(column, row);
    if (array.address == 0) {
        return false;
    }
    const uint32_t rows[] = {row};
    return Attached(column, rows) ||
           const_cast<ColumnStore&>(*this).Arrays(column.column->type).Borrowed(array);
}

void ColumnStore::SetArray(ColumnRef column, uint32_t row, const ArrayRef& array) {
    const uint64_t key = ArrayKey(column, row);
    if (const auto found = m_arrays.find(key); found != m_arrays.end()) {
        const ArrayRef old = found->second;
        m_arrays.erase(found);
        if (old.address != array.address) {
            Arrays(column.column->type).Free(old);
        }
    }
    if (array.address != 0) {
        m_arrays[key] = array;
    }
    Stage(column, row, std::as_bytes(std::span(&array, 1)));
    m_arrayWrites[ArrayKey(column, 0)].push_back({m_stamp, row});
}

std::vector<uint32_t> ColumnStore::ArrayRows(ColumnRef column, Ordinal since, Ordinal until) const {
    std::vector<uint32_t> rows;
    const auto found = m_arrayWrites.find(ArrayKey(column, 0));
    if (found == m_arrayWrites.end()) {
        return rows;
    }
    const std::vector<ArrayWrite>& writes = found->second;
    for (auto at = std::ranges::upper_bound(writes, since, {}, &ArrayWrite::stamp);
         at != writes.end() && at->stamp <= until; ++at) {
        rows.push_back(at->row);
    }
    std::ranges::sort(rows);
    rows.erase(std::ranges::unique(rows).begin(), rows.end());
    return rows;
}

void ColumnStore::LogArrayRows(ColumnRef column, std::span<const uint32_t> rows) {
    std::vector<ArrayWrite>& writes = m_arrayWrites[ArrayKey(column, 0)];
    for (const uint32_t row : rows) {
        if (row != kNoRow) {
            writes.push_back({m_stamp, row});
        }
    }
}

void ColumnStore::TrimArrayWrites(Ordinal through,
                                  std::span<const std::pair<ColumnRef, Ordinal>> held) {
    for (auto& [key, writes] : m_arrayWrites) {
        Ordinal kept = through;
        for (const auto& [column, seen] : held) {
            if (ArrayKey(column, 0) == key) {
                kept = std::min(kept, seen);
            }
        }
        writes.erase(writes.begin(),
                     std::ranges::upper_bound(writes, kept, {}, &ArrayWrite::stamp));
    }
}

void ColumnStore::SetCounts(uint32_t prims, uint32_t worlds) {
    m_primCount = prims;
    m_worldCount = worlds;
    if (worlds + 1 > m_worlds.Rows()) {
        m_worlds.Allocate(worlds + 1 - m_worlds.Rows());
    }
    m_worldStamps.resize(worlds + 1, 0);
}

void ColumnStore::Place(std::byte* data, uint32_t offset, uint64_t address) const {
    std::memcpy(data + offset, &address, sizeof(address));
}

void ColumnStore::Flush() {
    ERNIE_ZONE("ColumnStore/Flush");
    LandAuthored(false);

    struct Scattered {
        ColumnTable* table;
        uint32_t column;
        uint32_t count;
        uint32_t words;
        uint64_t rows;
        uint64_t cells;
    };

    std::vector<Scattered> scattered;
    std::vector<std::byte>& payload = m_payload;
    payload.clear();
    const auto flush = [&](ColumnTable& table) {
        table.Flush(
            [&](uint32_t column, std::span<const uint32_t> rows, std::span<const std::byte> cells) {
                const uint64_t at = payload.size();
                const std::span<const std::byte> rowBytes = std::as_bytes(rows);
                payload.insert(payload.end(), rowBytes.begin(), rowBytes.end());
                payload.insert(payload.end(), cells.begin(), cells.end());
                scattered.push_back({&table, column, uint32_t(rows.size()),
                                     uint32_t(cells.size() / rows.size() / sizeof(uint32_t)), at,
                                     at + rowBytes.size()});
            });
    };
    for (Entry& entry : m_entries) {
        flush(*entry.table);
    }
    flush(m_prims);
    flush(m_slots);
    flush(m_worlds);
    std::vector<CopyRun> runs;
    uint32_t total = 0;
    device::Queue& queue = m_device.GetQueue();
    for (const auto& [type, heap] : m_arrayHeaps) {
        const std::span<const std::byte> staged = heap->StagedBytes();
        if (staged.empty()) {
            continue;
        }
        const uint64_t base = queue.Upload(staged.data(), staged.size()).Address();
        for (const ArrayPool::StagedCopy& copy : heap->TakeStaged(base)) {
            runs.push_back({copy.dst, copy.source, total, 0});
            total += copy.words;
        }
    }
    if (!runs.empty()) {
        const std::span<const std::byte> table = std::as_bytes(std::span(runs));
        Dispatch("CopyWords",
                 CopyWordsParams{.runs = {queue.Upload(table.data(), table.size()).Address()},
                                 .runCount = uint32_t(runs.size()),
                                 .total = total},
                 total);
    }
    if (scattered.empty()) {
        return;
    }
    const Ref<device::Buffer> uploaded =
        m_device.CreateBuffer({.label = "scattered cells"}, std::span<const std::byte>(payload));
    for (const Scattered& scatter : scattered) {
        CellsParams params{};
        params.column.address = scatter.table->Pages(scatter.column);
        params.rows.address = uploaded->Address() + scatter.rows;
        params.cells.address = uploaded->Address() + scatter.cells;
        params.count = scatter.count;
        params.words = scatter.words;
        Dispatch("ScatterCells", params, uint64_t(scatter.count) * scatter.words);
    }
}

AuthoredData ColumnStore::Data() const {
    AuthoredData data{};
    data.prims.schema.pages.address = m_prims.Pages(kPrimSchema);
    data.prims.parent.pages.address = m_prims.Pages(kPrimParent);
    data.prims.world.pages.address = m_prims.Pages(kPrimWorld);
    data.prims.flags.pages.address = m_prims.Pages(kPrimFlags);
    data.prims.slots.pages.address = m_prims.Pages(kPrimSlots);
    data.prims.apis.pages.address = m_prims.Pages(kPrimApis);
    data.rows.pages.address = m_slots.Pages(0);
    data.worldStamps.pages.address = m_worlds.Pages(0);
    data.textureBindings.address = m_textureBindings;

    std::byte* schemas = reinterpret_cast<std::byte*>(&data.schemas);
    uint32_t base = 0;
    for (uint32_t id = 0; id < uint32_t(SchemaId::Count); ++id) {
        const SchemaId schema = SchemaId(id);
        const SchemaInfo& info = Info(schema);
        const Entry& entry = m_entries[id];
        const Span<const ColumnInfo> columns = Columns(schema);
        for (uint32_t column = 0; column < Info(schema).columnCount; ++column) {
            const ColumnInfo& at = columns[column];
            Place(schemas + base, at.offset, entry.table->Pages(column));
            if (entry.stamps[column] != kNoStamp) {
                Place(schemas + base, at.stampOffset, entry.table->Pages(entry.stamps[column]));
            }
        }
        Place(schemas + base, info.dataBytes - sizeof(uint64_t),
              entry.table->Pages(Info(schema).columnCount));
        base += info.dataBytes;
    }
    data.primCount = m_primCount;
    data.worldCount = m_worldCount;
    return data;
}

uint64_t ColumnStore::DeviceData() {
    ERNIE_ZONE("ColumnStore/DeviceData");
    Flush();
    if (!m_deviceData) {
        m_deviceData =
            m_device.CreateBuffer({.bytes = sizeof(AuthoredData), .label = "scene data"});
    }
    const AuthoredData data = Data();
    const std::span<const std::byte> bytes = std::as_bytes(std::span(&data, 1));
    if (m_uploaded.size() != bytes.size() ||
        std::memcmp(m_uploaded.data(), bytes.data(), bytes.size()) != 0) {
        m_device.GetQueue().Write(*m_deviceData, &data);
        m_uploaded.assign(bytes.begin(), bytes.end());
    }
    return m_deviceData->Address();
}

}
