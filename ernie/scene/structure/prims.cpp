// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/structure/prims.h"

#include <algorithm>
#include <set>

#include "ernie/util/error.h"

namespace ernie {

namespace {

std::span<const std::byte> Word(const uint32_t& value) {
    return std::as_bytes(std::span(&value, 1));
}

uint64_t BlockApis(const BlockShape& shape, uint32_t index) {
    const uint64_t applied = shape.apis.empty() ? 0 : shape.apis[index];
    return (Info(shape.schemas[index]).apis | applied) & ~kMultipleApis;
}

uint64_t InstanceKey(uint32_t prim, SchemaId api) {
    return (uint64_t(prim) << 6) | Info(api).api;
}

}

bool Prims::Alive(uint32_t prim) const {
    return prim < m_records.size() && (m_records[prim].flags & kPrimAlive) != 0;
}

uint32_t Prims::Generation(uint32_t prim) const {
    if (prim >= m_records.size()) {
        throw Error("No prim {}", prim);
    }
    return m_records[prim].generation;
}

const PrimRecord& Prims::Record(uint32_t prim) const {
    if (!Alive(prim)) {
        throw Error("No prim {}", prim);
    }
    return m_records[prim];
}

uint32_t Prims::FindRow(uint32_t prim, SchemaId table) const {
    const PrimRecord& record = Record(prim);
    const uint32_t slot = PrimSlot(record.schema, record.apis, table);
    return slot == kNoSlot ? kNoRow : m_slotRows[record.slots + slot];
}

std::vector<uint32_t> Prims::Rows(std::span<const uint32_t> prims, SchemaId table) const {
    std::vector<uint32_t> rows;
    rows.reserve(prims.size());
    for (const uint32_t prim : prims) {
        rows.push_back(Alive(prim) ? FindRow(prim, table) : kNoRow);
    }
    return rows;
}

uint32_t Prims::Row(uint32_t prim, SchemaId table) const {
    const uint32_t row = FindRow(prim, table);
    if (row == kNoRow) {
        throw Error("Prim {}, a {}, has no {} data", prim, Info(Record(prim).schema).name,
                    Info(table).name);
    }
    return row;
}

bool Prims::HasApi(uint32_t prim, SchemaId api) const {
    return ernie::HasApi(Record(prim).apis, api);
}

std::span<const ApiInstance> Prims::Instances(uint32_t prim, SchemaId api) const {
    const auto found = m_instances.find(InstanceKey(prim, api));
    return found == m_instances.end() ? std::span<const ApiInstance>() : found->second;
}

uint32_t Prims::InstanceRow(uint32_t prim, SchemaId api, uint32_t instance) const {
    for (const ApiInstance& entry : Instances(prim, api)) {
        if (entry.instance == instance) {
            return entry.row;
        }
    }
    return kNoRow;
}

void Prims::Check(const BlockShape& shape, std::span<const uint32_t> parents) const {
    if (shape.parents.size() != shape.schemas.size()) {
        throw Error("A block of {} prims with {} parents", shape.schemas.size(),
                    shape.parents.size());
    }
    if (!shape.apis.empty() && shape.apis.size() != shape.schemas.size()) {
        throw Error("A block of {} prims with {} API masks", shape.schemas.size(),
                    shape.apis.size());
    }
    bool worlds = false;
    for (uint32_t i = 0; i < shape.schemas.size(); ++i) {
        const SchemaId schema = shape.schemas[i];
        if (uint32_t(schema) >= uint32_t(SchemaId::Count)) {
            throw Error("No schema {}", uint32_t(schema));
        }
        if (Info(schema).kind != SchemaKind::Concrete) {
            throw Error("{} is {}, so no prim is one", Info(schema).name,
                        Info(schema).kind == SchemaKind::Api ? "an API schema" : "abstract");
        }
        for (const SchemaId api : kApiSchemas) {
            if (ernie::HasApi(BlockApis(shape, i), api) && !Includes(schema, api) &&
                !AppliesTo(api, schema)) {
                throw Error("{} doesn't apply to {}", Info(api).name, Info(schema).name);
            }
        }
        const uint32_t parent = shape.parents[i];
        if (parent != kNoPrim && parent >= i) {
            throw Error("Prim {} of a block has parent {}, which doesn't come before it", i,
                        parent);
        }
        if (IsA(schema, SchemaId::World)) {
            worlds = true;
            for (uint32_t up = parent; up != kNoPrim; up = shape.parents[up]) {
                if (IsA(shape.schemas[up], SchemaId::World)) {
                    throw Error("A World inside a World; worlds don't nest");
                }
            }
        }
    }
    for (const uint32_t parent : parents) {
        if (parent == kNoPrim) {
            continue;
        }
        const PrimRecord& record = Record(parent);
        if (worlds && record.world != kGlobalWorld) {
            throw Error("Worlds added inside world {}; worlds don't nest", record.world);
        }
    }
}

uint32_t Prims::AcquireSlots(uint32_t count) {
    if (count == 0) {
        return kNoRow;
    }
    std::vector<uint32_t>& free = m_freeSlots[count];
    if (!free.empty()) {
        const uint32_t first = free.back();
        free.pop_back();
        return first;
    }
    const uint32_t first = m_store.Slots().Allocate(count);
    m_slotRows.resize(m_store.Slots().Rows(), kNoRow);
    return first;
}

PrimBlock Prims::AllocateOne(SchemaId schema, uint64_t apis) {
    PrimBlock block;
    block.count = 1;
    block.copies = 1;
    if (m_freePrims.empty()) {
        block.firstPrim = m_store.Prims().Allocate(1);
        m_records.resize(m_store.Prims().Rows());
    } else {
        block.firstPrim = m_freePrims.back();
        m_freePrims.pop_back();
    }
    const std::vector<SchemaId> tables = PrimTables(schema, apis);
    block.slotsPerCopy = uint32_t(tables.size());
    block.firstSlot = AcquireSlots(block.slotsPerCopy);
    for (const SchemaId table : tables) {
        block.tables.push_back({table, m_store.Schema(table).Acquire(), 1});
    }
    return block;
}

PrimBlock Prims::AllocateBlock(const BlockShape& shape, uint32_t copies, bool pageAligned) {
    PrimBlock block;
    block.count = uint32_t(shape.schemas.size());
    block.copies = copies;
    block.firstPrim = m_store.Prims().Allocate(block.count * copies);
    m_records.resize(m_store.Prims().Rows());
    std::map<SchemaId, uint32_t> perCopy;
    for (uint32_t i = 0; i < block.count; ++i) {
        const std::vector<SchemaId> tables = PrimTables(shape.schemas[i], BlockApis(shape, i));
        for (const SchemaId table : tables) {
            ++perCopy[table];
        }
        block.slotsPerCopy += uint32_t(tables.size());
    }
    if (block.slotsPerCopy > 0) {
        block.firstSlot = m_store.Slots().Allocate(block.slotsPerCopy * copies);
    }
    for (const auto& [table, rows] : perCopy) {
        block.tables.push_back(
            {table, m_store.Schema(table).Allocate(rows * copies, pageAligned), rows});
    }
    return block;
}

PrimBlock Prims::Add(const BlockShape& shape, std::span<const uint32_t> parents,
                     uint32_t reuseWorld, bool pageAligned) {
    Check(shape, parents);
    const uint32_t count = uint32_t(shape.schemas.size());
    const uint32_t copies = uint32_t(parents.size());
    if (count == 0 || copies == 0) {
        return {};
    }
    PrimBlock block = count == 1 && copies == 1 && !pageAligned
                          ? AllocateOne(shape.schemas[0], BlockApis(shape, 0))
                          : AllocateBlock(shape, copies, pageAligned);
    m_slotRows.resize(m_store.Slots().Rows(), kNoRow);

    for (const SchemaId schema : shape.schemas) {
        block.worldsPerCopy += IsA(schema, SchemaId::World) ? 1 : 0;
    }
    if (block.worldsPerCopy > 0) {
        block.firstWorld = reuseWorld == kGlobalWorld ? m_worldCount : reuseWorld;
        m_worldCount = std::max(m_worldCount, block.firstWorld + block.worldsPerCopy * copies);
    }
    m_store.SetCounts(Size(), m_worldCount);

    std::vector<std::vector<uint32_t>> ordinals(count);
    std::vector<uint32_t> slotOffsets(count);
    std::map<SchemaId, uint32_t> used;
    uint32_t offset = 0;
    std::vector<std::vector<SchemaId>> primTables(count);
    for (uint32_t i = 0; i < count; ++i) {
        slotOffsets[i] = offset;
        primTables[i] = PrimTables(shape.schemas[i], BlockApis(shape, i));
        for (const SchemaId table : primTables[i]) {
            ordinals[i].push_back(used[table]++);
            ++offset;
        }
    }
    std::map<SchemaId, const BlockTable*> tables;
    for (const BlockTable& table : block.tables) {
        tables[table.table] = &table;
    }

    std::set<uint32_t> worlds;
    std::vector<uint32_t> copyWorlds(count);
    for (uint32_t copy = 0; copy < copies; ++copy) {
        uint32_t worldIndex = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t up = shape.parents[i];
            PrimRecord record;
            record.schema = shape.schemas[i];
            record.parent = up == kNoPrim ? parents[copy] : block.Prim(copy, up);
            record.world = up != kNoPrim              ? copyWorlds[up]
                           : parents[copy] == kNoPrim ? kGlobalWorld
                                                      : m_records[parents[copy]].world;
            if (IsA(record.schema, SchemaId::World)) {
                record.world = block.firstWorld + copy * block.worldsPerCopy + worldIndex++;
            }
            copyWorlds[i] = record.world;
            worlds.insert(record.world);
            record.flags = kPrimAlive;
            record.apis = BlockApis(shape, i);
            const std::vector<SchemaId>& slots = primTables[i];
            record.slots = slots.empty()
                               ? kNoRow
                               : block.firstSlot + copy * block.slotsPerCopy + slotOffsets[i];
            std::vector<uint32_t> rows;
            for (uint32_t slot = 0; slot < slots.size(); ++slot) {
                const BlockTable& table = *tables.at(slots[slot]);
                rows.push_back(table.first + copy * table.perCopy + ordinals[i][slot]);
            }
            Write(block.Prim(copy, i), record, rows);
        }
    }
    if (block.slotsPerCopy > 0) {
        const uint32_t slots = block.slotsPerCopy * copies;
        m_store.Slots().Stage(0, block.firstSlot,
                              std::as_bytes(std::span(m_slotRows.data() + block.firstSlot, slots)));
    }
    for (const uint32_t world : worlds) {
        m_store.StampWorld(world);
    }
    m_live += count * copies;
    return block;
}

void Prims::Write(uint32_t prim, const PrimRecord& record, std::span<const uint32_t> rows) {
    const uint32_t generation = m_records[prim].generation;
    m_records[prim] = record;
    m_records[prim].generation = generation;
    const std::vector<SchemaId> tables = PrimTables(record.schema, record.apis);
    for (uint32_t slot = 0; slot < tables.size(); ++slot) {
        m_slotRows[record.slots + slot] = rows[slot];
        m_store.Schema(tables[slot])
            .Stage(m_store.PrimsColumn(tables[slot]), rows[slot], Word(prim));
    }
    StageRecord(prim);
}

void Prims::StageRecord(uint32_t prim) {
    const PrimRecord& record = m_records[prim];
    const uint32_t values[] = {uint32_t(record.schema), record.parent, record.world, record.flags,
                               record.slots};
    for (uint32_t column = kPrimSchema; column <= kPrimSlots; ++column) {
        m_store.Prims().Stage(column, prim, Word(values[column]));
    }
    m_store.Prims().Stage(kPrimApis, prim, std::as_bytes(std::span(&record.apis, 1)));
}

void Prims::CheckRemovable(uint32_t prim) const {
    for (const SchemaId table : PrimTables(Record(prim).schema, Record(prim).apis)) {
        if (m_store.Pinned(table)) {
            throw Error("{} data is mapped or attached, so prims with it can't be removed until "
                        "the mapping closes or the attachment is released",
                        Info(table).name);
        }
    }
}

void Prims::Reparent(uint32_t prim, uint32_t parent) {
    m_records[prim].parent = parent;
    StageRecord(prim);
    m_store.StampWorld(m_records[prim].world);
}

void Prims::Renew(uint32_t prim) {
    ++m_records[prim].generation;
}

void Prims::Remove(uint32_t prim) {
    CheckRemovable(prim);
    const PrimRecord record = Record(prim);
    const std::vector<SchemaId> tables = PrimTables(record.schema, record.apis);
    for (uint32_t slot = 0; slot < tables.size(); ++slot) {
        if (!IsMultipleApi(tables[slot])) {
            ReleaseRow(tables[slot], m_slotRows[record.slots + slot]);
            continue;
        }
        const auto found = m_instances.find(InstanceKey(prim, tables[slot]));
        for (const ApiInstance& entry : found->second) {
            ReleaseRow(tables[slot], entry.row);
        }
        m_instances.erase(found);
    }
    if (!tables.empty()) {
        m_freeSlots[uint32_t(tables.size())].push_back(record.slots);
    }
    PrimRecord& removed = m_records[prim];
    removed = {};
    removed.generation = record.generation + 1;
    StageRecord(prim);
    m_freePrims.push_back(prim);
    m_store.StampWorld(record.world);
    --m_live;
}

BlockTable Prims::Apply(std::span<const uint32_t> prims, SchemaId api) {
    const BlockTable rows{api, m_store.Schema(api).Allocate(uint32_t(prims.size())), 1};
    for (uint32_t i = 0; i < prims.size(); ++i) {
        const uint32_t row = rows.first + i;
        m_store.Schema(api).Stage(m_store.PrimsColumn(api), row, Word(prims[i]));
        Regroup(prims[i], m_records[prims[i]].apis | ApiBit(api), api, row);
    }
    return rows;
}

void Prims::Unapply(std::span<const uint32_t> prims, SchemaId api) {
    for (const uint32_t prim : prims) {
        ReleaseRow(api, Row(prim, api));
        Regroup(prim, m_records[prim].apis & ~ApiBit(api), SchemaId::None, kNoRow);
    }
}

void Prims::Regroup(uint32_t prim, uint64_t apis, SchemaId api, uint32_t row) {
    const std::vector<SchemaId> before = PrimTables(m_records[prim].schema, m_records[prim].apis);
    const std::vector<SchemaId> after = PrimTables(m_records[prim].schema, apis);
    const uint32_t first = AcquireSlots(uint32_t(after.size()));
    PrimRecord& record = m_records[prim];
    for (uint32_t slot = 0; slot < after.size(); ++slot) {
        m_slotRows[first + slot] =
            after[slot] == api
                ? row
                : m_slotRows[record.slots + PrimSlot(record.schema, record.apis, after[slot])];
    }
    if (!before.empty()) {
        m_freeSlots[uint32_t(before.size())].push_back(record.slots);
    }
    record.slots = first;
    record.apis = apis;
    if (!after.empty()) {
        m_store.Slots().Stage(0, first,
                              std::as_bytes(std::span(m_slotRows.data() + first, after.size())));
    }
    StageRecord(prim);
    m_store.StampWorld(record.world);
}

void Prims::MarkSession(uint32_t prim) {
    m_records[prim].flags |= kPrimSession;
    StageRecord(prim);
}

bool Prims::Session(uint32_t prim) const {
    for (uint32_t at = prim; at != kNoPrim; at = m_records[at].parent) {
        if ((m_records[at].flags & kPrimSession) != 0) {
            return true;
        }
    }
    return false;
}

void Prims::ReleaseRow(SchemaId table, uint32_t row) {
    for (uint32_t index = 0; index < Info(table).columnCount; ++index) {
        const ColumnRef column{table, &Columns(table)[index], index};
        if (IsArray(column.column->type) && m_store.Array(column, row).address != 0) {
            m_store.SetArray(column, row, {});
        }
    }
    m_store.DropOverrides(table, row);
    m_store.Schema(table).Stage(m_store.PrimsColumn(table), row, Word(kNoPrim));
    m_store.Schema(table).Release(row);
}

BlockTable Prims::ApplyInstance(std::span<const uint32_t> prims, SchemaId api, uint32_t instance) {
    const BlockTable rows{api, m_store.Schema(api).Allocate(uint32_t(prims.size())), 1};
    const ColumnRef name = FindOwnColumn(api, "instance");
    for (uint32_t i = 0; i < prims.size(); ++i) {
        const uint32_t prim = prims[i];
        const uint32_t row = rows.first + i;
        m_store.Schema(api).Stage(m_store.PrimsColumn(api), row, Word(prim));
        m_store.Stage(name, row, Word(instance));
        std::vector<ApiInstance>& listed = m_instances[InstanceKey(prim, api)];
        listed.insert(listed.begin(), {instance, row});
        if (listed.size() == 1) {
            Regroup(prim, m_records[prim].apis | ApiBit(api), api, row);
        } else {
            SetHead(prim, api, row);
        }
    }
    StageLinks(prims, api);
    return rows;
}

void Prims::UnapplyInstance(std::span<const uint32_t> prims, SchemaId api, uint32_t instance) {
    for (const uint32_t prim : prims) {
        const auto found = m_instances.find(InstanceKey(prim, api));
        if (found == m_instances.end()) {
            continue;
        }
        std::vector<ApiInstance>& listed = found->second;
        const auto entry = std::ranges::find(listed, instance, &ApiInstance::instance);
        if (entry == listed.end()) {
            continue;
        }
        const bool head = entry == listed.begin();
        ReleaseRow(api, entry->row);
        listed.erase(entry);
        if (listed.empty()) {
            m_instances.erase(found);
            Regroup(prim, m_records[prim].apis & ~ApiBit(api), SchemaId::None, kNoRow);
            continue;
        }
        if (head) {
            SetHead(prim, api, listed.front().row);
        }
        StageLinks(std::span(&prim, 1), api);
    }
}

void Prims::StageLinks(std::span<const uint32_t> prims, SchemaId api) {
    const ColumnRef next = FindOwnColumn(api, "next");
    for (const uint32_t prim : prims) {
        const std::span<const ApiInstance> listed = Instances(prim, api);
        for (size_t i = 0; i < listed.size(); ++i) {
            const uint32_t following = i + 1 < listed.size() ? listed[i + 1].row : kNoRow;
            m_store.Stage(next, listed[i].row, Word(following));
        }
    }
}

void Prims::SetHead(uint32_t prim, SchemaId api, uint32_t row) {
    const PrimRecord& record = m_records[prim];
    const uint32_t slot = record.slots + ApiSlot(record.schema, record.apis, api);
    m_slotRows[slot] = row;
    m_store.Slots().Stage(0, slot, Word(row));
    m_store.StampWorld(record.world);
}

}
