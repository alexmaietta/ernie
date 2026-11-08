// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/publish/publisher.h"

#include <algorithm>
#include <cstring>

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/columns/kernels/changes.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"

namespace ernie {

void Publisher::Finish() {
    for (const Scheduled& scheduled : m_scheduled) {
        scheduled.table->Finish();
    }
}

bool Publisher::Pending() const {
    return m_structure.GetColumnStore().WriteSequence() != m_writes ||
           m_structure.Epoch() != m_epoch ||
           std::ranges::any_of(
               m_scheduled, [](const Scheduled& scheduled) { return scheduled.table->Pending(); });
}

void Publisher::Add(Table& table) {
    for (const TableId output : table.TableOutputs()) {
        for (const TableId input : table.TableInputs()) {
            if (input == output) {
                throw Error("Table {} reads its own output", table.Name());
            }
        }
        for (const Scheduled& earlier : m_scheduled) {
            for (const TableId input : earlier.table->TableInputs()) {
                if (input == output) {
                    throw Error("Table {} writes a table {} reads, so add {} first", table.Name(),
                                earlier.table->Name(), table.Name());
                }
            }
        }
    }
    m_scheduled.push_back({&table});
}

Ordinal Publisher::SealedStamp(Ordinal ordinal) const {
    const auto after = std::ranges::upper_bound(m_sealed, ordinal, {}, &Sealed::ordinal);
    return after == m_sealed.begin() ? 0 : std::prev(after)->stamp;
}

Ordinal Publisher::SealedOrdinal(Ordinal stamp, Ordinal sealing) const {
    const auto found = std::ranges::lower_bound(m_sealed, stamp, {}, &Sealed::stamp);
    return found == m_sealed.end() ? sealing : found->ordinal;
}

ChangeSet Publisher::Build(Ordinal since, Ordinal until, Ordinal sealing,
                           std::span<const ColumnRef> columns,
                           std::vector<Ref<device::Buffer>>* reused) {
    ColumnStore& store = m_structure.GetColumnStore();
    ChangeSet changes;
    changes.since = since == 0 ? 0 : SealedOrdinal(since, sealing);
    changes.until = SealedOrdinal(until, sealing);
    changes.sinceStamp = since;
    changes.untilStamp = until;
    changes.resync = since == 0;
    for (StructureEntry entry : m_structure.Log()) {
        if (entry.ordinal <= since || entry.ordinal > until) {
            continue;
        }
        entry.ordinal = SealedOrdinal(entry.ordinal, sealing);
        if (entry.api != SchemaId::None) {
            changes.apis.push_back(entry);
        } else {
            (entry.added ? changes.added : changes.removed).push_back(entry);
        }
    }
    changes.structure = !changes.added.empty() || !changes.removed.empty();
    const uint32_t worlds = m_structure.GetPrims().WorldCount();
    for (uint32_t slot = 0; slot <= worlds; ++slot) {
        const uint32_t world = SlotWorld(slot);
        const Ordinal stamp = store.WorldStamp(world);
        if (stamp > since && stamp <= until) {
            changes.worlds.push_back(world);
        }
    }

    device::Queue& queue = store.GetDevice().GetQueue();
    if (reused != nullptr) {
        reused->resize(columns.size());
    }
    for (size_t i = 0; i < columns.size(); ++i) {
        const ColumnRef column = columns[i];
        if (IsArray(column.column->type)) {
            changes.arrays.emplace_back(column, store.ArrayRows(column, since, until));
        }
        if (!store.Tracked(column)) {
            continue;
        }
        const ColumnTable& table = store.Schema(column.schema);
        const uint32_t rows = table.Rows();
        const uint64_t bytes =
            (uint64_t(kChangedRowsOffset) + std::max(rows, 1u)) * sizeof(uint32_t);
        Ref<device::Buffer> buffer = reused != nullptr ? (*reused)[i] : nullptr;
        if (!buffer || buffer->Bytes() < bytes) {
            buffer = store.GetDevice().CreateBuffer({.bytes = bytes, .label = "changed rows"});
            if (reused != nullptr) {
                (*reused)[i] = buffer;
            }
        }
        device::CommandEncoder encoder = queue.CreateEncoder();
        encoder.Fill(buffer->Slice(0, kChangedRowsOffset * sizeof(uint32_t)));
        queue.Submit(encoder.Finish());
        if (rows > 0) {
            CompactParams params{};
            params.stamps.address = table.Pages(store.StampColumn(column));
            params.tablePrims.address = table.Pages(store.PrimsColumn(column.schema));
            params.out.address = buffer->Address();
            params.since = since;
            params.until = until;
            params.rows = rows;
            store.Dispatch("CompactChanged", params, rows);
        }
        changes.columns.emplace_back(column, buffer, rows);
    }
    return changes;
}

void Publisher::RunTables(Ordinal stamp, Ordinal ordinal) {
    ColumnStore& store = m_structure.GetColumnStore();
    for (Scheduled& scheduled : m_scheduled) {
        Table& table = *scheduled.table;
        const bool written = std::ranges::any_of(table.Inputs(), [&](ColumnRef input) {
            return store.LastWrite(input) > scheduled.writes;
        });
        const bool restructured = table.Structural() && m_structure.Epoch() != scheduled.epoch;
        const bool tabled = std::ranges::any_of(table.TableInputs(), [&](TableId id) {
            return m_versions.Version(id) > scheduled.logged;
        });
        if (scheduled.ran && !written && !restructured && !tabled && !table.Pending()) {
            continue;
        }
        ERNIE_ZONE("Publisher/Table");
        ChangeSet changes = Build(scheduled.ran ? scheduled.stamp : 0, stamp, ordinal,
                                  table.Inputs(), &scheduled.changed);
        for (const TableId id : table.TableInputs()) {
            if (m_versions.Version(id) > scheduled.logged) {
                changes.wrote.push_back(id);
                m_versions.Collect(id, scheduled.logged, changes.worlds);
            }
        }
        const uint64_t writes = store.WriteSequence();
        const uint64_t epoch = m_structure.Epoch();
        const uint64_t logged = m_versions.Sequence();
        changes.tables = &m_tables;
        ++table.m_runs;
        table.Run(changes, m_versions);
        table.Describe(m_tables);
        scheduled.writes = writes;
        scheduled.epoch = epoch;
        scheduled.logged = logged;
        scheduled.stamp = stamp;
        scheduled.ran = true;
    }
    Ordinal seen = stamp;
    for (const Scheduled& scheduled : m_scheduled) {
        seen = std::min(seen, scheduled.stamp);
    }
    store.TrimArrayWrites(seen);
    uint64_t logged = m_versions.Sequence();
    for (const Scheduled& scheduled : m_scheduled) {
        if (!scheduled.table->TableInputs().empty()) {
            logged = std::min(logged, scheduled.logged);
        }
    }
    m_versions.Trim(logged);
}

device::Operation Publisher::Publish(Ordinal ordinal, Stream producer) {
    ERNIE_ZONE("Publisher/Publish");
    if (!m_sealed.empty() && ordinal <= m_sealed.back().ordinal) {
        throw Error("Publishing ordinal {}, but {} is already published", ordinal,
                    m_sealed.back().ordinal);
    }
    ColumnStore& store = m_structure.GetColumnStore();
    device::Queue& queue = store.GetDevice().GetQueue();
    queue.Follow(producer);
    const Ordinal stamp = store.Stamp();
    device::Operation done;
    if (Pending()) {
        queue.BeginBatch();
        try {
            store.Flush();
            RunTables(stamp, ordinal);
            Seal();
            store.DeviceData();
        } catch (...) {
            queue.EndBatch();
            store.SetStamp(stamp + 1);
            throw;
        }
        done = queue.EndBatch();
    }
    m_sealed.push_back({ordinal, stamp});
    store.SetStamp(stamp + 1);
    m_writes = store.WriteSequence();
    m_epoch = m_structure.Epoch();
    return done;
}

ChangeSet Publisher::Changes(Ordinal since, std::span<const ColumnRef> columns) {
    if (since > Published()) {
        throw Error("Changes since {}, but only {} is published", since, Published());
    }
    const Ordinal until = m_sealed.empty() ? 0 : m_sealed.back().stamp;
    return Build(SealedStamp(since), until, Published(), columns);
}

void Publisher::Seal() {
    const AuthoredData data = m_structure.GetColumnStore().Data();
    m_tables.prims = {.schema = data.prims.schema,
                      .world = data.prims.world,
                      .flags = data.prims.flags,
                      .slots = data.prims.slots,
                      .apis = data.prims.apis,
                      .rows = data.rows,
                      .parent = data.prims.parent,
                      .count = data.primCount};
    RenderTables tables = m_tables;
    tables.worldCount = m_structure.GetPrims().WorldCount();
    m_tables.worldCount = tables.worldCount;
    auto pinned = std::make_shared<PinnedTables>();
    if (m_pinned && std::memcmp(&tables, &m_pinned->tables, sizeof(tables)) == 0) {
        pinned->header = m_pinned->header;
    } else {
        pinned->header = m_structure.GetColumnStore().GetDevice().CreateBuffer(
            {.label = "render tables"}, std::span<const RenderTables>(&tables, 1));
    }
    pinned->tables = tables;
    pinned->held.emplace_back(pinned->header);
    for (const Scheduled& scheduled : m_scheduled) {
        scheduled.table->Hold(pinned->held);
    }
    m_pinned = std::move(pinned);
}

Snapshot Publisher::Acquire(Stream consumer) {
    if (m_sealed.empty()) {
        throw Error("Nothing is published yet");
    }
    ColumnStore& store = m_structure.GetColumnStore();
    Snapshot snapshot;
    snapshot.ordinal = Published();
    store.DeviceData();
    if (!m_pinned) {
        Seal();
    }
    snapshot.tables = m_pinned->header->Address();
    snapshot.pinned = m_pinned;
    snapshot.ready = store.GetDevice().GetQueue().Mark();
    snapshot.ready.Wait(consumer);
    ++m_outstanding;
    return snapshot;
}

void Publisher::Release(const Snapshot& snapshot, Stream consumer) {
    if (m_outstanding == 0 || snapshot.ordinal == 0) {
        throw Error("Releasing a snapshot that wasn't acquired");
    }
    m_structure.GetColumnStore().GetDevice().GetQueue().Follow(consumer);
    --m_outstanding;
}

}
