// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/timeline/timeline.h"

#include <algorithm>
#include <cstring>

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels/changes.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"

namespace ernie {

void Timeline::Finish() {
    for (const Scheduled& scheduled : m_systems) {
        scheduled.system->Finish();
    }
}

bool Timeline::Pending() const {
    return m_catalog.GetStore().WriteSequence() != m_writes || m_catalog.Epoch() != m_epoch ||
           std::ranges::any_of(
               m_systems, [](const Scheduled& scheduled) { return scheduled.system->Pending(); });
}

void Timeline::Add(System& system) {
    for (const TableId output : system.TableOutputs()) {
        for (const TableId input : system.TableInputs()) {
            if (input == output) {
                throw Error("System {} reads its own table output", system.Name());
            }
        }
        for (const Scheduled& earlier : m_systems) {
            for (const TableId input : earlier.system->TableInputs()) {
                if (input == output) {
                    throw Error("System {} writes a table {} reads, so add {} first", system.Name(),
                                earlier.system->Name(), system.Name());
                }
            }
        }
    }
    m_systems.push_back({&system});
}

Ordinal Timeline::StampAt(Ordinal ordinal) const {
    const auto after = std::ranges::upper_bound(m_sealed, ordinal, {}, &Sealed::ordinal);
    return after == m_sealed.begin() ? 0 : std::prev(after)->stamp;
}

Ordinal Timeline::OrdinalAt(Ordinal stamp, Ordinal sealing) const {
    const auto found = std::ranges::lower_bound(m_sealed, stamp, {}, &Sealed::stamp);
    return found == m_sealed.end() ? sealing : found->ordinal;
}

ChangeSet Timeline::Build(Ordinal since, Ordinal until, Ordinal sealing,
                          std::span<const ColumnRef> columns,
                          std::vector<Ref<device::Buffer>>* reused) {
    Store& store = m_catalog.GetStore();
    ChangeSet changes;
    changes.since = since == 0 ? 0 : OrdinalAt(since, sealing);
    changes.until = OrdinalAt(until, sealing);
    changes.sinceStamp = since;
    changes.untilStamp = until;
    changes.resync = since == 0;
    for (StructureEntry entry : m_catalog.Log()) {
        if (entry.ordinal <= since || entry.ordinal > until) {
            continue;
        }
        entry.ordinal = OrdinalAt(entry.ordinal, sealing);
        if (entry.api != SchemaId::None) {
            changes.apis.push_back(entry);
        } else {
            (entry.added ? changes.added : changes.removed).push_back(entry);
        }
    }
    changes.structure = !changes.added.empty() || !changes.removed.empty();
    const uint32_t worlds = m_catalog.GetPrims().WorldCount();
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
        const Table& table = store.Schema(column.schema);
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

void Timeline::RunSystems(Ordinal stamp, Ordinal ordinal) {
    Store& store = m_catalog.GetStore();
    for (Scheduled& scheduled : m_systems) {
        System& system = *scheduled.system;
        const bool written = std::ranges::any_of(system.Inputs(), [&](ColumnRef input) {
            return store.LastWrite(input) > scheduled.writes;
        });
        const bool restructured = system.Structural() && m_catalog.Epoch() != scheduled.epoch;
        const bool tabled = std::ranges::any_of(system.TableInputs(), [&](TableId table) {
            return m_log.Version(table) > scheduled.tables;
        });
        if (scheduled.ran && !written && !restructured && !tabled && !system.Pending()) {
            continue;
        }
        ERNIE_ZONE("Timeline/System");
        ChangeSet changes = Build(scheduled.ran ? scheduled.stamp : 0, stamp, ordinal,
                                  system.Inputs(), &scheduled.changed);
        for (const TableId table : system.TableInputs()) {
            if (m_log.Version(table) > scheduled.tables) {
                changes.wrote.push_back(table);
                m_log.Collect(table, scheduled.tables, changes.worlds);
            }
        }
        const uint64_t writes = store.WriteSequence();
        const uint64_t epoch = m_catalog.Epoch();
        const uint64_t tables = m_log.Sequence();
        changes.tables = &m_tables;
        changes.log = &m_log;
        system.Run(changes, Stream::None());
        system.Describe(m_tables);
        scheduled.writes = writes;
        scheduled.epoch = epoch;
        scheduled.tables = tables;
        scheduled.stamp = stamp;
        scheduled.ran = true;
    }
    Ordinal seen = stamp;
    for (const Scheduled& scheduled : m_systems) {
        seen = std::min(seen, scheduled.stamp);
    }
    store.TrimArrayWrites(seen);
    uint64_t logged = m_log.Sequence();
    for (const Scheduled& scheduled : m_systems) {
        if (!scheduled.system->TableInputs().empty()) {
            logged = std::min(logged, scheduled.tables);
        }
    }
    m_log.Trim(logged);
}

device::Operation Timeline::Publish(Ordinal ordinal, Stream producer) {
    ERNIE_ZONE("Timeline/Publish");
    if (!m_sealed.empty() && ordinal <= m_sealed.back().ordinal) {
        throw Error("Publishing ordinal {}, but {} is already published", ordinal,
                    m_sealed.back().ordinal);
    }
    Store& store = m_catalog.GetStore();
    device::Queue& queue = store.GetDevice().GetQueue();
    queue.Follow(producer);
    const Ordinal stamp = store.Stamp();
    device::Operation done;
    if (Pending()) {
        queue.BeginBatch();
        try {
            store.Flush();
            RunSystems(stamp, ordinal);
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
    m_epoch = m_catalog.Epoch();
    return done;
}

ChangeSet Timeline::Changes(Ordinal since, std::span<const ColumnRef> columns) {
    if (since > Published()) {
        throw Error("Changes since {}, but only {} is published", since, Published());
    }
    const Ordinal until = m_sealed.empty() ? 0 : m_sealed.back().stamp;
    return Build(StampAt(since), until, Published(), columns);
}

void Timeline::Seal() {
    RenderTables tables = m_tables;
    tables.worldCount = m_catalog.GetPrims().WorldCount();
    m_tables.worldCount = tables.worldCount;
    auto pinned = std::make_shared<PinnedTables>();
    if (m_pinned && std::memcmp(&tables, &m_pinned->tables, sizeof(tables)) == 0) {
        pinned->header = m_pinned->header;
    } else {
        pinned->header = m_catalog.GetStore().GetDevice().CreateBuffer(
            {.label = "render tables"}, std::span<const RenderTables>(&tables, 1));
    }
    pinned->tables = tables;
    pinned->held.emplace_back(pinned->header);
    for (const Scheduled& scheduled : m_systems) {
        scheduled.system->Hold(pinned->held);
    }
    m_pinned = std::move(pinned);
}

Snapshot Timeline::Acquire(Stream consumer) {
    if (m_sealed.empty()) {
        throw Error("Nothing is published yet");
    }
    Store& store = m_catalog.GetStore();
    Snapshot snapshot;
    snapshot.ordinal = Published();
    snapshot.sceneData = store.DeviceData();
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

void Timeline::Release(const Snapshot& snapshot, Stream consumer) {
    if (m_outstanding == 0 || snapshot.ordinal == 0) {
        throw Error("Releasing a snapshot that wasn't acquired");
    }
    m_catalog.GetStore().GetDevice().GetQueue().Follow(consumer);
    --m_outstanding;
}

}
