// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/publish/change_set.h"

#include <algorithm>

#include "ernie/device/queue.h"
#include "ernie/scene/columns/kernels/changes.h"

namespace ernie {

device::BufferRange ChangedRows::Count() const {
    return m_buffer->Slice(0, sizeof(uint32_t));
}

device::BufferRange ChangedRows::Rows() const {
    return m_buffer->Slice(kChangedRowsOffset * sizeof(uint32_t),
                           std::max<uint64_t>(m_capacity, 1) * sizeof(uint32_t));
}

std::vector<uint32_t> ChangedRows::ToHost() const {
    if (!m_buffer) {
        return {};
    }
    device::Queue& queue = m_buffer->GetDevice().GetQueue();
    uint32_t count = 0;
    queue.Read(Count(), &count);
    std::vector<uint32_t> rows(count);
    if (count > 0) {
        queue.Read(m_buffer->Slice(kChangedRowsOffset * sizeof(uint32_t), count * sizeof(uint32_t)),
                   rows.data());
    }
    std::ranges::sort(rows);
    return rows;
}

const ChangedRows* ChangeSet::Rows(ColumnRef column) const {
    const auto found = std::ranges::find_if(
        columns, [&](const ChangedRows& rows) { return rows.Column().column == column.column; });
    return found == columns.end() ? nullptr : &*found;
}

std::span<const uint32_t> ChangeSet::ArrayRows(ColumnRef column) const {
    for (const auto& [at, rows] : arrays) {
        if (at.column == column.column) {
            return rows;
        }
    }
    return {};
}

bool ChangeSet::Wrote(TableId table) const {
    return std::ranges::find(wrote, table) != wrote.end();
}

bool ChangeSet::World(uint32_t world) const {
    return std::ranges::find(worlds, world) != worlds.end() ||
           std::ranges::find(worlds, kGlobalWorld) != worlds.end();
}

void TableVersions::Record(TableId table, std::span<const uint32_t> worlds) {
    ++m_sequence;
    m_versions[uint32_t(table)] = m_sequence;
    m_entries.push_back({m_sequence, table, {worlds.begin(), worlds.end()}});
}

void TableVersions::Collect(TableId table, uint64_t since, std::vector<uint32_t>& worlds) const {
    for (const Entry& entry : m_entries) {
        if (entry.table != table || entry.sequence <= since) {
            continue;
        }
        for (const uint32_t world : entry.worlds) {
            if (std::ranges::find(worlds, world) == worlds.end()) {
                worlds.push_back(world);
            }
        }
    }
}

void TableVersions::Trim(uint64_t seen) {
    std::erase_if(m_entries, [&](const Entry& entry) { return entry.sequence <= seen; });
}

}
