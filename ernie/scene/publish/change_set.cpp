// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/publish/change_set.h"

#include <algorithm>

#include "ernie/device/queue.h"
#include "ernie/scene/kernels/changes.h"

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

}
