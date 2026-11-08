// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/publish/table_writes.h"

#include <algorithm>

namespace ernie {

void TableWrites::Add(TableId table, std::span<const uint32_t> worlds) {
    ++m_sequence;
    m_versions[uint32_t(table)] = m_sequence;
    m_entries.push_back({m_sequence, table, {worlds.begin(), worlds.end()}});
}

void TableWrites::Collect(TableId table, uint64_t since, std::vector<uint32_t>& worlds) const {
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

void TableWrites::Trim(uint64_t seen) {
    std::erase_if(m_entries, [&](const Entry& entry) { return entry.sequence <= seen; });
}

}
