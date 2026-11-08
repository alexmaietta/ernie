// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/regions.h"

#include <algorithm>

#include "ernie/scene/schema/ids.h"

namespace ernie {

namespace {

uint32_t Room(uint32_t count) {
    return std::max(count + count / 2, count + 8);
}

}

void Regions::Assign(std::vector<std::vector<uint32_t>> lists) {
    m_items.clear();
    m_starts = {0};
    m_counts.clear();
    for (const std::vector<uint32_t>& list : lists) {
        const uint32_t room = Room(uint32_t(list.size()));
        m_items.insert(m_items.end(), list.begin(), list.end());
        m_items.resize(m_items.size() + room - list.size(), kNoRow);
        m_starts.push_back(uint32_t(m_items.size()));
        m_counts.push_back(uint32_t(list.size()));
    }
}

bool Regions::Replace(uint32_t region, std::span<const uint32_t> items) {
    if (items.size() > Capacity(region)) {
        return false;
    }
    const auto start = m_items.begin() + m_starts[region];
    std::copy(items.begin(), items.end(), start);
    std::fill(start + ptrdiff_t(items.size()), m_items.begin() + m_starts[region + 1], kNoRow);
    m_counts[region] = uint32_t(items.size());
    return true;
}

}
