// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace ernie {

class Regions {
  public:
    void Assign(std::vector<std::vector<uint32_t>> lists);
    bool Replace(uint32_t region, std::span<const uint32_t> items);

    uint32_t Size() const {
        return uint32_t(m_counts.size());
    }

    std::span<const uint32_t> Items() const {
        return m_items;
    }

    std::span<const uint32_t> Starts() const {
        return m_starts;
    }

    std::span<const uint32_t> Counts() const {
        return m_counts;
    }

    uint32_t Capacity(uint32_t region) const {
        return m_starts[region + 1] - m_starts[region];
    }

    std::span<const uint32_t> Items(uint32_t region) const {
        return std::span(m_items).subspan(m_starts[region], m_counts[region]);
    }

  private:
    std::vector<uint32_t> m_items;
    std::vector<uint32_t> m_starts;
    std::vector<uint32_t> m_counts;
};

}
