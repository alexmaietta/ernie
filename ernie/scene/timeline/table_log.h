// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace ernie {

enum class TableId : uint32_t {
    Transforms,
    Imageables,
    Meshes,
    MeshData,
    Tangents,
};

inline constexpr uint32_t kTableCount = 5;

class TableLog {
  public:
    void Wrote(TableId table, std::span<const uint32_t> worlds);

    uint64_t Sequence() const {
        return m_sequence;
    }

    uint64_t Version(TableId table) const {
        return m_versions[uint32_t(table)];
    }

    void Collect(TableId table, uint64_t since, std::vector<uint32_t>& worlds) const;
    void Trim(uint64_t seen);

  private:
    struct Entry {
        uint64_t sequence = 0;
        TableId table = TableId::Transforms;
        std::vector<uint32_t> worlds;
    };

    std::vector<Entry> m_entries;
    std::array<uint64_t, kTableCount> m_versions{};
    uint64_t m_sequence = 0;
};

}
