// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/operation.h"
#include "ernie/device/ref.h"
#include "ernie/scene/schema/schema.h"
#include "ernie/scene/structure/structure.h"

namespace ernie {

enum class TableId : uint32_t;

class TableVersions {
  public:
    void Record(TableId table, std::span<const uint32_t> worlds);

    uint64_t Sequence() const {
        return m_sequence;
    }

    uint64_t Version(TableId table) const {
        return uint32_t(table) < m_versions.size() ? m_versions[uint32_t(table)] : 0;
    }

    void Collect(TableId table, uint64_t since, std::vector<uint32_t>& worlds) const;
    void Trim(uint64_t seen);

  private:
    struct Entry {
        uint64_t sequence = 0;
        TableId table{};
        std::vector<uint32_t> worlds;
    };

    std::vector<Entry> m_entries;
    std::vector<uint64_t> m_versions;
    uint64_t m_sequence = 0;
};

class ChangedRows {
  public:
    ChangedRows() = default;

    ChangedRows(ColumnRef column, Ref<device::Buffer> buffer, uint32_t capacity)
        : m_column(column), m_buffer(std::move(buffer)), m_capacity(capacity) {}

    ColumnRef Column() const {
        return m_column;
    }

    uint32_t Capacity() const {
        return m_capacity;
    }

    bool Empty() const {
        return !m_buffer;
    }

    device::BufferRange Count() const;
    device::BufferRange Rows() const;
    std::vector<uint32_t> ToHost() const;

  private:
    ColumnRef m_column;
    Ref<device::Buffer> m_buffer;
    uint32_t m_capacity = 0;
};

struct ChangeSet {
    Ordinal since = 0;
    Ordinal until = 0;
    Ordinal sinceStamp = 0;
    Ordinal untilStamp = 0;
    bool resync = false;
    bool structure = false;
    std::vector<StructureEntry> added;
    std::vector<StructureEntry> removed;
    std::vector<StructureEntry> apis;
    std::vector<uint32_t> worlds;
    std::vector<ChangedRows> columns;
    std::vector<std::pair<ColumnRef, std::vector<uint32_t>>> arrays;
    std::vector<TableId> wrote;

    const ChangedRows* Rows(ColumnRef column) const;
    std::span<const uint32_t> ArrayRows(ColumnRef column) const;
    bool World(uint32_t world) const;
    bool Global() const;
    bool Wrote(TableId table) const;
};

}
