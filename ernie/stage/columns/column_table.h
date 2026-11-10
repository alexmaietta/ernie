// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/ref.h"
#include "ernie/stage/columns/device_heap.h"

namespace ernie {

struct TableColumn {
    uint32_t cellBytes = 4;
    uint32_t fill = 0;
};

class ColumnTable {
  public:
    ColumnTable(DeviceHeap& heap, std::string label, std::vector<TableColumn> columns);

    const std::string& Label() const {
        return m_label;
    }

    uint32_t Rows() const {
        return m_rows;
    }

    uint32_t Capacity() const {
        return m_capacity;
    }

    uint32_t ColumnCount() const {
        return uint32_t(m_columns.size());
    }

    uint32_t CellBytes(uint32_t column) const;
    uint32_t AddColumn(TableColumn column);

    uint64_t Layout() const {
        return m_layout;
    }

    uint32_t Allocate(uint32_t count, bool pageAligned = false);
    uint32_t Acquire();
    void Release(uint32_t row);

    uint64_t Pages(uint32_t column) const;
    std::vector<device::BufferRange> Ranges(uint32_t column, uint32_t first, uint32_t count) const;
    std::optional<device::BufferRange> Contiguous(uint32_t column, uint32_t first,
                                                  uint32_t count) const;

    void Redirect(uint32_t column, uint32_t first, uint32_t count, Ref<device::Buffer> buffer);
    Ref<device::Buffer> Restore(uint32_t column, uint32_t first);
    bool Redirected(uint32_t column, uint32_t first, uint32_t count) const;

    using Scatter = std::function<void(uint32_t column, std::span<const uint32_t> rows,
                                       std::span<const std::byte> cells)>;

    void Stage(uint32_t column, uint32_t first, std::span<const std::byte> bytes);
    bool Staged() const;
    void Flush(const Scatter& scatter = {});
    std::vector<std::byte> Read(uint32_t column, uint32_t first, uint32_t count);

  private:
    struct Span {
        uint32_t first = 0;
        uint32_t rows = 0;
        std::vector<HeapBlock> blocks;
    };

    struct Redirection {
        uint32_t first = 0;
        uint32_t rows = 0;
        Ref<device::Buffer> buffer;
    };

    struct Staging {
        std::vector<uint32_t> rows;
        std::vector<std::byte> bytes;
    };

    const Span& Find(uint32_t row) const;
    const Redirection* FindRedirection(uint32_t column, uint32_t row) const;
    void PointPages(uint32_t column, uint32_t first, uint32_t count, uint64_t address);
    void CheckRows(uint32_t first, uint32_t count) const;
    void Grow(uint32_t count);
    void WritePages(uint32_t firstPage, uint32_t pageCount);

    DeviceHeap& m_heap;
    std::string m_label;
    std::vector<TableColumn> m_columns;
    std::vector<Span> m_spans;
    std::vector<std::vector<uint64_t>> m_pages;
    Ref<device::Buffer> m_pageTable;
    uint32_t m_pageCapacity = 0;
    std::vector<Staging> m_staged;
    std::vector<std::vector<Redirection>> m_redirections;
    std::map<uint32_t, uint32_t> m_aligned;
    std::vector<uint32_t> m_free;
    uint32_t m_rows = 0;
    uint32_t m_capacity = 0;
    uint64_t m_layout = 0;
};

}
