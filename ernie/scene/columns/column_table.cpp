// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/columns/column_table.h"

#include <algorithm>
#include <utility>

#include <fmt/format.h>

#include "ernie/device/commands.h"
#include "ernie/device/kernels/shared.h"
#include "ernie/device/queue.h"
#include "ernie/scene/columns/kernels/paged.h"
#include "ernie/util/error.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"

namespace ernie {

namespace {

constexpr uint32_t kLargestSpan = 64 * 1024;
constexpr uint64_t kSpanAlignment = 256;

}

ColumnTable::ColumnTable(DeviceHeap& heap, std::string label, std::vector<TableColumn> columns)
    : m_heap(heap), m_label(std::move(label)), m_columns(std::move(columns)),
      m_pages(m_columns.size()), m_staged(m_columns.size()), m_redirections(m_columns.size()) {
    for (const TableColumn& column : m_columns) {
        if (column.cellBytes == 0 || column.cellBytes % 4 != 0) {
            throw Error("{} has a column of {} bytes; cells are whole words", m_label,
                        column.cellBytes);
        }
    }
}

uint32_t ColumnTable::CellBytes(uint32_t column) const {
    if (column >= m_columns.size()) {
        throw Error("{} has no column {}", m_label, column);
    }
    return m_columns[column].cellBytes;
}

uint32_t ColumnTable::Allocate(uint32_t count, bool pageAligned) {
    if (count == 0) {
        return m_rows;
    }
    if (pageAligned) {
        const uint32_t aligned = std::min(uint32_t(RoundUp(m_rows, kPageRows)), m_capacity);
        for (uint32_t row = m_rows; row < aligned; ++row) {
            m_free.push_back(row);
        }
        m_rows = aligned;
    }
    if (uint64_t(m_rows) + count > m_capacity) {
        for (uint32_t row = m_rows; row < m_capacity; ++row) {
            m_free.push_back(row);
        }
        m_rows = m_capacity;
        Grow(count);
    }
    const uint32_t first = m_rows;
    m_rows += count;
    if (pageAligned) {
        m_rows = std::min(uint32_t(RoundUp(m_rows, kPageRows)), m_capacity);
        m_aligned[first] = count;
    }
    return first;
}

uint32_t ColumnTable::Acquire() {
    if (m_free.empty()) {
        return Allocate(1);
    }
    const uint32_t row = m_free.back();
    m_free.pop_back();
    return row;
}

void ColumnTable::Release(uint32_t row) {
    CheckRows(row, 1);
    const auto aligned = m_aligned.upper_bound(row);
    if (aligned != m_aligned.begin() &&
        row < std::prev(aligned)->first + std::prev(aligned)->second) {
        return;
    }
    m_free.push_back(row);
}

void ColumnTable::Grow(uint32_t count) {
    ERNIE_ZONE("ColumnTable/Grow");
    const uint32_t rows = uint32_t(RoundUp(
        std::max(count, std::min(std::max(m_capacity, kPageRows), kLargestSpan)), kPageRows));
    Span span{m_capacity, rows, {}};
    device::Queue& queue = m_heap.GetDevice().GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    for (uint32_t column = 0; column < m_columns.size(); ++column) {
        const TableColumn& info = m_columns[column];
        const HeapBlock block = m_heap.Allocate(uint64_t(rows) * info.cellBytes, kSpanAlignment);
        encoder.Fill(block.Range(), info.fill);
        for (uint32_t page = 0; page < rows / kPageRows; ++page) {
            m_pages[column].push_back(block.Address() +
                                      uint64_t(page) * kPageRows * info.cellBytes);
        }
        span.blocks.push_back(block);
    }
    queue.Submit(encoder.Finish());

    const uint32_t firstPage = m_capacity / kPageRows;
    m_spans.push_back(std::move(span));
    m_capacity += rows;
    WritePages(firstPage, rows / kPageRows);
}

void ColumnTable::WritePages(uint32_t firstPage, uint32_t pageCount) {
    const uint32_t pages = m_capacity / kPageRows;
    device::Queue& queue = m_heap.GetDevice().GetQueue();
    if (pages > m_pageCapacity) {
        m_pageCapacity = std::max(pages, m_pageCapacity * 2);
        m_pageTable = m_heap.GetDevice().CreateBuffer(
            {.bytes = uint64_t(m_pageCapacity) * m_columns.size() * sizeof(uint64_t),
             .label = fmt::format("{} pages", m_label)});
        ++m_layout;
        firstPage = 0;
        pageCount = pages;
    }
    const uint64_t stride = uint64_t(m_pageCapacity) * sizeof(uint64_t);
    for (uint32_t column = 0; column < m_columns.size(); ++column) {
        queue.Write(m_pageTable->Slice(column * stride + firstPage * sizeof(uint64_t),
                                       uint64_t(pageCount) * sizeof(uint64_t)),
                    m_pages[column].data() + firstPage);
    }
}

uint64_t ColumnTable::Pages(uint32_t column) const {
    CellBytes(column);
    return m_pageTable
               ? m_pageTable->Address() + uint64_t(column) * m_pageCapacity * sizeof(uint64_t)
               : 0;
}

const ColumnTable::Redirection* ColumnTable::FindRedirection(uint32_t column, uint32_t row) const {
    for (const Redirection& redirection : m_redirections[column]) {
        if (row >= redirection.first && row < redirection.first + redirection.rows) {
            return &redirection;
        }
    }
    return nullptr;
}

void ColumnTable::PointPages(uint32_t column, uint32_t first, uint32_t count, uint64_t address) {
    const uint32_t firstPage = first / kPageRows;
    const uint32_t pageCount = (count + kPageRows - 1) / kPageRows;
    const uint64_t pageBytes = uint64_t(kPageRows) * m_columns[column].cellBytes;
    for (uint32_t page = 0; page < pageCount; ++page) {
        m_pages[column][firstPage + page] = address + page * pageBytes;
    }
    WritePages(firstPage, pageCount);
}

void ColumnTable::Redirect(uint32_t column, uint32_t first, uint32_t count,
                           Ref<device::Buffer> buffer) {
    const uint32_t cell = CellBytes(column);
    CheckRows(first, count);
    if (count == 0 || first % kPageRows != 0) {
        throw Error("Rows {} to {} of {} don't start a page, so they can't be redirected", first,
                    first + count, m_label);
    }
    const auto aligned = m_aligned.upper_bound(first);
    const bool inside = aligned != m_aligned.begin() &&
                        first + count <= std::prev(aligned)->first + std::prev(aligned)->second;
    const bool wholePages =
        (first + count) % kPageRows == 0 ||
        (inside && first + count == std::prev(aligned)->first + std::prev(aligned)->second);
    if (!inside || !wholePages) {
        throw Error("Rows {} to {} of {} weren't allocated on whole pages of their own", first,
                    first + count, m_label);
    }
    if (Find(first).first + Find(first).rows < first + count) {
        throw Error("Rows {} to {} of {} cross a span", first, first + count, m_label);
    }
    if (buffer->Bytes() < uint64_t(count) * cell) {
        throw Error("{} bytes for {} rows of {}-byte cells", buffer->Bytes(), count, cell);
    }
    for (uint32_t row = first; row < first + count; row += kPageRows) {
        if (FindRedirection(column, row)) {
            throw Error("Rows {} to {} of {} are already redirected", first, first + count,
                        m_label);
        }
    }
    Flush();
    PointPages(column, first, count, buffer->Address());
    m_redirections[column].push_back({first, count, std::move(buffer)});
    ++m_layout;
}

Ref<device::Buffer> ColumnTable::Restore(uint32_t column, uint32_t first) {
    CellBytes(column);
    std::vector<Redirection>& redirections = m_redirections[column];
    const auto found = std::ranges::find(redirections, first, &Redirection::first);
    if (found == redirections.end()) {
        throw Error("Rows from {} of {} aren't redirected", first, m_label);
    }
    Flush();
    const Redirection redirection = *found;
    redirections.erase(found);
    const Span& span = Find(first);
    PointPages(column, first, redirection.rows,
               span.blocks[column].Address() +
                   uint64_t(first - span.first) * m_columns[column].cellBytes);
    ++m_layout;
    return redirection.buffer;
}

bool ColumnTable::Redirected(uint32_t column, uint32_t first, uint32_t count) const {
    CellBytes(column);
    return std::ranges::any_of(m_redirections[column], [&](const Redirection& r) {
        return first < r.first + r.rows && r.first < first + count;
    });
}

const ColumnTable::Span& ColumnTable::Find(uint32_t row) const {
    const auto after = std::ranges::upper_bound(m_spans, row, {}, &Span::first);
    return *(after - 1);
}

void ColumnTable::CheckRows(uint32_t first, uint32_t count) const {
    if (uint64_t(first) + count > m_rows) {
        throw Error("Rows {} to {} of {}, which has {}", first, uint64_t(first) + count, m_label,
                    m_rows);
    }
}

std::vector<device::BufferRange> ColumnTable::Ranges(uint32_t column, uint32_t first,
                                                     uint32_t count) const {
    const uint64_t cell = CellBytes(column);
    CheckRows(first, count);
    std::vector<device::BufferRange> ranges;
    while (count > 0) {
        if (const Redirection* redirection = FindRedirection(column, first)) {
            const uint32_t taken = std::min(count, redirection->first + redirection->rows - first);
            ranges.push_back(redirection->buffer->Slice(uint64_t(first - redirection->first) * cell,
                                                        uint64_t(taken) * cell));
            first += taken;
            count -= taken;
            continue;
        }
        const Span& span = Find(first);
        uint32_t taken = std::min(count, span.first + span.rows - first);
        for (const Redirection& redirection : m_redirections[column]) {
            if (redirection.first > first) {
                taken = std::min(taken, redirection.first - first);
            }
        }
        ranges.push_back(
            span.blocks[column].Range().Slice((first - span.first) * cell, taken * cell));
        first += taken;
        count -= taken;
    }
    return ranges;
}

std::optional<device::BufferRange> ColumnTable::Contiguous(uint32_t column, uint32_t first,
                                                           uint32_t count) const {
    if (count == 0) {
        return std::nullopt;
    }
    const std::vector<device::BufferRange> ranges = Ranges(column, first, count);
    if (ranges.size() != 1) {
        return std::nullopt;
    }
    return ranges.front();
}

void ColumnTable::Stage(uint32_t column, uint32_t first, std::span<const std::byte> bytes) {
    const uint32_t cell = CellBytes(column);
    if (bytes.size() % cell != 0) {
        throw Error("{} bytes for column {} of {}, whose cells are {} bytes", bytes.size(), column,
                    m_label, cell);
    }
    const uint32_t count = uint32_t(bytes.size() / cell);
    CheckRows(first, count);
    Staging& staged = m_staged[column];
    for (uint32_t i = 0; i < count; ++i) {
        staged.rows.push_back(first + i);
    }
    staged.bytes.insert(staged.bytes.end(), bytes.begin(), bytes.end());
}

bool ColumnTable::Staged() const {
    return std::ranges::any_of(m_staged, [](const Staging& s) { return !s.rows.empty(); });
}

void ColumnTable::Flush(const Scatter& scatter) {
    constexpr uint32_t kScatterRuns = 16;
    device::Queue& queue = m_heap.GetDevice().GetQueue();
    for (uint32_t column = 0; column < m_columns.size(); ++column) {
        Staging& staged = m_staged[column];
        if (staged.rows.empty()) {
            continue;
        }
        ERNIE_ZONE("ColumnTable/Flush");
        const uint32_t cell = m_columns[column].cellBytes;
        std::vector<uint32_t> order(staged.rows.size());
        for (uint32_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        std::ranges::stable_sort(order, {}, [&](uint32_t i) { return staged.rows[i]; });

        std::vector<uint32_t> rows;
        std::vector<std::byte> cells;
        uint32_t runs = 0;
        for (uint32_t at = 0; at < order.size(); ++at) {
            const uint32_t i = order[at];
            const uint32_t row = staged.rows[i];
            if (at + 1 < order.size() && staged.rows[order[at + 1]] == row) {
                continue;
            }
            runs += rows.empty() || rows.back() + 1 != row ? 1 : 0;
            rows.push_back(row);
            const std::byte* bytes = staged.bytes.data() + size_t(i) * cell;
            cells.insert(cells.end(), bytes, bytes + cell);
        }
        staged = {};

        if (scatter && runs > kScatterRuns && cell % sizeof(uint32_t) == 0 &&
            m_redirections[column].empty()) {
            scatter(column, rows, cells);
            continue;
        }
        for (uint32_t first = 0; first < rows.size();) {
            uint32_t count = 1;
            while (first + count < rows.size() && rows[first + count] == rows[first] + count) {
                ++count;
            }
            uint64_t at = uint64_t(first) * cell;
            for (const device::BufferRange& range : Ranges(column, rows[first], count)) {
                queue.Write(range, cells.data() + at);
                at += range.Bytes();
            }
            first += count;
        }
    }
}

std::vector<std::byte> ColumnTable::Read(uint32_t column, uint32_t first, uint32_t count) {
    Flush();
    std::vector<std::byte> bytes(size_t(count) * CellBytes(column));
    uint64_t at = 0;
    for (const device::BufferRange& range : Ranges(column, first, count)) {
        m_heap.GetDevice().GetQueue().Read(range, bytes.data() + at);
        at += range.Bytes();
    }
    return bytes;
}

}
