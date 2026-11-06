// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/store/arrays.h"

#include <algorithm>
#include <numeric>
#include <utility>

#include "ernie/device/kernels/shared.h"
#include "ernie/device/queue.h"
#include "ernie/util/error.h"

namespace ernie {

namespace {

constexpr uint64_t kArrayAlignment = 16;

}

ArrayHeap::ArrayHeap(Heap& heap, std::string label, uint32_t itemBytes)
    : m_heap(heap), m_label(std::move(label)), m_itemBytes(itemBytes) {
    if (itemBytes == 0 || itemBytes % 4 != 0) {
        throw Error("{} holds items of {} bytes; items are whole words", m_label, itemBytes);
    }
}

uint32_t ArrayHeap::AddRun(HeapBlock block, uint32_t live) {
    if (!m_freeRuns.empty()) {
        const uint32_t run = m_freeRuns.back();
        m_freeRuns.pop_back();
        m_runs[run] = {block, live, {}};
        return run;
    }
    m_runs.push_back({block, live, {}});
    return uint32_t(m_runs.size() - 1);
}

ArrayRef ArrayHeap::Allocate(uint32_t count, uint32_t capacity) {
    capacity = std::max(capacity, count);
    if (capacity == 0) {
        return {};
    }
    const uint64_t bytes = RoundUp(uint64_t(capacity) * m_itemBytes, kArrayAlignment);
    const HeapBlock block = m_heap.Allocate(bytes, kArrayAlignment);
    const uint32_t run = AddRun(block, 1);
    m_arrays.emplace(block.Address(), Entry{run, 1});
    return {block.Address(), count, capacity};
}

std::vector<ArrayRef> ArrayHeap::AllocateRun(std::span<const uint32_t> counts) {
    std::vector<ArrayRef> refs(counts.size());
    uint64_t items = 0;
    uint32_t live = 0;
    for (const uint32_t count : counts) {
        items += count;
        live += count > 0 ? 1 : 0;
    }
    if (items == 0) {
        return refs;
    }
    const uint64_t unit = std::lcm(uint64_t(m_itemBytes), kArrayAlignment);
    const HeapBlock block = m_heap.Allocate(RoundUp(items * m_itemBytes, unit), unit);
    const uint32_t run = AddRun(block, live);
    uint64_t address = block.Address();
    for (uint32_t i = 0; i < counts.size(); ++i) {
        if (counts[i] == 0) {
            continue;
        }
        m_arrays.emplace(address, Entry{run, 1});
        refs[i] = {address, counts[i], counts[i]};
        address += uint64_t(counts[i]) * m_itemBytes;
    }
    return refs;
}

std::vector<ArrayRef> ArrayHeap::Borrow(Ref<device::Buffer> buffer, uint32_t arrays,
                                        uint32_t count) {
    const uint64_t stride = uint64_t(count) * m_itemBytes;
    if (arrays == 0 || count == 0 || buffer->Bytes() < stride * arrays) {
        throw Error("{} borrows {} arrays of {} items from {} bytes", m_label, arrays, count,
                    buffer->Bytes());
    }
    const uint32_t run = AddRun({buffer.Get(), 0, buffer->Bytes()}, arrays);
    m_runs[run].borrowed = std::move(buffer);
    std::vector<ArrayRef> refs;
    for (uint32_t i = 0; i < arrays; ++i) {
        const uint64_t address = m_runs[run].block.Address() + i * stride;
        if (!m_arrays.emplace(address, Entry{run, 1}).second) {
            throw Error("{} already holds an array at {:#x}", m_label, address);
        }
        refs.push_back({address, count, count});
    }
    return refs;
}

bool ArrayHeap::Borrowed(const ArrayRef& array) const {
    return array.address != 0 && m_runs[At(array).run].borrowed;
}

const ArrayHeap::Entry& ArrayHeap::At(const ArrayRef& array) const {
    const auto found = m_arrays.find(array.address);
    if (found == m_arrays.end()) {
        throw Error("{} holds no array at {:#x}", m_label, array.address);
    }
    return found->second;
}

ArrayRef ArrayHeap::Share(const ArrayRef& array) {
    if (array.address != 0) {
        At(array);
        ++m_arrays.at(array.address).references;
    }
    return array;
}

uint32_t ArrayHeap::References(const ArrayRef& array) const {
    return array.address == 0 ? 0 : At(array).references;
}

void ArrayHeap::Free(const ArrayRef& array) {
    if (array.address == 0) {
        return;
    }
    At(array);
    Entry& entry = m_arrays.at(array.address);
    if (--entry.references > 0) {
        return;
    }
    if (const auto staged = m_stagedAt.find(array.address); staged != m_stagedAt.end()) {
        m_staged[staged->second].bytes = 0;
        m_stagedAt.erase(staged);
    }
    Run& run = m_runs[entry.run];
    if (--run.live == 0) {
        if (run.borrowed) {
            run.borrowed = {};
        } else {
            m_heap.Free(run.block);
        }
        m_freeRuns.push_back(entry.run);
    }
    m_arrays.erase(array.address);
}

device::BufferRange ArrayHeap::Range(const ArrayRef& array) const {
    const Run& run = m_runs[At(array).run];
    return run.block.buffer->Slice(array.address - run.block.buffer->Address(),
                                   uint64_t(array.capacity) * m_itemBytes);
}

void ArrayHeap::Write(const ArrayRef& array, std::span<const std::byte> bytes) {
    if (bytes.empty()) {
        return;
    }
    if (bytes.size() > uint64_t(array.capacity) * m_itemBytes) {
        throw Error("{} bytes for an array of {} {}-byte items", bytes.size(), array.capacity,
                    m_itemBytes);
    }
    m_heap.GetDevice().GetQueue().Write(Range(array).Slice(0, bytes.size()), bytes.data());
}

void ArrayHeap::Stage(const ArrayRef& array, std::span<const std::byte> bytes) {
    if (bytes.size() % sizeof(uint32_t) != 0) {
        Write(array, bytes);
        return;
    }
    if (bytes.size() > uint64_t(array.capacity) * m_itemBytes) {
        throw Error("{} bytes for an array of {} {}-byte items", bytes.size(), array.capacity,
                    m_itemBytes);
    }
    if (const auto staged = m_stagedAt.find(array.address); staged != m_stagedAt.end()) {
        m_staged[staged->second].bytes = 0;
    }
    m_stagedAt[array.address] = m_staged.size();
    m_staged.push_back({array, m_stagedBytes.size(), bytes.size()});
    m_stagedBytes.insert(m_stagedBytes.end(), bytes.begin(), bytes.end());
}

std::vector<ArrayHeap::StagedCopy> ArrayHeap::TakeStaged(uint64_t base) {
    std::vector<StagedCopy> copies;
    for (const Staged& staged : m_staged) {
        if (staged.bytes > 0) {
            copies.push_back({Range(staged.array).Address(), base + staged.offset,
                              uint32_t(staged.bytes / sizeof(uint32_t))});
        }
    }
    m_staged.clear();
    m_stagedBytes.clear();
    m_stagedAt.clear();
    return copies;
}

std::vector<std::byte> ArrayHeap::Read(const ArrayRef& array) {
    for (const Staged& staged : m_staged) {
        if (staged.bytes > 0) {
            Write(staged.array, std::span(m_stagedBytes).subspan(staged.offset, staged.bytes));
        }
    }
    m_staged.clear();
    m_stagedBytes.clear();
    m_stagedAt.clear();
    std::vector<std::byte> bytes(size_t(array.count) * m_itemBytes);
    if (!bytes.empty()) {
        m_heap.GetDevice().GetQueue().Read(Range(array).Slice(0, bytes.size()), bytes.data());
    }
    return bytes;
}

}
