// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "ernie/stage/columns/device_heap.h"
#include "ernie/stage/schema/types.h"

namespace ernie {

class ArrayPool {
  public:
    struct StagedCopy {
        uint64_t dst;
        uint64_t source;
        uint32_t words;
    };

    ArrayPool(DeviceHeap& heap, std::string label, uint32_t itemBytes);

    uint32_t ItemBytes() const {
        return m_itemBytes;
    }

    uint32_t ArrayCount() const {
        return uint32_t(m_arrays.size());
    }

    ArrayRef Allocate(uint32_t count, uint32_t capacity = 0);
    std::vector<ArrayRef> AllocateRun(std::span<const uint32_t> counts);
    std::vector<ArrayRef> Borrow(Ref<device::Buffer> buffer, uint32_t arrays, uint32_t count);
    bool Borrowed(const ArrayRef& array) const;
    ArrayRef Share(const ArrayRef& array);
    void Free(const ArrayRef& array);
    uint32_t References(const ArrayRef& array) const;

    device::BufferRange Range(const ArrayRef& array) const;
    void Write(const ArrayRef& array, std::span<const std::byte> bytes);
    void Stage(const ArrayRef& array, std::span<const std::byte> bytes);

    std::span<const std::byte> StagedBytes() const {
        return m_stagedBytes;
    }

    std::vector<StagedCopy> TakeStaged(uint64_t base);
    std::vector<std::byte> Read(const ArrayRef& array);
    std::vector<std::vector<std::byte>> Read(std::span<const ArrayRef> arrays);

  private:
    struct Run {
        HeapBlock block;
        uint32_t live = 0;
        Ref<device::Buffer> borrowed;
    };

    struct Entry {
        uint32_t run = 0;
        uint32_t references = 0;
    };

    struct Staged {
        ArrayRef array;
        uint64_t offset = 0;
        uint64_t bytes = 0;
    };

    void FlushStaged();
    const Entry& Lookup(const ArrayRef& array) const;
    uint32_t AddRun(HeapBlock block, uint32_t live);

    DeviceHeap& m_heap;
    std::string m_label;
    uint32_t m_itemBytes;
    std::vector<Run> m_runs;
    std::vector<uint32_t> m_freeRuns;
    std::unordered_map<uint64_t, Entry> m_arrays;
    std::vector<Staged> m_staged;
    std::vector<std::byte> m_stagedBytes;
    std::unordered_map<uint64_t, size_t> m_stagedIndices;
};

}
