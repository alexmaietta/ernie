// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/algorithms/radix_sort.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "ernie/algorithms/kernels.h"
#include "ernie/algorithms/kernels/radix_sort.h"
#include "ernie/device/device.h"
#include "ernie/device/slang/dispatch.h"
#include "ernie/util/error.h"

namespace ernie::algorithms {

using device::BufferRange;
using device::CommandEncoder;
using device::Device;
using device::GroupCount;
using device::RequireRange;

namespace {

uint32_t RequireCapacity(uint32_t capacity) {
    if (capacity > RadixSort::kMaxItems) {
        throw Error("RadixSort of {} items; it sorts at most {}", capacity, RadixSort::kMaxItems);
    }
    return capacity;
}

uint32_t RequireKeyWords(uint32_t keyBytes) {
    if (keyBytes != 4 && keyBytes != 8) {
        throw Error("RadixSort of keys of {} bytes; it sorts keys of 4 or 8", keyBytes);
    }
    return keyBytes / 4;
}

}

RadixSort::RadixSort(Device& device, uint32_t capacity, uint32_t keyBytes)
    : m_capacity(RequireCapacity(capacity)), m_keyWords(RequireKeyWords(keyBytes)),
      m_program(device.CreateProgram({.modules = {"ernie.algorithms.kernels.radix_sort"},
                                      .library = &kAlgorithmsKernels,
                                      .entries = {"CountDigits", "Scatter"},
                                      .label = "radix sort"})),
      m_countDigits(&m_program->Entry("CountDigits")), m_scatter(&m_program->Entry("Scatter")),
      m_scan(device, kDigitBins * GroupCount(capacity, kSortTileItems)) {
    const uint64_t items = std::max(capacity, 1u);
    const uint64_t tiles = GroupCount(items, kSortTileItems);
    m_keys = device.CreateBuffer({.bytes = items * m_keyWords * 4, .label = "radix sort keys"});
    m_values = device.CreateBuffer({.bytes = items * 4, .label = "radix sort values"});
    m_counts = device.CreateBuffer({.bytes = kDigitBins * tiles * 4, .label = "radix sort counts"});
    m_unlimitedCount =
        device.CreateBuffer({.label = "radix sort unlimited count"}, std::vector{~0u});
}

void RadixSort::Sort(CommandEncoder& encoder, BufferRange keys, BufferRange values, uint32_t count,
                     std::optional<uint32_t> keyBits) const {
    Record(encoder, keys, values, count, {m_unlimitedCount->Address()}, keyBits);
}

void RadixSort::Sort(CommandEncoder& encoder, BufferRange keys, BufferRange values,
                     BufferRange count, uint32_t capacity, std::optional<uint32_t> keyBits) const {
    Record(encoder, keys, values, capacity, RequireRange<uint32_t>(*m_program, "Sort", count, 1),
           keyBits);
}

void RadixSort::Record(CommandEncoder& encoder, BufferRange keys, BufferRange values,
                       uint32_t count, Ptr<uint32_t> liveCount,
                       std::optional<uint32_t> keyBits) const {
    if (count > m_capacity) {
        throw Error("Sort of {} items; the RadixSort was made for at most {}", count, m_capacity);
    }
    const uint32_t bits = keyBits.value_or(m_keyWords * 32);
    if (bits == 0 || bits > m_keyWords * 32) {
        throw Error("Sort by {} bits of its keys; they have 1 to {}", bits, m_keyWords * 32);
    }
    const uint64_t keyBytes = uint64_t(count) * m_keyWords * 4;
    const uint64_t valueBytes = uint64_t(count) * 4;
    const uint32_t tiles = GroupCount(count, kSortTileItems);
    SortParams params{.keys = RequireRange<uint32_t>(*m_program, "Sort", keys, keyBytes / 4),
                      .values = RequireRange<uint32_t>(*m_program, "Sort", values, count),
                      .keysOut = {m_keys->Address()},
                      .valuesOut = {m_values->Address()},
                      .counts = {m_counts->Address()},
                      .deviceCount = liveCount,
                      .count = count,
                      .tiles = tiles,
                      .keyWords = m_keyWords};

    if (count == 0) {
        return;
    }

    const uint32_t passes = (bits + kDigitBits - 1) / kDigitBits;
    if (passes % 2 != 0) {
        encoder.Barrier();
        encoder.Copy(keys.Slice(0, keyBytes), m_keys->Slice(0, keyBytes));
        encoder.Copy(values.Slice(0, valueBytes), m_values->Slice(0, valueBytes));
        std::swap(params.keys, params.keysOut);
        std::swap(params.values, params.valuesOut);
    }

    for (uint32_t pass = 0; pass < passes; ++pass) {
        params.shift = pass * kDigitBits;
        params.mask = (1u << std::min(kDigitBits, bits - params.shift)) - 1;
        encoder.Barrier();
        encoder.Dispatch(*m_countDigits, params, {.x = tiles});
        m_scan.ExclusiveSum(encoder, *m_counts, *m_counts, kDigitBins * tiles);
        encoder.Barrier();
        encoder.Dispatch(*m_scatter, params, {.x = tiles});
        std::swap(params.keys, params.keysOut);
        std::swap(params.values, params.valuesOut);
    }
}

}
