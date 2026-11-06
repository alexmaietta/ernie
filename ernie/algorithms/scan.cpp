// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/algorithms/scan.h"

#include <algorithm>
#include <vector>

#include "ernie/algorithms/kernels.h"
#include "ernie/algorithms/kernels/scan.h"
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
    if (capacity > Scan::kMaxItems) {
        throw Error("Scan of {} items; it scans at most {}", capacity, Scan::kMaxItems);
    }
    return capacity;
}

}

Scan::Scan(Device& device, uint32_t capacity)
    : m_capacity(RequireCapacity(capacity)),
      m_program(device.CreateProgram({.modules = {"ernie.algorithms.kernels.scan"},
                                      .library = &kAlgorithmsKernels,
                                      .entries = {"SumTiles", "ScanTiles"},
                                      .label = "scan"})),
      m_sumTiles(&m_program->Entry("SumTiles")), m_scanTiles(&m_program->Entry("ScanTiles")) {
    uint64_t words = 0;
    for (uint32_t tiles = GroupCount(capacity, kScanTileItems); tiles > 1;
         tiles = GroupCount(tiles, kScanTileItems)) {
        m_levelOffsets.push_back(words * 4);
        words += tiles;
    }
    m_sums = device.CreateBuffer({.bytes = std::max<uint64_t>(words, 1) * 4, .label = "scan sums"});
    m_unlimitedCount = device.CreateBuffer({.label = "scan unlimited count"}, std::vector{~0u});
}

void Scan::ExclusiveSum(CommandEncoder& encoder, BufferRange input, BufferRange output,
                        uint32_t count) const {
    Record(encoder, input, output, count, {m_unlimitedCount->Address()});
}

void Scan::ExclusiveSum(CommandEncoder& encoder, BufferRange input, BufferRange output,
                        BufferRange count, uint32_t capacity) const {
    Record(encoder, input, output, capacity,
           RequireRange<uint32_t>(*m_program, "ExclusiveSum", count, 1));
}

void Scan::Record(CommandEncoder& encoder, BufferRange input, BufferRange output, uint32_t count,
                  Ptr<uint32_t> liveCount) const {
    if (count > m_capacity) {
        throw Error("ExclusiveSum of {} items; the Scan was made for at most {}", count,
                    m_capacity);
    }
    const ScanParams params{
        .input = RequireRange<uint32_t>(*m_program, "ExclusiveSum", input, count),
        .output = RequireRange<uint32_t>(*m_program, "ExclusiveSum", output, count),
        .deviceCount = liveCount,
        .count = count};

    if (count > 0) {
        RecordLevel(encoder, params, 0);
    }
}

void Scan::RecordLevel(CommandEncoder& encoder, ScanParams params, size_t level) const {
    const uint32_t tiles = GroupCount(params.count, kScanTileItems);
    if (tiles > 1) {
        params.sums = {m_sums->Address() + m_levelOffsets[level]};
        params.hasSums = 1;
        encoder.Barrier();
        encoder.Dispatch(*m_sumTiles, params, {.x = tiles});
        RecordLevel(encoder,
                    {.input = params.sums,
                     .output = params.sums,
                     .deviceCount = {m_unlimitedCount->Address()},
                     .count = tiles},
                    level + 1);
    }

    encoder.Barrier();
    encoder.Dispatch(*m_scanTiles, params, {.x = tiles});
}

}
