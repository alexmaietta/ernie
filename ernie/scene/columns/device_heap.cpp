// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/columns/device_heap.h"

#include <algorithm>
#include <utility>

#include <fmt/format.h>

#include "ernie/device/kernels/shared.h"

namespace ernie {

namespace {

constexpr uint64_t kFirstChunk = uint64_t(1) << 20;
constexpr uint64_t kLargestChunk = uint64_t(64) << 20;

}

DeviceHeap::DeviceHeap(device::Device& device, std::string label)
    : m_device(device), m_label(std::move(label)), m_nextChunk(kFirstChunk) {}

uint64_t DeviceHeap::Bytes() const {
    uint64_t bytes = 0;
    for (const Chunk& chunk : m_chunks) {
        bytes += chunk.buffer->Bytes();
    }
    return bytes;
}

HeapBlock DeviceHeap::Allocate(uint64_t bytes, uint64_t alignment) {
    if (bytes == 0) {
        return {};
    }
    bytes = RoundUp(bytes, alignment);
    std::vector<HeapBlock>& free = m_free[bytes];
    const auto aligned = std::ranges::find_if(
        free, [&](const HeapBlock& block) { return block.offset % alignment == 0; });
    if (aligned != free.end()) {
        const HeapBlock block = *aligned;
        free.erase(aligned);
        return block;
    }
    const auto fits = [&](const Chunk& chunk) {
        return RoundUp(chunk.used, alignment) + bytes <= chunk.buffer->Bytes();
    };
    if (m_chunks.empty() || !fits(m_chunks.back())) {
        const uint64_t chunkBytes = std::max(m_nextChunk, bytes);
        m_nextChunk = std::min(m_nextChunk * 2, kLargestChunk);
        m_chunks.push_back(
            {m_device.CreateBuffer(
                 {.bytes = chunkBytes,
                  .usage = device::BufferUsage::Interop | device::BufferUsage::AccelBuildInput,
                  .label = fmt::format("{} chunk {}", m_label, m_chunks.size())}),
             0});
    }
    Chunk& chunk = m_chunks.back();
    const uint64_t offset = RoundUp(chunk.used, alignment);
    chunk.used = offset + bytes;
    return {chunk.buffer.Get(), offset, bytes};
}

void DeviceHeap::Free(const HeapBlock& block) {
    if (block) {
        m_free[block.bytes].push_back(block);
    }
}

}
