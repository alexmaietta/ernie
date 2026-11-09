// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/device.h"
#include "ernie/device/ref.h"

namespace ernie {

struct HeapBlock {
    device::Buffer* buffer = nullptr;
    uint64_t offset = 0;
    uint64_t bytes = 0;

    explicit operator bool() const {
        return buffer != nullptr;
    }

    uint64_t Address() const {
        return buffer->Address() + offset;
    }

    device::BufferRange Range() const {
        return buffer->Slice(offset, bytes);
    }
};

class DeviceHeap {
  public:
    DeviceHeap(device::Device& device, std::string label);

    device::Device& GetDevice() const {
        return m_device;
    }

    uint64_t Bytes() const;
    HeapBlock Allocate(uint64_t bytes, uint64_t alignment);
    void Free(const HeapBlock& block);

  private:
    struct Chunk {
        Ref<device::Buffer> buffer;
        uint64_t used = 0;
    };

    device::Device& m_device;
    std::string m_label;
    std::vector<Chunk> m_chunks;
    std::map<uint64_t, std::vector<HeapBlock>> m_free;
    uint64_t m_nextChunk;
};

}
