// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <memory>

#include "ernie/device/queue.h"
#include "ernie/device/vulkan/device.h"

namespace ernie::device::vulkan {

class StreamInterop {
  public:
    virtual ~StreamInterop() = default;

    virtual void Follow(void* stream, uint64_t value) = 0;

    virtual void Wait(void* stream, uint64_t value) = 0;
};

std::unique_ptr<StreamInterop> CreateStreamInterop(VulkanDevice& device, VkSemaphore timeline,
                                                   VkSemaphore follow);

struct ExportedMemory {
    int fd = -1;
    uint64_t memoryBytes = 0;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

struct ImportedMemory {
    void* external = nullptr;
    uint64_t address = 0;
};

class MemoryInterop {
  public:
    virtual ~MemoryInterop() = default;

    virtual DLDevice Device() const = 0;
    virtual ImportedMemory Import(const ExportedMemory& memory) = 0;
    virtual void Release(ImportedMemory& memory) noexcept = 0;
    virtual void Copy(uint64_t dst, uint64_t src, uint64_t bytes, void* stream) = 0;
    virtual void* Stream() const = 0;
};

std::unique_ptr<MemoryInterop> CreateMemoryInterop(VulkanDevice& device);

}
