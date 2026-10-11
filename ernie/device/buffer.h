// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/vulkan_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace ernie {

/// Where a buffer lives and how the host reaches it.
enum class BufferMemory {
    Device,  ///< Device-local; `Write` and `Read` go through a
             ///< staging copy.
    Upload,  ///< Host-visible, written sequentially by the host each
             ///< frame.
    Readback,  ///< Host-visible and cached, for results the host
               ///< reads.
    /// Device-local in a dedicated allocation exportable as a POSIX
    /// file descriptor, for sharing with CUDA
    /// (`VulkanContext::SupportsExternalMemory`); `Write` and `Read`
    /// stage.
    Shared,
};

/// A Vulkan buffer with its memory, allocated through VMA.
class GpuBuffer {
  public:
    /// @throws std::runtime_error if allocation fails.
    GpuBuffer(const VulkanContext& context, VkDeviceSize size,
              VkBufferUsageFlags usage, BufferMemory memory);
    ~GpuBuffer();
    GpuBuffer(GpuBuffer&& other) noexcept;
    GpuBuffer& operator=(GpuBuffer&& other) noexcept;
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    VkBuffer Handle() const {
        return buffer_;
    }

    VkDeviceSize Size() const {
        return size_;
    }

    /// Copies `bytes` to the buffer at `offset`, waiting for the copy
    /// when staging is needed.
    /// @throws std::out_of_range if the range does not fit.
    void Write(std::span<const std::byte> bytes,
               VkDeviceSize offset = 0);

    /// Copies the buffer's contents at `offset` into `bytes`, after
    /// all submitted work that `VulkanContext::Submit` has waited
    /// for.
    /// @throws std::out_of_range if the range does not fit.
    void Read(std::span<std::byte> bytes,
              VkDeviceSize offset = 0) const;

    BufferMemory Memory() const {
        return memory_;
    }

    /// For `Shared` buffers, a new handle to the buffer's memory (a
    /// file descriptor, or a Win32 handle on Windows), owned by the
    /// caller until it imports it elsewhere
    /// (`CloseExternalHandle`).
    /// @throws std::logic_error for other buffers, or
    /// std::runtime_error if export fails.
    ExternalHandle ExportHandle() const;

    /// For `Shared` buffers, the size of the buffer's dedicated
    /// memory allocation.
    VkDeviceSize AllocationSize() const {
        return allocationSize_;
    }

    /// Keeps `object`, such as a CUDA import of the buffer's memory,
    /// until the buffer is destroyed, releasing it before the memory
    /// is freed.
    void Attach(std::shared_ptr<void> object) {
        attached_ = std::move(object);
    }

    const std::shared_ptr<void>& Attached() const {
        return attached_;
    }

    /// Expires when the buffer is destroyed or replaced by
    /// assignment, for work queued for it later (`FrameUploads`).
    std::weak_ptr<const void> Lifetime() const {
        return lifetime_;
    }

  private:
    void Release();

    const VulkanContext* context_ = nullptr;
    std::weak_ptr<const void>
        contextAlive_;  ///< `VulkanContext::Alive`.
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkDeviceMemory sharedMemory_ = VK_NULL_HANDLE;
    VkDeviceSize allocationSize_ = 0;
    std::shared_ptr<void> attached_;
    std::shared_ptr<const void> lifetime_;
    void* mapped_ = nullptr;
    VkDeviceSize size_ = 0;
    BufferMemory memory_ = BufferMemory::Device;
};

}  // namespace ernie
