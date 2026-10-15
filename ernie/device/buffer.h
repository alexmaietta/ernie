// SPDX-License-Identifier: BSD-3-Clause
/// @file buffer.h
/// @brief GPU buffers, their memory kinds, and growing them.
#pragma once

#include "ernie/device/vulkan_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace ernie {

/// @brief Where a buffer lives and how the host reaches it.
enum class BufferMemory {
    /// Device-local; `Write` and `Read` go through a staging copy.
    Device,
    /// Host-visible, written by the host each frame.
    Upload,
    /// Host-visible and cached, for results the host reads.
    Readback,
    /// Device-local and exportable, for sharing with CUDA.
    /// @see VulkanContext::SupportsExternalMemory
    Shared,
};

/// @brief A Vulkan buffer with its memory, allocated through VMA.
class GpuBuffer {
  public:
    /// @brief Allocates a buffer.
    /// @param context The device to allocate on.
    /// @param size    The size in bytes.
    /// @param usage   Vulkan usage flags.
    /// @param memory  Where the memory lives.
    /// @throws std::runtime_error If allocation fails.
    GpuBuffer(const VulkanContext& context, VkDeviceSize size,
              VkBufferUsageFlags usage, BufferMemory memory);
    ~GpuBuffer();
    GpuBuffer(GpuBuffer&& other) noexcept;
    GpuBuffer& operator=(GpuBuffer&& other) noexcept;
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    /// @return The Vulkan handle.
    VkBuffer Handle() const {
        return buffer_;
    }

    /// @return The size in bytes.
    VkDeviceSize Size() const {
        return size_;
    }

    /// @brief Copies `bytes` into the buffer, waiting for the copy.
    /// @param bytes  The data to copy.
    /// @param offset Where in the buffer to start, in bytes.
    /// @throws std::out_of_range If the range does not fit.
    void Write(std::span<const std::byte> bytes,
               VkDeviceSize offset = 0);

    /// @brief Copies the buffer's contents into `bytes`.
    /// @param bytes  Where to copy to; its size sets how much.
    /// @param offset Where in the buffer to start, in bytes.
    /// @note Sees all work that `VulkanContext::Submit` waited for.
    /// @throws std::out_of_range If the range does not fit.
    void Read(std::span<std::byte> bytes,
              VkDeviceSize offset = 0) const;

    /// @return Where the memory lives.
    BufferMemory Memory() const {
        return memory_;
    }

    /// @brief The buffer's GPU address, which shaders read as a
    /// pointer in place of a binding.
    /// @return The device address.
    VkDeviceAddress Address() const;

    /// @brief Exports a new handle to a `Shared` buffer's memory.
    /// @return A file descriptor (a Win32 handle on Windows), owned
    /// by
    ///         the caller until imported (`CloseExternalHandle`).
    /// @throws std::logic_error If the buffer is not `Shared`.
    /// @throws std::runtime_error If the export fails.
    ExternalHandle ExportHandle() const;

    /// @return The size of a `Shared` buffer's dedicated allocation.
    VkDeviceSize AllocationSize() const {
        return allocationSize_;
    }

    /// @brief Keeps `object` alive until the buffer's memory is
    /// freed.
    /// @param object Something that uses the memory, such as a CUDA
    ///               import of it; released first.
    void Attach(std::shared_ptr<void> object) {
        attached_ = std::move(object);
    }

    /// @return The object passed to `Attach`, or null.
    const std::shared_ptr<void>& Attached() const {
        return attached_;
    }

    /// @return A token that expires when the buffer is destroyed or
    ///         replaced, for work queued for it later
    ///         (`FrameUploads`).
    std::weak_ptr<const void> Lifetime() const {
        return lifetime_;
    }

  private:
    void Release();

    const VulkanContext* context_ = nullptr;
    /// Expires with the context (`VulkanContext::Alive`).
    std::weak_ptr<const void> contextAlive_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkDeviceMemory sharedMemory_ = VK_NULL_HANDLE;
    /// A `Shared` buffer's handles as the context tracks them
    /// (`VulkanContext::TrackExported`).
    std::shared_ptr<VulkanContext::ExportedBuffer> exported_;
    VkDeviceSize allocationSize_ = 0;
    std::shared_ptr<void> attached_;
    std::shared_ptr<const void> lifetime_;
    void* mapped_ = nullptr;
    VkDeviceSize size_ = 0;
    BufferMemory memory_ = BufferMemory::Device;
};

/// @brief Grows `buffer` so it holds at least `bytes`.
/// @param context The device the buffer lives on.
/// @param buffer  The buffer to keep or replace; may be null.
/// @param bytes   The size needed (at least 16 bytes are allocated).
/// @param usage   Usage flags for a new buffer.
/// @param memory  Memory kind for a new buffer.
/// @return True when a new buffer replaced the old one, so its
///         address changed and its contents are undefined.
/// @note Waits for the device before replacing a buffer, since
///       submitted work may still read the old one.
bool Reserve(
    const VulkanContext& context, std::unique_ptr<GpuBuffer>& buffer,
    VkDeviceSize bytes,
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    BufferMemory memory = BufferMemory::Device);

}  // namespace ernie
