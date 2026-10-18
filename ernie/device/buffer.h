// SPDX-License-Identifier: BSD-3-Clause
/// @file buffer.h
/// @brief Device buffers, allocated by what they are for, each with its device address.
#pragma once

#include "ernie/device/context.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace ernie {

/// @brief What a buffer's memory is for, from which its placement follows.
enum class MemoryUse : std::uint8_t {
    /// Read and written by the GPU only: device-local.
    Device,
    /// Written by the host, read by the GPU: mapped, device-local when the device offers it.
    Upload,
    /// Written by the GPU, read by the host: mapped and cached.
    Readback,
};

/// @brief A buffer and its memory. Every buffer takes device-address and transfer usage, so
/// shaders reach it by address and copies need no extra flags. Destroying one the GPU may still
/// use is an error: hand it to `DeviceContext::Retire` instead.
class Buffer {
  public:
    Buffer() = default;
    /// @throws DeviceError when the memory cannot be allocated.
    Buffer(DeviceContext& context, std::uint64_t size, MemoryUse use, std::string_view name,
           VkBufferUsageFlags usage = 0);
    ~Buffer();
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    VkBuffer Handle() const {
        return buffer_;
    }
    std::uint64_t Size() const {
        return size_;
    }
    VkDeviceAddress Address() const {
        return address_;
    }
    /// @return Its bytes on the host, for upload and readback buffers; empty otherwise.
    std::span<std::byte> Mapped() const {
        return mapped_;
    }
    /// @brief Makes host writes to a mapped range visible to the device, where memory is not
    /// coherent.
    void Flush(std::uint64_t offset, std::uint64_t size) const;
    /// @brief Makes device writes to a mapped range visible to the host, where memory is not
    /// coherent.
    void Invalidate(std::uint64_t offset, std::uint64_t size) const;

  private:
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    std::uint64_t size_ = 0;
    VkDeviceAddress address_ = 0;
    std::span<std::byte> mapped_;
};

}  // namespace ernie
