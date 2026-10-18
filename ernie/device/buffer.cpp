// SPDX-License-Identifier: BSD-3-Clause
/// @file buffer.cpp
/// @brief Allocating buffers through VMA by their use.
#include "ernie/device/buffer.h"

#include <utility>

namespace ernie {

Buffer::Buffer(DeviceContext& context, std::uint64_t size, MemoryUse use, std::string_view name,
               VkBufferUsageFlags usage)
    : allocator_(context.Allocator()), size_(size) {
    const VkBufferCreateInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                  .size = size,
                                  .usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo allocation{.usage = VMA_MEMORY_USAGE_AUTO};
    if (use == MemoryUse::Upload) {
        allocation.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                           VMA_ALLOCATION_CREATE_MAPPED_BIT;
    } else if (use == MemoryUse::Readback) {
        allocation.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    } else {
        allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    }
    VmaAllocationInfo allocated{};
    CheckVk(vmaCreateBuffer(allocator_, &info, &allocation, &buffer_, &allocation_, &allocated),
            "Allocating " + std::to_string(size) + " bytes for " + std::string(name));
    const VkBufferDeviceAddressInfo addressInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = buffer_};
    address_ = vkGetBufferDeviceAddress(context.Device(), &addressInfo);
    if (allocated.pMappedData != nullptr) {
        mapped_ = {static_cast<std::byte*>(allocated.pMappedData), static_cast<std::size_t>(size)};
    }
    context.Name(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<std::uint64_t>(buffer_), name);
}

Buffer::~Buffer() {
    if (buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
    }
}

Buffer::Buffer(Buffer&& other) noexcept {
    *this = std::move(other);
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        if (buffer_ != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator_, buffer_, allocation_);
        }
        allocator_ = std::exchange(other.allocator_, VK_NULL_HANDLE);
        buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
        allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
        size_ = std::exchange(other.size_, 0);
        address_ = std::exchange(other.address_, 0);
        mapped_ = std::exchange(other.mapped_, {});
    }
    return *this;
}

void Buffer::Flush(std::uint64_t offset, std::uint64_t size) const {
    CheckVk(vmaFlushAllocation(allocator_, allocation_, offset, size), "Flushing a buffer");
}

void Buffer::Invalidate(std::uint64_t offset, std::uint64_t size) const {
    CheckVk(vmaInvalidateAllocation(allocator_, allocation_, offset, size),
            "Invalidating a buffer");
}

}  // namespace ernie
