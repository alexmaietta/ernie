// SPDX-License-Identifier: BSD-3-Clause
/// @file buffer.cpp
/// @brief Allocating buffers through VMA by their use, and sharing them.
#include "ernie/device/buffer.h"

#include <atomic>
#include <utility>

namespace ernie {
namespace {

/// Every buffer's usage: on devices with ray queries, any may hold or feed an acceleration
/// structure, so geometry in columns builds one in place.
VkBufferCreateInfo BufferInfo(const DeviceContext& context, std::uint64_t size,
                              VkBufferUsageFlags usage) {
    const VkBufferUsageFlags acceleration =
        context.Capabilities().rayQuery
            ? VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                  VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
            : 0;
    return {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage | kBufferUsage | acceleration,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
}

void RequireExternal(const DeviceContext& context) {
    if (!context.Capabilities().externalMemory) {
        throw DeviceError(DeviceErrorKind::Unsupported,
                          context.Info().name + " cannot share memory with other APIs");
    }
}

}  // namespace

Buffer::Buffer(DeviceContext& context, std::uint64_t size, MemoryUse use, std::string_view name,
               VkBufferUsageFlags usage)
    : allocator_(context.Allocator()), device_(context.Device()), shared_(use == MemoryUse::Shared),
      size_(size) {
    const VkBufferCreateInfo info = BufferInfo(context, size, usage);
    VmaAllocationInfo allocated{};
    if (shared_) {
        RequireExternal(context);
        CreateExportableBuffer(allocator_, info, buffer_, allocation_, allocated);
        Finish(context, allocated, name);
        return;
    }
    VmaAllocationCreateInfo allocation{.usage = VMA_MEMORY_USAGE_AUTO};
    if (use == MemoryUse::Upload) {
        // Cached host memory: staging is written a few rows at a time, which write-combined or
        // device-local mappings turn into a bus transaction each.
        allocation.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    } else if (use == MemoryUse::Readback) {
        allocation.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    } else {
        allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    }
    CheckVk(vmaCreateBuffer(allocator_, &info, &allocation, &buffer_, &allocation_, &allocated),
            "Allocating " + std::to_string(size) + " bytes for " + std::string(name));
    Finish(context, allocated, name);
}

Buffer::Buffer(DeviceContext& context, ExternalHandle handle, std::uint64_t size,
               std::string_view name, SharedFrom from)
    : allocator_(context.Allocator()), device_(context.Device()), shared_(true), size_(size) {
    RequireExternal(context);
    VmaAllocationInfo allocated{};
    if (from == SharedFrom::Vulkan) {
        CreateImportedBuffer(allocator_, BufferInfo(context, size, 0), handle, buffer_, allocation_,
                             allocated);
    } else {
        CreateImportedBuffer(device_, context.Physical(), BufferInfo(context, size, 0), handle,
                             buffer_, memory_);
        allocated.size = size;
    }
    Finish(context, allocated, name);
}

void Buffer::Destroy() {
    if (memory_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, buffer_, nullptr);
        vkFreeMemory(device_, memory_, nullptr);
    } else if (buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
    }
}

void Buffer::Finish(DeviceContext& context, const VmaAllocationInfo& allocated,
                    std::string_view name) {
    const VkBufferDeviceAddressInfo addressInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = buffer_};
    address_ = vkGetBufferDeviceAddress(device_, &addressInfo);
    memorySize_ = allocated.size;
    static std::atomic<std::uint64_t> ids = 0;
    id_ = ++ids;
    if (allocated.pMappedData != nullptr) {
        mapped_ = {static_cast<std::byte*>(allocated.pMappedData), static_cast<std::size_t>(size_)};
    }
    context.Name(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<std::uint64_t>(buffer_), name);
}

Buffer::~Buffer() {
    Destroy();
}

Buffer::Buffer(Buffer&& other) noexcept {
    *this = std::move(other);
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        Destroy();
        allocator_ = std::exchange(other.allocator_, VK_NULL_HANDLE);
        device_ = std::exchange(other.device_, VK_NULL_HANDLE);
        shared_ = std::exchange(other.shared_, false);
        buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
        allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
        memory_ = std::exchange(other.memory_, VK_NULL_HANDLE);
        size_ = std::exchange(other.size_, 0);
        memorySize_ = std::exchange(other.memorySize_, 0);
        id_ = std::exchange(other.id_, 0);
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

ExternalHandle Buffer::Export() const {
    if (!shared_) {
        throw DeviceError(DeviceErrorKind::Unsupported, "Only shared buffers can be exported");
    }
    if (memory_ != VK_NULL_HANDLE) {
        return ExportMemory(device_, memory_, "a shared buffer");
    }
    VmaAllocationInfo allocated{};
    vmaGetAllocationInfo(allocator_, allocation_, &allocated);
    // Shared buffers have dedicated memory: the buffer starts it.
    return ExportMemory(device_, allocated.deviceMemory, "a shared buffer");
}

}  // namespace ernie
