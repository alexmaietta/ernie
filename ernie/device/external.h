// SPDX-License-Identifier: BSD-3-Clause
/// @file external.h
/// @brief Sharing memory and timeline semaphores with other APIs and contexts through OS handles:
/// file descriptors on Linux and macOS, NT handles on Windows. Only external.cpp sees the
/// platform's declarations, so <windows.h> stays out of the rest of Ernie.
#pragma once

#include "ernie/device/vulkan.h"

#include <cstdint>
#include <string_view>

namespace ernie {

#if defined(_WIN32)
constexpr const char* kExternalMemoryExtension = "VK_KHR_external_memory_win32";
constexpr const char* kExternalSemaphoreExtension = "VK_KHR_external_semaphore_win32";
constexpr VkExternalMemoryHandleTypeFlagBits kExternalMemoryType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
constexpr VkExternalSemaphoreHandleTypeFlagBits kExternalSemaphoreType =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
constexpr const char* kExternalMemoryExtension = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
constexpr const char* kExternalSemaphoreExtension = VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
constexpr VkExternalMemoryHandleTypeFlagBits kExternalMemoryType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
constexpr VkExternalSemaphoreHandleTypeFlagBits kExternalSemaphoreType =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

/// @brief An OS handle to exported memory or a semaphore, closed when destroyed unless an import
/// took it.
class ExternalHandle {
  public:
    ExternalHandle() = default;
    explicit ExternalHandle(std::intptr_t value) : value_(value) {}
    ~ExternalHandle();
    ExternalHandle(ExternalHandle&& other) noexcept : value_(other.Release()) {}
    ExternalHandle& operator=(ExternalHandle&& other) noexcept;
    ExternalHandle(const ExternalHandle&) = delete;
    ExternalHandle& operator=(const ExternalHandle&) = delete;

    std::intptr_t Get() const {
        return value_;
    }
    explicit operator bool() const {
        return value_ != kNone;
    }
    /// @brief Marks the handle imported: an import takes file descriptors, not NT handles, which
    /// stay the importer's to close.
    void Imported() {
#if !defined(_WIN32)
        Release();
#endif
    }
    /// @brief Gives up the handle without closing it.
    std::intptr_t Release() {
        const std::intptr_t value = value_;
        value_ = kNone;
        return value;
    }

  private:
    static constexpr std::intptr_t kNone = -1;
    std::intptr_t value_ = kNone;
};

/// @return Whether the device exports and imports Ernie's buffers.
bool SharesMemory(VkPhysicalDevice physical);
/// @return Whether the device exports and imports timeline semaphores.
bool SharesTimelines(VkPhysicalDevice physical);

/// @brief Allocates a buffer with dedicated memory that `ExportMemory` can share.
void CreateExportableBuffer(VmaAllocator allocator, const VkBufferCreateInfo& info,
                            VkBuffer& buffer, VmaAllocation& allocation,
                            VmaAllocationInfo& allocated);
/// @brief Creates a buffer on memory imported from `handle`, taking the handle where the platform
/// does (file descriptors).
void CreateImportedBuffer(VmaAllocator allocator, const VkBufferCreateInfo& info,
                          ExternalHandle& handle, VkBuffer& buffer, VmaAllocation& allocation,
                          VmaAllocationInfo& allocated);
/// @brief Creates a buffer bound to device memory another API allocated, such as CUDA's virtual
/// memory, which no Vulkan dedicated allocation made, taking `handle` where the platform does.
/// @param size The size of the shared allocation, which the buffer spans.
void CreateImportedBuffer(VkDevice device, VkPhysicalDevice physical,
                          const VkBufferCreateInfo& info, ExternalHandle& handle, VkBuffer& buffer,
                          VkDeviceMemory& memory);
ExternalHandle ExportMemory(VkDevice device, VkDeviceMemory memory, std::string_view name);

ExternalHandle ExportSemaphore(VkDevice device, VkSemaphore semaphore, std::string_view name);
/// @brief Makes `semaphore` share the payload behind `handle`, taking the handle where the
/// platform does.
void ImportSemaphore(VkDevice device, VkSemaphore semaphore, ExternalHandle& handle,
                     std::string_view name);

}  // namespace ernie
