// SPDX-License-Identifier: BSD-3-Clause
/// @file external.cpp
/// @brief Exporting and importing memory and semaphores through file descriptors or NT handles.
#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "ernie/device/external.h"
#include "ernie/util/on_failure.h"

#include <string>
#include <utility>

namespace ernie {
namespace {

void CreateDedicatedBuffer(VmaAllocator allocator, const VkBufferCreateInfo& info,
                           void* allocateNext, VkBuffer& buffer, VmaAllocation& allocation,
                           VmaAllocationInfo& allocated, const std::string& what) {
    VkExternalMemoryBufferCreateInfo external{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext = info.pNext,
        .handleTypes = static_cast<VkExternalMemoryHandleTypeFlags>(kExternalMemoryType)};
    VkBufferCreateInfo shared = info;
    shared.pNext = &external;
    const VmaAllocationCreateInfo create{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    CheckVk(vmaCreateDedicatedBuffer(allocator, &shared, &create, allocateNext, &buffer,
                                     &allocation, &allocated),
            what);
}

}  // namespace

ExternalHandle::~ExternalHandle() {
    if (value_ != kNone) {
#if defined(_WIN32)
        CloseHandle(reinterpret_cast<HANDLE>(value_));
#else
        close(static_cast<int>(value_));
#endif
    }
}

ExternalHandle& ExternalHandle::operator=(ExternalHandle&& other) noexcept {
    if (this != &other) {
        ExternalHandle old(std::exchange(value_, other.Release()));
    }
    return *this;
}

bool SharesMemory(VkPhysicalDevice physical) {
    const VkPhysicalDeviceExternalBufferInfo buffer{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
        .usage = kBufferUsage,
        .handleType = kExternalMemoryType};
    VkExternalBufferProperties properties{.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
    vkGetPhysicalDeviceExternalBufferProperties(physical, &buffer, &properties);
    const VkExternalMemoryFeatureFlags both =
        VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT | VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
    return (properties.externalMemoryProperties.externalMemoryFeatures & both) == both;
}

bool SharesTimelines(VkPhysicalDevice physical) {
    const VkSemaphoreTypeCreateInfo timeline{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                             .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
    const VkPhysicalDeviceExternalSemaphoreInfo semaphore{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
        .pNext = &timeline,
        .handleType = kExternalSemaphoreType};
    VkExternalSemaphoreProperties properties{.sType =
                                                 VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
    vkGetPhysicalDeviceExternalSemaphoreProperties(physical, &semaphore, &properties);
    const VkExternalSemaphoreFeatureFlags both =
        VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT | VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT;
    return (properties.externalSemaphoreFeatures & both) == both;
}

void CreateExportableBuffer(VmaAllocator allocator, const VkBufferCreateInfo& info,
                            VkBuffer& buffer, VmaAllocation& allocation,
                            VmaAllocationInfo& allocated) {
    VkExportMemoryAllocateInfo exported{
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes = static_cast<VkExternalMemoryHandleTypeFlags>(kExternalMemoryType)};
    CreateDedicatedBuffer(allocator, info, &exported, buffer, allocation, allocated,
                          "Allocating " + std::to_string(info.size) + " shareable bytes");
}

void CreateImportedBuffer(VmaAllocator allocator, const VkBufferCreateInfo& info,
                          ExternalHandle& handle, VkBuffer& buffer, VmaAllocation& allocation,
                          VmaAllocationInfo& allocated) {
    const std::string what = "Importing " + std::to_string(info.size) + " bytes";
#if defined(_WIN32)
    VkImportMemoryWin32HandleInfoKHR imported{
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
        .handleType = kExternalMemoryType,
        .handle = reinterpret_cast<HANDLE>(handle.Get())};
    CreateDedicatedBuffer(allocator, info, &imported, buffer, allocation, allocated, what);
#else
    VkImportMemoryFdInfoKHR imported{.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
                                     .handleType = kExternalMemoryType,
                                     .fd = static_cast<int>(handle.Get())};
    CreateDedicatedBuffer(allocator, info, &imported, buffer, allocation, allocated, what);
    // The memory owns the descriptor now.
    handle.Release();
#endif
}

void CreateImportedBuffer(VkDevice device, VkPhysicalDevice physical,
                          const VkBufferCreateInfo& info, ExternalHandle& handle, VkBuffer& buffer,
                          VkDeviceMemory& memory) {
    const std::string what = "Importing " + std::to_string(info.size) + " bytes";
    const VkExternalMemoryBufferCreateInfo external{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = static_cast<VkExternalMemoryHandleTypeFlags>(kExternalMemoryType)};
    VkBufferCreateInfo shared = info;
    shared.pNext = &external;
    CheckVk(vkCreateBuffer(device, &shared, nullptr, &buffer), what);
    const OnFailure freeBuffer([&] { vkDestroyBuffer(device, buffer, nullptr); });
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    VkPhysicalDeviceMemoryProperties types{};
    vkGetPhysicalDeviceMemoryProperties(physical, &types);
    std::uint32_t type = 0;
    while (type < types.memoryTypeCount &&
           ((requirements.memoryTypeBits & (1u << type)) == 0 ||
            (types.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0)) {
        ++type;
    }
    if (type == types.memoryTypeCount) {
        throw DeviceError(DeviceErrorKind::Unsupported, what + ": no device-local memory fits");
    }
#if defined(_WIN32)
    const VkImportMemoryWin32HandleInfoKHR imported{
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
        .handleType = kExternalMemoryType,
        .handle = reinterpret_cast<HANDLE>(handle.Get())};
#else
    const VkImportMemoryFdInfoKHR imported{.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
                                           .handleType = kExternalMemoryType,
                                           .fd = static_cast<int>(handle.Get())};
#endif
    const VkMemoryAllocateFlagsInfo flags{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                          .pNext = &imported,
                                          .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    const VkMemoryAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                        .pNext = &flags,
                                        .allocationSize = info.size,
                                        .memoryTypeIndex = type};
    CheckVk(vkAllocateMemory(device, &allocate, nullptr, &memory), what);
    handle.Imported();
    const OnFailure freeMemory([&] { vkFreeMemory(device, memory, nullptr); });
    CheckVk(vkBindBufferMemory(device, buffer, memory, 0), what);
}

ExternalHandle ExportMemory(VkDevice device, VkDeviceMemory memory, std::string_view name) {
    const std::string what = "Exporting " + std::string(name);
#if defined(_WIN32)
    const VkMemoryGetWin32HandleInfoKHR info{.sType =
                                                 VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
                                             .memory = memory,
                                             .handleType = kExternalMemoryType};
    HANDLE handle = nullptr;
    CheckVk(vkGetMemoryWin32HandleKHR(device, &info, &handle), what);
    return ExternalHandle(reinterpret_cast<std::intptr_t>(handle));
#else
    const VkMemoryGetFdInfoKHR info{.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
                                    .memory = memory,
                                    .handleType = kExternalMemoryType};
    int fd = -1;
    CheckVk(vkGetMemoryFdKHR(device, &info, &fd), what);
    return ExternalHandle(fd);
#endif
}

ExternalHandle ExportSemaphore(VkDevice device, VkSemaphore semaphore, std::string_view name) {
    const std::string what = "Exporting " + std::string(name);
#if defined(_WIN32)
    const VkSemaphoreGetWin32HandleInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
        .semaphore = semaphore,
        .handleType = kExternalSemaphoreType};
    HANDLE handle = nullptr;
    CheckVk(vkGetSemaphoreWin32HandleKHR(device, &info, &handle), what);
    return ExternalHandle(reinterpret_cast<std::intptr_t>(handle));
#else
    const VkSemaphoreGetFdInfoKHR info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
                                       .semaphore = semaphore,
                                       .handleType = kExternalSemaphoreType};
    int fd = -1;
    CheckVk(vkGetSemaphoreFdKHR(device, &info, &fd), what);
    return ExternalHandle(fd);
#endif
}

void ImportSemaphore(VkDevice device, VkSemaphore semaphore, ExternalHandle& handle,
                     std::string_view name) {
    const std::string what = "Importing " + std::string(name);
#if defined(_WIN32)
    const VkImportSemaphoreWin32HandleInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR,
        .semaphore = semaphore,
        .handleType = kExternalSemaphoreType,
        .handle = reinterpret_cast<HANDLE>(handle.Get())};
    CheckVk(vkImportSemaphoreWin32HandleKHR(device, &info), what);
#else
    const VkImportSemaphoreFdInfoKHR info{.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
                                          .semaphore = semaphore,
                                          .handleType = kExternalSemaphoreType,
                                          .fd = static_cast<int>(handle.Get())};
    CheckVk(vkImportSemaphoreFdKHR(device, &info), what);
    handle.Release();
#endif
}

}  // namespace ernie
