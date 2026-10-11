// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/gpu_timer.h"

#include <vk_mem_alloc.h>
#include <volk.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace ernie {

/// Memory or a semaphore exported for another API, such as CUDA: a
/// POSIX file descriptor, or a Win32 NT handle on Windows.
#if defined(_WIN32)
using ExternalHandle = void*;
inline constexpr VkExternalMemoryHandleTypeFlagBits
    kExternalMemoryHandleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
inline constexpr VkExternalSemaphoreHandleTypeFlagBits
    kExternalSemaphoreHandleType =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
using ExternalHandle = int;
inline constexpr VkExternalMemoryHandleTypeFlagBits
    kExternalMemoryHandleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
inline constexpr VkExternalSemaphoreHandleTypeFlagBits
    kExternalSemaphoreHandleType =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

/// Closes an exported handle its importer did not take: a file
/// descriptor an import failed with, or any Win32 handle (CUDA
/// imports them without taking ownership).
void CloseExternalHandle(ExternalHandle handle);

/// Options for `VulkanContext::Create`.
struct VulkanContextOptions {
    /// Enable the Khronos validation layer when it is installed; its
    /// messages are collected (see
    /// `VulkanContext::TakeValidationMessages`) and errors are also
    /// printed to stderr.
    bool validation = false;
    /// Prefer the first device whose name contains this text; empty
    /// prefers a discrete GPU, then an integrated one, then anything
    /// else.
    std::string preferredDevice;
    /// Extra instance extensions, e.g. the ones a window system needs
    /// for presentation.
    std::vector<std::string> instanceExtensions;
    /// Extra device extensions, e.g. `VK_KHR_swapchain`.
    std::vector<std::string> deviceExtensions;
};

/// A wait for or a signal of a timeline semaphore value in a
/// submission.
struct TimelinePoint {
    VkSemaphore semaphore = VK_NULL_HANDLE;
    std::uint64_t value = 0;
    /// Stages that wait for the value, or that must finish before it
    /// is signalled.
    VkPipelineStageFlags2 stages =
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
};

/// Owns a Vulkan instance, device, graphics and compute queue, and
/// memory allocator.
///
/// Requires Vulkan 1.3 with dynamic rendering, synchronization2,
/// demote to helper invocation, timeline semaphores and shader draw
/// parameters, plus push descriptors. Ray queries are enabled when
/// the device supports them. Vulkan is loaded at run time through
/// volk: the system loader on Linux and Windows, MoltenVK on macOS
/// (with the portability extensions it needs). Several contexts may
/// exist at once, on the same or different devices: core functions go
/// through the loader's dispatch, and extension functions come from
/// the context that enabled them (e.g. `PushDescriptorSet`), never
/// from volk's globals.
class VulkanContext {
  public:
    /// @throws std::runtime_error if Vulkan cannot be loaded or no
    /// device meets the requirements.
    static std::unique_ptr<VulkanContext>
    Create(const VulkanContextOptions& options = {});

    ~VulkanContext();
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    VkInstance Instance() const {
        return instance_;
    }

    VkPhysicalDevice PhysicalDevice() const {
        return physicalDevice_;
    }

    VkDevice Device() const {
        return device_;
    }

    VkQueue Queue() const {
        return queue_;
    }

    std::uint32_t QueueFamily() const {
        return queueFamily_;
    }

    VmaAllocator Allocator() const {
        return allocator_;
    }

    /// Expires when the context is destroyed, for objects that may
    /// outlive it, such as buffers behind arrays a scene still holds
    /// after its renderer is gone: they then leave their handles to
    /// the destroyed device rather than destroy them.
    std::weak_ptr<const void> Alive() const {
        return alive_;
    }

    /// Device name, e.g. "NVIDIA GeForce RTX 5070 Laptop GPU".
    const std::string& DeviceName() const {
        return deviceName_;
    }

    /// Device UUID, which CUDA reports too, for matching devices
    /// across APIs.
    const std::array<std::uint8_t, VK_UUID_SIZE>& DeviceUuid() const {
        return deviceUuid_;
    }

    const VkPhysicalDeviceLimits& Limits() const {
        return limits_;
    }

    bool SupportsRayQuery() const {
        return rayQuery_;
    }

    bool SupportsAnisotropy() const {
        return anisotropy_;
    }

    /// Whether one indirect draw call may draw several commands
    /// (`multiDrawIndirect`); without it, callers issue one call per
    /// command.
    bool SupportsMultiDrawIndirect() const {
        return multiDrawIndirect_;
    }

    /// Whether buffers can be allocated with `BufferMemory::Shared`:
    /// memory and semaphores exported for CUDA interop, as POSIX file
    /// descriptors (VK_KHR_external_memory_fd and
    /// VK_KHR_external_semaphore_fd) or, on Windows, Win32 handles
    /// (their _win32 counterparts).
    bool SupportsExternalMemory() const {
        return exportMemory_ != nullptr &&
               exportSemaphore_ != nullptr;
    }

    /// Whether shaders can sample linear images (`ImageDesc::linear`)
    /// of `format`, which compute APIs then write in place.
    bool SupportsLinearSampling(VkFormat format) const {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice_, format,
                                            &properties);
        return (properties.linearTilingFeatures &
                VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
    }

    /// A new handle to `memory`, allocated exportable as
    /// `kExternalMemoryHandleType`, which the importer takes
    /// (`CloseExternalHandle`).
    /// @throws std::runtime_error without `SupportsExternalMemory`,
    /// or if the export fails.
    ExternalHandle ExportMemory(VkDeviceMemory memory) const;

    /// A new handle to `semaphore`, created exportable
    /// (`CreateTimelineSemaphore`), as `ExportMemory`.
    /// @throws std::runtime_error as `ExportMemory`.
    ExternalHandle ExportSemaphore(VkSemaphore semaphore) const;

    /// The index of a device-local memory type among `typeBits`.
    /// @throws std::runtime_error if there is none.
    std::uint32_t DeviceLocalMemoryType(std::uint32_t typeBits) const;

    /// Creates a timeline semaphore at `initial`, exportable as a
    /// POSIX file descriptor when `exportable` is set (which needs
    /// `SupportsExternalMemory`). The caller destroys it.
    /// @throws std::runtime_error if creation fails.
    VkSemaphore
    CreateTimelineSemaphore(std::uint64_t initial = 0,
                            bool exportable = false) const;

    bool ValidationEnabled() const {
        return validation_;
    }

    /// `vkCmdPushDescriptorSetKHR` for this device.
    PFN_vkCmdPushDescriptorSetKHR PushDescriptorSet() const {
        return pushDescriptorSet_;
    }

    /// Validation and error messages reported since the last call,
    /// oldest first.
    std::vector<std::string> TakeValidationMessages();

    /// Records commands with `record` into a one-off command buffer,
    /// submits it and waits for the device to finish it. Its writes
    /// are then visible to the host and to later submissions. The
    /// submission waits for `waits` before starting and signals
    /// `signals` when done.
    /// @throws std::runtime_error if submission fails.
    void Submit(const std::function<void(VkCommandBuffer)>& record,
                std::span<const TimelinePoint> waits = {},
                std::span<const TimelinePoint> signals = {}) const;

    /// Like `Submit`, but returns once the work is submitted, without
    /// waiting for the device; its command buffer is recycled after
    /// the device finishes it. Use `signals` to tell other work (such
    /// as CUDA) when it is done, or `WaitForSubmission` with the
    /// returned number.
    /// @throws std::runtime_error if submission fails.
    std::uint64_t
    SubmitAsync(const std::function<void(VkCommandBuffer)>& record,
                std::span<const TimelinePoint> waits = {},
                std::span<const TimelinePoint> signals = {}) const;

    /// Waits until the device has finished every `SubmitAsync`
    /// submission.
    void WaitForSubmissions() const;

    /// Waits up to `timeoutNs` nanoseconds (0 polls) for
    /// `SubmitAsync` submission `submission`; true once the device
    /// has finished it.
    bool
    WaitForSubmission(std::uint64_t submission,
                      std::uint64_t timeoutNs = UINT64_MAX) const;

    /// Whether `SubmitAsync` work is still pending on the device.
    bool HasPendingSubmissions() const;

    /// The Tracy Vulkan context of a profiling build (ERNIE_TRACY)
    /// for `ERNIE_GPU_ZONE`; null otherwise. GPU zones are calibrated
    /// against the CPU's clock when the device has
    /// VK_KHR_calibrated_timestamps (or the EXT).
    void* GpuProfiler() const {
        return gpuProfiler_;
    }

    /// The timer of `ERNIE_GPU_ZONE`s for Ernie's own profiler
    /// (ernie/util/profiler.h); it times nothing on devices without
    /// timestamps or host query reset.
    GpuTimer& Timer() const {
        return *timer_;
    }

    /// Records into `commands`, outside a rendering pass, the
    /// collection of finished GPU zones for Tracy, and hands those
    /// Ernie's profiler timed to it; nothing without either. `Submit`
    /// and `SubmitAsync` collect at the start of every command
    /// buffer, so only command buffers recorded elsewhere need it,
    /// such as an app's frames.
    void CollectGpuProfile(VkCommandBuffer commands) const;

  private:
    std::shared_ptr<const void> alive_ =
        std::make_shared<char>();  ///< See `Alive`.
    VulkanContext() = default;
    void Report(std::string message);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queueFamily_ = 0;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT destroyMessenger_ = nullptr;
    PFN_vkCmdPushDescriptorSetKHR pushDescriptorSet_ = nullptr;
    // vkGetMemoryFdKHR and vkGetSemaphoreFdKHR, or their Win32
    // counterparts on Windows: kept as void functions so this header
    // needs no Win32 types (and no <windows.h>).
    PFN_vkVoidFunction exportMemory_ = nullptr;
    PFN_vkVoidFunction exportSemaphore_ = nullptr;
    std::string deviceName_;
    std::array<std::uint8_t, VK_UUID_SIZE> deviceUuid_{};
    VkPhysicalDeviceLimits limits_{};
    bool rayQuery_ = false;
    bool anisotropy_ = false;
    bool multiDrawIndirect_ = false;
    bool validation_ = false;
    mutable std::mutex submitMutex_;
    void* gpuProfiler_ =
        nullptr;  ///< A tracy::VkCtx in a profiling build.
    std::unique_ptr<GpuTimer> timer_;
    mutable std::mutex profilerMutex_;
    /// Timeline signalled by every `SubmitAsync`, and its command
    /// buffers awaiting recycling.
    VkSemaphore submissions_ = VK_NULL_HANDLE;
    mutable std::uint64_t submitted_ = 0;
    mutable std::vector<std::pair<VkCommandBuffer, std::uint64_t>>
        pending_;

    VkCommandBuffer
    Record(const std::function<void(VkCommandBuffer)>& record) const;
    void RecyclePending() const;
    std::unique_ptr<std::function<void(std::string)>> sink_;
    std::mutex messagesMutex_;
    std::vector<std::string> messages_;
};

/// Loads Vulkan (the system loader, or MoltenVK on macOS without one)
/// and returns its `vkGetInstanceProcAddr`, e.g. for
/// `glfwInitVulkanLoader` so a window system uses the same Vulkan as
/// Ernie.
/// @throws std::runtime_error if Vulkan is not available.
PFN_vkGetInstanceProcAddr LoadVulkanLoader();

/// Throws `std::runtime_error` naming `what` and the result unless
/// `result` is `VK_SUCCESS`.
void CheckVk(VkResult result, const char* what);

}  // namespace ernie
