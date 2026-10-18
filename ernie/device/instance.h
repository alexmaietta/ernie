// SPDX-License-Identifier: BSD-3-Clause
/// @file instance.h
/// @brief The Vulkan instance: the loader, optional validation and the messages it reports.
#pragma once

#include "ernie/device/vulkan.h"

#include <mutex>
#include <string>
#include <vector>

namespace ernie {

struct InstanceOptions {
    /// The Khronos validation layer with synchronization validation, when installed.
    bool validation = false;
    /// Extensions a window system needs.
    std::vector<std::string> extensions;
};

/// @brief A Vulkan 1.3 instance. Each owns its messages: two contexts in one process keep theirs
/// apart.
class Instance {
  public:
    /// @throws DeviceError when no Vulkan loader or no Vulkan 1.3 driver is installed.
    explicit Instance(const InstanceOptions& options = {});
    ~Instance();
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;

    VkInstance Handle() const {
        return instance_;
    }
    std::vector<VkPhysicalDevice> PhysicalDevices() const;
    /// @return Whether the validation layer was found and enabled.
    bool Validating() const {
        return messenger_ != VK_NULL_HANDLE;
    }
    /// @return The validation layer's warnings and errors since the last call.
    std::vector<std::string> TakeMessages();
    /// @brief Names a device's object, when validating; otherwise does nothing.
    void Name(VkDevice device, VkObjectType type, std::uint64_t handle,
              std::string_view name) const;
    /// @brief Opens a named region of commands for captures and validation messages, when
    /// validating; otherwise does nothing.
    void BeginLabel(VkCommandBuffer commands, std::string_view name) const;
    void EndLabel(VkCommandBuffer commands) const;

  private:
    static VKAPI_ATTR VkBool32 VKAPI_CALL Report(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                 VkDebugUtilsMessageTypeFlagsEXT types,
                                                 const VkDebugUtilsMessengerCallbackDataEXT* data,
                                                 void* self);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    // This instance's own: volk's globals hold whichever instance loaded last, and only those
    // with validation have these.
    PFN_vkDestroyDebugUtilsMessengerEXT destroyMessenger_ = nullptr;
    PFN_vkSetDebugUtilsObjectNameEXT setName_ = nullptr;
    PFN_vkCmdBeginDebugUtilsLabelEXT beginLabel_ = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT endLabel_ = nullptr;
    std::mutex mutex_;
    std::vector<std::string> messages_;
};

}  // namespace ernie
