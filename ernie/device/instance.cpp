// SPDX-License-Identifier: BSD-3-Clause
/// @file instance.cpp
/// @brief Creating the instance, with validation configured through layer settings.
#include "ernie/device/instance.h"

#include "ernie/util/algorithm.h"
#include "ernie/util/on_failure.h"

#include <algorithm>
#include <array>
#include <string_view>
#include <utility>

namespace ernie {
namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

template <typename Properties, typename Name>
bool Has(const std::vector<Properties>& list, std::string_view name, Name Properties::* field) {
    return std::ranges::any_of(list, [&](const Properties& p) { return name == p.*field; });
}

std::vector<VkExtensionProperties> Extensions(const char* layer = nullptr) {
    std::uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(layer, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateInstanceExtensionProperties(layer, &count, extensions.data());
    return extensions;
}

bool HasValidationLayer() {
    std::uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    return Has(layers, kValidationLayer, &VkLayerProperties::layerName);
}

}  // namespace

Instance::Instance(const InstanceOptions& options) {
    if (volkInitialize() != VK_SUCCESS) {
        throw DeviceError(DeviceErrorKind::Unsupported, "No Vulkan loader is installed");
    }
    std::vector<const char*> extensions =
        Map(options.extensions, [](const std::string& name) { return name.c_str(); });
    const std::vector<VkExtensionProperties> available = Extensions();
    VkInstanceCreateFlags flags = 0;
    // Drivers layered over Metal (MoltenVK, Mesa on macOS) are listed only with portability
    // enumeration.
    if (Has(available, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
            &VkExtensionProperties::extensionName)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    const bool validation = options.validation && HasValidationLayer();
    const bool settings =
        validation && Has(Extensions(kValidationLayer), VK_EXT_LAYER_SETTINGS_EXTENSION_NAME,
                          &VkExtensionProperties::extensionName);
    if (validation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    if (settings) {
        extensions.push_back(VK_EXT_LAYER_SETTINGS_EXTENSION_NAME);
    }
    // Synchronization validation catches missing barriers, which ordinary validation does not.
    const VkBool32 on = VK_TRUE;
    const std::array layerSettings{
        VkLayerSettingEXT{kValidationLayer, "validate_sync", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1,
                          &on},
    };
    const VkLayerSettingsCreateInfoEXT layerSettingsInfo{
        .sType = VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT,
        .settingCount = static_cast<std::uint32_t>(layerSettings.size()),
        .pSettings = layerSettings.data()};
    const VkApplicationInfo application{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                        .pApplicationName = "ernie",
                                        .pEngineName = "ernie",
                                        .apiVersion = VK_API_VERSION_1_3};
    const VkInstanceCreateInfo info{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                    .pNext = settings ? &layerSettingsInfo : nullptr,
                                    .flags = flags,
                                    .pApplicationInfo = &application,
                                    .enabledLayerCount = validation ? 1u : 0u,
                                    .ppEnabledLayerNames = validation ? &kValidationLayer : nullptr,
                                    .enabledExtensionCount =
                                        static_cast<std::uint32_t>(extensions.size()),
                                    .ppEnabledExtensionNames = extensions.data()};
    CheckVk(vkCreateInstance(&info, nullptr, &instance_), "Creating a Vulkan 1.3 instance");
    const OnFailure freeInstance([&] { vkDestroyInstance(instance_, nullptr); });
    // Device functions too, through the loader's dispatch: they then work for every device, where
    // volkLoadDevice would tie the process to one.
    volkLoadInstance(instance_);
    if (validation) {
        const auto load = [&](const char* name) { return vkGetInstanceProcAddr(instance_, name); };
        const auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            load("vkCreateDebugUtilsMessengerEXT"));
        destroyMessenger_ = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            load("vkDestroyDebugUtilsMessengerEXT"));
        setName_ = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
            load("vkSetDebugUtilsObjectNameEXT"));
        beginLabel_ = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
            load("vkCmdBeginDebugUtilsLabelEXT"));
        endLabel_ =
            reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(load("vkCmdEndDebugUtilsLabelEXT"));
        const VkDebugUtilsMessengerCreateInfoEXT messenger{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
            .pfnUserCallback = &Instance::Report,
            .pUserData = this};
        CheckVk(createMessenger(instance_, &messenger, nullptr, &messenger_),
                "Creating the validation messenger");
    }
}

Instance::~Instance() {
    if (messenger_ != VK_NULL_HANDLE) {
        destroyMessenger_(instance_, messenger_, nullptr);
    }
    vkDestroyInstance(instance_, nullptr);
}

void Instance::Name(VkDevice device, VkObjectType type, std::uint64_t handle,
                    std::string_view name) const {
    if (setName_ == nullptr) {
        return;
    }
    const std::string text(name);
    const VkDebugUtilsObjectNameInfoEXT info{.sType =
                                                 VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                                             .objectType = type,
                                             .objectHandle = handle,
                                             .pObjectName = text.c_str()};
    setName_(device, &info);
}

void Instance::BeginLabel(VkCommandBuffer commands, std::string_view name) const {
    if (beginLabel_ == nullptr) {
        return;
    }
    const std::string text(name);
    const VkDebugUtilsLabelEXT label{.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
                                     .pLabelName = text.c_str()};
    beginLabel_(commands, &label);
}

void Instance::EndLabel(VkCommandBuffer commands) const {
    if (endLabel_ != nullptr) {
        endLabel_(commands);
    }
}

std::vector<VkPhysicalDevice> Instance::PhysicalDevices() const {
    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());
    return devices;
}

std::vector<std::string> Instance::TakeMessages() {
    const std::lock_guard lock(mutex_);
    return std::exchange(messages_, {});
}

VkBool32 Instance::Report(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
                          const VkDebugUtilsMessengerCallbackDataEXT* data, void* self) {
    auto& instance = *static_cast<Instance*>(self);
    const std::lock_guard lock(instance.mutex_);
    instance.messages_.emplace_back(data->pMessage);
    return VK_FALSE;
}

}  // namespace ernie
