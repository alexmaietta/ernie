// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/vulkan/api.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <variant>

#include <volk.h>

#include "ernie/util/error.h"
#include "ernie/util/library.h"
#include "ernie/util/log.h"
#include "ernie/util/version.h"

namespace ernie::device::vulkan {
namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

std::mutex g_volkMutex;
std::atomic<bool> g_validationEnabled{false};
std::atomic<uint64_t> g_validationMessageCount{0};

bool HasLayer(const char* name) {
    const std::vector<VkLayerProperties> layers =
        Enumerate<VkLayerProperties>(vkEnumerateInstanceLayerProperties);
    return std::any_of(layers.begin(), layers.end(), [name](const VkLayerProperties& layer) {
        return std::strcmp(layer.layerName, name) == 0;
    });
}

static_assert(DecodeVersion(VK_MAKE_API_VERSION(0, 1, 4, 341)) == Version{1, 4, 341});

Loaded<Api> Load() {
    if (volkInitialize() != VK_SUCCESS) {
        return "no Vulkan loader was found";
    }
    Api api;
    api.instanceVersion = DecodeVersion(volkGetInstanceVersion());
    if (api.instanceVersion < kMinimumInstanceVersion) {
        return TooOld("the Vulkan loader", api.instanceVersion, kMinimumInstanceVersion);
    }
    Log(LogLevel::Debug, "the Vulkan loader supports Vulkan {}", api.instanceVersion.ToString());
    return api;
}

const Loaded<Api>& Cached() {
    static const Loaded<Api> loaded = LogUnavailable("Vulkan", Load());
    return loaded;
}

}

bool Available() {
    return std::holds_alternative<Api>(Cached());
}

const Api& GetApi() {
    return RequireLoaded(Cached());
}

const char* ResultName(VkResult result) {
    switch (result) {
#define ERNIE_RESULT(name)                                                                         \
    case name:                                                                                     \
        return #name;
        ERNIE_RESULT(VK_SUCCESS)
        ERNIE_RESULT(VK_NOT_READY)
        ERNIE_RESULT(VK_TIMEOUT)
        ERNIE_RESULT(VK_INCOMPLETE)
        ERNIE_RESULT(VK_ERROR_OUT_OF_HOST_MEMORY)
        ERNIE_RESULT(VK_ERROR_OUT_OF_DEVICE_MEMORY)
        ERNIE_RESULT(VK_ERROR_INITIALIZATION_FAILED)
        ERNIE_RESULT(VK_ERROR_DEVICE_LOST)
        ERNIE_RESULT(VK_ERROR_MEMORY_MAP_FAILED)
        ERNIE_RESULT(VK_ERROR_LAYER_NOT_PRESENT)
        ERNIE_RESULT(VK_ERROR_EXTENSION_NOT_PRESENT)
        ERNIE_RESULT(VK_ERROR_FEATURE_NOT_PRESENT)
        ERNIE_RESULT(VK_ERROR_INCOMPATIBLE_DRIVER)
        ERNIE_RESULT(VK_ERROR_TOO_MANY_OBJECTS)
        ERNIE_RESULT(VK_ERROR_FORMAT_NOT_SUPPORTED)
        ERNIE_RESULT(VK_ERROR_FRAGMENTED_POOL)
        ERNIE_RESULT(VK_ERROR_OUT_OF_POOL_MEMORY)
        ERNIE_RESULT(VK_ERROR_INVALID_EXTERNAL_HANDLE)
        ERNIE_RESULT(VK_ERROR_FRAGMENTATION)
        ERNIE_RESULT(VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS)
#undef ERNIE_RESULT
    default:
        return "VK_ERROR_UNKNOWN";
    }
}

void Check(VkResult result, const char* what) {
    if (result == VK_ERROR_DEVICE_LOST) {
        throw DeviceLostError("{} failed: the device was lost (VK_ERROR_DEVICE_LOST)", what);
    }
    if (result != VK_SUCCESS) {
        throw ApiError(what, ResultName(result), result);
    }
}

Extensions::Extensions(VkPhysicalDevice device)
    : m_extensions(
          Enumerate<VkExtensionProperties>([device](uint32_t* count, VkExtensionProperties* items) {
              if (device != VK_NULL_HANDLE) {
                  vkEnumerateDeviceExtensionProperties(device, nullptr, count, items);
              } else {
                  vkEnumerateInstanceExtensionProperties(nullptr, count, items);
              }
          })) {}

bool Extensions::Has(const char* name) const {
    return std::any_of(m_extensions.begin(), m_extensions.end(),
                       [name](const VkExtensionProperties& extension) {
                           return std::strcmp(extension.extensionName, name) == 0;
                       });
}

namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL ReportValidation(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                VkDebugUtilsMessageTypeFlagsEXT,
                                                const VkDebugUtilsMessengerCallbackDataEXT* data,
                                                void*) {
    g_validationMessageCount.fetch_add(1, std::memory_order_relaxed);
    const LogLevel level = severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
                               ? LogLevel::Error
                               : LogLevel::Warning;
    Log(level, "Vulkan validation: {}", data->pMessage);
    return VK_FALSE;
}

void CreateValidationMessenger(VkInstance instance) {
    VkDebugUtilsMessengerCreateInfoEXT info{
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    info.pfnUserCallback = ReportValidation;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    Check(vkCreateDebugUtilsMessengerEXT(instance, &info, nullptr, &messenger),
          "vkCreateDebugUtilsMessengerEXT");
}

std::vector<const char*> ValidationLayers(bool validation, bool debugUtils) {
    if (!validation) {
        return {};
    }
    if (!debugUtils) {
        Log(LogLevel::Warning, "validation was asked for, but {} is not available",
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        return {};
    }
    if (!HasLayer(kValidationLayer)) {
        Log(LogLevel::Warning, "validation was asked for, but {} is not installed",
            kValidationLayer);
        return {};
    }
    Log(LogLevel::Debug, "validating Vulkan calls with {}", kValidationLayer);
    return {kValidationLayer};
}

VkInstance CreateInstance(bool validation) {
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "Ernie";
    application.pEngineName = "Ernie";
    application.apiVersion = VK_API_VERSION_1_3;
    const Extensions available;
    const bool debugUtils = available.Has(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    const std::vector<const char*> layers = ValidationLayers(validation, debugUtils);

    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &application;
    info.enabledLayerCount = uint32_t(layers.size());
    info.ppEnabledLayerNames = layers.data();
    std::vector<const char*> extensions;
    if (available.Has(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }
    if (debugUtils) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    for (const char* surface : kSurfaceExtensions) {
        if (available.Has(surface)) {
            extensions.push_back(surface);
        }
    }
    info.enabledExtensionCount = uint32_t(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();
    VkInstance instance = VK_NULL_HANDLE;
    Check(vkCreateInstance(&info, nullptr, &instance), "vkCreateInstance");
    volkLoadInstanceOnly(instance);

    if (!layers.empty()) {
        CreateValidationMessenger(instance);
        g_validationEnabled.store(true, std::memory_order_relaxed);
    }
    return instance;
}

}

VkInstance Instance(bool validation) {
    static VkInstance instances[2] = {};
    const std::lock_guard lock(g_volkMutex);
    VkInstance& instance = instances[validation ? 1 : 0];
    if (instance == VK_NULL_HANDLE) {
        instance = CreateInstance(validation);
    }
    return instance;
}

bool ValidationEnabled() {
    return g_validationEnabled.load(std::memory_order_relaxed);
}

uint64_t ValidationMessageCount() {
    return g_validationMessageCount.load(std::memory_order_relaxed);
}

std::vector<VkPhysicalDevice> PhysicalDevices(VkInstance instance) {
    return Enumerate<VkPhysicalDevice>([instance](uint32_t* count, VkPhysicalDevice* devices) {
        Check(vkEnumeratePhysicalDevices(instance, count, devices), "vkEnumeratePhysicalDevices");
    });
}

void LoadDeviceFunctions(VolkDeviceTable& functions, VkDevice device) {
    std::lock_guard lock(g_volkMutex);
    volkLoadDeviceTable(&functions, device);
}

}
