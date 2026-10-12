// SPDX-License-Identifier: BSD-3-Clause
// Vulkan's Win32 types, for memory and semaphores exported to CUDA as
// Win32 handles: here only, as they bring in <windows.h> and its
// macros.
#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include "ernie/device/vulkan_context.h"

#include "ernie/device/gpu_profile.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <dlfcn.h>
#endif
#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace ernie {

void CloseExternalHandle(ExternalHandle handle) {
#if defined(_WIN32)
    if (handle != nullptr) {
        CloseHandle(handle);
    }
#else
    if (handle >= 0) {
        close(handle);
    }
#endif
}

namespace {

constexpr const char* kValidationLayer =
    "VK_LAYER_KHRONOS_validation";

bool HasLayer(const char* name) {
    std::uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    return std::any_of(layers.begin(), layers.end(),
                       [name](const VkLayerProperties& layer) {
                           return std::strcmp(layer.layerName,
                                              name) == 0;
                       });
}

std::vector<VkExtensionProperties> InstanceExtensions() {
    std::uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count,
                                           extensions.data());
    return extensions;
}

std::vector<VkExtensionProperties>
DeviceExtensions(VkPhysicalDevice device) {
    std::uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count,
                                         nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count,
                                         extensions.data());
    return extensions;
}

bool Has(const std::vector<VkExtensionProperties>& extensions,
         const char* name) {
    return std::any_of(extensions.begin(), extensions.end(),
                       [name](const auto& extension) {
                           return std::strcmp(extension.extensionName,
                                              name) == 0;
                       });
}

void LoadVulkan() {
    if (volkGetInstanceVersion() != 0) {
        return;
    }
    if (volkInitialize() == VK_SUCCESS) {
        return;
    }
#if defined(__APPLE__)
    // Without a Vulkan loader installed, load MoltenVK directly: an
    // installed one, the one shipped beside the binary Ernie is in
    // (the Python package's), or the build's.
    std::vector<std::string> paths = {"libMoltenVK.dylib"};
    Dl_info self{};
    if (dladdr(reinterpret_cast<const void*>(&LoadVulkan), &self) !=
            0 &&
        self.dli_fname != nullptr) {
        const std::string binary = self.dli_fname;
        paths.push_back(
            binary.substr(0, binary.find_last_of('/') + 1) +
            "libMoltenVK.dylib");
    }
    paths.emplace_back(ERNIE_MOLTENVK_PATH);
    for (const std::string& path : paths) {
        if (void* library =
                dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)) {
            if (auto getProc =
                    reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                        dlsym(library, "vkGetInstanceProcAddr"))) {
                volkInitializeCustom(getProc);
                return;
            }
        }
    }
#endif
    throw std::runtime_error("Vulkan is not available: no Vulkan "
                             "loader or driver was found");
}

VKAPI_ATTR VkBool32 VKAPI_CALL OnDebugMessage(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        const std::string message =
            data->pMessage != nullptr ? data->pMessage : "";
        if (severity >=
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
            std::fprintf(stderr, "Vulkan validation: %s\n",
                         message.c_str());
        }
        static_cast<std::function<void(std::string)>*>(user)
            ->operator()(message);
    }
    return VK_FALSE;
}

std::string ResultName(VkResult result) {
    switch (result) {
    case VK_ERROR_OUT_OF_HOST_MEMORY:
        return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
        return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:
        return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_LAYER_NOT_PRESENT:
        return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
        return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:
        return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
        return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_OUT_OF_DATE_KHR:
        return "VK_ERROR_OUT_OF_DATE_KHR";
    case VK_ERROR_SURFACE_LOST_KHR:
        return "VK_ERROR_SURFACE_LOST_KHR";
    default:
        return "VkResult " + std::to_string(static_cast<int>(result));
    }
}

int DeviceRank(VkPhysicalDeviceType type) {
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return 0;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return 1;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return 2;
    default:
        return 3;
    }
}

}  // namespace

PFN_vkGetInstanceProcAddr LoadVulkanLoader() {
    LoadVulkan();
    return vkGetInstanceProcAddr;
}

void CheckVk(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(what) +
                                 " failed: " + ResultName(result));
    }
}

std::unique_ptr<VulkanContext>
VulkanContext::Create(const VulkanContextOptions& options) {
    LoadVulkan();
    if (volkGetInstanceVersion() < VK_API_VERSION_1_3) {
        throw std::runtime_error(
            "Ernie needs Vulkan 1.3; the installed loader is older");
    }
    std::unique_ptr<VulkanContext> context(new VulkanContext());
    VulkanContext& c = *context;

    const std::vector<VkExtensionProperties> available =
        InstanceExtensions();
    std::vector<const char*> extensions;
    const auto add = [&extensions](const char* name) {
        if (std::none_of(extensions.begin(), extensions.end(),
                         [name](const char* added) {
                             return std::strcmp(added, name) == 0;
                         })) {
            extensions.push_back(name);
        }
    };
    for (const std::string& extension : options.instanceExtensions) {
        add(extension.c_str());
    }
    // Every instance enables the loader's surface and debug
    // extensions it has, so each reload of the shared function table
    // (`volkLoadInstance` below) keeps the functions other contexts
    // use: a headless renderer's instance otherwise nulls a window's
    // vkDestroySurfaceKHR.
    for (const char* shared :
         {"VK_KHR_surface", "VK_KHR_wayland_surface",
          "VK_KHR_xlib_surface", "VK_KHR_xcb_surface",
          "VK_KHR_win32_surface", "VK_EXT_metal_surface",
          VK_EXT_DEBUG_UTILS_EXTENSION_NAME}) {
        if (Has(available, shared)) {
            add(shared);
        }
    }
    VkInstanceCreateFlags flags = 0;
    if (Has(available,
            VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(
            VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    std::vector<const char*> layers;
    c.validation_ = options.validation &&
                    HasLayer(kValidationLayer) &&
                    Has(available, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (c.validation_) {
        layers.push_back(kValidationLayer);
        add(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "Ernie";
    app.pEngineName = "Ernie";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instanceInfo{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.flags = flags;
    instanceInfo.pApplicationInfo = &app;
    instanceInfo.enabledLayerCount =
        static_cast<std::uint32_t>(layers.size());
    instanceInfo.ppEnabledLayerNames = layers.data();
    instanceInfo.enabledExtensionCount =
        static_cast<std::uint32_t>(extensions.size());
    instanceInfo.ppEnabledExtensionNames = extensions.data();
    CheckVk(vkCreateInstance(&instanceInfo, nullptr, &c.instance_),
            "vkCreateInstance");
    // Device functions resolve to the loader's dispatching entry
    // points rather than one device's driver, so several contexts
    // (and devices) can be alive at once.
    volkLoadInstance(c.instance_);

    if (c.validation_) {
        c.sink_ = std::make_unique<std::function<void(std::string)>>(
            [&c](std::string message) {
                c.Report(std::move(message));
            });
        VkDebugUtilsMessengerCreateInfoEXT messengerInfo{
            VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        messengerInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        messengerInfo.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        messengerInfo.pfnUserCallback = OnDebugMessage;
        messengerInfo.pUserData = c.sink_.get();
        const auto createMessenger =
            reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(
                    c.instance_, "vkCreateDebugUtilsMessengerEXT"));
        c.destroyMessenger_ =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(
                    c.instance_, "vkDestroyDebugUtilsMessengerEXT"));
        CheckVk(createMessenger(c.instance_, &messengerInfo, nullptr,
                                &c.messenger_),
                "vkCreateDebugUtilsMessengerEXT");
    }

    std::uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(c.instance_, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(c.instance_, &deviceCount,
                               devices.data());

    struct Candidate {
        VkPhysicalDevice device;
        std::uint32_t family;
        int rank;
    };
    std::vector<Candidate> candidates;
    std::string rejected;
    for (VkPhysicalDevice device : devices) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        const std::vector<VkExtensionProperties> deviceExtensions =
            DeviceExtensions(device);
        VkPhysicalDeviceVulkan13Features features13{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features features12{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        features12.pNext = &features13;
        VkPhysicalDeviceVulkan11Features features11{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        features11.pNext = &features12;
        VkPhysicalDeviceFeatures2 features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &features11;
        vkGetPhysicalDeviceFeatures2(device, &features);

        std::uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount,
                                                 nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount,
                                                 families.data());
        std::uint32_t family = familyCount;
        for (std::uint32_t f = 0; f < familyCount; ++f) {
            if ((families[f].queueFlags &
                 (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
                (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
                family = f;
                break;
            }
        }
        bool extensionsPresent = Has(
            deviceExtensions, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        for (const std::string& extension :
             options.deviceExtensions) {
            extensionsPresent =
                extensionsPresent &&
                Has(deviceExtensions, extension.c_str());
        }
        const bool suitable =
            properties.apiVersion >= VK_API_VERSION_1_3 &&
            family < familyCount && features13.dynamicRendering &&
            features13.synchronization2 &&
            features13.shaderDemoteToHelperInvocation &&
            features12.timelineSemaphore &&
            features12.shaderOutputLayer &&
            features11.shaderDrawParameters && extensionsPresent;
        if (!suitable) {
            rejected +=
                std::string(" ") + properties.deviceName + ";";
            continue;
        }
        int rank = DeviceRank(properties.deviceType);
        if (!options.preferredDevice.empty() &&
            std::string(properties.deviceName)
                    .find(options.preferredDevice) !=
                std::string::npos) {
            rank = -1;
        }
        candidates.push_back({device, family, rank});
    }
    if (candidates.empty()) {
        throw std::runtime_error("No Vulkan 1.3 device with dynamic "
                                 "rendering, synchronization2, "
                                 "demote to helper invocation, "
                                 "timeline semaphores, shader draw "
                                 "parameters and push "
                                 "descriptors; rejected:" +
                                 rejected);
    }
    const Candidate best =
        *std::min_element(candidates.begin(), candidates.end(),
                          [](const Candidate& a, const Candidate& b) {
                              return a.rank < b.rank;
                          });
    c.physicalDevice_ = best.device;
    c.queueFamily_ = best.family;

    VkPhysicalDeviceIDProperties ids{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &ids;
    vkGetPhysicalDeviceProperties2(c.physicalDevice_, &properties);
    c.deviceName_ = properties.properties.deviceName;
    c.limits_ = properties.properties.limits;
    std::copy(std::begin(ids.deviceUUID), std::end(ids.deviceUUID),
              c.deviceUuid_.begin());

    const std::vector<VkExtensionProperties> deviceExtensions =
        DeviceExtensions(c.physicalDevice_);
    std::vector<const char*> enabled = {
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
    for (const std::string& extension : options.deviceExtensions) {
        enabled.push_back(extension.c_str());
    }
    if (Has(deviceExtensions, "VK_KHR_portability_subset")) {
        enabled.push_back("VK_KHR_portability_subset");
    }
    // Memory and semaphores shared with CUDA: as file descriptors, or
    // Win32 handles on Windows.
#if defined(_WIN32)
    const char* const externalMemoryExtension =
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME;
    const char* const externalSemaphoreExtension =
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME;
#else
    const char* const externalMemoryExtension =
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
    const char* const externalSemaphoreExtension =
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
#endif
    const bool externalMemory =
        Has(deviceExtensions, externalMemoryExtension) &&
        Has(deviceExtensions, externalSemaphoreExtension);
    if (externalMemory) {
        enabled.push_back(externalMemoryExtension);
        enabled.push_back(externalSemaphoreExtension);
    }
#if ERNIE_TRACY
    // Profiling builds put GPU zones on the CPU's timeline with
    // calibrated timestamps.
    const bool calibratedKhr =
        Has(deviceExtensions,
            VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
    const bool calibratedExt =
        Has(deviceExtensions,
            VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
    if (calibratedKhr) {
        enabled.push_back(
            VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
    } else if (calibratedExt) {
        enabled.push_back(
            VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
    }
#endif

    // Pipelines' statistics for profiling shaders, when asked for.
    const char* statistics = std::getenv("ERNIE_PIPELINE_STATISTICS");
    c.pipelineStatistics_ =
        statistics != nullptr && *statistics != '\0' &&
        std::string_view(statistics) != "0" &&
        Has(deviceExtensions,
            VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    if (c.pipelineStatistics_) {
        enabled.push_back(
            VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    }

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationSupport{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR rayQuerySupport{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    const bool rayExtensions =
        Has(deviceExtensions,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
        Has(deviceExtensions, VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
        Has(deviceExtensions,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    VkPhysicalDeviceVulkan12Features supported12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 supported{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    supported.pNext = &supported12;
    if (rayExtensions) {
        supported12.pNext = &accelerationSupport;
        accelerationSupport.pNext = &rayQuerySupport;
    }
    vkGetPhysicalDeviceFeatures2(c.physicalDevice_, &supported);
    if (supported12.bufferDeviceAddress != VK_TRUE ||
        supported12.scalarBlockLayout != VK_TRUE ||
        supported12.runtimeDescriptorArray != VK_TRUE ||
        supported12.shaderSampledImageArrayNonUniformIndexing !=
            VK_TRUE ||
        supported12.descriptorBindingPartiallyBound != VK_TRUE ||
        supported12.descriptorBindingSampledImageUpdateAfterBind !=
            VK_TRUE ||
        supported12.descriptorBindingUpdateUnusedWhilePending !=
            VK_TRUE ||
        supported12.descriptorBindingVariableDescriptorCount !=
            VK_TRUE) {
        throw std::runtime_error(
            "The Vulkan device lacks buffer device addresses, scalar "
            "block layout or descriptor indexing (Vulkan 1.2), which "
            "Ernie's shaders need");
    }
    c.rayQuery_ =
        rayExtensions && accelerationSupport.accelerationStructure &&
        rayQuerySupport.rayQuery && supported12.bufferDeviceAddress;
    if (c.rayQuery_) {
        enabled.push_back(
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        enabled.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        enabled.push_back(
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    }

    VkPhysicalDeviceRayQueryFeaturesKHR rayQuery{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    rayQuery.rayQuery = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    acceleration.accelerationStructure = VK_TRUE;
    acceleration.pNext = &rayQuery;
    VkPhysicalDeviceVulkan13Features features13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;
    features13.shaderDemoteToHelperInvocation = VK_TRUE;
    features13.pNext = c.rayQuery_ ? &acceleration : nullptr;
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR executables{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    if (c.pipelineStatistics_) {
        executables.pipelineExecutableInfo = VK_TRUE;
        executables.pNext = features13.pNext;
        features13.pNext = &executables;
    }
    VkPhysicalDeviceVulkan12Features features12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features12.timelineSemaphore = VK_TRUE;
    // Layered rendering: a batch's views draw their G-buffers in one
    // pass, each into its layer.
    features12.shaderOutputLayer = VK_TRUE;
    // Shaders read buffers through their addresses and textures from
    // a bindless heap (`BindlessHeap`), not one push binding each.
    features12.bufferDeviceAddress = VK_TRUE;
    features12.scalarBlockLayout = VK_TRUE;
    features12.descriptorIndexing = VK_TRUE;
    features12.runtimeDescriptorArray = VK_TRUE;
    features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    features12.descriptorBindingPartiallyBound = VK_TRUE;
    features12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
    features12.descriptorBindingVariableDescriptorCount = VK_TRUE;
    // For the profiler's GPU zones (GpuTimer), which free their
    // queries from the host.
    features12.hostQueryReset = supported12.hostQueryReset;
    features12.pNext = &features13;
    VkPhysicalDeviceVulkan11Features features11{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    features11.shaderDrawParameters = VK_TRUE;
    features11.pNext = &features12;
    VkPhysicalDeviceFeatures2 features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.features.samplerAnisotropy =
        supported.features.samplerAnisotropy;
    c.anisotropy_ = supported.features.samplerAnisotropy == VK_TRUE;
    // For BC1 to BC7 textures read from DDS and KTX2 files as stored.
    features.features.textureCompressionBC =
        supported.features.textureCompressionBC;
    c.textureCompressionBc_ =
        supported.features.textureCompressionBC == VK_TRUE;
    features.features.fillModeNonSolid =
        supported.features.fillModeNonSolid;
    // For shadow maps that clamp casters in front of their near
    // plane onto it.
    features.features.depthClamp = supported.features.depthClamp;
    // For virtual shadow maps, whose casters' fragments write their
    // pages with atomics (`VirtualShadowMap`).
    features.features.fragmentStoresAndAtomics =
        supported.features.fragmentStoresAndAtomics;
    c.fragmentAtomics_ =
        supported.features.fragmentStoresAndAtomics == VK_TRUE;
    // For shadow casters, which clip to a virtual shadow map level's
    // backed pages (frame.slang's `shadowClipDistances`).
    features.features.shaderClipDistance =
        supported.features.shaderClipDistance;
    c.depthClamp_ = supported.features.depthClamp == VK_TRUE;
    features.features.multiDrawIndirect =
        supported.features.multiDrawIndirect;
    c.multiDrawIndirect_ =
        supported.features.multiDrawIndirect == VK_TRUE;
    // For virtual textures backed by sparse images (`SparseImage`):
    // tiles bound on the queue, and sampling clamped to the finest
    // resident mip (min LOD), so filtering spans tiles in hardware.
    {
        std::uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(
            c.physicalDevice_, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(
            c.physicalDevice_, &familyCount, families.data());
        c.sparseResidency_ =
            supported.features.sparseBinding == VK_TRUE &&
            supported.features.sparseResidencyImage2D == VK_TRUE &&
            supported.features.shaderResourceMinLod == VK_TRUE &&
            (families[c.queueFamily_].queueFlags &
             VK_QUEUE_SPARSE_BINDING_BIT) != 0;
        if (c.sparseResidency_) {
            features.features.sparseBinding = VK_TRUE;
            features.features.sparseResidencyImage2D = VK_TRUE;
            features.features.shaderResourceMinLod = VK_TRUE;
        }
    }
    features.pNext = &features11;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = c.queueFamily_;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkDeviceCreateInfo deviceInfo{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.pNext = &features;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount =
        static_cast<std::uint32_t>(enabled.size());
    deviceInfo.ppEnabledExtensionNames = enabled.data();
    CheckVk(vkCreateDevice(c.physicalDevice_, &deviceInfo, nullptr,
                           &c.device_),
            "vkCreateDevice");
    vkGetDeviceQueue(c.device_, c.queueFamily_, 0, &c.queue_);
    c.pushDescriptorSet_ =
        reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
            vkGetDeviceProcAddr(c.device_,
                                "vkCmdPushDescriptorSetKHR"));
    if (externalMemory) {
#if defined(_WIN32)
        c.exportMemory_ = vkGetDeviceProcAddr(
            c.device_, "vkGetMemoryWin32HandleKHR");
        c.exportSemaphore_ = vkGetDeviceProcAddr(
            c.device_, "vkGetSemaphoreWin32HandleKHR");
#else
        c.exportMemory_ =
            vkGetDeviceProcAddr(c.device_, "vkGetMemoryFdKHR");
        c.exportSemaphore_ =
            vkGetDeviceProcAddr(c.device_, "vkGetSemaphoreFdKHR");
#endif
    }

    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.flags =
        VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorInfo.physicalDevice = c.physicalDevice_;
    allocatorInfo.device = c.device_;
    allocatorInfo.instance = c.instance_;
    allocatorInfo.pVulkanFunctions = &functions;
    CheckVk(vmaCreateAllocator(&allocatorInfo, &c.allocator_),
            "vmaCreateAllocator");

    VkCommandPoolCreateInfo poolInfo{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = c.queueFamily_;
    CheckVk(vkCreateCommandPool(c.device_, &poolInfo, nullptr,
                                &c.commandPool_),
            "vkCreateCommandPool");
    c.submissions_ = c.CreateTimelineSemaphore();
    {
        std::uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(
            c.physicalDevice_, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(
            c.physicalDevice_, &familyCount, families.data());
        const std::uint32_t bits =
            supported12.hostQueryReset
                ? families[c.queueFamily_].timestampValidBits
                : 0;
        c.timer_ = std::make_unique<GpuTimer>(
            c.device_, bits, c.limits_.timestampPeriod);
    }
    c.bindless_ = std::make_unique<BindlessHeap>(c);
#if ERNIE_TRACY
    if (c.limits_.timestampComputeAndGraphics) {
        // Tracy records into its command buffer more than once, so
        // its pool allows resets.
        VkCommandPoolCreateInfo resettable{
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        resettable.flags =
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        resettable.queueFamilyIndex = c.queueFamily_;
        VkCommandPool pool = VK_NULL_HANDLE;
        CheckVk(vkCreateCommandPool(c.device_, &resettable, nullptr,
                                    &pool),
                "vkCreateCommandPool");
        VkCommandBufferAllocateInfo allocateInfo{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocateInfo.commandPool = pool;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = 1;
        VkCommandBuffer commands = VK_NULL_HANDLE;
        CheckVk(vkAllocateCommandBuffers(c.device_, &allocateInfo,
                                         &commands),
                "vkAllocateCommandBuffers");
        PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT domains =
            calibratedKhr
                ? vkGetPhysicalDeviceCalibrateableTimeDomainsKHR
            : calibratedExt
                ? vkGetPhysicalDeviceCalibrateableTimeDomainsEXT
                : nullptr;
        PFN_vkGetCalibratedTimestampsEXT timestamps =
            calibratedKhr   ? vkGetCalibratedTimestampsKHR
            : calibratedExt ? vkGetCalibratedTimestampsEXT
                            : nullptr;
        tracy::VkCtx* profiler = TracyVkContextCalibrated(
            c.physicalDevice_, c.device_, c.queue_, commands, domains,
            timestamps);
        vkDestroyCommandPool(c.device_, pool, nullptr);
        const std::string name = "Vulkan: " + c.deviceName_;
        TracyVkContextName(profiler, name.c_str(),
                           static_cast<std::uint16_t>(name.size()));
        c.gpuProfiler_ = profiler;
    }
#endif
    return context;
}

void VulkanContext::CollectGpuProfile(
    [[maybe_unused]] VkCommandBuffer commands) const {
    timer_->Collect();
#if ERNIE_TRACY
    if (gpuProfiler_ != nullptr) {
        std::lock_guard lock(profilerMutex_);
        TracyVkCollect(static_cast<tracy::VkCtx*>(gpuProfiler_),
                       commands);
    }
#endif
}

VulkanContext::~VulkanContext() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        // Shared buffers still alive go with the device, their memory
        // kept by CUDA's imports of it.
        {
            const std::lock_guard lock(exportedMutex_);
            for (const std::weak_ptr<ExportedBuffer>& weak :
                 exported_) {
                const std::shared_ptr<ExportedBuffer> exported =
                    weak.lock();
                if (exported == nullptr) {
                    continue;
                }
                const std::lock_guard held(exported->mutex);
                if (exported->buffer != VK_NULL_HANDLE) {
                    vkDestroyBuffer(device_, exported->buffer,
                                    nullptr);
                    vkFreeMemory(device_, exported->memory, nullptr);
                    exported->buffer = VK_NULL_HANDLE;
                    exported->memory = VK_NULL_HANDLE;
                }
            }
        }
        timer_.reset();
        bindless_.reset();
#if ERNIE_TRACY
        if (gpuProfiler_ != nullptr) {
            TracyVkDestroy(static_cast<tracy::VkCtx*>(gpuProfiler_));
        }
#endif
        if (submissions_ != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, submissions_, nullptr);
        }
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        vmaDestroyAllocator(allocator_);
        vkDestroyDevice(device_, nullptr);
    }
    if (messenger_ != VK_NULL_HANDLE) {
        destroyMessenger_(instance_, messenger_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

void VulkanContext::TrackExported(
    std::weak_ptr<ExportedBuffer> exported) const {
    const std::lock_guard lock(exportedMutex_);
    std::erase_if(exported_,
                  [](const std::weak_ptr<ExportedBuffer>& weak) {
                      return weak.expired();
                  });
    exported_.push_back(std::move(exported));
}

std::vector<std::string> VulkanContext::TakeValidationMessages() {
    std::lock_guard lock(messagesMutex_);
    return std::exchange(messages_, {});
}

void VulkanContext::Report(std::string message) {
    std::lock_guard lock(messagesMutex_);
    messages_.push_back(std::move(message));
}

std::uint32_t
VulkanContext::DeviceLocalMemoryType(std::uint32_t typeBits) const {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &properties);
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) != 0 &&
            (properties.memoryTypes[i].propertyFlags &
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
            return i;
        }
    }
    throw std::runtime_error("No device-local memory type");
}

ExternalHandle
VulkanContext::ExportMemory(VkDeviceMemory memory) const {
    if (!SupportsExternalMemory()) {
        throw std::runtime_error(
            "The Vulkan device cannot export memory");
    }
#if defined(_WIN32)
    VkMemoryGetWin32HandleInfoKHR info{
        VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    info.memory = memory;
    info.handleType = kExternalMemoryHandleType;
    HANDLE handle = nullptr;
    const auto exportMemory =
        reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
            exportMemory_);
    CheckVk(exportMemory(device_, &info, &handle),
            "vkGetMemoryWin32HandleKHR");
    return handle;
#else
    VkMemoryGetFdInfoKHR info{
        VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    info.memory = memory;
    info.handleType = kExternalMemoryHandleType;
    int fd = -1;
    const auto exportMemory =
        reinterpret_cast<PFN_vkGetMemoryFdKHR>(exportMemory_);
    CheckVk(exportMemory(device_, &info, &fd), "vkGetMemoryFdKHR");
    return fd;
#endif
}

ExternalHandle
VulkanContext::ExportSemaphore(VkSemaphore semaphore) const {
    if (!SupportsExternalMemory()) {
        throw std::runtime_error(
            "The Vulkan device cannot export semaphores");
    }
#if defined(_WIN32)
    VkSemaphoreGetWin32HandleInfoKHR info{
        VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
    info.semaphore = semaphore;
    info.handleType = kExternalSemaphoreHandleType;
    HANDLE handle = nullptr;
    const auto exportSemaphore =
        reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
            exportSemaphore_);
    CheckVk(exportSemaphore(device_, &info, &handle),
            "vkGetSemaphoreWin32HandleKHR");
    return handle;
#else
    VkSemaphoreGetFdInfoKHR info{
        VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    info.semaphore = semaphore;
    info.handleType = kExternalSemaphoreHandleType;
    int fd = -1;
    const auto exportSemaphore =
        reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(exportSemaphore_);
    CheckVk(exportSemaphore(device_, &info, &fd),
            "vkGetSemaphoreFdKHR");
    return fd;
#endif
}

VkSemaphore
VulkanContext::CreateTimelineSemaphore(std::uint64_t initial,
                                       bool exportable) const {
    VkSemaphoreTypeCreateInfo type{
        VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type.initialValue = initial;
    VkExportSemaphoreCreateInfo exported{
        VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    exported.handleTypes = kExternalSemaphoreHandleType;
    if (exportable) {
        if (!SupportsExternalMemory()) {
            throw std::runtime_error(
                "The Vulkan device cannot export semaphores");
        }
        type.pNext = &exported;
    }
    VkSemaphoreCreateInfo info{
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    info.pNext = &type;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    CheckVk(vkCreateSemaphore(device_, &info, nullptr, &semaphore),
            "vkCreateSemaphore");
    return semaphore;
}

VkCommandBuffer VulkanContext::Record(
    const std::function<void(VkCommandBuffer)>& record) const {
    VkCommandBufferAllocateInfo allocateInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocateInfo.commandPool = commandPool_;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    CheckVk(
        vkAllocateCommandBuffers(device_, &allocateInfo, &commands),
        "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo begin{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commands, &begin);
    CollectGpuProfile(commands);
    record(commands);
    // Make every write visible to the host and to later submissions.
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT |
                           VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT |
                            VK_ACCESS_2_MEMORY_WRITE_BIT |
                            VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commands, &dependency);
    vkEndCommandBuffer(commands);
    return commands;
}

namespace {

std::vector<VkSemaphoreSubmitInfo>
SubmitInfos(std::span<const TimelinePoint> points) {
    std::vector<VkSemaphoreSubmitInfo> infos;
    for (const TimelinePoint& point : points) {
        if (point.semaphore == VK_NULL_HANDLE) {
            continue;
        }
        VkSemaphoreSubmitInfo info{
            VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        info.semaphore = point.semaphore;
        info.value = point.value;
        info.stageMask = point.stages;
        infos.push_back(info);
    }
    return infos;
}

}  // namespace

void VulkanContext::WaitIdle() const {
    vkDeviceWaitIdle(device_);
}

void VulkanContext::Submit(
    const std::function<void(VkCommandBuffer)>& record,
    std::span<const TimelinePoint> waits,
    std::span<const TimelinePoint> signals) const {
    ERNIE_ZONE("Submit (waits)");
    std::lock_guard lock(submitMutex_);
    RecyclePending();
    VkCommandBuffer commands = Record(record);
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    CheckVk(vkCreateFence(device_, &fenceInfo, nullptr, &fence),
            "vkCreateFence");
    const std::vector<VkSemaphoreSubmitInfo> waitInfos =
        SubmitInfos(waits);
    const std::vector<VkSemaphoreSubmitInfo> signalInfos =
        SubmitInfos(signals);
    VkCommandBufferSubmitInfo commandInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    commandInfo.commandBuffer = commands;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount =
        static_cast<std::uint32_t>(waitInfos.size());
    submit.pWaitSemaphoreInfos = waitInfos.data();
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &commandInfo;
    submit.signalSemaphoreInfoCount =
        static_cast<std::uint32_t>(signalInfos.size());
    submit.pSignalSemaphoreInfos = signalInfos.data();
    const VkResult result = vkQueueSubmit2(queue_, 1, &submit, fence);
    if (result == VK_SUCCESS) {
        vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
    }
    vkDestroyFence(device_, fence, nullptr);
    vkFreeCommandBuffers(device_, commandPool_, 1, &commands);
    CheckVk(result, "vkQueueSubmit2");
}

std::uint64_t VulkanContext::SubmitAsync(
    const std::function<void(VkCommandBuffer)>& record,
    std::span<const TimelinePoint> waits,
    std::span<const TimelinePoint> signals) const {
    ERNIE_ZONE("SubmitAsync");
    std::lock_guard lock(submitMutex_);
    RecyclePending();
    VkCommandBuffer commands = Record(record);
    std::vector<TimelinePoint> allSignals(signals.begin(),
                                          signals.end());
    allSignals.push_back({submissions_, submitted_ + 1});
    const std::vector<VkSemaphoreSubmitInfo> waitInfos =
        SubmitInfos(waits);
    const std::vector<VkSemaphoreSubmitInfo> signalInfos =
        SubmitInfos(allSignals);
    VkCommandBufferSubmitInfo commandInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    commandInfo.commandBuffer = commands;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount =
        static_cast<std::uint32_t>(waitInfos.size());
    submit.pWaitSemaphoreInfos = waitInfos.data();
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &commandInfo;
    submit.signalSemaphoreInfoCount =
        static_cast<std::uint32_t>(signalInfos.size());
    submit.pSignalSemaphoreInfos = signalInfos.data();
    const VkResult result =
        vkQueueSubmit2(queue_, 1, &submit, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        vkFreeCommandBuffers(device_, commandPool_, 1, &commands);
        CheckVk(result, "vkQueueSubmit2");
    }
    pending_.emplace_back(commands, ++submitted_);
    return submitted_;
}

bool VulkanContext::WaitForSubmission(std::uint64_t submission,
                                      std::uint64_t timeoutNs) const {
    ERNIE_ZONE("Wait for submission");
    VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wait.semaphoreCount = 1;
    wait.pSemaphores = &submissions_;
    wait.pValues = &submission;
    return vkWaitSemaphores(device_, &wait, timeoutNs) == VK_SUCCESS;
}

void VulkanContext::RecyclePending() const {
    if (pending_.empty()) {
        return;
    }
    std::uint64_t done = 0;
    vkGetSemaphoreCounterValue(device_, submissions_, &done);
    auto finished = std::partition(
        pending_.begin(), pending_.end(),
        [done](const auto& entry) { return entry.second > done; });
    for (auto it = finished; it != pending_.end(); ++it) {
        vkFreeCommandBuffers(device_, commandPool_, 1, &it->first);
    }
    pending_.erase(finished, pending_.end());
}

void VulkanContext::WaitForSubmissions() const {
    ERNIE_ZONE("Wait for every submission");
    std::lock_guard lock(submitMutex_);
    if (pending_.empty()) {
        return;
    }
    VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wait.semaphoreCount = 1;
    wait.pSemaphores = &submissions_;
    wait.pValues = &submitted_;
    vkWaitSemaphores(device_, &wait, UINT64_MAX);
    RecyclePending();
}

bool VulkanContext::HasPendingSubmissions() const {
    std::lock_guard lock(submitMutex_);
    RecyclePending();
    return !pending_.empty();
}

}  // namespace ernie
