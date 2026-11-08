// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/vulkan/device.h"

#include <algorithm>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include <fmt/format.h>

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/device/slang/compiler.h"
#include "ernie/device/vulkan/interop.h"
#include "ernie/util/disk_cache.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/text.h"

namespace ernie::device::vulkan {

namespace {

std::array<uint8_t, 16> DeviceUuid(VkPhysicalDevice device) {
    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    vkGetPhysicalDeviceProperties2(device, &properties);
    std::array<uint8_t, 16> uuid{};
    std::copy(std::begin(id.deviceUUID), std::end(id.deviceUUID), uuid.begin());
    return uuid;
}

}

std::optional<ComputeFamily> ComputeQueueFamily(VkPhysicalDevice device, VkQueueFlags flags) {
    const std::vector<VkQueueFamilyProperties> families = Enumerate<VkQueueFamilyProperties>(
        [device](uint32_t* count, VkQueueFamilyProperties* items) {
            vkGetPhysicalDeviceQueueFamilyProperties(device, count, items);
        });
    for (uint32_t family = 0; family < uint32_t(families.size()); ++family) {
        if ((families[family].queueFlags & flags) == flags) {
            return ComputeFamily{family, families[family]};
        }
    }
    return std::nullopt;
}

namespace {

#if defined(_WIN32)
constexpr VkTimeDomainKHR kHostTimeDomain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR;

uint64_t HostClockNs(uint64_t ticks) {
    static const uint64_t frequency = [] {
        LARGE_INTEGER value{};
        QueryPerformanceFrequency(&value);
        return uint64_t(value.QuadPart);
    }();
    return ticks / frequency * 1'000'000'000 + ticks % frequency * 1'000'000'000 / frequency;
}
#else
constexpr VkTimeDomainKHR kHostTimeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;

uint64_t HostClockNs(uint64_t ticks) {
    return ticks;
}
#endif

constexpr VkBufferUsageFlags kStorageUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

VkPhysicalDevice FindPhysicalDevice(VkInstance instance, const AdapterInfo& adapter) {
    for (VkPhysicalDevice device : PhysicalDevices(instance)) {
        if (DeviceUuid(device) == adapter.uuid) {
            return device;
        }
    }
    throw Error("the Vulkan adapter {} is no longer present", adapter.name);
}

bool ExportsBuffers(VkPhysicalDevice device) {
    VkPhysicalDeviceExternalBufferInfo info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
    info.usage = kStorageUsage;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkExternalBufferProperties properties{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
    vkGetPhysicalDeviceExternalBufferProperties(device, &info, &properties);
    return (properties.externalMemoryProperties.externalMemoryFeatures &
            VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0;
}

bool ExportsTimelines(VkPhysicalDevice device) {
    VkSemaphoreTypeCreateInfo timeline{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkPhysicalDeviceExternalSemaphoreInfo info{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO, &timeline};
    info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkExternalSemaphoreProperties properties{VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
    vkGetPhysicalDeviceExternalSemaphoreProperties(device, &info, &properties);
    return (properties.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) !=
           0;
}

const char* CalibratedTimestampsExtension(VkPhysicalDevice device, const Extensions& extensions) {
#if defined(__APPLE__)
    return nullptr;
#endif
    const bool khr = extensions.Has(VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
    if (!khr && !extensions.Has(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)) {
        return nullptr;
    }

    const auto list = khr ? vkGetPhysicalDeviceCalibrateableTimeDomainsKHR
                          : vkGetPhysicalDeviceCalibrateableTimeDomainsEXT;
    const std::vector<VkTimeDomainKHR> domains = Enumerate<VkTimeDomainKHR>(
        [&](uint32_t* count, VkTimeDomainKHR* items) { list(device, count, items); });
    const auto has = [&](VkTimeDomainKHR domain) {
        return std::ranges::find(domains, domain) != domains.end();
    };
    if (!has(VK_TIME_DOMAIN_DEVICE_KHR) || !has(kHostTimeDomain)) {
        return nullptr;
    }
    return khr ? VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME
               : VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME;
}

struct DeviceExtensions {
    std::vector<const char*> names;
    const char* calibrated = nullptr;
    bool sharesSemaphores = false;
    bool sharesMemory = false;
    bool capturesStatistics = false;
    bool presentation = false;
};

DeviceExtensions ChooseExtensions(VkPhysicalDevice device, const AdapterInfo& adapter,
                                  const DeviceDesc& desc, bool hardware) {
    DeviceExtensions chosen;
    if (hardware) {
        chosen.names.assign(std::begin(kRayQueryExtensions), std::end(kRayQueryExtensions));
    }
    const Extensions available(device);
    chosen.calibrated = desc.profiling ? CalibratedTimestampsExtension(device, available) : nullptr;
    if (chosen.calibrated != nullptr) {
        chosen.names.push_back(chosen.calibrated);
    }
    if (available.Has("VK_KHR_portability_subset")) {
        chosen.names.push_back("VK_KHR_portability_subset");
    }
#if !defined(_WIN32)
    chosen.sharesSemaphores =
        available.Has(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME) && ExportsTimelines(device);
    if (chosen.sharesSemaphores) {
        chosen.names.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
    }
    chosen.sharesMemory =
        available.Has(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) && ExportsBuffers(device);
    if (chosen.sharesMemory) {
        chosen.names.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    }
#endif
    chosen.presentation = desc.presentation;
    if (chosen.presentation) {
        chosen.names.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
    chosen.capturesStatistics = desc.profiling && adapter.features.kernelStatistics;
    if (chosen.capturesStatistics) {
        chosen.names.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    }
    return chosen;
}

VkDevice CreateLogicalDevice(VkPhysicalDevice physicalDevice, uint32_t queueFamily,
                             uint32_t queueCount, const DeviceExtensions& extensions, bool hardware,
                             bool textures, bool blockCompression, bool anisotropy) {
    const std::vector<const char*>& names = extensions.names;
    FeatureChain enabled;
    enabled.Link([&](const char* name) {
        return std::ranges::any_of(
            names, [name](const char* each) { return std::strcmp(each, name) == 0; });
    });
    enabled.executables.pipelineExecutableInfo = extensions.capturesStatistics ? VK_TRUE : VK_FALSE;
    enabled.base.features.shaderInt64 = VK_TRUE;
    enabled.vulkan12.bufferDeviceAddress = VK_TRUE;
    enabled.vulkan12.timelineSemaphore = VK_TRUE;
    enabled.vulkan13.synchronization2 = VK_TRUE;
    enabled.vulkan13.subgroupSizeControl = VK_TRUE;
    enabled.vulkan13.computeFullSubgroups = VK_TRUE;
    enabled.vulkan13.dynamicRendering = extensions.presentation ? VK_TRUE : VK_FALSE;
    enabled.accelerationStructure.accelerationStructure = hardware ? VK_TRUE : VK_FALSE;
    enabled.rayQuery.rayQuery = hardware ? VK_TRUE : VK_FALSE;

    const VkBool32 bindless = textures ? VK_TRUE : VK_FALSE;
    enabled.vulkan12.runtimeDescriptorArray = bindless;
    enabled.vulkan12.descriptorBindingPartiallyBound = bindless;
    enabled.vulkan12.descriptorBindingSampledImageUpdateAfterBind = bindless;
    enabled.vulkan12.descriptorBindingUpdateUnusedWhilePending = bindless;
    enabled.vulkan12.shaderSampledImageArrayNonUniformIndexing = bindless;
    enabled.base.features.textureCompressionBC = textures && blockCompression ? VK_TRUE : VK_FALSE;
    enabled.base.features.samplerAnisotropy = textures && anisotropy ? VK_TRUE : VK_FALSE;

    const float priorities[2] = {queueCount > 1 ? 0.5f : 1.0f, 1.0f};
    VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueFamilyIndex = queueFamily;
    queue.queueCount = queueCount;
    queue.pQueuePriorities = priorities;
    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enabled.base};
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue;
    info.enabledExtensionCount = uint32_t(names.size());
    info.ppEnabledExtensionNames = names.data();
    VkDevice device = VK_NULL_HANDLE;
    Check(vkCreateDevice(physicalDevice, &info, nullptr, &device), "vkCreateDevice");
    return device;
}

VmaAllocator CreateAllocator(VkInstance instance, VkPhysicalDevice physicalDevice,
                             VkDevice device) {
    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo info{};
    info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    info.vulkanApiVersion = VK_API_VERSION_1_3;
    info.physicalDevice = physicalDevice;
    info.device = device;
    info.instance = instance;
    info.pVulkanFunctions = &functions;
    VmaAllocator allocator = VK_NULL_HANDLE;
    Check(vmaCreateAllocator(&info, &allocator), "vmaCreateAllocator");
    return allocator;
}

std::string PipelineCacheKey(const AdapterInfo& adapter) {
    const std::string_view uuid(reinterpret_cast<const char*>(adapter.uuid.data()),
                                adapter.uuid.size());
    return fmt::format("vulkan pipelines {} {} driver {}", adapter.name, Digest(uuid),
                       adapter.driverVersion.ToString());
}

}

VulkanDevice::VulkanDevice(const AdapterInfo& adapter, const DeviceDesc& desc, Traversal traversal)
    : Device(adapter, desc, traversal) {
    GetApi();
    try {
        m_labels = Extensions().Has(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        m_instance = vulkan::Instance(desc.validation);
        m_physicalDevice = FindPhysicalDevice(m_instance, adapter);
        const VkQueueFlags flags =
            VK_QUEUE_COMPUTE_BIT | (desc.presentation ? VK_QUEUE_GRAPHICS_BIT : 0);
        const std::optional<ComputeFamily> family = ComputeQueueFamily(m_physicalDevice, flags);
        if (!family) {
            throw Error("the Vulkan adapter {} has no {} queue", adapter.name,
                        desc.presentation ? "graphics and compute" : "compute");
        }

        m_queueFamily = family->index;
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(m_physicalDevice, &properties);
        m_vendorId = properties.vendorID;
        m_timestampPeriod = properties.limits.timestampPeriod;
        const uint32_t bits = family->properties.timestampValidBits;
        m_timestampMask = bits >= 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1;

        const bool hardware = traversal == Traversal::Hardware;
        const DeviceExtensions extensions =
            ChooseExtensions(m_physicalDevice, adapter, desc, hardware);
        m_sharesSemaphores = extensions.sharesSemaphores;
        m_sharesMemory = extensions.sharesMemory;
        m_capturesStatistics = extensions.capturesStatistics;

        const bool textures = GetSampling() == Sampling::Hardware;
        const uint32_t queueCount = desc.presentation && family->properties.queueCount > 1 ? 2 : 1;
        m_device = CreateLogicalDevice(m_physicalDevice, m_queueFamily, queueCount, extensions,
                                       hardware, textures, adapter.features.blockCompression,
                                       adapter.features.anisotropicFiltering);
        LoadDeviceFunctions(m_functions, m_device);
        if (extensions.calibrated != nullptr) {
            m_calibratedTimestamps = std::string_view(extensions.calibrated).starts_with("VK_KHR")
                                         ? m_functions.vkGetCalibratedTimestampsKHR
                                         : m_functions.vkGetCalibratedTimestampsEXT;
        }
        m_functions.vkGetDeviceQueue(m_device, m_queueFamily, 0, &m_queueHandle);
        m_allocator = CreateAllocator(m_instance, m_physicalDevice, m_device);

        const std::optional<std::string> saved =
            DiskCache(KernelCacheDirectory() / "vulkan").Load(PipelineCacheKey(Adapter()));
        VkPipelineCacheCreateInfo pipelines{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        pipelines.initialDataSize = saved ? saved->size() : 0;
        pipelines.pInitialData = saved ? saved->data() : nullptr;
        Check(m_functions.vkCreatePipelineCache(m_device, &pipelines, nullptr, &m_pipelineCache),
              "vkCreatePipelineCache");

        CreateDescriptors(textures);

        Log(LogLevel::Debug, "created a Vulkan device on queue family {}{}{}", m_queueFamily,
            extensions.names.empty() ? "" : ", with ",
            Join({extensions.names.begin(), extensions.names.end()}));
        m_queue = CreateQueue(*this, m_queueHandle, true);
        if (queueCount > 1) {
            VkQueue display = VK_NULL_HANDLE;
            m_functions.vkGetDeviceQueue(m_device, m_queueFamily, 1, &display);
            m_displayQueue = CreateQueue(*this, display, false);
        }
    } catch (...) {
        Destroy();
        throw;
    }
}

void VulkanDevice::CreateDescriptors(bool textures) {
    const uint32_t textureSlots = textures ? Adapter().limits.maxTextures : 0;
    if (textures) {
        CreateSamplers();
    }

    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = kTextureBinding;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[1].descriptorCount = textureSlots;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = kSamplerBinding;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[2].descriptorCount = uint32_t(m_samplers.size());
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].pImmutableSamplers = m_samplers.data();
    const VkDescriptorBindingFlags flags[3] = {
        0,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
        0};
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bindingFlags.bindingCount = 3;
    bindingFlags.pBindingFlags = flags;
    VkDescriptorSetLayoutCreateInfo set{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set.bindingCount = textures ? 3 : 1;
    set.pBindings = bindings;
    if (textures) {
        set.pNext = &bindingFlags;
        set.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    }
    Check(m_functions.vkCreateDescriptorSetLayout(m_device, &set, nullptr, &m_globalsLayout),
          "vkCreateDescriptorSetLayout");

    const VkPushConstantRange params{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                     Adapter().limits.maxParamBytes};
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &m_globalsLayout;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &params;
    Check(m_functions.vkCreatePipelineLayout(m_device, &layout, nullptr, &m_pipelineLayout),
          "vkCreatePipelineLayout");

    const VkDescriptorPoolSize sizes[3] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
                                           {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, textureSlots},
                                           {VK_DESCRIPTOR_TYPE_SAMPLER, kTextureSamplers}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = 1;
    pool.poolSizeCount = textures ? 3 : 1;
    pool.pPoolSizes = sizes;
    if (textures) {
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    }
    Check(m_functions.vkCreateDescriptorPool(m_device, &pool, nullptr, &m_descriptorPool),
          "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = m_descriptorPool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &m_globalsLayout;
    Check(m_functions.vkAllocateDescriptorSets(m_device, &allocate, &m_globals),
          "vkAllocateDescriptorSets");
}

void VulkanDevice::Name(VkObjectType type, uint64_t object, const std::string& name) const {
    if (!m_labels || name.empty()) {
        return;
    }
    VkDebugUtilsObjectNameInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    info.objectType = type;
    info.objectHandle = object;
    info.pObjectName = name.c_str();
    Check(vkSetDebugUtilsObjectNameEXT(m_device, &info), "vkSetDebugUtilsObjectNameEXT");
}

bool VulkanDevice::ToNanoseconds(std::span<uint64_t> timestamps) const {
    const auto calibrated = CalibratedTimestamps();
    for (uint64_t& time : timestamps) {
        const auto ticks = int64_t(time & m_timestampMask);
        time = calibrated
                   ? calibrated->second +
                         uint64_t(double(ticks - int64_t(calibrated->first & m_timestampMask)) *
                                  m_timestampPeriod)
                   : uint64_t(double(ticks) * m_timestampPeriod);
    }
    return calibrated.has_value();
}

std::optional<std::pair<uint64_t, uint64_t>> VulkanDevice::CalibratedTimestamps() const {
    if (m_calibratedTimestamps == nullptr) {
        return std::nullopt;
    }
    const VkCalibratedTimestampInfoKHR domains[2] = {
        {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, nullptr, VK_TIME_DOMAIN_DEVICE_KHR},
        {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, nullptr, kHostTimeDomain}};
    uint64_t times[2] = {};
    uint64_t deviation = 0;
    Check(m_calibratedTimestamps(m_device, 2, domains, times, &deviation),
          "vkGetCalibratedTimestampsKHR");
    return std::pair{times[0], HostClockNs(times[1])};
}

VulkanDevice::~VulkanDevice() {
    Destroy();
}

void VulkanDevice::Destroy() {
    if (m_displayQueue) {
        CloseQueue(*m_displayQueue);
        m_displayQueue.reset();
    }
    if (m_device != VK_NULL_HANDLE) {
        m_functions.vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        m_functions.vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        m_functions.vkDestroyDescriptorSetLayout(m_device, m_globalsLayout, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
        m_pipelineLayout = VK_NULL_HANDLE;
        m_globalsLayout = VK_NULL_HANDLE;
        for (VkSampler& sampler : m_samplers) {
            m_functions.vkDestroySampler(m_device, sampler, nullptr);
            sampler = VK_NULL_HANDLE;
        }
    }
    if (m_pipelineCache != VK_NULL_HANDLE) {
        const std::vector<char> data = Enumerate<char, size_t>([this](size_t* bytes, char* items) {
            m_functions.vkGetPipelineCacheData(m_device, m_pipelineCache, bytes, items);
        });
        if (!data.empty()) {
            DiskCache(KernelCacheDirectory() / "vulkan")
                .Store(PipelineCacheKey(Adapter()), {data.data(), data.size()});
        }
        m_functions.vkDestroyPipelineCache(m_device, m_pipelineCache, nullptr);
        m_pipelineCache = VK_NULL_HANDLE;
    }
    ReleaseSpares();
    m_memoryInterop.reset();
    if (m_interopPool != VK_NULL_HANDLE) {
        vmaDestroyPool(m_allocator, m_interopPool);
        m_interopPool = VK_NULL_HANDLE;
    }
    if (m_allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(m_allocator);
        m_allocator = VK_NULL_HANDLE;
    }
    if (m_device != VK_NULL_HANDLE) {
        m_functions.vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
    }
}

VkBufferUsageFlags VulkanDevice::UsageFlags(BufferUsage usage) const {
    // NOLINTNEXTLINE(bugprone-signed-bitwise)
    VkBufferUsageFlags flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (Has(usage, BufferUsage::Indirect)) {
        flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    }
    if (GetTraversal() == Traversal::Hardware) {
        if (Has(usage, BufferUsage::AccelBuildInput)) {
            flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        }
        if (Has(usage, BufferUsage::AccelStorage)) {
            flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
        }
    }
    return flags;
}

namespace {

VkAccelerationStructureTypeKHR VulkanType(AccelLevel level) {
    return level == AccelLevel::Top ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
                                    : VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
}

}

void VulkanBuild::Describe(const AccelInput& input) {
    info.type = VulkanType(GetLevel(input));
    info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    if (Refittable(input)) {
        info.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    }
    info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.geometryCount = 1;
    info.pGeometries = &geometry;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    std::visit(Overloaded{
                   [&](const AccelTriangles& t) {
                       geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                       VkAccelerationStructureGeometryTrianglesDataKHR& triangles =
                           geometry.geometry.triangles;
                       triangles.sType =
                           VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                       triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
                       triangles.vertexData.deviceAddress = t.vertices.Address();
                       triangles.vertexStride = t.stride;
                       triangles.maxVertex = t.vertexCount > 0 ? t.vertexCount - 1 : 0;
                       triangles.indexType =
                           t.indices ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_NONE_KHR;
                       triangles.indexData.deviceAddress = t.indices ? t.indices->Address() : 0;
                       primitives = t.count;
                   },
                   [&](const AccelBoxes& b) {
                       geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
                       geometry.geometry.aabbs.sType =
                           VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
                       geometry.geometry.aabbs.data.deviceAddress = b.boxes.Address();
                       geometry.geometry.aabbs.stride = 6 * sizeof(float);
                       primitives = b.count;
                   },
                   [&](const AccelInstances& i) {
                       geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
                       geometry.geometry.instances.sType =
                           VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
                       geometry.geometry.instances.arrayOfPointers = VK_FALSE;
                       geometry.geometry.instances.data.deviceAddress = i.instances.Address();
                       primitives = i.count;
                   },
               },
               input);
}

AccelSizes VulkanDevice::HardwareBuildSizes(const AccelInput& input) const {
    VulkanBuild build;
    build.Describe(input);
    VkAccelerationStructureBuildSizesInfoKHR sizes{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    m_functions.vkGetAccelerationStructureBuildSizesKHR(
        m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build.info, &build.primitives,
        &sizes);
    return {.storage = sizes.accelerationStructureSize,
            .scratch = sizes.buildScratchSize,
            .updateScratch = Refittable(input) ? sizes.updateScratchSize : 0};
}

Ref<Accel> VulkanDevice::NewAccel(Ref<Buffer> storage, AccelLevel level, AccelSizes sizes,
                                  std::string label) {
    return Ref<Accel>(new VulkanAccel(*this, std::move(storage), level, sizes, std::move(label)));
}

VulkanAccel::VulkanAccel(VulkanDevice& owner, Ref<Buffer> storage, AccelLevel level,
                         AccelSizes sizes, std::string label)
    : Accel(owner, std::move(storage), level, sizes, std::move(label)), m_owner(owner) {
    VkAccelerationStructureCreateInfoKHR info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    info.buffer = static_cast<VulkanBuffer&>(Storage()).Handle();
    info.size = Storage().Bytes();
    info.type = VulkanType(level);
    const VolkDeviceTable& vk = owner.Functions();
    Check(vk.vkCreateAccelerationStructureKHR(owner.Handle(), &info, nullptr, &m_handle),
          "vkCreateAccelerationStructureKHR");
    VkAccelerationStructureDeviceAddressInfoKHR address{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    address.accelerationStructure = m_handle;
    m_address = vk.vkGetAccelerationStructureDeviceAddressKHR(owner.Handle(), &address);
    owner.Name(VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR, uint64_t(m_handle), Label());
}

VulkanAccel::~VulkanAccel() {
    m_owner.Functions().vkDestroyAccelerationStructureKHR(m_owner.Handle(), m_handle, nullptr);
}

uint64_t VulkanDevice::BufferAddress(VkBuffer buffer) const {
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buffer;
    return m_functions.vkGetBufferDeviceAddress(m_device, &info);
}

VmaPool VulkanDevice::InteropPool() {
    const std::lock_guard lock(m_interopMutex);
    if (m_interopPool != VK_NULL_HANDLE) {
        return m_interopPool;
    }

    VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkBufferCreateInfo sample{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
    sample.size = 65536;
    sample.usage = UsageFlags(BufferUsage::None);
    VmaAllocationCreateInfo wanted{};
    wanted.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    uint32_t type = 0;
    Check(vmaFindMemoryTypeIndexForBufferInfo(m_allocator, &sample, &wanted, &type),
          "vmaFindMemoryTypeIndexForBufferInfo");

    m_exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VmaPoolCreateInfo pool{};
    pool.memoryTypeIndex = type;
    pool.pMemoryAllocateNext = &m_exportInfo;
    Check(vmaCreatePool(m_allocator, &pool, &m_interopPool), "vmaCreatePool");
    return m_interopPool;
}

MemoryInterop* VulkanDevice::GetMemoryInterop() {
    const std::lock_guard lock(m_interopMutex);
    if (!m_interopTried) {
        m_interopTried = true;
        if (m_sharesMemory) {
            try {
                m_memoryInterop = CreateMemoryInterop(*this);
            } catch (const UnsupportedError& error) {
                Log(LogLevel::Debug, "{} shares no memory: {}", Adapter().alias, error.what());
            }
        }
    }
    return m_memoryInterop.get();
}

std::optional<DLDevice> VulkanDevice::SharedDevice() {
    const MemoryInterop* interop = GetMemoryInterop();
    return interop != nullptr ? std::optional(interop->Device()) : std::nullopt;
}

Stream VulkanDevice::CopyStream() {
    MemoryInterop* interop = GetMemoryInterop();
    return interop != nullptr ? Stream::Compute(interop->Stream()) : Stream::None();
}

void VulkanDevice::ExternalCopy(BufferRange dst, uint64_t source, Stream producer) {
    MemoryInterop& interop = *GetMemoryInterop();
    const ExternalAddress external = dst.GetBuffer().External(interop.Device().device_type);
    interop.Copy(external.address + dst.Offset(), source, dst.Bytes(),
                 producer.IsCompute() ? producer.Handle() : nullptr);
}

VulkanBuffer::~VulkanBuffer() {
    if (m_external != nullptr && m_owner.KeepSpare({m_buffer, m_allocation, m_external,
                                                    m_externalAddress, Bytes(), Usage()})) {
        return;
    }
    if (m_external != nullptr) {
        ImportedMemory imported{m_external, m_externalAddress};
        m_owner.GetMemoryInterop()->Release(imported);
    }
    vmaDestroyBuffer(m_owner.Allocator(), m_buffer, m_allocation);
}

namespace {

ExportedMemory ExportMemory(const VulkanDevice& owner, VkBuffer buffer, VmaAllocation allocation,
                            uint64_t bytes) {
    VmaAllocationInfo info{};
    vmaGetAllocationInfo(owner.Allocator(), allocation, &info);
    VkMemoryGetFdInfoKHR get{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    get.memory = info.deviceMemory;
    get.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    ExportedMemory exported;
    Check(owner.Functions().vkGetMemoryFdKHR(owner.Handle(), &get, &exported.fd),
          "vkGetMemoryFdKHR");

    VkMemoryRequirements requirements{};
    owner.Functions().vkGetBufferMemoryRequirements(owner.Handle(), buffer, &requirements);
    exported.memoryBytes = std::max<uint64_t>(info.size, requirements.size);
    exported.offset = info.offset;
    exported.bytes = bytes;
    return exported;
}

}

ExternalAddress VulkanBuffer::External(DLDeviceType api) {
    MemoryInterop* interop = m_owner.GetMemoryInterop();
    if (interop == nullptr || interop->Device().device_type != api ||
        !Has(Usage(), BufferUsage::Interop)) {
        throw UnsupportedError("Buffer \"{}\" on {} isn't shared with device type {}", Label(),
                               m_owner.Adapter().alias, int(api));
    }

    const std::lock_guard lock(m_externalMutex);
    if (m_external == nullptr) {
        const ImportedMemory imported =
            interop->Import(ExportMemory(m_owner, m_buffer, m_allocation, Bytes()));
        m_external = imported.external;
        m_externalAddress = imported.address;
    }
    return {interop->Device(), m_externalAddress};
}

namespace {

VmaAllocationCreateInfo MemoryAllocation(Memory memory) {
    constexpr VkMemoryPropertyFlags kCoherent =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VmaAllocationCreateInfo allocation{};
    switch (memory) {
    case Memory::Device:
        allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        break;
    case Memory::Upload:
        allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        allocation.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                           VMA_ALLOCATION_CREATE_MAPPED_BIT;
        allocation.requiredFlags = kCoherent;
        break;
    case Memory::Readback:
        allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        allocation.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        allocation.requiredFlags = kCoherent;
        allocation.preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        break;
    case Memory::Shared:
        allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        allocation.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                           VMA_ALLOCATION_CREATE_MAPPED_BIT;
        allocation.requiredFlags = kCoherent | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case Memory::Virtual:
        break;
    }
    return allocation;
}

}

bool VulkanDevice::KeepSpare(const SpareBuffer& spare) {
    constexpr size_t kMostSpares = 32;
    const std::lock_guard lock(m_spareMutex);
    if (m_sparesClosed || m_spares.size() >= kMostSpares) {
        return false;
    }
    m_spares.push_back(spare);
    return true;
}

std::optional<VulkanDevice::SpareBuffer> VulkanDevice::TakeSpare(uint64_t bytes,
                                                                 BufferUsage usage) {
    const std::lock_guard lock(m_spareMutex);
    const auto found = std::ranges::find_if(m_spares, [&](const SpareBuffer& spare) {
        return spare.bytes == bytes && spare.usage == usage;
    });
    if (found == m_spares.end()) {
        return std::nullopt;
    }
    const SpareBuffer spare = *found;
    m_spares.erase(found);
    return spare;
}

void VulkanDevice::ReleaseSpares() {
    std::vector<SpareBuffer> spares;
    {
        const std::lock_guard lock(m_spareMutex);
        m_sparesClosed = true;
        spares.swap(m_spares);
    }
    for (const SpareBuffer& spare : spares) {
        ImportedMemory imported{spare.external, spare.externalAddress};
        m_memoryInterop->Release(imported);
        vmaDestroyBuffer(m_allocator, spare.buffer, spare.allocation);
    }
}

std::vector<MemoryHeap> VulkanDevice::Heaps() const {
    const VkPhysicalDeviceMemoryProperties* properties = nullptr;
    vmaGetMemoryProperties(m_allocator, &properties);
    std::vector<VmaBudget> budgets(properties->memoryHeapCount);
    vmaGetHeapBudgets(m_allocator, budgets.data());
    std::vector<MemoryHeap> heaps;
    for (uint32_t i = 0; i < properties->memoryHeapCount; ++i) {
        const VkMemoryHeap& heap = properties->memoryHeaps[i];
        heaps.push_back({.usage = budgets[i].usage,
                         .budget = budgets[i].budget,
                         .size = heap.size,
                         .deviceLocal = (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0});
    }
    return heaps;
}

Ref<Buffer> VulkanDevice::NewBuffer(const BufferDesc& desc) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = desc.bytes;
    info.usage = UsageFlags(desc.usage);
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    const bool interop = Has(desc.usage, BufferUsage::Interop) && GetMemoryInterop() != nullptr;
    if (interop) {
        if (desc.memory != Memory::Device) {
            throw UnsupportedError("Buffer \"{}\" asks for interop, which only device memory has",
                                   desc.label);
        }
        external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        info.pNext = &external;
    }

    if (interop) {
        if (const std::optional<SpareBuffer> spare = TakeSpare(desc.bytes, desc.usage)) {
            vmaSetAllocationName(m_allocator, spare->allocation, desc.label.c_str());
            Name(VK_OBJECT_TYPE_BUFFER, uint64_t(spare->buffer), desc.label);
            return Ref<Buffer>(new VulkanBuffer(*this, desc, spare->buffer, spare->allocation,
                                                nullptr, spare->external, spare->externalAddress));
        }
    }

    VmaAllocationCreateInfo allocation = MemoryAllocation(desc.memory);
    if (interop) {
        allocation.pool = InteropPool();
        allocation.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocated = VK_NULL_HANDLE;
    VmaAllocationInfo allocatedInfo{};
    Check(vmaCreateBufferWithAlignment(m_allocator, &info, &allocation, kBufferAlignment, &buffer,
                                       &allocated, &allocatedInfo),
          "vmaCreateBufferWithAlignment");
    vmaSetAllocationName(m_allocator, allocated, desc.label.c_str());
    Name(VK_OBJECT_TYPE_BUFFER, uint64_t(buffer), desc.label);
    return Ref<Buffer>(new VulkanBuffer(*this, desc, buffer, allocated, allocatedInfo.pMappedData));
}

namespace {

std::string Value(const VkPipelineExecutableStatisticKHR& statistic) {
    switch (statistic.format) {
    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
        return statistic.value.b32 == VK_TRUE ? "true" : "false";
    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
        return std::to_string(statistic.value.i64);
    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
        return std::to_string(statistic.value.u64);
    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
        return fmt::format("{}", statistic.value.f64);
    default:
        return "";
    }
}

struct KnownStatistic {
    const char* name;
    std::optional<uint32_t> KernelStats::* field;
};

constexpr KnownStatistic kKnownStatistics[] = {
    {"Register Count", &KernelStats::registers},
    {"VGPRs", &KernelStats::registers},
    {"Scratch Memory Size", &KernelStats::spillBytes},
    {"Shared Memory Size", &KernelStats::sharedBytes},
    {"Workgroup Memory Size", &KernelStats::sharedBytes},
};

VkShaderModule CreateShaderModule(const VolkDeviceTable& functions, VkDevice device,
                                  const KernelCode& code) {
    std::vector<uint32_t> words(code.code.size() / 4);
    std::memcpy(words.data(), code.code.data(), words.size() * 4);
    VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader.codeSize = words.size() * 4;
    shader.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    Check(functions.vkCreateShaderModule(device, &shader, nullptr, &module),
          "vkCreateShaderModule");
    return module;
}

KernelStats ReadStatistics(const VolkDeviceTable& functions, VkDevice device, VkPipeline pipeline) {
    const VkPipelineExecutableInfoKHR executable{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
                                                 nullptr, pipeline, 0};
    uint32_t count = 0;
    Check(functions.vkGetPipelineExecutableStatisticsKHR(device, &executable, &count, nullptr),
          "vkGetPipelineExecutableStatisticsKHR");
    std::vector<VkPipelineExecutableStatisticKHR> statistics(
        count, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
    Check(functions.vkGetPipelineExecutableStatisticsKHR(device, &executable, &count,
                                                         statistics.data()),
          "vkGetPipelineExecutableStatisticsKHR");

    KernelStats stats{.cached = true};
    for (const VkPipelineExecutableStatisticKHR& statistic : statistics) {
        stats.driver.push_back({statistic.name, Value(statistic)});
        for (const KnownStatistic& known : kKnownStatistics) {
            if (std::string_view(statistic.name) == known.name &&
                statistic.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR) {
                stats.*known.field = uint32_t(statistic.value.u64);
            }
        }
    }
    return stats;
}

}

KernelStats VulkanKernel::Load(const KernelCode& code, const DiskCache&, const DeviceGlobals&) {
    const VolkDeviceTable& functions = m_owner.Functions();
    const VkDevice device = m_owner.Handle();
    m_owner.Globals();
    const VkShaderModule module = CreateShaderModule(functions, device, code);

    VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline.stage.module = module;
    pipeline.stage.pName = Name().c_str();
    pipeline.layout = m_owner.PipelineLayout();
    if (code.shape.groupSize.x % m_owner.Adapter().features.waveSizeMax == 0) {
        pipeline.stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    }
    if (m_owner.CapturesStatistics()) {
        pipeline.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    }
    const VkResult created = functions.vkCreateComputePipelines(device, m_owner.PipelineCache(), 1,
                                                                &pipeline, nullptr, &m_pipeline);
    functions.vkDestroyShaderModule(device, module, nullptr);
    Check(created, "vkCreateComputePipelines");
    m_owner.Name(VK_OBJECT_TYPE_PIPELINE, uint64_t(m_pipeline),
                 fmt::format("{} of program \"{}\"", Name(), GetProgram().Label()));

    if (!m_owner.CapturesStatistics()) {
        return {.cached = true};
    }
    return ReadStatistics(functions, device, m_pipeline);
}

VulkanKernel::~VulkanKernel() {
    m_owner.Functions().vkDestroyPipeline(m_owner.Handle(), m_pipeline, nullptr);
}

VkDescriptorSet VulkanDevice::Globals() {
    std::call_once(m_globalsWritten, [this] {
        VkDescriptorBufferInfo buffer{};
        buffer.buffer = static_cast<const VulkanBuffer&>(GlobalsBuffer()).Handle();
        buffer.range = sizeof(DeviceGlobals);
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = m_globals;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.pBufferInfo = &buffer;
        m_functions.vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
    });
    return m_globals;
}

namespace {

VkSamplerAddressMode VulkanAddressMode(uint32_t wrap) {
    switch (TextureWrap(wrap)) {
    case TextureWrap::Repeat:
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case TextureWrap::Mirror:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case TextureWrap::Clamp:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    default:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    }
}

VkFormat VulkanFormat(const TextureDesc& desc) {
    const bool encoded = desc.colorSpace == TexelColorSpace::Srgb;
    const auto pick = [&](VkFormat linear, VkFormat srgb) {
        if (encoded && srgb == VK_FORMAT_UNDEFINED) {
            throw Error("texture \"{}\" is sRGB in a format with no sRGB variant", desc.label);
        }
        return encoded ? srgb : linear;
    };
    switch (desc.format) {
    case TexelFormat::R8:
        return pick(VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SRGB);
    case TexelFormat::Rg8:
        return pick(VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SRGB);
    case TexelFormat::Rgba8:
        return pick(VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB);
    case TexelFormat::R16:
        return pick(VK_FORMAT_R16_UNORM, VK_FORMAT_UNDEFINED);
    case TexelFormat::Rg16:
        return pick(VK_FORMAT_R16G16_UNORM, VK_FORMAT_UNDEFINED);
    case TexelFormat::Rgba16:
        return pick(VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_UNDEFINED);
    case TexelFormat::R16f:
        return pick(VK_FORMAT_R16_SFLOAT, VK_FORMAT_UNDEFINED);
    case TexelFormat::Rg16f:
        return pick(VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_UNDEFINED);
    case TexelFormat::Rgba16f:
        return pick(VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_UNDEFINED);
    case TexelFormat::R32f:
        return pick(VK_FORMAT_R32_SFLOAT, VK_FORMAT_UNDEFINED);
    case TexelFormat::Rg32f:
        return pick(VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_UNDEFINED);
    case TexelFormat::Rgba32f:
        return pick(VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_UNDEFINED);
    case TexelFormat::Bc1:
        return pick(VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_BC1_RGBA_SRGB_BLOCK);
    case TexelFormat::Bc3:
        return pick(VK_FORMAT_BC3_UNORM_BLOCK, VK_FORMAT_BC3_SRGB_BLOCK);
    case TexelFormat::Bc4:
        return pick(VK_FORMAT_BC4_UNORM_BLOCK, VK_FORMAT_UNDEFINED);
    case TexelFormat::Bc5:
        return pick(VK_FORMAT_BC5_UNORM_BLOCK, VK_FORMAT_UNDEFINED);
    }
    return VK_FORMAT_UNDEFINED;
}

}

void VulkanDevice::CreateSamplers() {
    const bool anisotropic = Adapter().features.anisotropicFiltering;
    for (uint32_t wrap = 0; wrap < m_samplers.size(); ++wrap) {
        VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        info.anisotropyEnable = anisotropic ? VK_TRUE : VK_FALSE;
        info.maxAnisotropy = anisotropic ? std::min(16.0f, Adapter().limits.maxAnisotropy) : 1.0f;
        info.addressModeU = VulkanAddressMode(wrap >> kTextureWrapBits);
        info.addressModeV = VulkanAddressMode(wrap & ((1u << kTextureWrapBits) - 1));
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.maxLod = VK_LOD_CLAMP_NONE;
        info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        Check(m_functions.vkCreateSampler(m_device, &info, nullptr, &m_samplers[wrap]),
              "vkCreateSampler");
    }
}

uint32_t VulkanDevice::AcquireTextureSlot(VkImageView view) {
    const std::lock_guard lock(m_textureMutex);
    uint32_t slot = 0;
    if (!m_freeTextureSlots.empty()) {
        slot = m_freeTextureSlots.back();
        m_freeTextureSlots.pop_back();
    } else if (m_textureSlots < Adapter().limits.maxTextures) {
        slot = m_textureSlots++;
    } else {
        throw Error("{} holds at most {} textures at once", Adapter().alias,
                    Adapter().limits.maxTextures);
    }

    VkDescriptorImageInfo image{};
    image.imageView = view;
    image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = Globals();
    write.dstBinding = kTextureBinding;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &image;
    m_functions.vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
    return slot;
}

void VulkanDevice::ReleaseTextureSlot(uint32_t slot) {
    const std::lock_guard lock(m_textureMutex);
    m_freeTextureSlots.push_back(slot);
}

Ref<Texture> VulkanDevice::NewTexture(const TextureDesc& desc) {
    return Ref<Texture>(new VulkanTexture(*this, desc));
}

VulkanTexture::VulkanTexture(VulkanDevice& owner, const TextureDesc& desc)
    : Texture(owner, desc), m_owner(owner) {
    const VkFormat format = VulkanFormat(desc);
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(owner.PhysicalDevice(), format, &properties);
    constexpr VkFormatFeatureFlags kNeeded = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                             VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                             VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if ((properties.optimalTilingFeatures & kNeeded) != kNeeded) {
        throw UnsupportedError("{} can't filter the format of texture \"{}\"",
                               owner.Adapter().alias, desc.label);
    }

    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {desc.width, desc.height, 1};
    info.mipLevels = desc.levels;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo allocation{};
    allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    Check(vmaCreateImage(owner.Allocator(), &info, &allocation, &m_image, &m_allocation, nullptr),
          "vmaCreateImage");

    try {
        vmaSetAllocationName(owner.Allocator(), m_allocation, desc.label.c_str());
        owner.Name(VK_OBJECT_TYPE_IMAGE, uint64_t(m_image), desc.label);
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = m_image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = format;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, desc.levels, 0, 1};
        Check(owner.Functions().vkCreateImageView(owner.Handle(), &view, nullptr, &m_view),
              "vkCreateImageView");
        m_slot = owner.AcquireTextureSlot(m_view);
    } catch (...) {
        Destroy();
        throw;
    }
    m_handle = TextureHandle(*m_slot, desc.wrapS, desc.wrapT);
}

VulkanTexture::~VulkanTexture() {
    Destroy();
}

void VulkanTexture::Destroy() noexcept {
    if (m_slot) {
        m_owner.ReleaseTextureSlot(*m_slot);
        m_slot.reset();
    }
    m_owner.Functions().vkDestroyImageView(m_owner.Handle(), m_view, nullptr);
    vmaDestroyImage(m_owner.Allocator(), m_image, m_allocation);
    m_view = VK_NULL_HANDLE;
    m_image = VK_NULL_HANDLE;
    m_allocation = VK_NULL_HANDLE;
}

std::unique_ptr<Kernel> VulkanDevice::NewKernel() {
    return std::make_unique<VulkanKernel>(*this);
}

}
