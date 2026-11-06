// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include <volk.h>

#include <vk_mem_alloc.h>

#include "ernie/device/accel.h"
#include "ernie/device/device.h"
#include "ernie/device/operation.h"
#include "ernie/device/program.h"
#include "ernie/device/texture.h"
#include "ernie/device/vulkan/api.h"

namespace ernie::device::vulkan {

class MemoryInterop;

inline constexpr const char* kRayQueryExtensions[] = {
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};

// Bindless texture slots; adapters with fewer sampled images after binding offer fewer.
inline constexpr uint32_t kMaxTextures = 65536;

struct FeatureChain {
    VkPhysicalDeviceFeatures2 base{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan12Features vulkan12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features vulkan13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructure{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR rayQuery{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    VkPhysicalDeviceShaderClockFeaturesKHR clock{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR executables{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};

    template <class Has> void Link(Has has) {
        base.pNext = &vulkan12;
        vulkan12.pNext = &vulkan13;
        void** tail = &vulkan13.pNext;
        const auto chain = [&tail](auto& features, bool present) {
            if (present) {
                *tail = &features;
                tail = &features.pNext;
            }
        };
        const bool rayQueries = std::ranges::all_of(kRayQueryExtensions, has);
        chain(accelerationStructure, rayQueries);
        chain(rayQuery, rayQueries);
        chain(atomicFloat, has(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME));
        chain(clock, has(VK_KHR_SHADER_CLOCK_EXTENSION_NAME));
        chain(executables, has(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME));
        *tail = nullptr;
    }
};

struct ComputeFamily {
    uint32_t index;
    VkQueueFamilyProperties properties;
};

std::optional<ComputeFamily> ComputeQueueFamily(VkPhysicalDevice device,
                                                VkQueueFlags flags = VK_QUEUE_COMPUTE_BIT);

std::vector<AdapterInfo> ListAdapters();

class VulkanDevice : public Device {
  public:
    VulkanDevice(const AdapterInfo& adapter, const DeviceDesc& desc, Traversal traversal);
    ~VulkanDevice() override;

    VkInstance InstanceHandle() const {
        return m_instance;
    }

    VkDevice Handle() const {
        return m_device;
    }

    VkPhysicalDevice PhysicalDevice() const {
        return m_physicalDevice;
    }

    VkQueue QueueHandle() const {
        return m_queueHandle;
    }

    uint32_t QueueFamily() const {
        return m_queueFamily;
    }

    Queue& DisplayQueue() {
        return m_displayQueue ? *m_displayQueue : GetQueue();
    }

    bool SharesDisplayQueue() const {
        return !m_displayQueue;
    }

    const VolkDeviceTable& Functions() const {
        return m_functions;
    }

    VkPipelineLayout PipelineLayout() const {
        return m_pipelineLayout;
    }

    VkPipelineCache PipelineCache() const {
        return m_pipelineCache;
    }

    VmaAllocator Allocator() const {
        return m_allocator;
    }

    bool SharesSemaphores() const {
        return m_sharesSemaphores;
    }

    bool SharesExportedMemory() const {
        return m_sharesMemory;
    }

    uint32_t VendorId() const {
        return m_vendorId;
    }

    bool HasTimestamps() const {
        return m_timestampMask != 0;
    }

    bool CapturesStatistics() const {
        return m_capturesStatistics;
    }

    bool Labels() const {
        return m_labels;
    }

    void Name(VkObjectType type, uint64_t object, const std::string& name) const;
    bool ToNanoseconds(std::span<uint64_t> timestamps) const;
    std::optional<std::pair<uint64_t, uint64_t>> CalibratedTimestamps() const;

    struct SpareBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        void* external = nullptr;
        uint64_t externalAddress = 0;
        uint64_t bytes = 0;
        BufferUsage usage = BufferUsage::None;
    };

    uint64_t BufferAddress(VkBuffer buffer) const;
    MemoryInterop* GetMemoryInterop();
    bool KeepSpare(const SpareBuffer& spare);
    std::optional<DLDevice> SharedDevice() override;

    VkDescriptorSet Globals();
    uint32_t AcquireTextureSlot(VkImageView view);
    void ReleaseTextureSlot(uint32_t slot);

  protected:
    AccelSizes HardwareBuildSizes(const AccelInput& input) const override;
    Ref<Accel> NewAccel(Ref<Buffer> storage, AccelLevel level, AccelSizes sizes,
                        std::string label) override;
    void ExternalCopy(BufferRange dst, uint64_t source, Stream producer) override;
    Stream CopyStream() override;
    Ref<Buffer> NewBuffer(const BufferDesc& desc) override;
    std::vector<MemoryHeap> Heaps() const override;
    Ref<Texture> NewTexture(const TextureDesc& desc) override;
    std::unique_ptr<Kernel> NewKernel() override;

  private:
    void CreateDescriptors(bool textures);
    void Destroy();
    VkBufferUsageFlags UsageFlags(BufferUsage usage) const;
    VmaPool InteropPool();
    std::optional<SpareBuffer> TakeSpare(uint64_t bytes, BufferUsage usage);
    void ReleaseSpares();
    void CreateSamplers();

    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queueHandle = VK_NULL_HANDLE;
    uint32_t m_queueFamily = 0;
    std::unique_ptr<Queue> m_displayQueue;
    VolkDeviceTable m_functions{};
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkPipelineCache m_pipelineCache = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_globalsLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    PFN_vkGetCalibratedTimestampsKHR m_calibratedTimestamps = nullptr;
    double m_timestampPeriod = 0;
    uint64_t m_timestampMask = 0;
    bool m_capturesStatistics = false;
    bool m_labels = false;
    bool m_sharesSemaphores = false;
    bool m_sharesMemory = false;
    VkExportMemoryAllocateInfo m_exportInfo{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    VmaPool m_interopPool = VK_NULL_HANDLE;
    std::mutex m_interopMutex;
    bool m_interopTried = false;
    std::unique_ptr<MemoryInterop> m_memoryInterop;
    std::mutex m_spareMutex;
    std::vector<SpareBuffer> m_spares;
    bool m_sparesClosed = false;
    uint32_t m_vendorId = 0;
    VkDescriptorSet m_globals = VK_NULL_HANDLE;
    std::once_flag m_globalsWritten;
    std::array<VkSampler, kTextureSamplers> m_samplers{};
    std::mutex m_textureMutex;
    std::vector<uint32_t> m_freeTextureSlots;
    uint32_t m_textureSlots = 0;
};

struct VulkanBuild {
    VkAccelerationStructureGeometryKHR geometry{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    VkAccelerationStructureBuildGeometryInfoKHR info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    uint32_t primitives = 0;

    void Describe(const AccelInput& input);
};

class VulkanAccel : public Accel {
  public:
    VulkanAccel(VulkanDevice& owner, Ref<Buffer> storage, AccelLevel level, AccelSizes sizes,
                std::string label);
    ~VulkanAccel() override;

    VkAccelerationStructureKHR Handle() const {
        return m_handle;
    }

  private:
    VulkanDevice& m_owner;
    VkAccelerationStructureKHR m_handle = VK_NULL_HANDLE;
};

class VulkanBuffer : public Buffer {
  public:
    VulkanBuffer(VulkanDevice& owner, const BufferDesc& desc, VkBuffer buffer,
                 VmaAllocation allocation, void* mapped, void* external = nullptr,
                 uint64_t externalAddress = 0)
        : Buffer(owner, desc), m_owner(owner), m_buffer(buffer), m_allocation(allocation),
          m_external(external), m_externalAddress(externalAddress) {
        m_address = owner.BufferAddress(buffer);
        m_mapped = mapped;
    }

    ~VulkanBuffer() override;

    VkBuffer Handle() const {
        return m_buffer;
    }

    ExternalAddress External(DLDeviceType api) override;

  private:
    VulkanDevice& m_owner;
    VkBuffer m_buffer;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    std::mutex m_externalMutex;
    void* m_external = nullptr;
    uint64_t m_externalAddress = 0;
};

class VulkanKernel : public Kernel {
  public:
    explicit VulkanKernel(VulkanDevice& owner) : m_owner(owner) {}

    ~VulkanKernel() override;

    VkPipeline Pipeline() const {
        return m_pipeline;
    }

  protected:
    KernelStats Load(const KernelCode& code, const DiskCache& cache,
                     const DeviceGlobals& globals) override;

  private:
    VulkanDevice& m_owner;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
};

class VulkanTexture : public Texture {
  public:
    VulkanTexture(VulkanDevice& owner, const TextureDesc& desc);
    ~VulkanTexture() override;

    VkImage Image() const {
        return m_image;
    }

    uint64_t KernelHandle(TextureWrap wrapS, TextureWrap wrapT) override {
        return uint64_t(1) << 32 | TextureHandle(*m_slot, wrapS, wrapT);
    }

  private:
    void Destroy() noexcept;

    VulkanDevice& m_owner;
    VkImage m_image = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    VkImageView m_view = VK_NULL_HANDLE;
    std::optional<uint32_t> m_slot;
};

std::unique_ptr<Queue> CreateQueue(VulkanDevice& device, VkQueue handle, bool runsKernels);

struct NativeSubmit;

Operation SubmitNative(Queue& queue, const NativeSubmit& submit);

VkResult PresentNative(Queue& queue, const VkPresentInfoKHR& present);

VkQueue NativeQueue(Queue& queue);

}
