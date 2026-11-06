// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include "ernie/device/kernels/shared.h"
#include "ernie/device/queue.h"
#include "ernie/device/vulkan/device.h"
#include "ernie/device/vulkan/interop.h"
#include "ernie/device/vulkan/native.h"
#include "ernie/util/error.h"
#include "ernie/util/profiler.h"

namespace ernie::device::vulkan {
namespace {

void Barrier(const VolkDeviceTable& functions, VkCommandBuffer commands,
             VkPipelineStageFlags2 stages, VkAccessFlags2 accesses) {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    barrier.dstStageMask = stages;
    barrier.dstAccessMask = accesses;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    functions.vkCmdPipelineBarrier2(commands, &dependency);
}

class VulkanQueue : public Queue {
  public:
    VulkanQueue(VulkanDevice& device, VkQueue handle, bool runsKernels)
        : Queue(device, runsKernels), m_vulkan(device), m_handle(handle) {
        m_timeline = CreateTimeline();
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = device.QueueFamily();
        const VkResult created = Functions().vkCreateCommandPool(Handle(), &pool, nullptr, &m_pool);
        if (created != VK_SUCCESS) {
            Functions().vkDestroySemaphore(Handle(), m_timeline, nullptr);
            Check(created, "vkCreateCommandPool");
        }
    }

    ~VulkanQueue() override {
        Functions().vkQueueWaitIdle(m_handle);
        Functions().vkDestroyCommandPool(Handle(), m_pool, nullptr);
        m_interop.reset();
        if (m_follow != VK_NULL_HANDLE) {
            Functions().vkDestroySemaphore(Handle(), m_follow, nullptr);
        }
        Functions().vkDestroySemaphore(Handle(), m_timeline, nullptr);
    }

    Operation SubmitNative(const NativeSubmit& submit) {
        return RunNative(&submit);
    }

    VkResult Present(const VkPresentInfoKHR& present) {
        const std::unique_lock submitLock = LockSubmissions();
        return Functions().vkQueuePresentKHR(m_handle, &present);
    }

    VkQueue QueueHandle() const {
        return m_handle;
    }

  protected:
    void Execute(const CommandBuffer& queued, const CommandBuffer& commands,
                 const Timestamps* timestamps, uint64_t value) override {
        const auto* queries = static_cast<const QueryTimestamps*>(timestamps);
        const VkCommandBuffer first = Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        Translate(first, queued, nullptr);
        VkCommandBuffer second = VK_NULL_HANDLE;
        if (!commands.Reusable()) {
            Translate(first, commands, queries);
            End(first);
        } else {
            End(first);
            second = RecordReusable(commands, queries);
        }

        SubmitBuffers(first, second, value, nullptr);
        m_running.emplace_back(value, first);
    }

    void ExecuteNative(const CommandBuffer& queued, uint64_t value, const void* native) override {
        const VkCommandBuffer first = Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        Translate(first, queued, nullptr);
        Barrier(Functions(), first, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        End(first);
        SubmitBuffers(first, VK_NULL_HANDLE, value, static_cast<const NativeSubmit*>(native));
        m_running.emplace_back(value, first);
    }

    bool SharesStreams() override {
        return Interop() != nullptr;
    }

    void FollowStream(void* stream) override {
        Interop()->Follow(stream, ++m_followValue);
    }

    void StreamWait(void* stream, uint64_t value) override {
        Interop()->Wait(stream, value);
    }

    void FollowQueue(Queue& other, uint64_t value) override {
        const VkSemaphore timeline = static_cast<VulkanQueue&>(other).m_timeline;
        const auto found = std::ranges::find(m_queueWaits, timeline, &QueueWait::timeline);
        if (found != m_queueWaits.end()) {
            found->value = std::max(found->value, value);
        } else {
            m_queueWaits.push_back({timeline, value});
        }
    }

    std::shared_ptr<Timestamps> CreateTimestamps(uint32_t count) override {
        return m_vulkan.HasTimestamps() ? std::make_shared<QueryTimestamps>(*this, count) : nullptr;
    }

    uint64_t Finished() const override {
        uint64_t value = 0;
        Check(Functions().vkGetSemaphoreCounterValue(Handle(), m_timeline, &value),
              "vkGetSemaphoreCounterValue");
        return value;
    }

    void WaitFor(uint64_t value) const override {
        VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wait.semaphoreCount = 1;
        wait.pSemaphores = &m_timeline;
        wait.pValues = &value;
        Check(Functions().vkWaitSemaphores(Handle(), &wait, UINT64_MAX), "vkWaitSemaphores");
    }

  private:
    struct QueryTimestamps : Timestamps {
        QueryTimestamps(const VulkanQueue& owner, uint32_t slots) : queue(owner), count(slots) {
            VkQueryPoolCreateInfo create{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            create.queryType = VK_QUERY_TYPE_TIMESTAMP;
            create.queryCount = slots;
            Check(owner.Functions().vkCreateQueryPool(owner.Handle(), &create, nullptr, &pool),
                  "vkCreateQueryPool");
        }

        ~QueryTimestamps() override {
            queue.Functions().vkDestroyQueryPool(queue.Handle(), pool, nullptr);
        }

        QueryTimestamps(const QueryTimestamps&) = delete;
        QueryTimestamps& operator=(const QueryTimestamps&) = delete;

        bool Read(std::span<uint64_t> ns) const override {
            Check(queue.Functions().vkGetQueryPoolResults(queue.Handle(), pool, 0, count,
                                                          ns.size_bytes(), ns.data(),
                                                          sizeof(uint64_t), VK_QUERY_RESULT_64_BIT),
                  "vkGetQueryPoolResults");
            return queue.m_vulkan.ToNanoseconds(ns);
        }

        const VulkanQueue& queue;
        uint32_t count;
        VkQueryPool pool = VK_NULL_HANDLE;
    };

    const VolkDeviceTable& Functions() const {
        return m_vulkan.Functions();
    }

    VkDevice Handle() const {
        return m_vulkan.Handle();
    }

    VkCommandBuffer RecordReusable(const CommandBuffer& commands, const QueryTimestamps* queries) {
        std::shared_ptr<void>& recording = Recording(commands);
        if (!recording) {
            const VkCommandBuffer kept = Begin(0);
            Translate(kept, commands, queries);
            End(kept);
            recording = std::shared_ptr<void>(kept, [this](void* handed) {
                const std::lock_guard lock(m_returnedMutex);
                m_returned.push_back(static_cast<VkCommandBuffer>(handed));
            });
        }
        return static_cast<VkCommandBuffer>(recording.get());
    }

    void SubmitBuffers(VkCommandBuffer first, VkCommandBuffer second, uint64_t value,
                       const NativeSubmit* native) {
        std::vector<VkCommandBufferSubmitInfo> buffers = {
            {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, first}};
        if (second != VK_NULL_HANDLE) {
            buffers.push_back({VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, second});
        }
        VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signal.semaphore = m_timeline;
        signal.value = value;
        signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        std::vector<VkSemaphoreSubmitInfo> signals = {signal};
        std::vector<VkSemaphoreSubmitInfo> waits;
        if (native != nullptr) {
            for (const VkCommandBuffer commands : native->commands) {
                buffers.push_back(
                    {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, commands});
            }
            waits.assign(native->waits.begin(), native->waits.end());
            signals.insert(signals.end(), native->signals.begin(), native->signals.end());
        }
        for (const QueueWait& queueWait : m_queueWaits) {
            VkSemaphoreSubmitInfo& wait = waits.emplace_back();
            wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            wait.semaphore = queueWait.timeline;
            wait.value = queueWait.value;
            wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        }
        m_queueWaits.clear();
        if (m_followValue > m_followed) {
            VkSemaphoreSubmitInfo& wait = waits.emplace_back();
            wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            wait.semaphore = m_follow;
            wait.value = m_followValue;
            wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            m_followed = m_followValue;
        }

        VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submit.commandBufferInfoCount = uint32_t(buffers.size());
        submit.pCommandBufferInfos = buffers.data();
        submit.signalSemaphoreInfoCount = uint32_t(signals.size());
        submit.pSignalSemaphoreInfos = signals.data();
        submit.waitSemaphoreInfoCount = uint32_t(waits.size());
        submit.pWaitSemaphoreInfos = waits.data();
        Check(Functions().vkQueueSubmit2(m_handle, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit2");
    }

    VkCommandBuffer Begin(VkCommandBufferUsageFlags flags) {
        const VkCommandBuffer recorded = TakeCommandBuffer();
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = flags;
        Check(Functions().vkBeginCommandBuffer(recorded, &begin), "vkBeginCommandBuffer");
        return recorded;
    }

    void End(VkCommandBuffer recorded) {
        Barrier(Functions(), recorded, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        Check(Functions().vkEndCommandBuffer(recorded), "vkEndCommandBuffer");
    }

    void Translate(VkCommandBuffer recorded, const CommandBuffer& commands,
                   const QueryTimestamps* timestamps) {
        const VolkDeviceTable& vk = Functions();
        if (timestamps != nullptr) {
            vk.vkCmdResetQueryPool(recorded, timestamps->pool, 0, timestamps->count);
        }
        bool barrier = true;
        const std::span<const Command> list = commands.Commands();
        for (size_t at = 0; at < list.size(); ++at) {
            const Command& command = list[at];
            if (std::holds_alternative<BarrierCommand>(command)) {
                barrier = true;
                continue;
            }
            if (const auto* zone = std::get_if<ZoneCommand>(&command)) {
                if (timestamps != nullptr) {
                    vk.vkCmdWriteTimestamp2(recorded, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                            timestamps->pool,
                                            2 * zone->index + (zone->begin ? 0 : 1));
                }
                Label(recorded, commands, *zone);
                continue;
            }
            if (barrier) {
                Barrier(vk, recorded, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
                barrier = false;
            }
            if (std::holds_alternative<BuildAccelCommand>(command)) {
                size_t end = at + 1;
                while (end < list.size() && std::holds_alternative<BuildAccelCommand>(list[end])) {
                    ++end;
                }
                RecordBuilds(recorded, list.subspan(at, end - at));
                at = end - 1;
                continue;
            }
            Record(recorded, commands, command);
        }
    }

    void RecordBuilds(VkCommandBuffer recorded, std::span<const Command> commands) {
        ERNIE_ZONE("Vulkan/RecordBuilds");
        std::vector<VulkanBuild> builds(commands.size());
        std::vector<VkAccelerationStructureBuildGeometryInfoKHR> infos;
        std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
        std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> pointers;
        infos.reserve(commands.size());
        ranges.reserve(commands.size());
        for (size_t i = 0; i < commands.size(); ++i) {
            const BuildAccelCommand& c = std::get<BuildAccelCommand>(commands[i]);
            VulkanBuild& build = builds[i];
            build.Describe(c.input);
            build.info.dstAccelerationStructure = static_cast<const VulkanAccel&>(*c.dst).Handle();
            if (c.update) {
                build.info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
                build.info.srcAccelerationStructure = build.info.dstAccelerationStructure;
            }
            build.info.scratchData.deviceAddress = c.scratch->Address() + c.scratchOffset;
            infos.push_back(build.info);
            ranges.push_back({build.primitives, 0, 0, 0});
        }

        for (const VkAccelerationStructureBuildRangeInfoKHR& range : ranges) {
            pointers.push_back(&range);
        }
        Functions().vkCmdBuildAccelerationStructuresKHR(recorded, uint32_t(infos.size()),
                                                        infos.data(), pointers.data());
    }

    void Label(VkCommandBuffer recorded, const CommandBuffer& commands, const ZoneCommand& zone) {
        if (!m_vulkan.Labels()) {
            return;
        }
        if (!zone.begin) {
            vkCmdEndDebugUtilsLabelEXT(recorded);
            return;
        }
        VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
        label.pLabelName = commands.Zones()[zone.index].Name().data();
        vkCmdBeginDebugUtilsLabelEXT(recorded, &label);
    }

    VkCommandBuffer TakeCommandBuffer() {
        const uint64_t finished = Finished();
        while (!m_running.empty() && m_running.front().first <= finished) {
            m_free.push_back(m_running.front().second);
            m_running.pop_front();
        }
        {
            const std::lock_guard lock(m_returnedMutex);
            m_free.insert(m_free.end(), m_returned.begin(), m_returned.end());
            m_returned.clear();
        }

        if (!m_free.empty()) {
            const VkCommandBuffer reused = m_free.back();
            m_free.pop_back();
            return reused;
        }

        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = m_pool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        VkCommandBuffer allocated = VK_NULL_HANDLE;
        Check(Functions().vkAllocateCommandBuffers(Handle(), &allocate, &allocated),
              "vkAllocateCommandBuffers");
        return allocated;
    }

    void Record(VkCommandBuffer recorded, const CommandBuffer& commands, const Command& command) {
        const VolkDeviceTable& vk = Functions();
        std::visit(
            Overloaded{
                [&](const CopyCommand& c) {
                    const VkBufferCopy region{c.srcOffset, c.dstOffset, c.bytes};
                    vk.vkCmdCopyBuffer(recorded, Native(*c.src), Native(*c.dst), 1, &region);
                },
                [&](const TextureCopyCommand& c) { RecordTextureCopy(recorded, c); },
                [&](const FillCommand& c) {
                    vk.vkCmdFillBuffer(recorded, Native(*c.dst), c.offset, c.bytes, c.value);
                },
                [&](const UpdateCommand& c) {
                    vk.vkCmdUpdateBuffer(recorded, Native(*c.dst), c.offset, c.bytes,
                                         commands.Data(c));
                },
                [&](const BuildAccelCommand&) { RecordBuilds(recorded, std::span(&command, 1)); },
                [&](const BarrierCommand&) {},
                [&](const ZoneCommand&) {},
                [&](const DispatchCommand& c) {
                    Bind(recorded, *c.kernel, c.paramBytes, commands.Data(c));
                    if (c.counts != nullptr) {
                        vk.vkCmdDispatchIndirect(recorded, Native(*c.counts), c.countsOffset);
                    } else {
                        vk.vkCmdDispatch(recorded, c.groups.x, c.groups.y, c.groups.z);
                    }
                },
            },
            command);
    }

    void RecordTextureCopy(VkCommandBuffer recorded, const TextureCopyCommand& c) {
        // The copy replaces the whole level, so its old contents are discarded.
        // Textures stay in the general layout, which copies and samples both take.
        const VolkDeviceTable& vk = Functions();
        const VkImage image = static_cast<const VulkanTexture&>(*c.dst).Image();
        const TextureDesc& desc = c.dst->Desc();
        VkImageMemoryBarrier2 discard{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        discard.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        discard.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        discard.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        discard.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        discard.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        discard.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        discard.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        discard.image = image;
        discard.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, c.level, 1, 0, 1};
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &discard;
        vk.vkCmdPipelineBarrier2(recorded, &dependency);

        VkBufferImageCopy region{};
        region.bufferOffset = c.srcOffset;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, c.level, 0, 1};
        region.imageExtent = {std::max(desc.width >> c.level, 1u),
                              std::max(desc.height >> c.level, 1u), 1};
        vk.vkCmdCopyBufferToImage(recorded, Native(*c.src), image, VK_IMAGE_LAYOUT_GENERAL, 1,
                                  &region);
    }

    void Bind(VkCommandBuffer recorded, const Kernel& kernel, uint32_t paramBytes,
              const void* params) {
        const VolkDeviceTable& vk = Functions();
        const VkPipelineLayout layout = m_vulkan.PipelineLayout();
        const VkDescriptorSet globals = m_vulkan.Globals();
        vk.vkCmdBindPipeline(recorded, VK_PIPELINE_BIND_POINT_COMPUTE,
                             static_cast<const VulkanKernel&>(kernel).Pipeline());
        vk.vkCmdBindDescriptorSets(recorded, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &globals,
                                   0, nullptr);
        if (paramBytes > 0) {
            vk.vkCmdPushConstants(recorded, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                  uint32_t(RoundUp(paramBytes, 4)), params);
        }
    }

    static VkBuffer Native(Buffer& buffer) {
        return static_cast<VulkanBuffer&>(buffer).Handle();
    }

    VkSemaphore CreateTimeline() {
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkExportSemaphoreCreateInfo exported{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
        exported.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        if (m_vulkan.SharesSemaphores()) {
            type.pNext = &exported;
        }
        VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        VkSemaphore created = VK_NULL_HANDLE;
        Check(Functions().vkCreateSemaphore(Handle(), &semaphore, nullptr, &created),
              "vkCreateSemaphore");
        return created;
    }

    StreamInterop* Interop() {
        const std::lock_guard lock(m_interopMutex);
        if (!m_interopTried) {
            m_interopTried = true;
            m_follow = CreateTimeline();
            try {
                m_interop = CreateStreamInterop(m_vulkan, m_timeline, m_follow);
            } catch (const UnsupportedError&) {
            }
        }
        return m_interop.get();
    }

    struct QueueWait {
        VkSemaphore timeline;
        uint64_t value;
    };

    VulkanDevice& m_vulkan;
    VkQueue m_handle;
    std::vector<QueueWait> m_queueWaits;
    VkSemaphore m_timeline = VK_NULL_HANDLE;
    VkSemaphore m_follow = VK_NULL_HANDLE;
    uint64_t m_followValue = 0;
    uint64_t m_followed = 0;
    std::mutex m_interopMutex;
    bool m_interopTried = false;
    std::unique_ptr<StreamInterop> m_interop;
    VkCommandPool m_pool = VK_NULL_HANDLE;
    std::deque<std::pair<uint64_t, VkCommandBuffer>> m_running;
    std::vector<VkCommandBuffer> m_free;
    std::mutex m_returnedMutex;
    std::vector<VkCommandBuffer> m_returned;
};

}

std::unique_ptr<Queue> CreateQueue(VulkanDevice& device, VkQueue handle, bool runsKernels) {
    return std::make_unique<VulkanQueue>(device, handle, runsKernels);
}

VkQueue NativeQueue(Queue& queue) {
    return static_cast<VulkanQueue&>(queue).QueueHandle();
}

Operation SubmitNative(Queue& queue, const NativeSubmit& submit) {
    return static_cast<VulkanQueue&>(queue).SubmitNative(submit);
}

VkResult PresentNative(Queue& queue, const VkPresentInfoKHR& present) {
    return static_cast<VulkanQueue&>(queue).Present(present);
}

}
