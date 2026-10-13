// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/bindless.h"

#include "ernie/device/vulkan_context.h"

#include <array>
#include <stdexcept>
#include <utility>

namespace ernie {

namespace {

constexpr VkDescriptorType TypeOf(BindlessHeap::Kind kind) {
    return kind == BindlessHeap::Kind::Sampler ||
                   kind == BindlessHeap::Kind::ComparisonSampler
               ? VK_DESCRIPTOR_TYPE_SAMPLER
               : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
}

}  // namespace

BindlessHeap::Slot::~Slot() {
    Free();
}

BindlessHeap::Slot::Slot(Slot&& other) noexcept
    : heap_(std::exchange(other.heap_, nullptr)), kind_(other.kind_),
      index_(other.index_) {}

BindlessHeap::Slot&
BindlessHeap::Slot::operator=(Slot&& other) noexcept {
    if (this != &other) {
        Free();
        heap_ = std::exchange(other.heap_, nullptr);
        kind_ = other.kind_;
        index_ = other.index_;
    }
    return *this;
}

void BindlessHeap::Slot::Free() {
    if (heap_ != nullptr) {
        heap_->Release(kind_, index_);
        heap_ = nullptr;
    }
}

BindlessHeap::BindlessHeap(const VulkanContext& context)
    : context_(context) {
    std::array<VkDescriptorSetLayoutBinding, kKinds> bindings{};
    std::array<VkDescriptorBindingFlags, kKinds> flags{};
    std::array<VkDescriptorPoolSize, 2> sizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 0},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, 0}};
    for (std::uint32_t k = 0; k < kKinds; ++k) {
        const auto kind = static_cast<Kind>(k);
        const VkDescriptorType type = TypeOf(kind);
        const std::uint32_t count = type == VK_DESCRIPTOR_TYPE_SAMPLER
                                        ? kSamplers
                                        : kTextures;
        bindings[k] = {k, type, count, VK_SHADER_STAGE_ALL, nullptr};
        flags[k] =
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
        arrays_[k].capacity = count;
        sizes[type == VK_DESCRIPTOR_TYPE_SAMPLER ? 1 : 0]
            .descriptorCount += count;
    }
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bindingFlags.bindingCount = kKinds;
    bindingFlags.pBindingFlags = flags.data();
    VkDescriptorSetLayoutCreateInfo layoutInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.pNext = &bindingFlags;
    layoutInfo.flags =
        VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = kKinds;
    layoutInfo.pBindings = bindings.data();
    CheckVk(vkCreateDescriptorSetLayout(context.Device(), &layoutInfo,
                                        nullptr, &layout_),
            "vkCreateDescriptorSetLayout (bindless heap)");

    VkDescriptorPoolCreateInfo poolInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
    poolInfo.pPoolSizes = sizes.data();
    CheckVk(vkCreateDescriptorPool(context.Device(), &poolInfo,
                                   nullptr, &pool_),
            "vkCreateDescriptorPool (bindless heap)");

    VkDescriptorSetAllocateInfo allocate{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = pool_;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &layout_;
    CheckVk(
        vkAllocateDescriptorSets(context.Device(), &allocate, &set_),
        "vkAllocateDescriptorSets (bindless heap)");
}

BindlessHeap::~BindlessHeap() {
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(context_.Device(), pool_, nullptr);
    }
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context_.Device(), layout_,
                                     nullptr);
    }
}

std::uint32_t BindlessHeap::Take(Kind kind) {
    Array& array = arrays_[static_cast<std::uint32_t>(kind)];
    // Retired slots whose frames have passed.
    while (!array.retiring.empty() &&
           array.retiring.front().second <= frame_) {
        array.free.push_back(array.retiring.front().first);
        array.retiring.pop_front();
    }
    if (array.free.empty() && array.next == array.capacity &&
        !array.retiring.empty()) {
        // Full: once the device is idle, nothing in flight reads
        // them.
        vkDeviceWaitIdle(context_.Device());
        for (const auto& [index, frame] : array.retiring) {
            array.free.push_back(index);
        }
        array.retiring.clear();
    }
    std::uint32_t index = 0;
    if (!array.free.empty()) {
        index = array.free.back();
        array.free.pop_back();
    } else if (array.next < array.capacity) {
        index = array.next++;
    } else {
        throw std::runtime_error("The bindless heap is full: " +
                                 std::to_string(array.capacity) +
                                 " slots in use");
    }
    ++array.inUse;
    return index;
}

void BindlessHeap::Release(Kind kind, std::uint32_t index) {
    const std::lock_guard lock(mutex_);
    // Shared: freed with its last holder.
    const auto k = static_cast<std::uint32_t>(kind);
    if (const auto at = sharedAt_[k].find(index);
        at != sharedAt_[k].end()) {
        if (--at->second.holders > 0) {
            return;
        }
        // Unless forgotten (and its handle maybe shared anew).
        if (const auto it = shared_[k].find(at->second.held);
            it != shared_[k].end() && it->second == index) {
            shared_[k].erase(it);
        }
        sharedAt_[k].erase(at);
    }
    Array& array = arrays_[k];
    array.retiring.emplace_back(index, frame_ + kRetireFrames);
    --array.inUse;
}

void BindlessHeap::Write(Kind kind, std::uint32_t index,
                         const VkDescriptorImageInfo& info) {
    VkWriteDescriptorSet write{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set_;
    write.dstBinding = static_cast<std::uint32_t>(kind);
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = TypeOf(kind);
    write.pImageInfo = &info;
    vkUpdateDescriptorSets(context_.Device(), 1, &write, 0, nullptr);
}

BindlessHeap::Slot BindlessHeap::AddImage(Kind kind, VkImageView view,
                                          VkImageLayout layout) {
    if (TypeOf(kind) != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {
        throw std::invalid_argument(
            "BindlessHeap::AddImage takes a texture kind");
    }
    return Share(kind,
                 Held{reinterpret_cast<std::uint64_t>(view),
                      static_cast<std::uint32_t>(layout)},
                 VkDescriptorImageInfo{VK_NULL_HANDLE, view, layout});
}

BindlessHeap::Slot BindlessHeap::AddSampler(VkSampler sampler,
                                            bool comparison) {
    return Share(comparison ? Kind::ComparisonSampler : Kind::Sampler,
                 Held{reinterpret_cast<std::uint64_t>(sampler), 0},
                 VkDescriptorImageInfo{sampler, VK_NULL_HANDLE,
                                       VK_IMAGE_LAYOUT_UNDEFINED});
}

BindlessHeap::Slot
BindlessHeap::Share(Kind kind, const Held& held,
                    const VkDescriptorImageInfo& info) {
    const std::lock_guard lock(mutex_);
    const auto k = static_cast<std::uint32_t>(kind);
    // Not written again: frames in flight may read the slot, and an
    // update to a descriptor in use invalidates their commands. A
    // destroyed object's slots are forgotten (`Forget`), so a handle
    // reused after it never finds them.
    if (const auto it = shared_[k].find(held);
        it != shared_[k].end()) {
        ++sharedAt_[k].at(it->second).holders;
        return Slot(this, kind, it->second);
    }
    const std::uint32_t index = Take(kind);
    shared_[k].emplace(held, index);
    sharedAt_[k].emplace(index, SharedSlot{held, 1});
    Write(kind, index, info);
    return Slot(this, kind, index);
}

void BindlessHeap::Forget(std::uint64_t object) {
    const std::lock_guard lock(mutex_);
    for (auto& shared : shared_) {
        std::erase_if(shared, [object](const auto& entry) {
            return entry.first.object == object;
        });
    }
}

void BindlessHeap::AdvanceFrame() {
    const std::lock_guard lock(mutex_);
    ++frame_;
}

void BindlessHeap::Bind(VkCommandBuffer commands,
                        VkPipelineBindPoint point,
                        VkPipelineLayout pipelineLayout) const {
    vkCmdBindDescriptorSets(commands, point, pipelineLayout, 1, 1,
                            &set_, 0, nullptr);
}

std::uint32_t BindlessHeap::InUse(Kind kind) const {
    const std::lock_guard lock(mutex_);
    return arrays_[static_cast<std::uint32_t>(kind)].inUse;
}

}  // namespace ernie
