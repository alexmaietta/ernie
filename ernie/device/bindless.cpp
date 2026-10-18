// SPDX-License-Identifier: BSD-3-Clause
/// @file bindless.cpp
/// @brief The bindless descriptor set and its slot free lists.
#include "ernie/device/bindless.h"

#include <algorithm>
#include <array>

namespace ernie {
namespace {

constexpr std::array kTypes{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                            VK_DESCRIPTOR_TYPE_SAMPLER};
constexpr std::array kNames{"texture", "storage image", "sampler"};

}  // namespace

BindlessHeap::BindlessHeap(DeviceContext& context) : context_(context) {
    VkPhysicalDeviceDescriptorIndexingProperties indexing{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                           .pNext = &indexing};
    vkGetPhysicalDeviceProperties2(context.Physical(), &properties);
    // Generous where the device allows; within what every desktop driver and lavapipe offer.
    const std::array capacities{
        std::min(indexing.maxDescriptorSetUpdateAfterBindSampledImages, 65536u),
        std::min(indexing.maxDescriptorSetUpdateAfterBindStorageImages, 4096u),
        std::min(indexing.maxDescriptorSetUpdateAfterBindSamplers, 256u)};

    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    std::array<VkDescriptorPoolSize, 3> sizes{};
    const VkDescriptorBindingFlags flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                           VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                                           VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
    const std::array bindingFlags{flags, flags, flags};
    for (std::uint32_t i = 0; i < 3; ++i) {
        slots_[i].capacity = capacities[i];
        bindings[i] = {.binding = i,
                       .descriptorType = kTypes[i],
                       .descriptorCount = capacities[i],
                       .stageFlags = VK_SHADER_STAGE_ALL};
        sizes[i] = {kTypes[i], capacities[i]};
    }
    const VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = 3,
        .pBindingFlags = bindingFlags.data()};
    const VkDescriptorSetLayoutCreateInfo layout{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &flagsInfo,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = 3,
        .pBindings = bindings.data()};
    CheckVk(vkCreateDescriptorSetLayout(context.Device(), &layout, nullptr, &layout_),
            "Creating the bindless layout");
    const VkDescriptorPoolCreateInfo pool{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                          .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
                                          .maxSets = 1,
                                          .poolSizeCount = 3,
                                          .pPoolSizes = sizes.data()};
    CheckVk(vkCreateDescriptorPool(context.Device(), &pool, nullptr, &pool_),
            "Creating the bindless pool");
    const VkDescriptorSetAllocateInfo allocate{.sType =
                                                   VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                               .descriptorPool = pool_,
                                               .descriptorSetCount = 1,
                                               .pSetLayouts = &layout_};
    CheckVk(vkAllocateDescriptorSets(context.Device(), &allocate, &set_),
            "Allocating the bindless set");
}

BindlessHeap::~BindlessHeap() {
    // Slots freed late write to this heap: they must come back before it goes.
    context_.WaitIdle();
    vkDestroyDescriptorPool(context_.Device(), pool_, nullptr);
    vkDestroyDescriptorSetLayout(context_.Device(), layout_, nullptr);
}

BindlessIndex BindlessHeap::AddTexture(const Image& image) {
    const BindlessIndex index = Take(0);
    Write(0, index, {.imageView = image.View(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL});
    return index;
}

BindlessIndex BindlessHeap::AddStorageImage(const Image& image) {
    const BindlessIndex index = Take(1);
    Write(1, index, {.imageView = image.View(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL});
    return index;
}

BindlessIndex BindlessHeap::AddSampler(const Sampler& sampler) {
    const BindlessIndex index = Take(2);
    Write(2, index, {.sampler = sampler.Handle()});
    return index;
}

void BindlessHeap::RemoveTexture(BindlessIndex index) {
    Remove(0, index);
}

void BindlessHeap::RemoveStorageImage(BindlessIndex index) {
    Remove(1, index);
}

void BindlessHeap::RemoveSampler(BindlessIndex index) {
    Remove(2, index);
}

BindlessIndex BindlessHeap::Take(std::uint32_t binding) {
    const std::lock_guard lock(mutex_);
    Slots& slots = slots_[binding];
    if (!slots.free.empty()) {
        const std::uint32_t index = slots.free.back();
        slots.free.pop_back();
        return {index};
    }
    if (slots.next == slots.capacity) {
        throw DeviceError(DeviceErrorKind::OutOfMemory,
                          std::string("The bindless heap holds no more ") + kNames[binding] + "s");
    }
    return {slots.next++};
}

void BindlessHeap::Write(std::uint32_t binding, BindlessIndex index,
                         const VkDescriptorImageInfo& info) {
    const VkWriteDescriptorSet write{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                     .dstSet = set_,
                                     .dstBinding = binding,
                                     .dstArrayElement = index.value,
                                     .descriptorCount = 1,
                                     .descriptorType = kTypes[binding],
                                     .pImageInfo = &info};
    vkUpdateDescriptorSets(context_.Device(), 1, &write, 0, nullptr);
}

void BindlessHeap::Remove(std::uint32_t binding, BindlessIndex index) {
    context_.Retire([this, binding, index] {
        const std::lock_guard lock(mutex_);
        slots_[binding].free.push_back(index.value);
    });
}

}  // namespace ernie
