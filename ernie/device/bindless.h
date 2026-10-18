// SPDX-License-Identifier: BSD-3-Clause
/// @file bindless.h
/// @brief One heap of every image and sampler shaders reach, by index.
#pragma once

#include "ernie/device/image.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <vector>

namespace ernie {

/// @brief A slot of the bindless heap: what shaders index its arrays with.
struct BindlessIndex {
    std::uint32_t value = 0;
    bool operator==(const BindlessIndex&) const = default;
};

/// @brief The bindless heap: one descriptor set of three arrays, updated after bind and partially
/// bound, which every kernel binds as set 0:
///
///     [[vk::binding(0, 0)]] Texture2D textures[];
///     [[vk::binding(1, 0)]] RWTexture2D<float4> images[];
///     [[vk::binding(2, 0)]] SamplerState samplers[];
///
/// Indices come from free lists and return to them once the GPU is done with their slot. On
/// WebGPU the same indices pick from texture arrays.
class BindlessHeap {
  public:
    explicit BindlessHeap(DeviceContext& context);
    ~BindlessHeap();
    BindlessHeap(const BindlessHeap&) = delete;
    BindlessHeap& operator=(const BindlessHeap&) = delete;

    /// @throws DeviceError when the array is full.
    BindlessIndex AddTexture(const DeviceImage& image);
    BindlessIndex AddStorageImage(const DeviceImage& image);
    BindlessIndex AddSampler(const Sampler& sampler);
    /// @brief Frees slots once every queue's work submitted so far is done.
    void RemoveTexture(BindlessIndex index);
    void RemoveStorageImage(BindlessIndex index);
    void RemoveSampler(BindlessIndex index);

    VkDescriptorSetLayout Layout() const {
        return layout_;
    }
    VkDescriptorSet Set() const {
        return set_;
    }

  private:
    /// @brief The slots of one array.
    struct Slots {
        std::uint32_t capacity = 0;
        std::uint32_t next = 0;
        std::vector<std::uint32_t> free;
    };

    /// Takes a slot of `binding` and writes its descriptor, under the lock: the set's updates
    /// need it to themselves.
    BindlessIndex Add(std::uint32_t binding, const VkDescriptorImageInfo& info);
    void Remove(std::uint32_t binding, BindlessIndex index);

    DeviceContext& context_;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    std::mutex mutex_;
    std::array<Slots, 3> slots_{};
};

}  // namespace ernie
