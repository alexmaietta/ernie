// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <volk.h>

#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace ernie {

class VulkanContext;

/// Every pass's textures and samplers in one descriptor set (set 1 of
/// every `PipelineLayout`), each at an index shaders take from their
/// uniforms: bindless descriptor indexing, so a pass's long-lived
/// textures cost it no binding of its own (bindless.slang declares
/// the arrays). Buffers need none: shaders read them through their
/// addresses (`GpuBuffer::Address`).
///
/// A slot's descriptor is written when it is added, while the set may
/// be bound (update after bind), and a freed slot is reused only a
/// few frames later (`AdvanceFrame`), or once the device is idle when
/// the heap runs out, so work in flight never sees it change.
///
/// Reference: Khronos, VK_EXT_descriptor_indexing (core in Vulkan
/// 1.2),
/// https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_descriptor_indexing.html
class BindlessHeap {
  public:
    /// The heap's arrays, in bindless.slang's bindings.
    enum class Kind : std::uint32_t {
        Texture2D = 0,          ///< Texture2D<float4>
        Texture2DArray = 1,     ///< Texture2DArray<float4>
        Sampler = 2,            ///< SamplerState
        ComparisonSampler = 3,  ///< SamplerComparisonState
    };
    static constexpr std::uint32_t kKinds = 4;
    static constexpr std::uint32_t kTextures = 4096;
    static constexpr std::uint32_t kSamplers = 256;
    /// Frames a freed slot waits before reuse.
    static constexpr std::uint64_t kRetireFrames = 3;

    /// A slot of the heap, freed when destroyed; move-only. Its index
    /// is what a shader reads its array at.
    class Slot {
      public:
        Slot() = default;
        ~Slot();
        Slot(Slot&& other) noexcept;
        Slot& operator=(Slot&& other) noexcept;
        Slot(const Slot&) = delete;
        Slot& operator=(const Slot&) = delete;

        std::uint32_t Index() const {
            return index_;
        }
        bool Valid() const {
            return heap_ != nullptr;
        }

      private:
        friend class BindlessHeap;
        Slot(BindlessHeap* heap, Kind kind, std::uint32_t index)
            : heap_(heap), kind_(kind), index_(index) {}
        void Free();

        BindlessHeap* heap_ = nullptr;
        Kind kind_ = Kind::Texture2D;
        std::uint32_t index_ = 0;
    };

    /// @throws std::runtime_error if the set cannot be created.
    explicit BindlessHeap(const VulkanContext& context);
    ~BindlessHeap();
    BindlessHeap(const BindlessHeap&) = delete;
    BindlessHeap& operator=(const BindlessHeap&) = delete;

    /// A slot holding `view` (of `Texture2D` or `Texture2DArray`
    /// kind), sampled in `layout`.
    /// @throws std::runtime_error when the kind's array is full of
    /// slots in use.
    Slot AddImage(Kind kind, VkImageView view,
                  VkImageLayout layout =
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    /// A slot holding `sampler`, a comparison sampler or not.
    Slot AddSampler(VkSampler sampler, bool comparison = false);

    /// Lets slots freed `kRetireFrames` frames ago be reused; the
    /// renderer calls it once a frame.
    void AdvanceFrame();

    VkDescriptorSetLayout Layout() const {
        return layout_;
    }

    /// Binds the heap as set 1 of `pipelineLayout`.
    void Bind(VkCommandBuffer commands, VkPipelineBindPoint point,
              VkPipelineLayout pipelineLayout) const;

    /// Slots in use of `kind`, for tests.
    std::uint32_t InUse(Kind kind) const;

  private:
    struct Array {
        std::uint32_t capacity = 0;
        std::uint32_t next = 0;  ///< Never used from here on.
        std::vector<std::uint32_t> free;
        /// Freed slots and the frame they may be reused from.
        std::deque<std::pair<std::uint32_t, std::uint64_t>> retiring;
        std::uint32_t inUse = 0;
    };

    std::uint32_t Take(Kind kind);
    void Release(Kind kind, std::uint32_t index);
    void Write(Kind kind, std::uint32_t index,
               const VkDescriptorImageInfo& info);

    const VulkanContext& context_;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    mutable std::mutex mutex_;
    Array arrays_[kKinds];
    std::uint64_t frame_ = 0;
};

}  // namespace ernie
