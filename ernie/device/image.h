// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/vulkan_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ernie {

/// Description of a 2D image or 2D image array.
struct ImageDesc {
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    std::uint32_t layers = 1;
    std::uint32_t mips = 1;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    /// Allocate in a dedicated allocation exportable as a POSIX file
    /// descriptor, for CUDA to read the image in place
    /// (`VulkanContext::SupportsExternalMemory`).
    bool shared = false;
    /// View the image as a 2D array even with one layer, for shaders
    /// that declare an array.
    bool array = false;
    /// With several layers, also view them in consecutive groups of
    /// this many as 2D arrays (`GroupView`), such as one view's
    /// spectral bands in a batch of views; 0 for none.
    std::uint32_t groupLayers = 0;
    /// Lay the texels out linearly, rows `RowPitch` bytes apart,
    /// rather than in the device's own tiling: with `shared`, compute
    /// APIs write the image as a plain pitched array in place, and
    /// shaders sample that same memory
    /// (`VulkanContext::SupportsLinearSampling`). One layer and one
    /// mip.
    bool linear = false;
    /// How the image's view maps its channels, such as a grey image's
    /// red to all three.
    VkComponentMapping components{};
};

/// A device-local Vulkan image with a view of all its mips and
/// layers, allocated through VMA.
///
/// The image tracks one layout for all its subresources; `Transition`
/// records the barrier that moves it to a new one.
class GpuImage {
  public:
    /// @throws std::runtime_error if allocation fails.
    GpuImage(const VulkanContext& context, const ImageDesc& desc);
    ~GpuImage();
    GpuImage(GpuImage&& other) noexcept;
    GpuImage& operator=(GpuImage&& other) noexcept;
    GpuImage(const GpuImage&) = delete;
    GpuImage& operator=(const GpuImage&) = delete;

    VkImage Handle() const {
        return image_;
    }

    /// View of every mip and layer (`VK_IMAGE_VIEW_TYPE_2D_ARRAY`
    /// when there are several layers).
    VkImageView View() const {
        return view_;
    }

    /// View of one layer's first mip, e.g. to render into one cascade
    /// of a shadow map.
    /// @throws std::out_of_range for a layer the image does not have.
    VkImageView LayerView(std::uint32_t layer) const;

    /// View of mip `mip` alone, of every layer as a 2D array for
    /// array images, such as one level of a depth pyramid of a batch
    /// of views; a 2D view of it otherwise.
    /// @throws std::out_of_range for a mip the image does not have.
    VkImageView MipView(std::uint32_t mip) const;

    /// View of layer group `group` (`ImageDesc::groupLayers` layers
    /// from `group` times that) as a 2D array; with a single group,
    /// the view of every layer.
    /// @throws std::out_of_range for a group the image does not have.
    VkImageView GroupView(std::uint32_t group) const;

    const ImageDesc& Desc() const {
        return desc_;
    }

    VkImageLayout Layout() const {
        return layout_;
    }

    /// Bytes of device memory the image occupies.
    VkDeviceSize Bytes() const {
        return bytes_;
    }

    /// Records a barrier moving every subresource to `layout`; waits
    /// for and makes visible all earlier work, which is simple rather
    /// than minimal.
    void Transition(VkCommandBuffer commands, VkImageLayout layout);

    /// Uploads tightly packed texels for each mip of layer 0
    /// (`mipData[m]` holds mip `m`), or of every layer in turn
    /// (`mipData[l * mips + m]` holds mip `m` of layer `l`), and
    /// leaves the image ready for sampling in fragment and compute
    /// shaders.
    /// @throws std::invalid_argument if the number of mips or a mip's
    /// size does not match.
    void Upload(std::span<const std::span<const std::byte>> mipData);

    /// For shared images, a new handle to the image's memory (a file
    /// descriptor, or a Win32 handle on Windows), owned by the caller
    /// until it imports it elsewhere (`CloseExternalHandle`).
    /// @throws std::logic_error for other images, or
    /// std::runtime_error if export fails.
    ExternalHandle ExportHandle() const;

    /// For shared images, the size of the image's dedicated memory
    /// allocation.
    VkDeviceSize AllocationSize() const {
        return sharedMemory_ != VK_NULL_HANDLE ? bytes_ : 0;
    }

    /// For linear images, the bytes from one row's first texel to the
    /// next's, and from the memory's start to the first texel; zero
    /// for others.
    VkDeviceSize RowPitch() const {
        return rowPitch_;
    }
    VkDeviceSize TexelOffset() const {
        return texelOffset_;
    }

    /// Keeps `object`, such as a CUDA import of the image's memory,
    /// until the image is destroyed, releasing it before the memory
    /// is freed.
    void Attach(std::shared_ptr<void> object) {
        attached_ = std::move(object);
    }

    const std::shared_ptr<void>& Attached() const {
        return attached_;
    }

    /// Bytes per texel of the formats Ernie uploads and reads back.
    /// @throws std::invalid_argument for other formats.
    static std::uint32_t TexelBytes(VkFormat format);

    /// Bytes of one 4 x 4 block of a BC1 to BC7 format; 0 for formats
    /// that are not block-compressed.
    static std::uint32_t BlockBytes(VkFormat format);

    /// Tightly packed bytes of a `width` x `height` mip of `format`,
    /// in whole blocks for block-compressed formats.
    /// @throws std::invalid_argument for a format `TexelBytes` and
    /// `BlockBytes` do not know.
    static VkDeviceSize MipBytes(VkFormat format, std::uint32_t width,
                                 std::uint32_t height);

  private:
    void Release();

    const VulkanContext* context_ = nullptr;
    ImageDesc desc_;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkDeviceMemory sharedMemory_ = VK_NULL_HANDLE;
    std::shared_ptr<void> attached_;
    VkImageView view_ = VK_NULL_HANDLE;
    std::vector<VkImageView> layerViews_;
    std::vector<VkImageView> groupViews_;
    std::vector<VkImageView> mipViews_;
    VkImageLayout layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkDeviceSize bytes_ = 0;
    VkDeviceSize rowPitch_ = 0;
    VkDeviceSize texelOffset_ = 0;
};

/// Aspect of an image format: depth for depth formats, colour
/// otherwise.
VkImageAspectFlags AspectOf(VkFormat format);

/// Sampler filtering and addressing.
struct SamplerDesc {
    VkFilter filter = VK_FILTER_LINEAR;
    VkSamplerMipmapMode mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    VkSamplerAddressMode addressMode =
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    float maxAnisotropy = 1.0f;  ///< Above 1 enables anisotropic
                                 ///< filtering where supported.
    bool compare =
        false;  ///< Depth comparison (`VK_COMPARE_OP_LESS`) for
                ///< shadow maps.
    /// Addressing along v (and w) when it differs from
    /// `addressMode`'s along u.
    std::optional<VkSamplerAddressMode> addressModeV;
    /// What `VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER` reads outside
    /// the image.
    VkBorderColor borderColor =
        VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
};

/// An owned `VkSampler`.
class Sampler {
  public:
    Sampler(const VulkanContext& context, const SamplerDesc& desc);
    ~Sampler();
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    VkSampler Handle() const {
        return sampler_;
    }

  private:
    const VulkanContext& context_;
    VkSampler sampler_ = VK_NULL_HANDLE;
};

}  // namespace ernie
