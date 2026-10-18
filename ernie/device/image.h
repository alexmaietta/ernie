// SPDX-License-Identifier: BSD-3-Clause
/// @file image.h
/// @brief Device images and samplers, described in Ernie's terms.
#pragma once

#include "ernie/device/context.h"

#include <glm/vec3.hpp>

#include <cstdint>
#include <string_view>

namespace ernie {

/// @brief The formats Ernie stores images in.
enum class ImageFormat : std::uint8_t {
    Rgba8Unorm,
    Rgba8Srgb,
    R32Float,
    Rg32Float,
    Rgba16Float,
    Rgba32Float,
    R32Uint,
};

/// @return The bytes of one texel.
std::uint32_t TexelBytes(ImageFormat format);

/// @brief How an image is used, as flags.
enum class ImageUsage : std::uint8_t {
    Sampled = 1u << 0,
    Storage = 1u << 1,
    Attachment = 1u << 2,
};
constexpr ImageUsage operator|(ImageUsage a, ImageUsage b) {
    return static_cast<ImageUsage>(static_cast<std::uint8_t>(a) | static_cast<std::uint8_t>(b));
}
constexpr bool Any(ImageUsage usage, ImageUsage flags) {
    return (static_cast<std::uint8_t>(usage) & static_cast<std::uint8_t>(flags)) != 0;
}

/// @brief An image: 2D, or 3D when its extent has depth.
struct ImageDescription {
    ImageFormat format = ImageFormat::Rgba8Unorm;
    glm::uvec3 extent{1, 1, 1};
    std::uint32_t mips = 1;
    ImageUsage usage = ImageUsage::Sampled;
};

/// @brief An image, its memory and a view of it all. Images stay in the general layout, which
/// every use accepts; the frame graph will choose layouts per pass. Destroying one the GPU may
/// still use is an error: retire it instead.
class DeviceImage {
  public:
    /// @throws DeviceError when the memory cannot be allocated or the format is unsupported.
    DeviceImage(DeviceContext& context, const ImageDescription& description, std::string_view name);
    ~DeviceImage();
    DeviceImage(const DeviceImage&) = delete;
    DeviceImage& operator=(const DeviceImage&) = delete;

    const ImageDescription& Description() const {
        return description_;
    }
    VkImage Handle() const {
        return image_;
    }
    VkImageView View() const {
        return view_;
    }
    /// @return The ticket of the transition into the general layout, which first uses wait on.
    Ticket Ready() const {
        return ready_;
    }

  private:
    DeviceContext& context_;
    ImageDescription description_;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    Ticket ready_;
};

enum class Filter : std::uint8_t { Nearest, Linear };
enum class Address : std::uint8_t { Repeat, Clamp, Mirror };

struct SamplerDescription {
    Filter filter = Filter::Linear;
    Address address = Address::Repeat;
    bool operator==(const SamplerDescription&) const = default;
};

/// @brief A sampler.
class Sampler {
  public:
    Sampler(DeviceContext& context, const SamplerDescription& description, std::string_view name);
    ~Sampler();
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    VkSampler Handle() const {
        return sampler_;
    }

  private:
    VkDevice device_;
    VkSampler sampler_ = VK_NULL_HANDLE;
};

/// @return The Vulkan format of an Ernie format.
VkFormat ToVkFormat(ImageFormat format);

}  // namespace ernie
