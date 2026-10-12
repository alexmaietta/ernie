// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/pipeline.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/device/vulkan_context.h"
#include "ernie/util/texture_bake.h"

#include <memory>

namespace ernie {

/// BC7 encoding on the GPU (`BlockEncoder`), a thread a block
/// (texture/bc7.slang): bc7enc's fits of modes 1, 6 and 7 at its
/// uber level 1, with four partitions fit where bc7enc fits one. It
/// encodes a 4096 x 4096 image in about a tenth of a second (the
/// CPU's bc7enc about 1.3 s), decoding as close to the source as the
/// CPU's and DirectXTex's encoders or closer.
///
/// Reference: R. Geldreich, "bc7enc_rdo" (bc7enc.cpp), MIT or public
/// domain. https://github.com/richgel999/bc7enc_rdo
class GpuBlockEncoder : public BlockEncoder {
  public:
    /// Its pipeline compiled by `compiler`; `context` must outlive
    /// it.
    /// @throws std::runtime_error if its pipeline cannot be made.
    GpuBlockEncoder(const VulkanContext& context,
                    const ShaderCompiler& compiler);
    ~GpuBlockEncoder() override;
    GpuBlockEncoder(const GpuBlockEncoder&) = delete;
    GpuBlockEncoder& operator=(const GpuBlockEncoder&) = delete;

    bool Encodes(TextureFormat format) const override {
        return format == TextureFormat::Bc7 ||
               format == TextureFormat::Bc7Srgb;
    }

    /// Encodes on the device, waiting for it.
    /// @throws std::invalid_argument for a format it does not encode.
    void Encode(std::span<const std::uint8_t> rgba,
                std::uint32_t width, std::uint32_t height,
                TextureFormat format, std::byte* out) override;

  private:
    const VulkanContext& context_;
    std::unique_ptr<ComputePipeline> pipeline_;
};

}  // namespace ernie
