// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/image.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/session/frame_params.h"
#include "ernie/util/atmosphere.h"

#include <memory>
#include <optional>
#include <vector>

namespace ernie {

/// A planet's atmosphere on the GPU for the rasterizer
/// (atmosphere.slang): its medium, each constituent's coefficients
/// per band and densities by altitude, and the tables built from it,
/// the transmittance to the top of the atmosphere and Hillaire's
/// multiple scattering. They are rebuilt only when the atmosphere's
/// model or the spectral bands change; shading reads them every
/// frame.
///
/// Both tables are float32 arrays with a layer per four bands: their
/// values span many orders of magnitude, and the shaders filter them
/// linearly.
class AtmosphereLuts {
  public:
    AtmosphereLuts(const VulkanContext& context,
                   const ShaderCompiler& compiler);

    /// Sets the atmosphere to draw, or none, for `bands`: builds its
    /// model and medium when they changed (waiting for the device
    /// first, as recorded tables may still read the old medium), and
    /// marks the tables for rebuilding.
    void Set(const std::optional<AtmosphereParams>& params,
             const SpectralBands& bands);

    /// Records the passes rebuilding the tables when they are stale,
    /// leaving them ready for sampling; nothing otherwise.
    void Record(VkCommandBuffer commands) const;

    /// Whether there is an atmosphere.
    bool Active() const {
        return banded_.has_value();
    }

    const AtmosphereParams& Params() const {
        return *params_;
    }

    /// The model averaged over the bands, for an active atmosphere.
    const BandedAtmosphere& Banded() const {
        return *banded_;
    }

    /// The tables, or 1x1 placeholders without an atmosphere.
    const GpuImage& Transmittance() const {
        return *transmittance_;
    }

    const GpuImage& MultipleScattering() const {
        return *multipleScattering_;
    }

    /// atmosphere.slang's medium, densities and phase tables, or
    /// small placeholders without an atmosphere.
    const GpuBuffer& Medium() const {
        return *medium_;
    }

    const GpuBuffer& Densities() const {
        return *densities_;
    }

    const GpuBuffer& Phases() const {
        return *phases_;
    }

    /// The atmosphere's model from its settings (`EarthAtmosphere`,
    /// `MarsAtmosphere`).
    static AtmosphereModel ModelOf(const AtmosphereParams& params);

    /// Relative density samples per constituent, evenly from the
    /// ground to the top.
    static constexpr std::uint32_t kDensitySamples = 1024;

  private:
    void Upload();
    void Placeholders();
    std::unique_ptr<GpuBuffer>
    Buffer(const std::vector<float>& values) const;

    const VulkanContext& context_;
    const ShaderCompiler& compiler_;
    PipelineLayout layout_;
    Sampler sampler_;
    /// Compiled when first recorded.
    mutable std::unique_ptr<ComputePipeline> transmittancePipeline_;
    mutable std::unique_ptr<ComputePipeline> multiplePipeline_;
    std::optional<AtmosphereParams> params_;
    std::optional<BandedAtmosphere> banded_;
    std::unique_ptr<GpuBuffer> medium_;
    std::unique_ptr<GpuBuffer> densities_;
    std::unique_ptr<GpuBuffer> phases_;
    std::unique_ptr<GpuImage> transmittance_;
    std::unique_ptr<GpuImage> multipleScattering_;
    mutable bool stale_ = false;
};

}  // namespace ernie
