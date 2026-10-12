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
/// the transmittance to the top of the atmosphere, Hillaire's
/// multiple scattering and the sky's irradiance on level ground. They
/// are rebuilt only when the atmosphere's model or the spectral bands
/// change; shading reads them every frame.
///
/// The tables are float32 arrays with a layer per four bands: their
/// values span many orders of magnitude, and the shaders filter them
/// linearly.
class AtmosphereLuts {
  public:
    AtmosphereLuts(const VulkanContext& context,
                   const ShaderCompiler& compiler);

    /// Sets the atmosphere to draw, or none, for `bands` and a sun of
    /// angular diameter `sunAngle` degrees: builds its model and
    /// medium when they changed (waiting for the device first, as
    /// recorded tables may still read the old medium), and marks the
    /// tables for rebuilding; the planet's shadow alone when only the
    /// sun's size changed.
    void Set(const std::optional<AtmosphereParams>& params,
             const SpectralBands& bands, float sunAngle = 0.53f);

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

    /// The sky's irradiance on level ground per unit sun irradiance,
    /// by sun zenith cosine and height (atmosphere.slang's
    /// `skyIrradiance`).
    const GpuImage& SkyIrradiance() const {
        return *skyIrradiance_;
    }

    /// The bending (radians) of rays leaving the atmosphere, by
    /// apparent zenith angle and height (atmosphere.slang's
    /// `refractionDeviation`).
    const GpuImage& Refraction() const {
        return *refraction_;
    }

    /// The fraction of the sun's light reaching points behind the
    /// planet, through its atmosphere and past it, by distance and
    /// separation from the sun (atmosphere.slang's `planetShadow`):
    /// the transmittance table's image, in the layers after its own.
    const GpuImage& PlanetShadow() const {
        return *transmittance_;
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
    /// Writes the shadow settings from the sun's size, the clouds and
    /// the bands.
    void ShadowSettings();
    std::unique_ptr<GpuBuffer>
    Buffer(const std::vector<float>& values) const;

    const VulkanContext& context_;
    const ShaderCompiler& compiler_;
    PipelineLayout layout_;
    Sampler sampler_;
    /// Compiled when first recorded.
    mutable std::unique_ptr<ComputePipeline> transmittancePipeline_;
    mutable std::unique_ptr<ComputePipeline> multiplePipeline_;
    mutable std::unique_ptr<ComputePipeline> irradiancePipeline_;
    mutable std::unique_ptr<ComputePipeline> refractionPipeline_;
    mutable std::unique_ptr<ComputePipeline> shadowPipeline_;
    std::optional<AtmosphereParams> params_;
    std::optional<BandedAtmosphere> banded_;
    std::unique_ptr<GpuBuffer> medium_;
    std::unique_ptr<GpuBuffer> densities_;
    std::unique_ptr<GpuBuffer> phases_;
    std::unique_ptr<GpuImage> transmittance_;
    std::unique_ptr<GpuImage> multipleScattering_;
    std::unique_ptr<GpuImage> skyIrradiance_;
    std::unique_ptr<GpuImage> refraction_;
    /// atmosphere_luts.slang's shadow settings.
    std::unique_ptr<GpuBuffer> shadowSettings_;
    /// The buffers' addresses, which the tables' passes read them by.
    mutable std::unique_ptr<GpuBuffer> addresses_;
    float sunAngle_ = 0.0f;
    mutable bool stale_ = false;
    /// Whether the planet's shadow alone needs rebuilding.
    mutable bool shadowStale_ = false;
};

}  // namespace ernie
