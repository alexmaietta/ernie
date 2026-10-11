// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/atmosphere_luts.h"

#include "ernie/device/gpu_profile.h"
#include "ernie/kernel/embedded_shaders.h"
#include "ernie/scene/tokens.h"
#include "ernie/util/profile.h"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace ernie {

namespace {

constexpr VkShaderStageFlags kCompute = VK_SHADER_STAGE_COMPUTE_BIT;

/// atmosphere_luts.slang's bindings.
constexpr DescriptorBinding kBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {4, VK_DESCRIPTOR_TYPE_SAMPLER, kCompute},
    {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
};

/// atmosphere.slang's table sizes.
constexpr std::uint32_t kTransmittanceWidth = 256;
constexpr std::uint32_t kTransmittanceHeight = 64;
constexpr std::uint32_t kMultipleScatteringSize = 32;

constexpr VkBufferUsageFlags kStorage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

std::unique_ptr<GpuImage> TableImage(const VulkanContext& context,
                                     std::uint32_t width,
                                     std::uint32_t height,
                                     std::uint32_t layers) {
    return std::make_unique<GpuImage>(
        context, ImageDesc{width, height, layers, 1,
                           VK_FORMAT_R32G32B32A32_SFLOAT,
                           VK_IMAGE_USAGE_SAMPLED_BIT |
                               VK_IMAGE_USAGE_STORAGE_BIT,
                           false, true});
}

std::uint32_t Groups(std::uint32_t size) {
    return (size + 7) / 8;
}

}  // namespace

AtmosphereLuts::AtmosphereLuts(const VulkanContext& context,
                               const ShaderCompiler& compiler)
    : context_(context), compiler_(compiler),
      layout_(context, kBindings),
      sampler_(context,
               SamplerDesc{VK_FILTER_LINEAR,
                           VK_SAMPLER_MIPMAP_MODE_NEAREST,
                           VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                           1.0f, false}),
      transmittance_(TableImage(context, 1, 1, 1)),
      multipleScattering_(TableImage(context, 1, 1, 1)) {
    // The placeholder tables are never written; shading reads them
    // only without an atmosphere, so their contents do not matter.
    Placeholders();
    stale_ = true;
}

void AtmosphereLuts::Placeholders() {
    const std::vector<float> zeros(8, 0.0f);
    medium_ = Buffer(zeros);
    densities_ = Buffer(zeros);
    phases_ = Buffer(zeros);
}

std::unique_ptr<GpuBuffer>
AtmosphereLuts::Buffer(const std::vector<float>& values) const {
    auto buffer = std::make_unique<GpuBuffer>(
        context_, values.size() * sizeof(float), kStorage,
        BufferMemory::Upload);
    buffer->Write(std::as_bytes(std::span(values)));
    return buffer;
}

AtmosphereModel
AtmosphereLuts::ModelOf(const AtmosphereParams& params) {
    if (params.preset == tokens::kMars) {
        MarsAtmosphereOptions options;
        options.dustOpticalDepth = params.dustOpticalDepth;
        return MarsAtmosphere(params.planetRadius, options);
    }
    if (params.preset == tokens::kEarth) {
        EarthAtmosphereOptions options;
        options.ozoneColumn = params.ozoneColumn;
        options.aerosolOpticalDepth = params.hazeOpticalDepth;
        return EarthAtmosphere(params.planetRadius, options);
    }
    throw std::invalid_argument("No atmosphere preset '" +
                                params.preset + "'");
}

void AtmosphereLuts::Set(
    const std::optional<AtmosphereParams>& params,
    const SpectralBands& bands) {
    const bool sameModel =
        params.has_value() == params_.has_value() &&
        (!params || params->SameModel(*params_)) &&
        (!banded_ || banded_->bands == bands);
    params_ = params;
    if (sameModel) {
        return;
    }
    ERNIE_ZONE("Atmosphere model");
    // Tables recorded before may still be reading the old medium.
    vkDeviceWaitIdle(context_.Device());
    if (!params) {
        banded_.reset();
        Placeholders();
        transmittance_ = TableImage(context_, 1, 1, 1);
        multipleScattering_ = TableImage(context_, 1, 1, 1);
        stale_ = true;
        return;
    }
    banded_ = BandAverage(ModelOf(*params), bands);
    Upload();
    const auto groups =
        static_cast<std::uint32_t>((bands.count + 3) / 4);
    transmittance_ = TableImage(context_, kTransmittanceWidth,
                                kTransmittanceHeight, groups);
    multipleScattering_ =
        TableImage(context_, kMultipleScatteringSize,
                   kMultipleScatteringSize, groups);
    stale_ = true;
}

void AtmosphereLuts::Upload() {
    const BandedAtmosphere& banded = *banded_;
    const auto bandCount =
        static_cast<std::size_t>(banded.bands.count);
    const std::size_t groups = (bandCount + 3) / 4;
    const std::size_t constituents = banded.constituents.size();
    const double top = banded.topAltitude;

    // atmosphere.slang's medium, in kilometres: per constituent its
    // scattering and absorption per band group, then the ground's
    // albedo per band group.
    std::vector<float> medium(
        4 * (2 + 2 * constituents * groups + groups), 0.0f);
    medium[0] = static_cast<float>(banded.bottomRadius * 1e-3);
    medium[1] =
        static_cast<float>((banded.bottomRadius + top) * 1e-3);
    medium[2] = static_cast<float>(constituents);
    medium[3] = static_cast<float>(kDensitySamples);
    medium[4] = static_cast<float>(groups);
    medium[5] = static_cast<float>(bandCount);
    medium[6] = static_cast<float>(banded.phaseAngles);
    for (std::size_t c = 0; c < constituents; ++c) {
        const BandedConstituent& constituent = banded.constituents[c];
        const std::size_t scattering = 4 * (2 + c * 2 * groups);
        const std::size_t absorption = scattering + 4 * groups;
        for (std::size_t b = 0; b < bandCount; ++b) {
            medium[scattering + b] = constituent.scattering[b] * 1e3f;
            medium[absorption + b] = constituent.absorption[b] * 1e3f;
        }
    }
    const std::size_t albedo = 4 * (2 + 2 * constituents * groups);
    for (std::size_t b = 0; b < bandCount; ++b) {
        medium[albedo + b] =
            static_cast<float>(params_->groundAlbedo);
    }

    std::vector<float> densities(constituents * kDensitySamples);
    for (std::size_t c = 0; c < constituents; ++c) {
        for (std::uint32_t i = 0; i < kDensitySamples; ++i) {
            const double altitude = top * i / (kDensitySamples - 1);
            densities[c * kDensitySamples + i] = static_cast<float>(
                banded.constituents[c].density(altitude));
        }
    }

    // Each constituent's phase tables, band-major, back to back.
    std::vector<float> phases;
    for (const BandedConstituent& constituent : banded.constituents) {
        phases.insert(phases.end(), constituent.phase.begin(),
                      constituent.phase.end());
    }

    medium_ = Buffer(medium);
    densities_ = Buffer(densities);
    phases_ = Buffer(phases);
}

void AtmosphereLuts::Record(VkCommandBuffer commands) const {
    if (!stale_) {
        return;
    }
    stale_ = false;
    if (!Active()) {
        transmittance_->Transition(
            commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        multipleScattering_->Transition(
            commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Atmosphere tables");
    if (transmittancePipeline_ == nullptr) {
        const std::vector<ShaderModuleSource> modules{
            {"atmosphere", shaders::kAtmosphereSlang},
            {"atmosphere_luts", shaders::kAtmosphereLutsSlang}};
        transmittancePipeline_ = std::make_unique<ComputePipeline>(
            context_, layout_,
            compiler_.CompileSpirv(modules, "transmittanceMain"));
        multiplePipeline_ = std::make_unique<ComputePipeline>(
            context_, layout_,
            compiler_.CompileSpirv(modules,
                                   "multipleScatteringMain"));
    }
    transmittance_->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    multipleScattering_->Transition(commands,
                                    VK_IMAGE_LAYOUT_GENERAL);
    const DescriptorResource resources[] = {
        DescriptorResource::Of(*medium_),
        DescriptorResource::Of(*densities_),
        DescriptorResource::Of(transmittance_->View(),
                               VK_IMAGE_LAYOUT_GENERAL),
        DescriptorResource::Of(transmittance_->View(),
                               VK_IMAGE_LAYOUT_GENERAL),
        DescriptorResource::Of(sampler_),
        DescriptorResource::Of(multipleScattering_->View(),
                               VK_IMAGE_LAYOUT_GENERAL)};
    const std::uint32_t groups = transmittance_->Desc().layers;

    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      transmittancePipeline_->Handle());
    layout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE, resources);
    vkCmdDispatch(commands, Groups(kTransmittanceWidth),
                  Groups(kTransmittanceHeight), groups);

    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commands, &dependency);

    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      multiplePipeline_->Handle());
    layout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE, resources);
    vkCmdDispatch(commands, Groups(kMultipleScatteringSize),
                  Groups(kMultipleScatteringSize), groups);

    transmittance_->Transition(
        commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    multipleScattering_->Transition(
        commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

}  // namespace ernie
