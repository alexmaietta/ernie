// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/ground_drawable.h"

#include "ernie/kernel/embedded_shaders.h"
#include "ernie/raster/minimal_targets.h"
#include "ernie/util/profile.h"

#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie {

namespace {

struct GroundUniforms {
    float invViewProj[16];
    float plane[4];
    float checkerPhase[4];
    float colour0[4];
    float colour1[4];
    float viewport[4];
    std::uint32_t ids[4];
};
static_assert(sizeof(GroundUniforms) == 160, "GroundUniforms must match ground.slang");

/// Each view's uniforms start at a multiple of this, the largest uniform offset alignment.
constexpr VkDeviceSize kUniformsStride = 256;

/// The views' uniforms, which views posed on the GPU patch (`RecordViewPoses`).
constexpr VkBufferUsageFlags kUniformsUsage =
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

/// frame.slang's per-draw push constants, which the pass declares but leaves unused.
constexpr std::uint32_t kDrawConstantsSize = 96;

constexpr VkShaderStageFlags kGraphics = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

/// Frame (0 to 4), material (5 to 9) and ground (10) bindings; see ground.slang.
constexpr DescriptorBinding kBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kGraphics},
    {3, VK_DESCRIPTOR_TYPE_SAMPLER, kGraphics},
    {4, VK_DESCRIPTOR_TYPE_SAMPLER, kGraphics},
    {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kGraphics},
    {7, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kGraphics},
    {8, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kGraphics},
    {9, VK_DESCRIPTOR_TYPE_SAMPLER, kGraphics},
    {10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
};

}  // namespace

GroundDrawable::GroundDrawable(RasterRenderer& renderer)
    : renderer_(renderer), layout_(renderer.Context(), kBindings, kDrawConstantsSize),
      uniforms_(renderer.Context(), kUniformsStride, kUniformsUsage, BufferMemory::Device) {
    CreatePipeline(false);
}

void GroundDrawable::CreatePipeline(bool pushbroom) {
    const std::vector<ShaderModuleSource> modules{{"brdf", std::string(BrdfSlang())},
                                                  ProjectionModule(pushbroom),
                                                  {"frame", shaders::kFrameSlang},
                                                  {"surface", shaders::kSurfaceSlang},
                                                  {"ground", shaders::kGroundSlang}};
    const std::vector<std::uint32_t> vertex =
        renderer_.Compiler().CompileSpirv(modules, "vertexMain");
    const std::vector<std::uint32_t> fragment =
        renderer_.Compiler().CompileSpirv(modules, "fragmentMain");
    GraphicsPipelineDesc desc;
    desc.vertexSpirv = vertex;
    desc.fragmentSpirv = fragment;
    desc.colorFormats.assign(RenderTargets::kGBufferFormats.begin(),
                             RenderTargets::kGBufferFormats.end());
    desc.depthFormat = RenderTargets::kDepthFormat;
    pipelines_[pushbroom ? 1 : 0] =
        std::make_unique<GraphicsPipeline>(renderer_.Context(), layout_, desc);
    // Minimal shading's, for frame cameras only.
    if (!pushbroom && minimalPipelines_[0] == nullptr) {
        const char* const entries[] = {"minimalFragmentMain", "idsFragmentMain"};
        for (std::size_t i = 0; i < minimalPipelines_.size(); ++i) {
            // Kept alive until the pipeline is made: the desc only spans it.
            const std::vector<std::uint32_t> minimalFragment =
                renderer_.Compiler().CompileSpirv(modules, entries[i]);
            GraphicsPipelineDesc minimal;
            minimal.vertexSpirv = vertex;
            minimal.fragmentSpirv = minimalFragment;
            minimal.colorFormats.assign(MinimalTargets::kColorFormats.begin(),
                                        MinimalTargets::kColorFormats.end());
            minimal.depthFormat = MinimalTargets::kDepthFormat;
            minimalPipelines_[i] =
                std::make_unique<GraphicsPipeline>(renderer_.Context(), layout_, minimal);
        }
    }
}

GroundDrawable::~GroundDrawable() {
    vkDeviceWaitIdle(renderer_.Context().Device());
}

void GroundDrawable::SetParams(const GroundParams& params) {
    if (!(params.checkerSize > 0.0f) || !std::isfinite(params.checkerSize)) {
        throw std::invalid_argument("The ground's checker size must be positive and finite");
    }
    params_ = params;
}

void GroundDrawable::Prepare(const FrameParams& params, std::span<const ShadowCascade>) {
    ERNIE_ZONE("Ground prepare");
    prepared_ = false;
    if (!enabled_) {
        return;
    }
    pushbroom_ = params.camera.pushbroom;
    minimal_ = params.minimal;
    if (pipelines_[pushbroom_ ? 1 : 0] == nullptr ||
        (minimal_ != MinimalMode::Off && minimalPipelines_[0] == nullptr)) {
        CreatePipeline(pushbroom_);
    }
    GroundUniforms uniforms{};
    // Relative to the frame's origin, the checkerboard keeping its place in the world.
    uniforms.plane[0] = static_cast<float>(static_cast<double>(params_.height) - origin_.z);
    uniforms.plane[1] = params_.checkerSize;
    uniforms.plane[2] = params_.checker ? 1.0f : 0.0f;
    const glm::dvec2 squares = glm::dvec2(origin_) / static_cast<double>(params_.checkerSize);
    const glm::dvec2 phase = squares - 2.0 * glm::floor(squares / 2.0);
    uniforms.checkerPhase[0] = static_cast<float>(phase.x);
    uniforms.checkerPhase[1] = static_cast<float>(phase.y);
    for (int k = 0; k < 3; ++k) {
        uniforms.colour0[k] = params_.checkerColors[0][k];
        uniforms.colour1[k] = params_.checkerColors[1][k];
    }
    uniforms.viewport[0] = static_cast<float>(params.camera.width);
    uniforms.viewport[1] = static_cast<float>(params.camera.height);
    uniforms.ids[0] = renderer_.HasMaterial(params_.material) ? params_.material
                                                              : RasterRenderer::kDefaultMaterial;
    uniforms.ids[1] = params_.primId;
    const VkDeviceSize bytes = params.Views() * kUniformsStride;
    if (uniforms_.Size() < bytes) {
        vkDeviceWaitIdle(renderer_.Context().Device());
        uniforms_ = GpuBuffer(renderer_.Context(), bytes, kUniformsUsage, BufferMemory::Device);
    }
    for (std::size_t v = 0; v < params.Views(); ++v) {
        const glm::mat4 inverse = glm::inverse(params.View(v).ViewProjection());
        std::memcpy(uniforms.invViewProj, glm::value_ptr(inverse), sizeof(uniforms.invViewProj));
        renderer_.Uploads().Write(uniforms_, std::as_bytes(std::span(&uniforms, 1)),
                                  v * kUniformsStride);
    }
    prepared_ = true;
}

void GroundDrawable::RecordViewPoses(VkCommandBuffer commands) const {
    if (!prepared_) {
        return;
    }
    RasterRenderer::ViewFields fields;
    fields.inverseViewProjection = offsetof(GroundUniforms, invViewProj);
    renderer_.PatchViews(commands, uniforms_, 0, kUniformsStride, fields);
}

void GroundDrawable::RecordColor(VkCommandBuffer commands, const RasterFrame& frame,
                                 std::size_t view, DrawPhase phase) const {
    if (!prepared_ || phase == DrawPhase::Occludees) {
        return;
    }
    std::vector<DescriptorResource> resources = {frame.uniforms, frame.shadows, frame.shadowMap,
                                                 frame.shadowSampler, frame.clampSampler};
    renderer_.MaterialOrDefault(params_.material).AppendResources(resources);
    resources.push_back(
        DescriptorResource::Of(uniforms_, view * kUniformsStride, sizeof(GroundUniforms)));
    const GraphicsPipeline& pipeline =
        minimal_ != MinimalMode::Off
            ? *minimalPipelines_[minimal_ == MinimalMode::None ? 1 : 0]
            : *pipelines_[pushbroom_ ? 1 : 0];
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.Handle());
    layout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, resources);
    vkCmdDraw(commands, 3, 1, 0, 0);
}

}  // namespace ernie
