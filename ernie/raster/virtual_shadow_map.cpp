// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/virtual_shadow_map.h"

#include "ernie/device/gpu_profile.h"
#include "ernie/kernel/embedded_shaders.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

constexpr VkShaderStageFlags kCompute = VK_SHADER_STAGE_COMPUTE_BIT;

/// virtual_shadow.slang's bindings.
constexpr DescriptorBinding kBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kCompute},
    {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
};

/// virtual_shadow.slang's `VirtualShadow`.
struct BlockGpu {
    VkDeviceAddress physical;
    VkDeviceAddress pageTable;
    VkDeviceAddress requests;
    VkDeviceAddress counter;
    VkDeviceAddress levels;
    std::uint32_t pages;
    std::uint32_t levelCount;
    std::int32_t firstLevel;
    float bias;
    VkDeviceAddress rects;
    std::uint32_t filterLevels;
    std::uint32_t padding;
};
static_assert(sizeof(BlockGpu) == 72,
              "BlockGpu must match virtual_shadow.slang");

/// virtual_shadow.slang's `VirtualLevel`.
struct LevelGpu {
    glm::mat4 viewProjection;
    glm::vec4 info;
};
static_assert(sizeof(LevelGpu) == 80,
              "LevelGpu must match virtual_shadow.slang");

/// A light-space frame: z towards the light, x and y across it.
struct LightFrame {
    glm::dvec3 x;
    glm::dvec3 y;
    glm::dvec3 z;
};

LightFrame FrameOf(const glm::dvec3& toLight) {
    LightFrame frame;
    frame.z = glm::normalize(toLight);
    const glm::dvec3 up = std::abs(frame.z.z) > 0.99
                              ? glm::dvec3(0.0, 1.0, 0.0)
                              : glm::dvec3(0.0, 0.0, 1.0);
    frame.x = glm::normalize(glm::cross(up, frame.z));
    frame.y = glm::cross(frame.z, frame.x);
    return frame;
}

std::uint32_t Groups(std::uint32_t size) {
    return (size + 7) / 8;
}

/// Fills before the compute that reads them.
void TransferBarrier(VkCommandBuffer commands) {
    VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memory.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    memory.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memory.dstAccessMask =
        VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    vkCmdPipelineBarrier2(commands, &dependency);
}

}  // namespace

VirtualShadowMap::VirtualShadowMap(const VulkanContext& context,
                                   const ShaderCompiler& compiler,
                                   std::int32_t firstLevel,
                                   std::int32_t lastLevel,
                                   std::uint32_t pages)
    : context_(context), compiler_(compiler), firstLevel_(firstLevel),
      layout_(context, kBindings, sizeof(Constants)) {
    if (lastLevel < firstLevel ||
        static_cast<std::uint32_t>(lastLevel - firstLevel + 1) >
            kMaxLevels ||
        pages == 0 || pages > (1u << 20)) {
        throw std::invalid_argument("Virtual shadow maps need 1 to " +
                                    std::to_string(kMaxLevels) +
                                    " levels and 1 to 2^20 pages");
    }
    levelCount_ =
        static_cast<std::uint32_t>(lastLevel - firstLevel + 1);
    pages_ = pages;
    constexpr VkBufferUsageFlags kStorage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    const VkDeviceSize pageTexels = kPageTexels * kPageTexels;
    const VkDeviceSize entries =
        VkDeviceSize{kPagesAcross} * kPagesAcross * levelCount_;
    pool_ = std::make_unique<GpuBuffer>(
        context, pages * pageTexels * sizeof(std::uint32_t), kStorage,
        BufferMemory::Device);
    pageTable_ = std::make_unique<GpuBuffer>(
        context, entries * sizeof(std::uint32_t), kStorage,
        BufferMemory::Device);
    requests_ = std::make_unique<GpuBuffer>(
        context, entries * sizeof(std::uint32_t), kStorage,
        BufferMemory::Device);
    levels_ = std::make_unique<GpuBuffer>(
        context, levelCount_ * sizeof(LevelGpu), kStorage,
        BufferMemory::Upload);
    counter_ =
        std::make_unique<GpuBuffer>(context, sizeof(std::uint32_t),
                                    kStorage, BufferMemory::Readback);
    rects_ = std::make_unique<GpuBuffer>(
        context, levelCount_ * 4 * sizeof(std::uint32_t), kStorage,
        BufferMemory::Device);
    block_ = std::make_unique<GpuBuffer>(
        context, sizeof(BlockGpu),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | kStorage,
        BufferMemory::Upload);
    noDepth_ = std::make_unique<GpuImage>(
        context, ImageDesc{1, 1, 1, 1, VK_FORMAT_R32_SFLOAT,
                           VK_IMAGE_USAGE_SAMPLED_BIT});
    volumeBoxes_ = std::make_unique<GpuBuffer>(
        context, kMaxVolumeBoxes * 2 * sizeof(glm::vec4), kStorage,
        BufferMemory::Upload);
    SetLevels(FitLevels(glm::dvec3(0.0, 0.0, 1.0), glm::dvec3(0.0),
                        glm::dvec3(0.0), firstLevel, lastLevel),
              0.0f);
}

std::vector<VirtualShadowLevel> VirtualShadowMap::FitLevels(
    const glm::dvec3& toLight, const glm::dvec3& eye,
    const glm::dvec3& origin, std::int32_t firstLevel,
    std::int32_t lastLevel) {
    const LightFrame frame = FrameOf(toLight);
    // The camera in light space, from world coordinates, so the
    // snapped centres do not move with the origin.
    const glm::dvec3 camera(glm::dot(eye, frame.x),
                            glm::dot(eye, frame.y),
                            glm::dot(eye, frame.z));
    std::vector<VirtualShadowLevel> levels;
    for (std::int32_t level = firstLevel; level <= lastLevel;
         ++level) {
        VirtualShadowLevel out;
        out.level = level;
        out.radius = std::ldexp(1.0, level + 1);
        const double r = out.radius;
        out.texelWorldSize =
            4.0 * r / static_cast<double>(kVirtualTexels);
        // Centred on the camera in light space: x and y snapped to
        // the radius (whole pages: a radius is 4096 texels), z at its
        // depth, with +-1000 R of depth about it, as Unreal's
        // ZRangeScale; casters nearer the light clamp onto the near
        // plane (`GraphicsPipelineDesc::depthClamp`).
        const glm::dvec2 centre =
            glm::round(glm::dvec2(camera) / r) * r;
        const double zRange = 1000.0 * r;
        // Light space relative to the origin: p' = M (p - origin).
        const glm::dvec3 offset(glm::dot(origin, frame.x) - centre.x,
                                glm::dot(origin, frame.y) - centre.y,
                                glm::dot(origin, frame.z) - camera.z);
        glm::dmat4 m(0.0);
        const double sxy = 1.0 / (2.0 * r);
        const double sz = 0.5 / zRange;
        for (int column = 0; column < 3; ++column) {
            m[column][0] = frame.x[column] * sxy;
            m[column][1] = frame.y[column] * sxy;
            m[column][2] = frame.z[column] * sz;
        }
        m[3][0] = offset.x * sxy;
        m[3][1] = offset.y * sxy;
        m[3][2] = offset.z * sz + 0.5;
        m[3][3] = 1.0;
        out.viewProjection = glm::mat4(m);
        levels.push_back(out);
    }
    return levels;
}

std::int32_t VirtualShadowMap::LevelFor(double distance,
                                        double bias) {
    return static_cast<std::int32_t>(
        std::floor(std::log2(std::max(distance, 1e-6)) + bias));
}

void VirtualShadowMap::SetLevels(
    std::span<const VirtualShadowLevel> levels, float bias,
    FrameUploads* uploads, std::uint32_t filterLevels) {
    if (levels.size() != levelCount_) {
        throw std::invalid_argument(
            "Virtual shadow maps need one fit per level");
    }
    std::vector<LevelGpu> gpu;
    for (const VirtualShadowLevel& level : levels) {
        gpu.push_back(
            {level.viewProjection,
             glm::vec4(static_cast<float>(level.texelWorldSize),
                       static_cast<float>(level.radius),
                       static_cast<float>(level.level), 0.0f)});
    }
    const auto write = [uploads](const GpuBuffer& buffer,
                                 std::span<const std::byte> bytes) {
        if (uploads != nullptr) {
            uploads->Write(buffer, bytes);
        } else {
            const_cast<GpuBuffer&>(buffer).Write(bytes);
        }
    };
    write(*levels_, std::as_bytes(std::span(gpu)));
    const BlockGpu block{pool_->Address(),
                         pageTable_->Address(),
                         requests_->Address(),
                         counter_->Address(),
                         levels_->Address(),
                         pages_,
                         levelCount_,
                         firstLevel_,
                         bias,
                         rects_->Address(),
                         std::min(filterLevels, kMaxFilterLevels),
                         0};
    write(*block_, std::as_bytes(std::span(&block, 1)));
}

std::optional<glm::mat4> VirtualShadowMap::CasterCull(
    const VirtualShadowLevel& level,
    const glm::mat4& inverseViewProjection, const glm::vec3& eye,
    float bias, std::uint32_t filterLevels, bool first, bool last) {
    // Its receivers' distances, a little wider for the shaders' float
    // rounding of the level.
    const double near =
        first ? 0.0
              : 0.99 * std::exp2(level.level -
                                 static_cast<double>(filterLevels) -
                                 bias);
    const double far = 1.01 * std::exp2(level.level + 1.0 - bias);
    const auto unproject = [&](float x, float y, float z) {
        const glm::vec4 h =
            inverseViewProjection * glm::vec4(x, y, z, 1.0f);
        return glm::dvec3(h) / static_cast<double>(h.w);
    };
    // The frustum between view depths `near` (as a depth: the nearest
    // a point that far can be along the view) and `far`, which holds
    // every point of it at those distances: its corners along the
    // edges from the near plane.
    const glm::dvec3 from(eye);
    const glm::dvec3 forward = glm::normalize(
        unproject(0.0f, 0.0f, 0.5f) - unproject(0.0f, 0.0f, 1.0f));
    glm::dvec3 corners[4];
    glm::dvec3 rays[4];
    double slant = 1.0;
    for (int i = 0; i < 4; ++i) {
        const float x = (i & 1) != 0 ? 1.0f : -1.0f;
        const float y = (i & 2) != 0 ? 1.0f : -1.0f;
        corners[i] = unproject(x, y, 1.0f);
        rays[i] = glm::normalize(unproject(x, y, 0.5f) - corners[i]);
        slant = std::min(slant, glm::dot(rays[i], forward));
    }
    // Parallel rays (an orthographic camera) keep every depth from
    // the near plane.
    const double nearDepth = slant < 0.9999 ? near * slant : 0.0;
    glm::dvec2 low(std::numeric_limits<double>::infinity());
    glm::dvec2 high(-std::numeric_limits<double>::infinity());
    for (int i = 0; i < 4; ++i) {
        const double start = glm::dot(corners[i] - from, forward);
        const double along = glm::dot(rays[i], forward);
        if (far < start) {
            continue;
        }
        for (const double depth : {std::max(nearDepth, start), far}) {
            const glm::dvec3 point =
                corners[i] + rays[i] * ((depth - start) / along);
            const glm::dvec4 clip = glm::dmat4(level.viewProjection) *
                                    glm::dvec4(point, 1.0);
            low = glm::min(low, glm::dvec2(clip));
            high = glm::max(high, glm::dvec2(clip));
        }
    }
    // The coarsest level's four pages about the camera, which every
    // frame marks (`markMain`).
    const double page = 2.0 / kPagesAcross;
    if (last) {
        low = glm::min(low, glm::dvec2(-page));
        high = glm::max(high, glm::dvec2(page));
    }
    // A page beside, clip space's 2 over 128 pages.
    low = glm::max(low - page, glm::dvec2(-1.0));
    high = glm::min(high + page, glm::dvec2(1.0));
    if (!(low.x < high.x && low.y < high.y)) {
        return std::nullopt;
    }
    // x and y of the box to [-1, 1], z as it was.
    glm::dmat4 narrow(1.0);
    narrow[0][0] = 2.0 / (high.x - low.x);
    narrow[1][1] = 2.0 / (high.y - low.y);
    narrow[3][0] = -(high.x + low.x) / (high.x - low.x);
    narrow[3][1] = -(high.y + low.y) / (high.y - low.y);
    return glm::mat4(narrow * glm::dmat4(level.viewProjection));
}

std::uint32_t VirtualShadowMap::FilterLevelsFor(double tanRadius) {
    const double texels = 4096.0 * tanRadius;
    if (!(texels > 64.0)) {
        return 0;
    }
    return std::min(static_cast<std::uint32_t>(
                        std::ceil(std::log2(texels / 64.0))),
                    kMaxFilterLevels);
}

void VirtualShadowMap::SetVolumes(
    std::span<const std::pair<glm::vec3, glm::vec3>> boxes,
    FrameUploads* uploads) {
    std::vector<glm::vec4> corners;
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        const auto& [low, high] = boxes[i];
        if (i < kMaxVolumeBoxes) {
            corners.emplace_back(low, 0.0f);
            corners.emplace_back(high, 0.0f);
        } else {
            glm::vec4& last = corners[corners.size() - 2];
            last = glm::min(last, glm::vec4(low, 0.0f));
            corners.back() =
                glm::max(corners.back(), glm::vec4(high, 0.0f));
        }
    }
    volumeBoxCount_ = static_cast<std::uint32_t>(corners.size() / 2);
    if (corners.empty()) {
        return;
    }
    if (uploads != nullptr) {
        uploads->Write(*volumeBoxes_,
                       std::as_bytes(std::span(corners)));
    } else {
        volumeBoxes_->Write(std::as_bytes(std::span(corners)));
    }
}

void VirtualShadowMap::CreatePipelines() const {
    if (markPipeline_ != nullptr) {
        return;
    }
    const std::vector<ShaderModuleSource> modules{
        {"virtual_shadow", shaders::kVirtualShadowSlang}};
    markPipeline_ = std::make_unique<ComputePipeline>(
        context_, layout_,
        compiler_.CompileSpirv(modules, "markMain"));
    markVolumesPipeline_ = std::make_unique<ComputePipeline>(
        context_, layout_,
        compiler_.CompileSpirv(modules, "markVolumesMain"));
    allocatePipeline_ = std::make_unique<ComputePipeline>(
        context_, layout_,
        compiler_.CompileSpirv(modules, "allocateMain"));
    clearPipeline_ = std::make_unique<ComputePipeline>(
        context_, layout_,
        compiler_.CompileSpirv(modules, "clearMain"));
}

void VirtualShadowMap::Push(VkCommandBuffer commands,
                            VkImageView depth,
                            const Constants& constants) const {
    const DescriptorResource resources[] = {
        DescriptorResource::Of(*block_),
        DescriptorResource::Of(depth)};
    layout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE, resources);
    layout_.PushConstants(commands, &constants, sizeof(constants));
}

void VirtualShadowMap::Prepare(VkCommandBuffer commands) const {
    CreatePipelines();
    noDepth_->Transition(commands,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    // The flags start clear; `allocateMain` clears them after.
    if (!cleared_) {
        vkCmdFillBuffer(commands, requests_->Handle(), 0,
                        VK_WHOLE_SIZE, 0);
        TransferBarrier(commands);
        cleared_ = true;
    }
}

VirtualShadowMap::Constants
VirtualShadowMap::ConstantsOf(std::uint32_t width,
                              std::uint32_t height,
                              const glm::mat4& inverseViewProjection,
                              const glm::vec3& eye) const {
    static_assert(sizeof(Constants) == 112,
                  "Constants must match virtual_shadow.slang");
    Constants constants{};
    std::memcpy(constants.inverseViewProjection,
                &inverseViewProjection[0][0],
                sizeof(constants.inverseViewProjection));
    constants.eye[0] = eye.x;
    constants.eye[1] = eye.y;
    constants.eye[2] = eye.z;
    constants.sizes[0] = width;
    constants.sizes[1] = height;
    const VkDeviceAddress boxes = volumeBoxes_->Address();
    constants.volumes[0] = static_cast<std::uint32_t>(boxes);
    constants.volumes[1] = static_cast<std::uint32_t>(boxes >> 32);
    constants.volumes[2] = volumeBoxCount_;
    return constants;
}

void VirtualShadowMap::RecordMark(
    VkCommandBuffer commands, VkImageView depth, std::uint32_t width,
    std::uint32_t height, const glm::mat4& inverseViewProjection,
    const glm::vec3& eye) const {
    ERNIE_GPU_ZONE(context_, commands, "Virtual shadow marking");
    Prepare(commands);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      markPipeline_->Handle());
    Push(commands, depth,
         ConstantsOf(width, height, inverseViewProjection, eye));
    vkCmdDispatch(commands, Groups((width + 1) / 2),
                  Groups((height + 1) / 2), 1);
    if (volumeBoxCount_ > 0) {
        // Writing the same flags, in any order.
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          markVolumesPipeline_->Handle());
        vkCmdDispatch(commands, Groups((width + 7) / 8),
                      Groups((height + 7) / 8), 1);
    }
    ComputeBarrier(commands);
}

void VirtualShadowMap::RecordAllocate(
    VkCommandBuffer commands) const {
    ERNIE_GPU_ZONE(context_, commands, "Virtual shadow allocation");
    Prepare(commands);
    vkCmdFillBuffer(commands, counter_->Handle(), 0,
                    sizeof(std::uint32_t), 0);
    vkCmdFillBuffer(commands, rects_->Handle(), 0, VK_WHOLE_SIZE,
                    ~0u);
    TransferBarrier(commands);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      allocatePipeline_->Handle());
    Push(commands, noDepth_->View(),
         ConstantsOf(1, 1, glm::mat4(1.0f), glm::vec3(0.0f)));
    vkCmdDispatch(commands, Groups(kPagesAcross),
                  Groups(kPagesAcross), levelCount_);
    ComputeBarrier(commands);
}

void VirtualShadowMap::RecordClear(VkCommandBuffer commands) const {
    ERNIE_GPU_ZONE(context_, commands, "Virtual shadow clear");
    Prepare(commands);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      clearPipeline_->Handle());
    Push(commands, noDepth_->View(),
         ConstantsOf(1, 1, glm::mat4(1.0f), glm::vec3(0.0f)));
    vkCmdDispatch(commands, Groups(kPageTexels), Groups(kPageTexels),
                  pages_);
    ComputeBarrier(commands);
}

}  // namespace ernie
