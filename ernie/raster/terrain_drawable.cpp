// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/terrain_drawable.h"

#include "ernie/device/gpu_profile.h"
#include "ernie/kernel/embedded_shaders.h"
#include "ernie/util/profile.h"
#include "ernie/util/raster_checks.h"

#include <glm/gtc/packing.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ernie {

namespace {

constexpr std::size_t kInitialPatchCapacity = 256;

/// `frustum` with its points moved by `offset`: a point p is inside
/// the result where p - offset was inside `frustum`, as from
/// origin-relative to terrain coordinates.
Frustum Shifted(Frustum frustum, const glm::vec3& offset) {
    for (glm::vec4& plane : frustum.planes) {
        plane.w -= glm::dot(glm::vec3(plane), offset);
    }
    return frustum;
}

struct TerrainUniforms {
    float grid[4];
    float heightRange[4];
    std::uint32_t dims[4];
    float lodEye[4];
    std::uint32_t lodInfo[4];
    float morph[LodQuadtree::kMaxLevels * 2];
    std::uint32_t ids[4];
    float hillshade[4];
    /// The heights' and gradients' bindless heap slots; unused.
    std::uint32_t textures[4];
};
static_assert(sizeof(TerrainUniforms) == 256,
              "TerrainUniforms must match terrain.slang");
static_assert(sizeof(LodPatch) == 16,
              "LodPatch must match the shaders' uint4");

/// Each view's uniforms start at a multiple of this, the largest
/// uniform offset alignment.
constexpr VkDeviceSize kUniformsStride = 256;

struct DrawConstants {
    float lightViewProj[16];
    std::uint32_t first;
    std::uint32_t instanceOffset;
    std::uint32_t indexOffset;
    std::uint32_t records;
    std::uint32_t shadowFromFrame;
    std::uint32_t view;
    std::uint32_t shadowLayer;
    std::uint32_t padding;
};
static_assert(sizeof(DrawConstants) == 96,
              "DrawConstants must match frame.slang");

constexpr VkShaderStageFlags kGraphics =
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

/// Frame (0 and 1), material (5) and terrain (10 its uniforms, which
/// name its heights and gradients in the bindless heap; 12 its LOD
/// patches) bindings; see terrain.slang.
constexpr DescriptorBinding kBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
};

std::vector<ShaderModuleSource>
TerrainModules(bool pushbroom = false) {
    return {{"brdf", std::string(BrdfSlang())},
            ProjectionModule(pushbroom),
            {"bindless", shaders::kBindlessSlang},
            {"frame", shaders::kFrameSlang},
            {"terrain", shaders::kTerrainSlang},
            {"gbuffer", shaders::kGBufferSlang},
            {"surface", shaders::kSurfaceSlang}};
}

// Central-difference gradients (dz/dx east, dz/dy north) per cell,
// clamping at the raster's edges, and their mip chain as packed half
// floats: each level averages 2 x 2 of the one before (odd sizes
// repeat their last row or column), so a distant pixel shades by the
// mean slope it covers instead of one cell's, which aliased into
// salt-and-pepper speckle on fine heightfields.
/// terrain_update.slang's bindings: heights, the gradient mip written
/// and the one below, leaves.
constexpr DescriptorBinding kUpdateBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

/// terrain_horizon.slang's bindings: the heights, and the horizon map
/// to write.
constexpr DescriptorBinding kHorizonBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

/// terrain_sun_horizon.slang's bindings: the finer level of the
/// heights' maxima (or the heights), the level to write, every level
/// of the maxima, and the sun horizon map's layer to write.
constexpr DescriptorBinding kSunHorizonBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

/// terrain_sun_horizon.slang's constants.
struct SunHorizonConstants {
    std::uint32_t size[4];
    float values[4];
    float sun[4];
};

/// terrain_horizon.slang's constants.
struct HorizonConstants {
    std::uint32_t size[4];
    float values[4];
    float levels[4];
    float sky[4];
    std::uint32_t skyInfo[4];
};

/// The heights above the ground the horizon map's levels see from, in
/// cells: the ground, then heights for what stands on the terrain
/// (terrain_horizon.slang).
constexpr std::array<float, 4> kHorizonLevelCells{0.0f, 2.0f, 8.0f,
                                                  32.0f};

/// The horizon map's layers: two per level, then two for the
/// ground-level horizon over what stands on the terrain
/// (`horizonCastersMain`).
constexpr auto kHorizonLayers =
    static_cast<std::uint32_t>(2 * kHorizonLevelCells.size() + 2);

/// terrain_update.slang's constants.
struct UpdateConstants {
    std::uint32_t rect[4];
    std::uint32_t size[4];
    float values[4];
};

std::vector<std::vector<std::uint32_t>>
PackedGradientMips(const RasterView& raster) {
    std::size_t w = raster.width;
    std::size_t h = raster.height;
    const auto scale = static_cast<float>(0.5 / raster.cellSize);
    std::vector<glm::vec2> level(w * h);
    for (std::size_t row = 0; row < h; ++row) {
        const std::size_t north = row > 0 ? row - 1 : row;
        const std::size_t south = row + 1 < h ? row + 1 : row;
        for (std::size_t col = 0; col < w; ++col) {
            const std::size_t west = col > 0 ? col - 1 : col;
            const std::size_t east = col + 1 < w ? col + 1 : col;
            level[row * w + col] = glm::vec2(
                (raster(east, row) - raster(west, row)) * scale,
                (raster(col, north) - raster(col, south)) * scale);
        }
    }
    std::vector<std::vector<std::uint32_t>> mips;
    while (true) {
        std::vector<std::uint32_t> packed(level.size());
        for (std::size_t i = 0; i < level.size(); ++i) {
            packed[i] = glm::packHalf2x16(level[i]);
        }
        mips.push_back(std::move(packed));
        if (w == 1 && h == 1) {
            return mips;
        }
        const std::size_t nw = std::max<std::size_t>(w / 2, 1);
        const std::size_t nh = std::max<std::size_t>(h / 2, 1);
        std::vector<glm::vec2> next(nw * nh);
        for (std::size_t row = 0; row < nh; ++row) {
            const std::size_t r0 = std::min(2 * row, h - 1);
            const std::size_t r1 = std::min(2 * row + 1, h - 1);
            for (std::size_t col = 0; col < nw; ++col) {
                const std::size_t c0 = std::min(2 * col, w - 1);
                const std::size_t c1 = std::min(2 * col + 1, w - 1);
                next[row * nw + col] =
                    0.25f * (level[r0 * w + c0] + level[r0 * w + c1] +
                             level[r1 * w + c0] + level[r1 * w + c1]);
            }
        }
        level = std::move(next);
        w = nw;
        h = nh;
    }
}

}  // namespace

TerrainDrawable::TerrainDrawable(RasterRenderer& renderer)
    : renderer_(renderer),
      layout_(renderer.Context(), kBindings, sizeof(DrawConstants)),
      uniforms_(renderer.Context(), kUniformsStride,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                BufferMemory::Device),
      updateLayout_(renderer.Context(), kUpdateBindings,
                    sizeof(UpdateConstants)),
      horizonLayout_(renderer.Context(), kHorizonBindings,
                     sizeof(HorizonConstants)),
      sunHorizonLayout_(renderer.Context(), kSunHorizonBindings,
                        sizeof(SunHorizonConstants)) {
    const std::vector<ShaderModuleSource> update{
        {"terrain_update", shaders::kTerrainUpdateSlang}};
    const auto compile = [&](const char* entry) {
        return std::make_unique<ComputePipeline>(
            renderer_.Context(), updateLayout_,
            renderer_.Compiler().CompileSpirv(update, entry));
    };
    gradientsPipeline_ = compile("gradientsMain");
    gradientMipPipeline_ = compile("gradientMipMain");
    leafRangePipeline_ = compile("leafRangeMain");
    std::vector<ShaderModuleSource> modules = TerrainModules();
    modules.push_back(
        {"terrain_shadow", shaders::kTerrainShadowSlang});
    const std::vector<std::uint32_t> shadowVertex =
        renderer_.Compiler().CompileSpirv(modules, "shadowVertex");
    GraphicsPipelineDesc shadow;
    shadow.vertexSpirv = shadowVertex;
    shadow.depthFormat = VK_FORMAT_D32_SFLOAT;
    shadow.depthBias = true;
    shadow.depthClamp = true;
    shadowPipeline_ = std::make_unique<GraphicsPipeline>(
        renderer_.Context(), layout_, shadow);
    CreateGBufferPipeline(false);
    ReservePatches(kInitialPatchCapacity);
}

TerrainDrawable::~TerrainDrawable() {
    vkDeviceWaitIdle(renderer_.Context().Device());
}

void TerrainDrawable::CreateGBufferPipeline(bool pushbroom) {
    std::vector<ShaderModuleSource> modules =
        TerrainModules(pushbroom);
    modules.push_back({"terrain_color", shaders::kTerrainColorSlang});
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
    desc.depthCompare = RenderTargets::kDepthCompare;
    gbufferPipelines_[pushbroom ? 1 : 0] =
        std::make_unique<GraphicsPipeline>(renderer_.Context(),
                                           layout_, desc);
}

void TerrainDrawable::ReservePatches(std::size_t count) {
    if (patchBuffer_ != nullptr &&
        count * sizeof(LodPatch) <= patchBuffer_->Size()) {
        return;
    }
    const std::size_t capacity = std::max(
        count, patchBuffer_ != nullptr
                   ? static_cast<std::size_t>(
                         2 * patchBuffer_->Size() / sizeof(LodPatch))
                   : std::size_t{0});
    vkDeviceWaitIdle(renderer_.Context().Device());
    patchBuffer_ = std::make_unique<GpuBuffer>(
        renderer_.Context(), capacity * sizeof(LodPatch),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
}

void TerrainDrawable::ClearHeightfield() {
    vkDeviceWaitIdle(renderer_.Context().Device());
    heights_.reset();
    gradients_.reset();
    tree_ = LodQuadtree();
    patches_.clear();
    mainRanges_.clear();
    cascadeRanges_.clear();
    triangles_ = 0;
    prepared_ = false;
    width_ = 0;
    height_ = 0;
    deviceHeights_.reset();
    mappedHeights_ = Tensor();
    leaves_.reset();
    pendingRect_.reset();
    readbackPending_ = {};
    horizon_.reset();
    horizonDirty_ = true;
    heightMaxima_.reset();
    sunHorizon_.reset();
    renderer_.SetTerrainExtent(std::nullopt);
}

void TerrainDrawable::SetCastersExtent() {
    renderer_.SetTerrainExtent(glm::dvec4(
        originX_, originY_, static_cast<double>(width_) * pixelSize_,
        static_cast<double>(height_) * pixelSize_));
}

void TerrainDrawable::SetHeightfield(const RasterView& raster) {
    CheckRasterView(raster, "Heightfield", 2);
    // The horizon maps follow the new heights, at their size.
    horizon_.reset();
    horizonDirty_ = true;
    heightMaxima_.reset();
    sunHorizon_.reset();
    noData_ = raster.noData;
    const std::uint32_t maxSize =
        renderer_.Context().Limits().maxImageDimension2D;
    if (raster.width > maxSize || raster.height > maxSize) {
        throw std::invalid_argument(
            "Heightfield exceeds the device's maximum texture size");
    }

    vkDeviceWaitIdle(renderer_.Context().Device());
    heights_.reset();
    gradients_.reset();
    tree_ = LodQuadtree();
    patches_.clear();
    mainRanges_.clear();
    cascadeRanges_.clear();
    triangles_ = 0;
    prepared_ = false;
    deviceHeights_.reset();
    mappedHeights_ = Tensor();
    leaves_.reset();
    pendingRect_.reset();
    readbackPending_ = {};
    width_ = raster.width;
    height_ = raster.height;
    originX_ = raster.originX;
    originY_ = raster.originY;
    pixelSize_ = raster.cellSize;
    SetCastersExtent();
    float lowest = std::numeric_limits<float>::infinity();
    float highest = -std::numeric_limits<float>::infinity();
    for (std::size_t row = 0; row < height_; ++row) {
        for (std::size_t col = 0; col < width_; ++col) {
            const float value = raster(col, row);
            if (raster.IsValid(value)) {
                lowest = std::min(lowest, value);
                highest = std::max(highest, value);
            }
        }
    }
    minHeight_ = lowest <= highest ? lowest : 0.0f;
    maxHeight_ = lowest <= highest ? highest : 0.0f;

    const auto w = static_cast<std::uint32_t>(width_);
    const auto h = static_cast<std::uint32_t>(height_);
    std::vector<float> packedHeights(static_cast<std::size_t>(w) * h);
    for (std::uint32_t row = 0; row < h; ++row) {
        for (std::uint32_t col = 0; col < w; ++col) {
            packedHeights[static_cast<std::size_t>(row) * w + col] =
                raster(col, row);
        }
    }
    auto heights = std::make_unique<GpuImage>(
        renderer_.Context(),
        ImageDesc{w, h, 1, 1, VK_FORMAT_R32_SFLOAT,
                  VK_IMAGE_USAGE_SAMPLED_BIT});
    const std::span<const std::byte> heightMips[] = {
        std::as_bytes(std::span(packedHeights))};
    heights->Upload(heightMips);
    const std::vector<std::vector<std::uint32_t>> gradients =
        PackedGradientMips(raster);
    auto gradientImage = std::make_unique<GpuImage>(
        renderer_.Context(),
        ImageDesc{
            w, h, 1, static_cast<std::uint32_t>(gradients.size()),
            VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT});
    std::vector<std::span<const std::byte>> gradientMips;
    for (const std::vector<std::uint32_t>& mip : gradients) {
        gradientMips.push_back(std::as_bytes(std::span(mip)));
    }
    gradientImage->Upload(gradientMips);
    // The tree in the terrain's own coordinates (its north-west
    // corner at x = y = 0), small and exact in float wherever the
    // terrain lies; `LocalFrom` maps views into them.
    RasterView local = raster;
    local.originX = 0.0;
    local.originY = 0.0;
    tree_ = LodQuadtree(local, kLodLeafCells);
    heights_ = std::move(heights);
    gradients_ = std::move(gradientImage);
}

void TerrainDrawable::AllocateHeightfield(std::uint32_t width,
                                          std::uint32_t height,
                                          double cellSize) {
    const VulkanContext& context = renderer_.Context();
    vkDeviceWaitIdle(context.Device());
    patches_.clear();
    mainRanges_.clear();
    cascadeRanges_.clear();
    triangles_ = 0;
    prepared_ = false;
    width_ = width;
    height_ = height;
    horizon_.reset();
    horizonDirty_ = true;
    heightMaxima_.reset();
    sunHorizon_.reset();
    heights_ = std::make_unique<GpuImage>(
        context, ImageDesc{width, height, 1, 1, VK_FORMAT_R32_SFLOAT,
                           VK_IMAGE_USAGE_SAMPLED_BIT});
    const auto mips = static_cast<std::uint32_t>(
        std::bit_width(std::max(width, height)));
    gradients_ = std::make_unique<GpuImage>(
        context,
        ImageDesc{width, height, 1, mips, VK_FORMAT_R16G16_SFLOAT,
                  VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_STORAGE_BIT});
    mappedHeights_ = Tensor();
    deviceHeights_ = std::make_shared<GpuBuffer>(
        context, VkDeviceSize{width} * height * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Shared);
    tree_ = LodQuadtree(width, height, cellSize, kLodLeafCells);
    const VkDeviceSize leafBytes =
        VkDeviceSize{tree_.LeavesPerSide()} * tree_.LeavesPerSide() *
        sizeof(glm::vec2);
    leaves_ = std::make_unique<GpuBuffer>(
        context, leafBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        BufferMemory::Device);
    for (std::unique_ptr<GpuBuffer>& readback : readbacks_) {
        readback = std::make_unique<GpuBuffer>(
            context, leafBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            BufferMemory::Readback);
    }
    pendingRect_.reset();
    readbackPending_ = {};
}

void TerrainDrawable::SetHeightfield(const DeviceHeightfield& field) {
    if (compute_ == nullptr) {
        throw std::invalid_argument(
            "Heights on a GPU need compute interop");
    }
    const Tensor& tensor = field.heights;
    if (tensor.Location() != compute_->Location() ||
        tensor.Type() != DType::Float32 || tensor.Rank() != 2 ||
        !tensor.IsContiguous()) {
        throw std::invalid_argument(
            "Heights on a GPU must be a contiguous float32 (rows, "
            "cols) tensor on compute device " +
            std::to_string(compute_->Device()));
    }
    const auto width = static_cast<std::uint32_t>(tensor.Shape()[1]);
    const auto height = static_cast<std::uint32_t>(tensor.Shape()[0]);
    const std::uint32_t maxSize =
        renderer_.Context().Limits().maxImageDimension2D;
    if (width < 2 || height < 2 || width > maxSize ||
        height > maxSize) {
        throw std::invalid_argument(
            "Heightfield must be from 2 x 2 to the device's maximum "
            "texture size");
    }
    if (!(field.cellSize > 0.0)) {
        throw std::invalid_argument(
            "Heightfield cells must have a positive size");
    }
    std::array<std::uint32_t, 4> rect{0, 0, width, height};
    if (field.dirty) {
        const auto& [col, row, columns, rows] = *field.dirty;
        if (std::uint64_t{col} + columns > width ||
            std::uint64_t{row} + rows > height) {
            throw std::invalid_argument(
                "The heights' dirty rectangle must lie inside them");
        }
        rect = {col, row, col + columns, row + rows};
    }
    // A first field, or one of another shape, starts over; another
    // nodata value changes every cell's bounds.
    const bool reshape = deviceHeights_ == nullptr ||
                         width != width_ || height != height_ ||
                         field.cellSize != device_.cellSize;
    if (reshape) {
        AllocateHeightfield(width, height, field.cellSize);
    }
    if (reshape || field.noData != device_.noData) {
        rect = {0, 0, width, height};
    }
    device_ = field;
    device_.heights = Tensor();
    noData_ = field.noData;
    originX_ = field.originX;
    originY_ = field.originY;
    pixelSize_ = field.cellSize;
    SetCastersExtent();
    if (!reshape && !mappedHeights_.Empty() &&
        tensor.Data() == mappedHeights_.Data()) {
        // Written in place through `MapHeights`: the frame waits for
        // their stream.
        compute_->SignalVulkan(field.stream);
    } else {
        compute_->CopyToBuffer(tensor, *deviceHeights_, 0,
                               field.stream);
    }
    if (rect[0] < rect[2] && rect[1] < rect[3]) {
        pendingRect_ =
            pendingRect_
                ? std::array<std::uint32_t,
                             4>{std::min((*pendingRect_)[0], rect[0]),
                                std::min((*pendingRect_)[1], rect[1]),
                                std::max((*pendingRect_)[2], rect[2]),
                                std::max((*pendingRect_)[3], rect[3])}
                : rect;
    }
    if (reshape) {
        // Bounded at once, so the first frame selects patches from
        // the real heights.
        const TimelinePoint wait = compute_->VulkanWait();
        const std::span<const TimelinePoint> waits(
            &wait, wait.semaphore != VK_NULL_HANDLE ? 1 : 0);
        renderer_.Context().Submit(
            [&](VkCommandBuffer commands) {
                RecordHeightUpdate(commands, *pendingRect_,
                                   readbacks_[0].get());
            },
            waits);
        pendingRect_.reset();
        TakeLeaves(*readbacks_[0]);
    }
}

Tensor TerrainDrawable::MapHeights(std::uintptr_t stream) {
    if (compute_ == nullptr || deviceHeights_ == nullptr) {
        throw std::invalid_argument(
            "Mapping heights needs heights written on the GPU first");
    }
    // Nothing writes them while frames already submitted may still
    // copy them.
    compute_->WaitForVulkan(stream);
    mappedHeights_ =
        compute_->View(deviceHeights_, DType::Float32,
                       {static_cast<std::int64_t>(height_),
                        static_cast<std::int64_t>(width_)});
    return mappedHeights_;
}

void TerrainDrawable::RecordCull(VkCommandBuffer commands) const {
    if (pendingRect_) {
        // Its leaves come back to the host once this frame's slot is
        // next taken (`Prepare`).
        const std::uint32_t slot = renderer_.Uploads().Slot();
        RecordHeightUpdate(commands, *pendingRect_,
                           readbacks_[slot].get());
        readbackPending_[slot] = true;
        pendingRect_.reset();
        horizonDirty_ = true;
        heightMaximaDirty_ = true;
    }
    // The horizon map, after the heights it follows, when dome
    // occlusion reads it.
    if (horizonWanted_ && horizon_ != nullptr && horizonDirty_) {
        RecordHorizon(commands);
        horizonDirty_ = false;
    }
    // The sun horizon maps, after them too, when distant lights'
    // shadows read them.
    if (sunHorizon_ != nullptr && !sunAzimuths_.empty()) {
        RecordSunHorizon(commands);
    }
}

void TerrainDrawable::EnsureSunHorizon(std::uint32_t lights) {
    if (heightMaxPipeline_ == nullptr) {
        const std::vector<ShaderModuleSource> modules{
            {"terrain_sun_horizon",
             shaders::kTerrainSunHorizonSlang}};
        heightMaxPipeline_ = std::make_unique<ComputePipeline>(
            renderer_.Context(), sunHorizonLayout_,
            renderer_.Compiler().CompileSpirv(modules,
                                              "heightMaxMain"));
        sunHorizonPipeline_ = std::make_unique<ComputePipeline>(
            renderer_.Context(), sunHorizonLayout_,
            renderer_.Compiler().CompileSpirv(modules,
                                              "sunHorizonMain"));
    }
    const auto width = static_cast<std::uint32_t>(width_);
    const auto height = static_cast<std::uint32_t>(height_);
    if (heightMaxima_ == nullptr) {
        std::uint32_t mips = 1;
        while ((std::max(width, height) >> mips) > 0) {
            ++mips;
        }
        heightMaxima_ = std::make_unique<GpuImage>(
            renderer_.Context(),
            ImageDesc{width, height, 1, mips, VK_FORMAT_R32_SFLOAT,
                      VK_IMAGE_USAGE_SAMPLED_BIT |
                          VK_IMAGE_USAGE_STORAGE_BIT});
        heightMaximaDirty_ = true;
        sunHorizon_.reset();
    }
    if (sunHorizon_ != nullptr &&
        sunHorizon_->Desc().layers >= lights) {
        return;
    }
    // A layer for each light, all computed afresh.
    const std::uint32_t longest = std::max(width, height);
    sunHorizonScale_ =
        (longest + kMaxSunHorizon - 1) / kMaxSunHorizon;
    sunHorizon_ = std::make_unique<GpuImage>(
        renderer_.Context(),
        ImageDesc{(width + sunHorizonScale_ - 1) / sunHorizonScale_,
                  (height + sunHorizonScale_ - 1) / sunHorizonScale_,
                  std::max(lights, 1u), 1, VK_FORMAT_R32G32_SFLOAT,
                  VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_STORAGE_BIT,
                  false, true});
    sunHorizonAzimuths_.assign(sunHorizon_->Desc().layers,
                               std::nullopt);
}

void TerrainDrawable::RecordSunHorizon(
    VkCommandBuffer commands) const {
    // The layers whose lights turned, or all with new heights.
    std::vector<std::uint32_t> layers;
    for (std::uint32_t layer = 0; layer < sunAzimuths_.size();
         ++layer) {
        const std::optional<glm::vec2>& was =
            sunHorizonAzimuths_[layer];
        if (heightMaximaDirty_ || !was ||
            std::acos(std::clamp(glm::dot(*was, sunAzimuths_[layer]),
                                 -1.0f, 1.0f)) > kSunHorizonTurn) {
            layers.push_back(layer);
        }
    }
    if (layers.empty()) {
        return;
    }
    ERNIE_GPU_ZONE(renderer_.Context(), commands,
                   "Terrain sun horizon");
    const auto barrier = [&] {
        VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memory.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memory.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        VkDependencyInfo dependency{
            VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &memory;
        vkCmdPipelineBarrier2(commands, &dependency);
    };
    // Each pass writes one of `target` (a level of the maxima) and
    // `layer` (of the sun horizon maps), the other bound unused.
    const auto dispatch = [&](const ComputePipeline& pipeline,
                              VkImageView finer, VkImageLayout layout,
                              VkImageView target, std::uint32_t layer,
                              const SunHorizonConstants& constants) {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline.Handle());
        const DescriptorResource resources[] = {
            DescriptorResource::Of(finer, layout),
            DescriptorResource::Of(target, VK_IMAGE_LAYOUT_GENERAL),
            DescriptorResource::Of(heightMaxima_->View(),
                                   VK_IMAGE_LAYOUT_GENERAL),
            DescriptorResource::Of(sunHorizon_->LayerView(layer),
                                   VK_IMAGE_LAYOUT_GENERAL)};
        sunHorizonLayout_.Push(
            commands, VK_PIPELINE_BIND_POINT_COMPUTE, resources);
        sunHorizonLayout_.PushConstants(commands, &constants,
                                        sizeof(constants));
        vkCmdDispatch(commands, (constants.size[2] + 7) / 8,
                      (constants.size[3] + 7) / 8, 1);
    };
    heightMaxima_->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    sunHorizon_->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    if (heightMaximaDirty_) {
        // Level 0 from the heights, then each level from the last.
        const auto width = static_cast<std::uint32_t>(width_);
        const auto height = static_cast<std::uint32_t>(height_);
        SunHorizonConstants constants{};
        constants.size[0] = width;
        constants.size[1] = height;
        constants.size[2] = width;
        constants.size[3] = height;
        constants.values[0] = noData_.value_or(0.0f);
        constants.values[1] = noData_ ? 1.0f : 0.0f;
        dispatch(*heightMaxPipeline_, heights_->View(),
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 heightMaxima_->MipView(0), 0, constants);
        for (std::uint32_t mip = 1; mip < heightMaxima_->Desc().mips;
             ++mip) {
            barrier();
            constants.size[0] = std::max(width >> (mip - 1), 1u);
            constants.size[1] = std::max(height >> (mip - 1), 1u);
            constants.size[2] = std::max(width >> mip, 1u);
            constants.size[3] = std::max(height >> mip, 1u);
            constants.values[1] = 0.0f;
            dispatch(*heightMaxPipeline_,
                     heightMaxima_->MipView(mip - 1),
                     VK_IMAGE_LAYOUT_GENERAL,
                     heightMaxima_->MipView(mip), 0, constants);
        }
        heightMaximaDirty_ = false;
        barrier();
    }
    SunHorizonConstants constants{};
    constants.size[0] = static_cast<std::uint32_t>(width_);
    constants.size[1] = static_cast<std::uint32_t>(height_);
    constants.size[2] = sunHorizon_->Desc().width;
    constants.size[3] = sunHorizon_->Desc().height;
    constants.values[0] = static_cast<float>(sunHorizonScale_);
    constants.values[1] = static_cast<float>(pixelSize_);
    constants.sun[2] = static_cast<float>(heightMaxima_->Desc().mips);
    for (const std::uint32_t layer : layers) {
        constants.sun[0] = sunAzimuths_[layer].x;
        constants.sun[1] = sunAzimuths_[layer].y;
        dispatch(*sunHorizonPipeline_, heightMaxima_->MipView(0),
                 VK_IMAGE_LAYOUT_GENERAL, heightMaxima_->MipView(0),
                 layer, constants);
        sunHorizonAzimuths_[layer] = sunAzimuths_[layer];
    }
    sunHorizon_->Transition(commands,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    heightMaxima_->Transition(
        commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

float TerrainDrawable::HorizonLevel(std::size_t level) const {
    return kHorizonLevelCells[level] * static_cast<float>(pixelSize_);
}

void TerrainDrawable::EnsureHorizon() {
    if (horizonPipeline_ == nullptr) {
        const std::vector<ShaderModuleSource> modules{
            {"terrain_horizon", shaders::kTerrainHorizonSlang}};
        horizonPipeline_ = std::make_unique<ComputePipeline>(
            renderer_.Context(), horizonLayout_,
            renderer_.Compiler().CompileSpirv(modules,
                                              "horizonMain"));
        horizonCastersPipeline_ = std::make_unique<ComputePipeline>(
            renderer_.Context(), horizonLayout_,
            renderer_.Compiler().CompileSpirv(modules,
                                              "horizonCastersMain"));
    }
    if (horizon_ == nullptr) {
        const auto longest =
            static_cast<std::uint32_t>(std::max(width_, height_));
        horizonScale_ = (longest + kMaxHorizon - 1) / kMaxHorizon;
        const auto width = static_cast<std::uint32_t>(
            (width_ + horizonScale_ - 1) / horizonScale_);
        const auto height = static_cast<std::uint32_t>(
            (height_ + horizonScale_ - 1) / horizonScale_);
        horizon_ = std::make_unique<GpuImage>(
            renderer_.Context(),
            ImageDesc{width, height, kHorizonLayers, 1,
                      VK_FORMAT_R16G16B16A16_SFLOAT,
                      VK_IMAGE_USAGE_SAMPLED_BIT |
                          VK_IMAGE_USAGE_STORAGE_BIT,
                      false, true});
        horizonDirty_ = true;
    }
}

void TerrainDrawable::RecordHorizon(VkCommandBuffer commands) const {
    ERNIE_GPU_ZONE(renderer_.Context(), commands, "Terrain horizon");
    RecordHorizonPass(commands, *horizonPipeline_);
    // The ground level changed under the casters' layers too.
    castersStale_ = true;
}

void TerrainDrawable::RecordAfterShadows(
    VkCommandBuffer commands) const {
    const RasterRenderer::CastersView& casters = renderer_.Casters();
    if (!horizonWanted_ || horizon_ == nullptr || !casters.valid ||
        !(castersStale_ || renderer_.CastersChanged())) {
        return;
    }
    ERNIE_GPU_ZONE(renderer_.Context(), commands,
                   "Terrain horizon over casters");
    RecordHorizonPass(commands, *horizonCastersPipeline_);
    castersStale_ = false;
}

void TerrainDrawable::RecordHorizonPass(
    VkCommandBuffer commands, const ComputePipeline& pipeline) const {
    horizon_->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline.Handle());
    const DescriptorResource resources[] = {
        DescriptorResource::Of(heights_->View()),
        DescriptorResource::Of(horizon_->View(),
                               VK_IMAGE_LAYOUT_GENERAL),
        DescriptorResource::Of(
            renderer_.LightShadowMaps().View(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)};
    horizonLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                        resources);
    HorizonConstants constants{};
    constants.size[0] = static_cast<std::uint32_t>(width_);
    constants.size[1] = static_cast<std::uint32_t>(height_);
    constants.size[2] = horizon_->Desc().width;
    constants.size[3] = horizon_->Desc().height;
    constants.values[0] = static_cast<float>(horizonScale_);
    constants.values[1] = static_cast<float>(pixelSize_);
    constants.values[2] = noData_.value_or(0.0f);
    constants.values[3] = noData_ ? 1.0f : 0.0f;
    for (std::size_t level = 0; level < kHorizonLevelCells.size();
         ++level) {
        constants.levels[level] = HorizonLevel(level);
    }
    const RasterRenderer::CastersView& casters = renderer_.Casters();
    constants.sky[0] = casters.top;
    constants.sky[1] = casters.span;
    constants.sky[2] = static_cast<float>(origin_.z);
    constants.sky[3] = params_.exaggeration;
    constants.skyInfo[0] = casters.layer;
    constants.skyInfo[1] = casters.resolution;
    horizonLayout_.PushConstants(commands, &constants,
                                 sizeof(constants));
    vkCmdDispatch(commands, (constants.size[2] + 7) / 8,
                  (constants.size[3] + 7) / 8, 1);
    horizon_->Transition(commands,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void TerrainDrawable::RecordHeightUpdate(
    VkCommandBuffer commands,
    const std::array<std::uint32_t, 4>& rect,
    const GpuBuffer* readback) const {
    ERNIE_GPU_ZONE(renderer_.Context(), commands, "Terrain heights");
    const auto width = static_cast<std::uint32_t>(width_);
    const auto height = static_cast<std::uint32_t>(height_);
    const auto barrier = [&](VkPipelineStageFlags2 dstStage,
                             VkAccessFlags2 dstAccess) {
        VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        memory.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT |
                               VK_ACCESS_2_TRANSFER_WRITE_BIT;
        memory.dstStageMask = dstStage;
        memory.dstAccessMask = dstAccess;
        VkDependencyInfo dependency{
            VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &memory;
        vkCmdPipelineBarrier2(commands, &dependency);
    };
    const auto groups = [](std::uint32_t extent) {
        return (extent + 7) / 8;
    };

    // The heights image over the rectangle, from the copy of the
    // device's heights.
    heights_->Transition(commands,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy region{};
    region.bufferOffset =
        (VkDeviceSize{rect[1]} * width + rect[0]) * sizeof(float);
    region.bufferRowLength = width;
    region.bufferImageHeight = height;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {static_cast<std::int32_t>(rect[0]),
                          static_cast<std::int32_t>(rect[1]), 0};
    region.imageExtent = {rect[2] - rect[0], rect[3] - rect[1], 1};
    vkCmdCopyBufferToImage(
        commands, deviceHeights_->Handle(), heights_->Handle(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    heights_->Transition(commands,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    UpdateConstants constants{};
    constants.size[2] = kLodLeafCells;
    constants.size[3] = tree_.LeavesPerSide();
    constants.values[0] = static_cast<float>(0.5 / pixelSize_);
    constants.values[1] = device_.noData.value_or(0.0f);
    constants.values[2] = device_.noData ? 1.0f : 0.0f;
    const auto dispatch = [&](const ComputePipeline& pipeline,
                              std::uint32_t mip,
                              const std::array<std::uint32_t, 4>& at,
                              std::uint32_t w, std::uint32_t h) {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline.Handle());
        const DescriptorResource resources[] = {
            DescriptorResource::Of(*deviceHeights_),
            DescriptorResource::Of(gradients_->MipView(mip),
                                   VK_IMAGE_LAYOUT_GENERAL),
            DescriptorResource::Of(
                gradients_->MipView(mip > 0 ? mip - 1 : 0),
                VK_IMAGE_LAYOUT_GENERAL),
            DescriptorResource::Of(*leaves_)};
        updateLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                           resources);
        std::copy(at.begin(), at.end(), constants.rect);
        constants.size[0] = w;
        constants.size[1] = h;
        updateLayout_.PushConstants(commands, &constants,
                                    sizeof(constants));
        vkCmdDispatch(commands, groups(at[2] - at[0]),
                      groups(at[3] - at[1]), 1);
    };

    // Gradients: level 0 where a changed height is a neighbour, then
    // each mip over what those texels average into.
    gradients_->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    std::array<std::uint32_t, 4> at{
        rect[0] > 0 ? rect[0] - 1 : 0, rect[1] > 0 ? rect[1] - 1 : 0,
        std::min(rect[2] + 1, width), std::min(rect[3] + 1, height)};
    dispatch(*gradientsPipeline_, 0, at, width, height);
    for (std::uint32_t mip = 1; mip < gradients_->Desc().mips;
         ++mip) {
        barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_READ_BIT);
        const std::uint32_t below = mip - 1;
        const std::uint32_t mipWidth = std::max(width >> mip, 1u);
        const std::uint32_t mipHeight = std::max(height >> mip, 1u);
        at = {at[0] / 2, at[1] / 2,
              std::min((at[2] + 1) / 2, mipWidth),
              std::min((at[3] + 1) / 2, mipHeight)};
        dispatch(*gradientMipPipeline_, mip, at,
                 std::max(width >> below, 1u),
                 std::max(height >> below, 1u));
    }

    // The LOD leaves sharing a vertex with a changed cell; every
    // leaf, those past the raster's edge included, when every cell
    // changed.
    const std::uint32_t leaves = tree_.LeavesPerSide();
    const auto first = [](std::uint32_t cell) {
        return cell > 0 ? (cell - 1) / kLodLeafCells : 0;
    };
    const bool all =
        rect == std::array<std::uint32_t, 4>{0, 0, width, height};
    const std::array<std::uint32_t, 4> leafRect =
        all ? std::array<std::uint32_t, 4>{0, 0, leaves, leaves}
            : std::array<std::uint32_t, 4>{
                  first(rect[0]), first(rect[1]),
                  std::min((rect[2] - 1) / kLodLeafCells + 1, leaves),
                  std::min((rect[3] - 1) / kLodLeafCells + 1,
                           leaves)};
    dispatch(*leafRangePipeline_, 0, leafRect, width, height);
    barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_SHADER_READ_BIT |
                VK_ACCESS_2_TRANSFER_READ_BIT);
    gradients_->Transition(commands,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (readback != nullptr) {
        const VkBufferCopy copy{0, 0, leaves_->Size()};
        vkCmdCopyBuffer(commands, leaves_->Handle(),
                        readback->Handle(), 1, &copy);
        barrier(VK_PIPELINE_STAGE_2_HOST_BIT,
                VK_ACCESS_2_HOST_READ_BIT);
    }
}

void TerrainDrawable::TakeLeaves(const GpuBuffer& readback) {
    std::vector<glm::vec2> leaves(
        static_cast<std::size_t>(tree_.LeavesPerSide()) *
        tree_.LeavesPerSide());
    readback.Read(std::as_writable_bytes(std::span(leaves)));
    tree_.SetLeaves(leaves);
    const glm::vec2 range = tree_.HeightRange();
    minHeight_ = range.x <= range.y ? range.x : 0.0f;
    maxHeight_ = range.x <= range.y ? range.y : 0.0f;
}

std::uint64_t TerrainDrawable::GpuBytes() const {
    std::uint64_t bytes = uniforms_.Size() + patchBuffer_->Size();
    if (heights_ != nullptr) {
        bytes += heights_->Bytes() + gradients_->Bytes() +
                 (horizon_ != nullptr ? horizon_->Bytes() : 0);
    }
    if (deviceHeights_ != nullptr) {
        bytes += deviceHeights_->Size() + leaves_->Size();
    }
    return bytes;
}

std::pair<glm::vec3, glm::vec3>
TerrainDrawable::Bounds(float exaggeration) const {
    // Relative to the frame's origin, computed in double.
    const double halfPixel = 0.5 * pixelSize_;
    const glm::dvec3 low(
        originX_ + halfPixel,
        originY_ - static_cast<double>(height_) * pixelSize_ +
            halfPixel,
        static_cast<double>(minHeight_ * exaggeration));
    const glm::dvec3 high(
        originX_ + static_cast<double>(width_) * pixelSize_ -
            halfPixel,
        originY_ - halfPixel,
        static_cast<double>(maxHeight_ * exaggeration));
    const glm::vec3 a(low - origin_);
    const glm::vec3 b(high - origin_);
    return {glm::min(a, b), glm::max(a, b)};
}

std::optional<Bounds3> TerrainDrawable::Bounds() const {
    if (!HasHeightfield()) {
        return std::nullopt;
    }
    return Bounds(params_.exaggeration);
}

void TerrainDrawable::Prepare(
    const FrameParams& frame,
    std::span<const ShadowCascade> cascades) {
    ERNIE_ZONE("Terrain prepare");
    prepared_ = false;
    pushbroom_ = frame.camera.pushbroom;
    if (gbufferPipelines_[pushbroom_ ? 1 : 0] == nullptr) {
        CreateGBufferPipeline(pushbroom_);
    }
    // The horizon map for dome occlusion, when the frame shadows its
    // dome.
    horizonWanted_ = HasHeightfield() && renderer_.DomeShadowed();
    RasterRenderer::TerrainHorizon horizon;
    if (horizonWanted_) {
        EnsureHorizon();
        horizon.image = horizon_.get();
        horizon.grid = {
            static_cast<float>(originX_ - origin_.x),
            static_cast<float>(originY_ - origin_.y),
            static_cast<float>(pixelSize_ * horizonScale_),
            params_.exaggeration};
        horizon.info = {horizon_->Desc().width,
                        horizon_->Desc().height, params_.primId, 1};
        // The heights, for how far what stands on the terrain stands
        // above it.
        horizon.heights = heights_.get();
        horizon.heightsGrid = {static_cast<float>(pixelSize_),
                               static_cast<float>(origin_.z),
                               noData_.value_or(0.0f),
                               noData_ ? 1.0f : 0.0f};
        // z: whether the map holds the ground-level horizon over what
        // stands on the terrain.
        horizon.heightsSize = {static_cast<std::uint32_t>(width_),
                               static_cast<std::uint32_t>(height_),
                               renderer_.Casters().valid ? 1u : 0u,
                               0};
        for (std::size_t level = 0; level < kHorizonLevelCells.size();
             ++level) {
            horizon.levels[level] = HorizonLevel(level);
        }
    }
    renderer_.SetTerrainHorizon(horizon);
    // The sun horizon maps for the shadows of the distant lights that
    // cast them, a layer each.
    sunAzimuths_.clear();
    const std::vector<glm::vec3>& lights = renderer_.HorizonLights();
    if (HasHeightfield() && !lights.empty()) {
        EnsureSunHorizon(static_cast<std::uint32_t>(lights.size()));
        for (const glm::vec3& toLight : lights) {
            // Across the grid: columns east, rows south. A light
            // overhead has no azimuth, nor a horizon that hides it.
            const glm::vec2 across(toLight.x, -toLight.y);
            sunAzimuths_.push_back(glm::length(across) > 1e-6f
                                       ? glm::normalize(across)
                                       : glm::vec2(1.0f, 0.0f));
        }
        RasterRenderer::SunHorizon sun;
        sun.image = sunHorizon_.get();
        sun.grid = {static_cast<float>(originX_ - origin_.x),
                    static_cast<float>(originY_ - origin_.y),
                    static_cast<float>(pixelSize_ * sunHorizonScale_),
                    params_.exaggeration};
        sun.info = {sunHorizon_->Desc().width,
                    sunHorizon_->Desc().height, params_.primId, 1};
        sun.heights = {static_cast<float>(origin_.z),
                       static_cast<float>(pixelSize_), 0.0f, 0.0f};
        renderer_.SetSunHorizon(sun);
    }
    if (!HasHeightfield()) {
        return;
    }
    // The LOD bounds of device heights this slot's last frame
    // updated, now that it finished.
    const std::uint32_t slot = renderer_.Uploads().Slot();
    if (readbackPending_[slot]) {
        readbackPending_[slot] = false;
        TakeLeaves(*readbacks_[slot]);
    }
    const std::uint32_t material =
        renderer_.HasMaterial(params_.material)
            ? params_.material
            : RasterRenderer::kDefaultMaterial;
    TerrainUniforms uniforms{};
    // Relative to the frame's origin.
    uniforms.grid[0] = static_cast<float>(originX_ - origin_.x);
    uniforms.grid[1] = static_cast<float>(originY_ - origin_.y);
    uniforms.grid[2] = static_cast<float>(pixelSize_);
    uniforms.grid[3] = params_.exaggeration;
    uniforms.heightRange[0] = minHeight_;
    uniforms.heightRange[1] = maxHeight_;
    uniforms.heightRange[2] = static_cast<float>(origin_.z);
    uniforms.dims[0] = static_cast<std::uint32_t>(width_);
    uniforms.dims[1] = static_cast<std::uint32_t>(height_);
    uniforms.dims[2] = static_cast<std::uint32_t>(params_.colorMode);
    uniforms.dims[3] = params_.lod.showLevels ? 1u : 0u;
    const glm::vec3 towards = HillshadeDirection(params_);
    uniforms.hillshade[0] = towards.x;
    uniforms.hillshade[1] = towards.y;
    uniforms.hillshade[2] = towards.z;

    const LodSettings& lod = params_.lod;
    const std::size_t views = frame.Views();
    // The LOD eyes and frusta in the terrain's coordinates (`tree_`),
    // which do not move with the frame's origin, so a frozen LOD
    // stays put across a rebase.
    const glm::vec3 local = LocalFrom();
    if (!(lod.freeze && frozen_) || lodEyes_.size() != views) {
        lodEyes_.clear();
        lodFrusta_.clear();
        for (std::size_t v = 0; v < views; ++v) {
            const CameraView view = frame.View(v);
            lodEyes_.push_back(view.eye + local);
            lodFrusta_.push_back(Shifted(
                Frustum::FromMatrix(view.ViewProjection()), local));
        }
    }
    frozen_ = lod.freeze;
    std::vector<LodRange> ranges;
    firstRange_ = 0.0f;
    if (lod.enabled && !tree_.Empty()) {
        const float morphStart =
            std::clamp(lod.morphStart, 0.05f, 0.95f);
        firstRange_ = std::max(
            lod.detail * tree_.LeafSize(),
            tree_.MinFirstRange(params_.exaggeration, morphStart));
        ranges = tree_.Ranges(firstRange_, morphStart);
    }
    uniforms.lodInfo[0] = tree_.PatchCells();
    for (std::uint32_t level = 0; level < LodQuadtree::kMaxLevels;
         ++level) {
        // Levels that never morph get a start beyond any distance and
        // a zero slope.
        float start = std::numeric_limits<float>::max();
        float slope = 0.0f;
        if (level < ranges.size() &&
            std::isfinite(ranges[level].morphEnd)) {
            start = ranges[level].morphStart;
            slope = 1.0f / (ranges[level].morphEnd - start);
        }
        uniforms.morph[2 * level] = start;
        uniforms.morph[2 * level + 1] = slope;
    }
    uniforms.ids[0] = material;
    uniforms.ids[1] = params_.primId;
    uniforms.ids[2] = params_.linkGroup;
    uniforms.ids[3] = std::bit_cast<std::uint32_t>(
        frame.thermal.enable ? params_.temperature : 0.0f);
    // The heights and gradients in the bindless heap, in fresh slots:
    // the last ones retire once no frame in flight reads them.
    BindlessHeap& heap = renderer_.Context().Bindless();
    heapSlots_.clear();
    for (const GpuImage* image : {heights_.get(), gradients_.get()}) {
        heapSlots_.push_back(heap.AddImage(
            BindlessHeap::Kind::Texture2D, image->View()));
        uniforms.textures[heapSlots_.size() - 1] =
            heapSlots_.back().Index();
    }
    const VkDeviceSize bytes = views * kUniformsStride;
    if (uniforms_.Size() < bytes) {
        vkDeviceWaitIdle(renderer_.Context().Device());
        uniforms_ = GpuBuffer(renderer_.Context(), bytes,
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              BufferMemory::Device);
    }
    // Each view morphs its levels of detail around its own eye.
    for (std::size_t v = 0; v < views; ++v) {
        const glm::vec3 eye =
            lodEyes_[v] - local;  // relative to the frame's origin
        uniforms.lodEye[0] = eye.x;
        uniforms.lodEye[1] = eye.y;
        uniforms.lodEye[2] = eye.z;
        renderer_.Uploads().Write(
            uniforms_, std::as_bytes(std::span(&uniforms, 1)),
            v * kUniformsStride);
    }

    patches_.clear();
    mainRanges_.clear();
    for (std::size_t v = 0; v < views; ++v) {
        const auto first =
            static_cast<std::uint32_t>(patches_.size());
        if (ranges.empty()) {
            if (v == 0) {
                tree_.SelectAll(patches_);
            }
            mainRanges_.push_back(
                {0, static_cast<std::uint32_t>(patches_.size())});
            continue;
        }
        // Views posed on the GPU have only estimated poses here:
        // their levels follow the estimate, and nothing is left out
        // by a frustum the views may not have.
        tree_.Select(lodEyes_[v], ranges,
                     frame.posesOnGpu ? nullptr : &lodFrusta_[v],
                     params_.exaggeration, patches_);
        mainRanges_.push_back(
            {first,
             static_cast<std::uint32_t>(patches_.size()) - first});
    }
    triangles_ = 0;
    for (const DrawRange& range : mainRanges_) {
        triangles_ += tree_.TriangleCount(
            std::span(patches_).subspan(range.first, range.count));
    }
    cascadeRanges_.assign(cascades.size(), DrawRange{});
    for (std::size_t c = 0; c < cascades.size(); ++c) {
        if (ranges.empty()) {
            cascadeRanges_[c] = mainRanges_.front();
            continue;
        }
        const auto first =
            static_cast<std::uint32_t>(patches_.size());
        // A view the GPU poses, or moving across the shutter, draws
        // every patch: its matrix here is only an estimate, or one of
        // its sub-frames'.
        const ShadowCascade& cascade = cascades[c];
        const Frustum light = Shifted(
            Frustum::FromMatrix(cascade.CullViewProjection()), local);
        const bool culls = !cascade.moving && !cascade.gpuFitted;
        tree_.Select(lodEyes_.front(), ranges,
                     culls ? &light : nullptr, params_.exaggeration,
                     patches_);
        cascadeRanges_[c] = {
            first,
            static_cast<std::uint32_t>(patches_.size()) - first};
    }
    if (!patches_.empty()) {
        ReservePatches(patches_.size());
        renderer_.Uploads().Write(*patchBuffer_,
                                  std::as_bytes(std::span(patches_)));
    }
    prepared_ = true;
}

std::vector<DescriptorResource>
TerrainDrawable::Resources(const RasterFrame& frame,
                           std::size_t view) const {
    std::vector<DescriptorResource> resources = {frame.resources,
                                                 frame.shadows};
    renderer_.MaterialOrDefault(params_.material)
        .AppendResources(resources);
    resources.push_back(DescriptorResource::Of(
        uniforms_, view * kUniformsStride, sizeof(TerrainUniforms)));
    resources.push_back(DescriptorResource::Of(*patchBuffer_));
    return resources;
}

void TerrainDrawable::DrawPatches(VkCommandBuffer commands,
                                  DrawRange range,
                                  const glm::mat4& light,
                                  const RasterFrame& frame,
                                  bool shadow) const {
    if (range.count == 0) {
        return;
    }
    DrawConstants constants{};
    std::memcpy(constants.lightViewProj, glm::value_ptr(light),
                sizeof(constants.lightViewProj));
    constants.first = range.first;
    constants.shadowFromFrame =
        shadow && frame.shadowFromFrame ? 1u : 0u;
    constants.view = frame.view;
    constants.shadowLayer = shadow ? frame.shadowLayer : 0u;
    layout_.PushConstants(commands, &constants, sizeof(constants));
    const std::uint32_t cells = tree_.PatchCells();
    vkCmdDraw(commands, 6 * cells * cells, range.count, 0, 0);
}

void TerrainDrawable::RecordShadow(
    VkCommandBuffer commands, const RasterFrame& frame,
    std::size_t cascade, const glm::mat4& lightViewProjection,
    const ShadowCasters& casters) const {
    // Not into the casters' view: what stands on the terrain, which
    // the horizon map adds to the heights themselves (a coarse LOD
    // there would read as relief that is not).
    const RasterRenderer::CastersView& view = renderer_.Casters();
    if (!prepared_ || cascade >= cascadeRanges_.size() ||
        !casters.shared || !casters.Casts(params_.linkGroup) ||
        (view.valid && cascade == view.shadowView)) {
        return;
    }
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      shadowPipeline_->Handle());
    layout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                 Resources(frame, 0));
    DrawPatches(commands, cascadeRanges_[cascade],
                lightViewProjection, frame, true);
}

void TerrainDrawable::RecordColor(VkCommandBuffer commands,
                                  const RasterFrame& frame,
                                  std::size_t view,
                                  DrawPhase phase) const {
    if (!prepared_ || view >= mainRanges_.size() ||
        phase == DrawPhase::Occludees) {
        return;
    }
    vkCmdBindPipeline(
        commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
        gbufferPipelines_[pushbroom_ ? 1 : 0]->Handle());
    layout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                 Resources(frame, view));
    DrawPatches(commands, mainRanges_[view], glm::mat4(1.0f), frame);
}

}  // namespace ernie
