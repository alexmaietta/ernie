// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/overlay_pass.h"

#include "ernie/kernel/embedded_shaders.h"
#include "ernie/raster/drawable.h"
#include "ernie/util/profile.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

struct GpuSegment {
    float a[4];
    float b[4];
    std::uint32_t info[4];
};
static_assert(sizeof(GpuSegment) == 48,
              "GpuSegment must match overlay.slang");

struct GpuTriangle {
    float p[3][4];
    std::uint32_t info[4];
};
static_assert(sizeof(GpuTriangle) == 64,
              "GpuTriangle must match overlay.slang");

struct GpuSprite {
    float anchor[4];
    float rect[4];
    std::uint32_t info[4];
};
static_assert(sizeof(GpuSprite) == 48,
              "GpuSprite must match overlay.slang");

struct OverlayConstants {
    std::int32_t view[4];
    float viewport[4];
    std::uint32_t batch[4];
};
static_assert(sizeof(OverlayConstants) == 48,
              "OverlayConstants must match overlay.slang");

/// One `OverlayBatch` as overlay.slang's `Batch` reads it.
struct GpuBatch {
    std::uint32_t shape[4];
    std::uint32_t arrays[4];
    std::uint32_t extras[4];
    std::uint32_t style[4];
    float origin[4];
    float params[4];
};
static_assert(sizeof(GpuBatch) == 96,
              "GpuBatch must match overlay.slang");

/// overlay_sort.slang's push constants.
struct SortConstants {
    std::uint32_t batch[4];
    std::uint32_t vertices[4];
};

constexpr DescriptorBinding kSortBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

constexpr const char* kSortEntries[] = {
    "sortClearMain", "sortCountMain", "sortScanMain",
    "sortScatterMain"};

constexpr VkBufferUsageFlags kArgsUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

// overlay.slang's kBatchHas* flags, segments of a sphere's circles
// and sides of an arrow head's cone.
constexpr std::uint32_t kBatchHasColors = 1;
constexpr std::uint32_t kBatchHasAlpha = 2;
constexpr std::uint32_t kBatchHasSizes = 4;
constexpr std::uint32_t kBatchHasWorlds = 8;
constexpr std::uint32_t kBatchHasOrientations = 16;
constexpr std::uint32_t kCircleSegments = 48;
constexpr std::uint32_t kConeSides = 8;

/// Segments each item of a batch of `kind` draws.
std::uint32_t SegmentsPerItem(OverlayBatchKind kind) {
    switch (kind) {
    case OverlayBatchKind::Lines:
    case OverlayBatchKind::Arrows:
        return 1;
    case OverlayBatchKind::Markers:
        return 0;
    case OverlayBatchKind::Boxes:
        return 12;
    case OverlayBatchKind::Spheres:
        return 3 * kCircleSegments;
    case OverlayBatchKind::Frames:
        return 3;
    }
    return 0;
}

constexpr VkShaderStageFlags kGraphics =
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

constexpr DescriptorBinding kBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kGraphics},
    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
};

/// LdrColor's format, which headless products draw overlays into.
constexpr VkFormat kLdrFormat = VK_FORMAT_R8G8B8A8_UNORM;

std::uint32_t PackColor(const glm::vec4& color) {
    std::uint32_t packed = 0;
    for (int k = 0; k < 4; ++k) {
        const float clamped = std::clamp(color[k], 0.0f, 1.0f);
        packed |=
            static_cast<std::uint32_t>(std::lround(clamped * 255.0f))
            << (8 * k);
    }
    return packed;
}

void Info(std::uint32_t (&info)[4], const OverlayItem& item,
          MarkerShape shape) {
    info[0] = PackColor(item.color);
    info[1] = static_cast<std::uint32_t>(item.depth) |
              (static_cast<std::uint32_t>(shape) << 8);
    info[2] = static_cast<std::uint32_t>(item.world);
    info[3] = 0;
}

void Relative(float (&out)[4], const glm::dvec3& p,
              const glm::dvec3& origin, float w = 0.0f) {
    const glm::vec3 relative(p - origin);
    out[0] = relative.x;
    out[1] = relative.y;
    out[2] = relative.z;
    out[3] = w;
}

}  // namespace

OverlayPass::OverlayPass(const VulkanContext& context,
                         const ShaderCompiler& compiler,
                         FrameUploads& uploads,
                         VkFormat presentFormat)
    : context_(context), compiler_(compiler), uploads_(uploads),
      presentFormat_(presentFormat),
      layout_(context, kBindings, sizeof(OverlayConstants)),
      sortLayout_(context, kSortBindings, sizeof(SortConstants)) {
    CreatePipelines(false);
    const std::vector<ShaderModuleSource> sort{
        {"overlay_sort", shaders::kOverlaySortSlang}};
    for (std::size_t k = 0; k < sortPipelines_.size(); ++k) {
        sortPipelines_[k] = std::make_unique<ComputePipeline>(
            context, sortLayout_,
            compiler.CompileSpirv(sort, kSortEntries[k]));
    }
    // Placeholders, so the bindings always have a buffer.
    for (std::unique_ptr<GpuBuffer>& buffer : buffers_) {
        buffer = std::make_unique<GpuBuffer>(
            context, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            BufferMemory::Device);
    }
    batchData_ = std::make_shared<GpuBuffer>(
        context, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        BufferMemory::Device);
    batchInfos_ = std::make_unique<GpuBuffer>(
        context, sizeof(GpuBatch), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        BufferMemory::Device);
    sortData_ = std::make_unique<GpuBuffer>(
        context, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        BufferMemory::Device);
    drawArgs_ = std::make_unique<GpuBuffer>(context, 64, kArgsUsage,
                                            BufferMemory::Device);
}

void OverlayPass::SetComputeInterop(ComputeInterop* compute) {
    compute_ = compute;
}

void OverlayPass::CreatePipelines(bool pushbroom) {
    const std::vector<ShaderModuleSource> modules{
        ProjectionModule(pushbroom),
        {"overlay", shaders::kOverlaySlang}};
    const std::vector<std::uint32_t> fragment =
        compiler_.CompileSpirv(modules, "overlayFragment");
    const std::array<std::vector<std::uint32_t>, 6> vertices{
        compiler_.CompileSpirv(modules, "segmentVertex"),
        compiler_.CompileSpirv(modules, "triangleVertex"),
        compiler_.CompileSpirv(modules, "spriteVertex"),
        compiler_.CompileSpirv(modules, "batchSegmentVertex"),
        compiler_.CompileSpirv(modules, "batchTriangleVertex"),
        compiler_.CompileSpirv(modules, "batchSpriteVertex")};
    for (const VkFormat format : {kLdrFormat, presentFormat_}) {
        if (pipelines_.contains({format, pushbroom})) {
            continue;
        }
        std::array<std::unique_ptr<GraphicsPipeline>, 6>& pipelines =
            pipelines_[{format, pushbroom}];
        for (std::size_t kind = 0; kind < vertices.size(); ++kind) {
            GraphicsPipelineDesc desc;
            desc.vertexSpirv = vertices[kind];
            desc.fragmentSpirv = fragment;
            desc.colorFormat = format;
            desc.depthFormat = VK_FORMAT_UNDEFINED;
            desc.alphaBlend = true;
            pipelines[kind] = std::make_unique<GraphicsPipeline>(
                context_, layout_, desc);
        }
    }
}

OverlayPass::~OverlayPass() {
    vkDeviceWaitIdle(context_.Device());
}

void OverlayPass::Upload(std::unique_ptr<GpuBuffer>& buffer,
                         std::span<const std::byte> data) {
    if (data.empty()) {
        return;
    }
    if (buffer->Size() < data.size()) {
        // Earlier frames may still read the old buffer.
        vkDeviceWaitIdle(context_.Device());
        buffer = std::make_unique<GpuBuffer>(
            context_, std::max<VkDeviceSize>(data.size(), 4096),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    }
    uploads_.Write(*buffer, data);
}

void OverlayPass::Prepare(std::span<const Overlay* const> overlays,
                          const glm::dvec3& origin, bool pushbroom,
                          std::uint32_t worlds) {
    if (pushbroom) {
        CreatePipelines(true);
    }
    std::vector<std::pair<const Overlay*, std::uint64_t>> wanted;
    for (const Overlay* overlay : overlays) {
        if (overlay != nullptr && overlay->IsVisible() &&
            !overlay->Empty()) {
            wanted.emplace_back(overlay, overlay->Version());
        }
    }
    if (wanted == prepared_ && origin == preparedOrigin_ &&
        worlds == worlds_) {
        return;
    }
    ERNIE_ZONE("Overlay prepare");
    prepared_ = std::move(wanted);
    preparedOrigin_ = origin;
    worlds_ = worlds;
    std::vector<GpuSegment> segments;
    std::vector<GpuTriangle> triangles;
    std::vector<GpuSprite> sprites;
    for (const auto& [overlay, version] : prepared_) {
        for (const OverlaySegment& segment : overlay->Segments()) {
            GpuSegment& gpu = segments.emplace_back();
            Relative(gpu.a, segment.a, origin, segment.width);
            Relative(gpu.b, segment.b, origin);
            Info(gpu.info, segment.item, MarkerShape::Square);
        }
        for (const OverlayTriangle& triangle : overlay->Fills()) {
            GpuTriangle& gpu = triangles.emplace_back();
            for (int k = 0; k < 3; ++k) {
                Relative(gpu.p[k],
                         triangle.points[static_cast<std::size_t>(k)],
                         origin);
            }
            Info(gpu.info, triangle.item, MarkerShape::Square);
        }
        for (const OverlaySprite& sprite : overlay->Sprites()) {
            GpuSprite& gpu = sprites.emplace_back();
            Relative(gpu.anchor, sprite.anchor, origin);
            std::memcpy(gpu.rect, &sprite.rect[0], sizeof(gpu.rect));
            Info(gpu.info, sprite.item, sprite.shape);
        }
    }
    Upload(buffers_[0], std::as_bytes(std::span(segments)));
    Upload(buffers_[1], std::as_bytes(std::span(triangles)));
    Upload(buffers_[2], std::as_bytes(std::span(sprites)));
    counts_ = {static_cast<std::uint32_t>(segments.size()),
               static_cast<std::uint32_t>(triangles.size()),
               static_cast<std::uint32_t>(sprites.size())};
    PrepareBatches(origin);
}

void OverlayPass::PrepareBatches(const glm::dvec3& origin) {
    batchDraws_.clear();
    sortJobs_.clear();
    sortPending_ = false;
    // Words of sort data and draw arguments of the sorted batches.
    std::uint32_t sortTotal = 0;
    std::uint32_t argsTotal = 0;
    // Where each array goes in the batch buffer, in floats (or
    // int32s), and what fills it.
    struct Placed {
        const Tensor* array;
        std::uintptr_t stream;
        std::uint32_t first;
    };
    std::vector<Placed> placed;
    std::vector<GpuBatch> records;
    std::uint32_t total = 0;
    const auto place = [&](const Tensor& array,
                           std::uintptr_t stream) -> std::uint32_t {
        if (array.Empty()) {
            return 0;
        }
        const std::uint32_t first = total;
        total += static_cast<std::uint32_t>(array.Elements());
        placed.push_back({&array, stream, first});
        return first;
    };
    for (const auto& [overlay, version] : prepared_) {
        for (const OverlayBatch& batch : overlay->Batches()) {
            const auto count =
                static_cast<std::uint32_t>(batch.Count());
            GpuBatch& gpu = records.emplace_back();
            const std::uintptr_t stream = batch.stream;
            const bool alpha = !batch.colors.Empty() &&
                               batch.colors.Rank() > 1 &&
                               batch.colors.Shape().back() == 4;
            gpu.shape[0] = static_cast<std::uint32_t>(batch.kind);
            gpu.shape[1] = count;
            gpu.shape[2] = std::max(SegmentsPerItem(batch.kind), 1u);
            gpu.shape[3] =
                (batch.colors.Empty() ? 0 : kBatchHasColors) |
                (alpha ? kBatchHasAlpha : 0) |
                (batch.sizes.Empty() ? 0 : kBatchHasSizes) |
                (batch.worlds.Empty() ? 0 : kBatchHasWorlds) |
                (batch.orientations.Empty() ? 0
                                            : kBatchHasOrientations);
            gpu.arrays[0] = place(batch.points, stream);
            gpu.arrays[1] =
                place(!batch.ends.Empty()          ? batch.ends
                      : !batch.halfExtents.Empty() ? batch.halfExtents
                                                   : batch.radii,
                      stream);
            gpu.arrays[2] = place(batch.orientations, stream);
            gpu.arrays[3] = place(batch.colors, stream);
            gpu.extras[0] = place(batch.sizes, stream);
            gpu.extras[1] = place(batch.worlds, stream);
            const OverlayItem item{batch.style.color,
                                   batch.style.depth,
                                   batch.style.world};
            Info(gpu.style, item, batch.shape);
            Relative(gpu.origin, batch.origin, origin,
                     batch.kind == OverlayBatchKind::Markers
                         ? batch.style.size
                         : batch.style.width);
            gpu.params[0] = static_cast<float>(batch.head);
            gpu.params[1] = static_cast<float>(batch.length);
            const auto index =
                static_cast<std::uint32_t>(records.size() - 1);
            const std::size_t firstDraw = batchDraws_.size();
            if (batch.kind == OverlayBatchKind::Arrows) {
                batchDraws_.push_back(
                    {index, 1, count * 2 * kConeSides * 3});
            }
            if (batch.kind == OverlayBatchKind::Markers) {
                batchDraws_.push_back({index, 2, count * 6});
            } else {
                batchDraws_.push_back(
                    {index, 0, count * gpu.shape[2] * 6});
            }
            // With worlds and batched views, sorted by world: slots
            // for the shared items, each world's, and worlds no view
            // shows (overlay_sort.slang).
            if (worlds_ > 0 && !batch.worlds.Empty()) {
                SortJob job;
                job.batch = index;
                job.slots = worlds_ + 2;
                job.first = sortTotal;
                job.args = argsTotal;
                job.count = count;
                gpu.extras[2] = sortTotal + 2 * job.slots;
                sortTotal += 2 * job.slots + count;
                argsTotal += 2 * (job.slots - 1);
                for (std::size_t d = firstDraw;
                     d < batchDraws_.size(); ++d) {
                    BatchDraw& draw = batchDraws_[d];
                    const auto k =
                        static_cast<std::uint32_t>(d - firstDraw);
                    job.vertices[k] =
                        count > 0 ? draw.vertices / count : 0;
                    draw.sorted = true;
                    draw.args = job.args + k;
                }
                sortJobs_.push_back(job);
            }
        }
    }
    if (records.empty()) {
        return;
    }
    const bool onGpu =
        std::ranges::any_of(placed, [](const Placed& p) {
            return IsGpu(p.array->Location().kind);
        });
    if (onGpu && compute_ == nullptr) {
        batchDraws_.clear();
        throw std::invalid_argument(
            "Overlay arrays on a GPU need compute interop");
    }
    // Shared with the compute device to take its copies; grown with
    // room to spare, as batches are added every frame.
    const VkDeviceSize bytes = VkDeviceSize{total} * sizeof(float);
    const BufferMemory memory = compute_ != nullptr
                                    ? BufferMemory::Shared
                                    : BufferMemory::Device;
    if (batchData_->Size() < bytes ||
        (compute_ != nullptr && batchData_->Memory() != memory)) {
        vkDeviceWaitIdle(context_.Device());
        batchData_ = std::make_shared<GpuBuffer>(
            context_, std::max<VkDeviceSize>(2 * bytes, 4096),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memory);
    }
    for (const Placed& p : placed) {
        const VkDeviceSize offset =
            VkDeviceSize{p.first} * sizeof(float);
        if (IsGpu(p.array->Location().kind)) {
            compute_->CopyToBuffer(*p.array, *batchData_, offset,
                                   p.stream);
        } else {
            uploads_.Write(
                *batchData_,
                std::span(
                    static_cast<const std::byte*>(p.array->Data()),
                    static_cast<std::size_t>(p.array->Elements()) *
                        sizeof(float)),
                offset);
        }
    }
    const std::span<const std::byte> infos =
        std::as_bytes(std::span(records));
    if (batchInfos_->Size() < infos.size()) {
        vkDeviceWaitIdle(context_.Device());
        batchInfos_ = std::make_unique<GpuBuffer>(
            context_, std::max<VkDeviceSize>(2 * infos.size(), 4096),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    }
    uploads_.Write(*batchInfos_, infos);
    if (sortJobs_.empty()) {
        return;
    }
    const VkDeviceSize sortBytes =
        VkDeviceSize{sortTotal} * sizeof(std::uint32_t);
    const VkDeviceSize argsBytes =
        VkDeviceSize{argsTotal} * sizeof(VkDrawIndirectCommand);
    if (sortData_->Size() < sortBytes ||
        drawArgs_->Size() < argsBytes) {
        vkDeviceWaitIdle(context_.Device());
        sortData_ = std::make_unique<GpuBuffer>(
            context_, std::max<VkDeviceSize>(2 * sortBytes, 4096),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
        drawArgs_ = std::make_unique<GpuBuffer>(
            context_, std::max<VkDeviceSize>(2 * argsBytes, 4096),
            kArgsUsage, BufferMemory::Device);
    }
    sortPending_ = true;
}

void OverlayPass::RecordSort(VkCommandBuffer commands) const {
    const auto barrier = [commands](VkPipelineStageFlags2 dstStage,
                                    VkAccessFlags2 dstAccess) {
        VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memory.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        memory.dstStageMask = dstStage;
        memory.dstAccessMask = dstAccess;
        VkDependencyInfo dependency{
            VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &memory;
        vkCmdPipelineBarrier2(commands, &dependency);
    };
    const DescriptorResource resources[] = {
        DescriptorResource::Of(*batchData_),
        DescriptorResource::Of(*batchInfos_),
        DescriptorResource::Of(*sortData_),
        DescriptorResource::Of(*drawArgs_)};
    // Each step for every batch, then a barrier before the next.
    for (std::size_t step = 0; step < sortPipelines_.size(); ++step) {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          sortPipelines_[step]->Handle());
        sortLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                         resources);
        for (const SortJob& job : sortJobs_) {
            const SortConstants constants{
                {job.batch, job.slots, job.first, job.args},
                {job.vertices[0], job.vertices[1], 0, 0}};
            sortLayout_.PushConstants(commands, &constants,
                                      sizeof(constants));
            const std::uint32_t threads = step == 0   ? job.slots
                                          : step == 2 ? 1
                                                      : job.count;
            vkCmdDispatch(commands, (threads + 63) / 64, 1, 1);
        }
        barrier(step + 1 < sortPipelines_.size()
                    ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                    : VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                          VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                step + 1 < sortPipelines_.size()
                    ? VK_ACCESS_2_SHADER_READ_BIT |
                          VK_ACCESS_2_SHADER_WRITE_BIT
                    : VK_ACCESS_2_SHADER_READ_BIT |
                          VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    }
    sortPending_ = false;
}

void OverlayPass::Record(VkCommandBuffer commands,
                         const Target& target) const {
    if (Empty()) {
        return;
    }
    const auto found =
        pipelines_.find({target.format, target.pushbroom});
    if (found == pipelines_.end()) {
        throw std::invalid_argument(
            "Overlays cannot draw into format " +
            std::to_string(static_cast<int>(target.format)));
    }
    // The first draw since the batches changed sorts them by world,
    // outside the rendering.
    if (sortPending_) {
        RecordSort(commands);
    }
    RenderingTarget rendering;
    rendering.color = target.color;
    rendering.extent = target.extent;
    rendering.loadColor = true;
    BeginRendering(commands, rendering);
    const DescriptorResource resources[] = {
        target.uniforms,
        DescriptorResource::Of(target.depth),
        DescriptorResource::Of(*buffers_[0]),
        DescriptorResource::Of(*buffers_[1]),
        DescriptorResource::Of(*buffers_[2]),
        DescriptorResource::Of(*batchData_),
        DescriptorResource::Of(*batchInfos_),
        DescriptorResource::Of(*sortData_)};
    layout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                 resources);
    const bool srgb = target.format == VK_FORMAT_R8G8B8A8_SRGB ||
                      target.format == VK_FORMAT_B8G8R8A8_SRGB;
    OverlayConstants constants{
        {target.world, static_cast<std::int32_t>(target.depthLayer),
         srgb ? 1 : 0, target.orthographic ? 1 : 0},
        {static_cast<float>(target.extent.width),
         static_cast<float>(target.extent.height), target.nearPlane,
         target.farPlane},
        {0, 0, 0, 0}};
    layout_.PushConstants(commands, &constants, sizeof(constants));
    // Triangles first, so lines and markers on them stay visible.
    constexpr std::array<std::size_t, 3> kOrder{1, 0, 2};
    constexpr std::array<std::uint32_t, 3> kVertices{6, 3, 6};
    for (const std::size_t kind : kOrder) {
        if (counts_[kind] == 0) {
            continue;
        }
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          found->second[kind]->Handle());
        vkCmdDraw(commands, counts_[kind] * kVertices[kind], 1, 0, 0);
    }
    // Then the batches, each expanded from its arrays as it draws:
    // arrows' heads before their shafts. A view of one world draws a
    // sorted batch's shared items and its world's, in world order.
    const bool ofWorld =
        target.world >= 0 &&
        static_cast<std::uint32_t>(target.world) < worlds_;
    for (const BatchDraw& draw : batchDraws_) {
        const bool sorted = draw.sorted && ofWorld;
        constants.batch[0] = draw.batch;
        constants.batch[1] = sorted ? 1u : 0u;
        constants.batch[3] = worlds_ + 2;  // the batch's slots
        vkCmdBindPipeline(
            commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
            found->second[3 + draw.primitive]->Handle());
        if (!sorted) {
            layout_.PushConstants(commands, &constants,
                                  sizeof(constants));
            vkCmdDraw(commands, draw.vertices, 1, 0, 0);
            continue;
        }
        for (const std::uint32_t slot :
             {0u, static_cast<std::uint32_t>(target.world) + 1}) {
            constants.batch[2] = slot;
            layout_.PushConstants(commands, &constants,
                                  sizeof(constants));
            vkCmdDrawIndirect(commands, drawArgs_->Handle(),
                              VkDeviceSize{draw.args + 2 * slot} *
                                  sizeof(VkDrawIndirectCommand),
                              1, sizeof(VkDrawIndirectCommand));
        }
    }
    vkCmdEndRendering(commands);
}

std::uint64_t OverlayPass::GpuBytes() const {
    std::uint64_t bytes = 0;
    for (const std::unique_ptr<GpuBuffer>& buffer : buffers_) {
        bytes += buffer->Size();
    }
    return bytes + batchData_->Size() + batchInfos_->Size() +
           sortData_->Size() + drawArgs_->Size();
}

}  // namespace ernie
