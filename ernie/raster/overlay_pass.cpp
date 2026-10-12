// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/overlay_pass.h"

#include "ernie/device/gpu_profile.h"
#include "ernie/kernel/embedded_shaders.h"
#include "ernie/raster/drawable.h"
#include "ernie/raster/mesh_drawable.h"
#include "ernie/util/profile.h"

#include <glm/gtc/matrix_transform.hpp>

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
    std::uint32_t text[4];
    float place[4];
    std::uint32_t more[4];
};
static_assert(sizeof(GpuBatch) == 144,
              "GpuBatch must match overlay.slang");

/// overlay_sort.slang's push constants.
struct SortConstants {
    std::uint32_t batch[4];
    std::uint32_t vertices[4];
    float frame[3][4];
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
    {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
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
constexpr std::uint32_t kBatchItemAnchors = 32;
constexpr std::uint32_t kBatchScreen = 64;
constexpr std::uint32_t kBatchHasPoints = 128;
constexpr std::uint32_t kBatchHasHeads = 256;
constexpr std::uint32_t kBatchFade = 512;
constexpr std::uint32_t kBatchDirections = 1024;
constexpr std::uint32_t kBatchPointColors = 2048;
constexpr std::uint32_t kCircleSegments = 48;
constexpr std::uint32_t kConeSides = 8;
/// Segments of a capsule (overlay.slang's `capsuleSegment`): two
/// rings, four sides and four half circles over its caps.
constexpr std::uint32_t kCapsuleSegments = 2 * 24 + 4 + 4 * 12;
/// Characters of the longest number a `Numbers` item prints but its
/// decimals: a sign, ten digits and the point.
constexpr std::uint32_t kNumberChars = 12;
/// The words of each glyph in overlay.slang's `glyphs`.
constexpr std::size_t kGlyphWords = 10;

/// Segments each item of a batch of `kind` draws.
std::uint32_t SegmentsPerItem(OverlayBatchKind kind) {
    switch (kind) {
    case OverlayBatchKind::Lines:
    case OverlayBatchKind::Arrows:
        return 1;
    case OverlayBatchKind::Markers:
    case OverlayBatchKind::Numbers:
        return 0;
    case OverlayBatchKind::Boxes:
        return 12;
    case OverlayBatchKind::Spheres:
        return 3 * kCircleSegments;
    case OverlayBatchKind::Frames:
        return 3;
    case OverlayBatchKind::Trails:
        return 0;  // a trail's length less 1 (`PrepareBatches`)
    case OverlayBatchKind::Capsules:
        return kCapsuleSegments;
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
    {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
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

/// overlay.slang's `info` of an item: its anchor (1 + its index in
/// the anchor table, 0 for none) in place of the overlay's own.
void Info(std::uint32_t (&info)[4], const OverlayItem& item,
          MarkerShape shape, std::uint32_t anchor = 0, char glyph = 0,
          bool screen = false) {
    info[0] = PackColor(item.color);
    info[1] =
        static_cast<std::uint32_t>(item.depth) |
        (static_cast<std::uint32_t>(shape) << 8) |
        (static_cast<std::uint32_t>(static_cast<unsigned char>(glyph))
         << 16) |
        (screen ? 1u << 24 : 0u);
    info[2] = static_cast<std::uint32_t>(item.world);
    info[3] = anchor;
}

void Relative(float (&out)[4], const glm::dvec3& p,
              const glm::dvec3& origin, float w = 0.0f) {
    const glm::vec3 relative(p - origin);
    out[0] = relative.x;
    out[1] = relative.y;
    out[2] = relative.z;
    out[3] = w;
}

/// The font as overlay.slang's `glyphs` reads it: per character its
/// sixteen rows two to a word, its box and its advance.
std::vector<std::uint32_t> GlyphWords() {
    std::vector<std::uint32_t> words;
    for (const OverlayGlyph& glyph : OverlayGlyphs()) {
        for (std::size_t r = 0; r < glyph.rows.size(); r += 2) {
            words.push_back(glyph.rows[r] |
                            (std::uint32_t{glyph.rows[r + 1]} << 16));
        }
        words.push_back(glyph.box[0] | (glyph.box[1] << 8) |
                        (glyph.box[2] << 16) |
                        (std::uint32_t{glyph.box[3]} << 24));
        words.push_back(glyph.advance);
    }
    return words;
}

/// The (screen, world) a host item is grouped by.
using GroupKey = std::pair<bool, std::int32_t>;

/// Items of one primitive with their groups, sorted into runs and
/// appended to `groups`.
template <typename T, typename Group>
std::vector<T> Grouped(std::vector<std::pair<GroupKey, T>>& items,
                       std::size_t primitive,
                       std::vector<Group>& groups) {
    std::stable_sort(items.begin(), items.end(),
                     [](const auto& a, const auto& b) {
                         return a.first < b.first;
                     });
    std::vector<T> sorted;
    sorted.reserve(items.size());
    for (const auto& [key, item] : items) {
        if (sorted.empty() || groups.back().screen != key.first ||
            groups.back().world != key.second ||
            groups.back().primitive != primitive) {
            groups.push_back(
                {primitive, key.second, key.first,
                 static_cast<std::uint32_t>(sorted.size()), 0});
        }
        ++groups.back().count;
        sorted.push_back(item);
    }
    return sorted;
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
    countPipeline_ = std::make_unique<ComputePipeline>(
        context, sortLayout_,
        compiler.CompileSpirv(sort, "countArgsMain"));
    posePipeline_ = std::make_unique<ComputePipeline>(
        context, sortLayout_,
        compiler.CompileSpirv(sort, "posesMain"));
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
    anchorMatrices_ = std::make_unique<GpuBuffer>(
        context, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        BufferMemory::Device);
    anchorRecords_ = std::make_unique<GpuBuffer>(
        context, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        BufferMemory::Device);
    const std::vector<std::uint32_t> glyphs = GlyphWords();
    glyphs_ = std::make_unique<GpuBuffer>(
        context, glyphs.size() * sizeof(std::uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    uploads_.Write(*glyphs_, std::as_bytes(std::span(glyphs)));
}

void OverlayPass::SetComputeInterop(ComputeInterop* compute) {
    compute_ = compute;
}

void OverlayPass::SetAnchorResolver(OverlayAnchorResolver resolve,
                                    const MeshDrawable* meshes) {
    resolve_ = std::move(resolve);
    meshes_ = meshes;
}

void OverlayPass::CreatePipelines(bool pushbroom) {
    const std::vector<ShaderModuleSource> modules{
        ProjectionModule(pushbroom),
        {"overlay", shaders::kOverlaySlang}};
    const std::vector<std::uint32_t> fragment =
        compiler_.CompileSpirv(modules, "overlayFragment");
    const std::array<std::vector<std::uint32_t>, 7> vertices{
        compiler_.CompileSpirv(modules, "segmentVertex"),
        compiler_.CompileSpirv(modules, "triangleVertex"),
        compiler_.CompileSpirv(modules, "spriteVertex"),
        compiler_.CompileSpirv(modules, "batchSegmentVertex"),
        compiler_.CompileSpirv(modules, "batchTriangleVertex"),
        compiler_.CompileSpirv(modules, "batchSpriteVertex"),
        compiler_.CompileSpirv(modules, "batchNumberVertex")};
    for (const VkFormat format : {kLdrFormat, presentFormat_}) {
        if (pipelines_.contains({format, pushbroom})) {
            continue;
        }
        std::array<std::unique_ptr<GraphicsPipeline>, 7>& pipelines =
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

std::uint32_t OverlayPass::AnchorIndex(const Path& path) {
    const auto [it, added] = anchorIndices_.try_emplace(
        path, static_cast<std::uint32_t>(anchorPaths_.size()));
    if (added) {
        anchorPaths_.push_back(path);
    }
    return it->second;
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
        // GPU arrays may have been written since.
        CopyGpuArrays();
        return;
    }
    ERNIE_ZONE("Overlay prepare");
    prepared_ = std::move(wanted);
    preparedOrigin_ = origin;
    worlds_ = worlds;
    anchorPaths_.clear();
    anchorIndices_.clear();
    groups_.clear();
    std::vector<std::pair<GroupKey, GpuSegment>> segments;
    std::vector<std::pair<GroupKey, GpuTriangle>> triangles;
    std::vector<std::pair<GroupKey, GpuSprite>> sprites;
    for (const auto& [overlay, version] : prepared_) {
        // The overlay's anchors' indices in the table, plus 1.
        std::vector<std::uint32_t> anchors;
        for (const Path& path : overlay->Anchors()) {
            anchors.push_back(AnchorIndex(path) + 1);
        }
        const auto anchorOf = [&](const OverlayItem& item) {
            return item.anchor == 0 ? 0u : anchors[item.anchor - 1];
        };
        // Items posed under anchors are in their anchors' space.
        const auto originOf = [&](const OverlayItem& item) {
            return item.anchor == 0 ? origin : glm::dvec3(0.0);
        };
        for (const OverlaySegment& segment : overlay->Segments()) {
            GpuSegment gpu;
            const glm::dvec3 from = originOf(segment.item);
            Relative(gpu.a, segment.a, from, segment.width);
            Relative(gpu.b, segment.b, from, segment.dash);
            Info(gpu.info, segment.item, MarkerShape::Square,
                 anchorOf(segment.item));
            segments.emplace_back(GroupKey{false, segment.item.world},
                                  gpu);
        }
        for (const OverlayTriangle& triangle : overlay->Fills()) {
            GpuTriangle gpu;
            const glm::dvec3 from = originOf(triangle.item);
            for (int k = 0; k < 3; ++k) {
                Relative(gpu.p[k],
                         triangle.points[static_cast<std::size_t>(k)],
                         from);
            }
            Info(gpu.info, triangle.item, MarkerShape::Square,
                 anchorOf(triangle.item));
            triangles.emplace_back(
                GroupKey{false, triangle.item.world}, gpu);
        }
        for (const OverlaySprite& sprite : overlay->Sprites()) {
            GpuSprite gpu;
            Relative(gpu.anchor, sprite.anchor,
                     sprite.screen ? glm::dvec3(0.0)
                                   : originOf(sprite.item));
            std::memcpy(gpu.rect, &sprite.rect[0], sizeof(gpu.rect));
            Info(gpu.info, sprite.item, sprite.shape,
                 sprite.screen ? 0u : anchorOf(sprite.item),
                 sprite.glyph, sprite.screen);
            sprites.emplace_back(
                GroupKey{sprite.screen, sprite.item.world}, gpu);
        }
    }
    // Triangles first, so lines and markers on them stay visible.
    const std::vector<GpuTriangle> sortedTriangles =
        Grouped(triangles, 1, groups_);
    const std::vector<GpuSegment> sortedSegments =
        Grouped(segments, 0, groups_);
    const std::vector<GpuSprite> sortedSprites =
        Grouped(sprites, 2, groups_);
    Upload(buffers_[0], std::as_bytes(std::span(sortedSegments)));
    Upload(buffers_[1], std::as_bytes(std::span(sortedTriangles)));
    Upload(buffers_[2], std::as_bytes(std::span(sortedSprites)));
    hasScreen_ = std::ranges::any_of(
        groups_, [](const Group& group) { return group.screen; });
    PrepareBatches(origin);
}

void OverlayPass::PrepareBatches(const glm::dvec3& origin) {
    batchDraws_.clear();
    sortJobs_.clear();
    countJobs_.clear();
    poseJobs_.clear();
    poseRows_ = 0;
    sortPending_ = false;
    // Words of sort data and draw arguments of the sorted batches.
    std::uint32_t sortTotal = 0;
    std::uint32_t argsTotal = 0;
    // Where each array goes in the batch buffer, in floats (or
    // int32s), and what fills it.
    std::vector<Placed> placed;
    std::vector<GpuBatch> records;
    std::uint32_t total = 0;
    const auto place = [&](Placed p) -> std::uint32_t {
        const Tensor& array = p.Source();
        if (array.Empty()) {
            return 0;
        }
        p.first = total;
        total += static_cast<std::uint32_t>(array.Elements());
        placed.push_back(std::move(p));
        return placed.back().first;
    };
    for (const auto& [overlay, version] : prepared_) {
        for (std::size_t b = 0; b < overlay->Batches().size(); ++b) {
            const OverlayBatch& batch = overlay->Batches()[b];
            const auto count =
                static_cast<std::uint32_t>(batch.Count());
            GpuBatch& gpu = records.emplace_back();
            const std::uintptr_t stream = batch.stream;
            const auto array = [&](std::string_view name,
                                   const Tensor& values) {
                return place({overlay, b, name, &values, {}, stream});
            };
            const bool numbers =
                batch.kind == OverlayBatchKind::Numbers;
            const bool alpha = !batch.colors.Empty() &&
                               batch.colors.Rank() > 1 &&
                               batch.colors.Shape().back() == 4;
            const bool posed = !batch.poses.Empty();
            const auto trail =
                static_cast<std::uint32_t>(batch.TrailLength());
            // Segments each item draws: a trail's between its points.
            const std::uint32_t segments =
                batch.kind == OverlayBatchKind::Trails
                    ? trail - 1
                    : std::max(SegmentsPerItem(batch.kind), 1u);
            gpu.shape[0] = static_cast<std::uint32_t>(batch.kind);
            gpu.shape[1] = count;
            gpu.shape[2] = std::max(segments, 1u);
            gpu.shape[3] =
                (batch.colors.Empty() ? 0 : kBatchHasColors) |
                (alpha ? kBatchHasAlpha : 0) |
                (batch.sizes.Empty() ? 0 : kBatchHasSizes) |
                (batch.worlds.Empty() ? 0 : kBatchHasWorlds) |
                (batch.orientations.Empty() ? 0
                                            : kBatchHasOrientations) |
                (batch.anchors.size() > 1 || posed ? kBatchItemAnchors
                                                   : 0) |
                (batch.screen ? kBatchScreen : 0) |
                (batch.points.Empty() ? 0 : kBatchHasPoints) |
                (batch.heads.Empty() ? 0 : kBatchHasHeads) |
                (batch.fade ? kBatchFade : 0) |
                (batch.directions ? kBatchDirections : 0) |
                (trail > 1 && !batch.colors.Empty() &&
                         static_cast<std::uint32_t>(
                             batch.colors.Elements()) ==
                             count * trail * (alpha ? 4u : 3u)
                     ? kBatchPointColors
                     : 0);
            gpu.arrays[0] = array("points", batch.points);
            gpu.arrays[1] =
                !batch.ends.Empty() ? array("ends", batch.ends)
                : !batch.halfExtents.Empty()
                    ? array("half_extents", batch.halfExtents)
                : !batch.radii.Empty()
                    ? array("radii", batch.radii)
                    : array("values", batch.values);
            gpu.arrays[2] = array("orientations", batch.orientations);
            gpu.arrays[3] = array("colors", batch.colors);
            gpu.extras[0] = array("sizes", batch.sizes);
            gpu.extras[1] = array("worlds", batch.worlds);
            gpu.more[0] =
                batch.kind == OverlayBatchKind::Capsules
                    ? array("half_heights", batch.halfHeights)
                    : array("heads", batch.heads);
            gpu.more[2] = batch.count.Empty()
                              ? 0
                              : array("count", batch.count) + 1;
            gpu.more[3] = trail;
            if (posed) {
                // A run of anchor rows of its own, after the prims',
                // written from the poses on the GPU (`posesMain`).
                const std::uint32_t indices =
                    batch.poseIndices.Empty()
                        ? 0
                        : array("pose_indices", batch.poseIndices) +
                              1;
                gpu.more[1] = indices;
                poseJobs_.push_back(
                    {static_cast<std::uint32_t>(records.size() - 1),
                     poseRows_, count, array("poses", batch.poses),
                     indices, batch.poseFrame});
                poseRows_ += count;
            }
            // One anchor shared through the table; per-item ones a
            // run of rows of their own, item i's at the first + i.
            if (batch.anchors.size() == 1) {
                gpu.extras[3] = AnchorIndex(batch.anchors[0]) + 1;
            } else if (!batch.anchors.empty()) {
                gpu.extras[3] =
                    static_cast<std::uint32_t>(anchorPaths_.size()) +
                    1;
                anchorPaths_.insert(anchorPaths_.end(),
                                    batch.anchors.begin(),
                                    batch.anchors.end());
            }
            const OverlayItem item{batch.style.color,
                                   batch.style.depth,
                                   batch.style.world};
            Info(gpu.style, item, batch.shape);
            gpu.style[3] = PackColor(batch.style.background);
            // Arrays in anchors' space are not moved to the frame's
            // origin.
            Relative(gpu.origin, batch.origin,
                     batch.anchors.empty() && !posed && !batch.screen
                         ? origin
                         : glm::dvec3(0.0),
                     batch.kind == OverlayBatchKind::Markers ||
                             numbers
                         ? batch.style.size
                         : batch.style.width);
            gpu.params[0] = static_cast<float>(batch.head);
            gpu.params[1] = static_cast<float>(batch.length);
            // Numbers' alignment, or lines' dashes.
            gpu.params[2] =
                numbers ? batch.format.align.x : batch.style.dash;
            gpu.params[3] = batch.format.align.y;
            std::uint32_t chars = 0;
            if (numbers) {
                // The text's characters, prefix then suffix, as
                // floats among the arrays.
                std::vector<float> text;
                for (const char c :
                     batch.format.prefix + batch.format.suffix) {
                    text.push_back(static_cast<float>(
                        c >= 32 && c < 127 ? c : '?'));
                }
                gpu.text[0] =
                    text.empty()
                        ? 0
                        : place({overlay, b, "", nullptr,
                                 Tensor::Copy<float>(
                                     text, {static_cast<std::int64_t>(
                                               text.size())}),
                                 stream});
                gpu.text[1] = static_cast<std::uint32_t>(
                    batch.format.prefix.size());
                gpu.text[2] = static_cast<std::uint32_t>(
                    batch.format.suffix.size());
                gpu.text[3] =
                    static_cast<std::uint32_t>(batch.format.decimals);
                chars = static_cast<std::uint32_t>(text.size()) +
                        kNumberChars + gpu.text[3];
                gpu.place[0] = batch.placement.at.x;
                gpu.place[1] = batch.placement.at.y;
                gpu.place[2] = batch.placement.offset.x;
                gpu.place[3] = batch.placement.offset.y;
                if (batch.screen) {
                    const glm::vec2 align = batch.placement.Align();
                    gpu.params[2] = align.x;
                    gpu.params[3] = align.y;
                }
                // Each item's sprites: its backing box, then a glyph
                // per character.
                gpu.shape[2] = chars + 1;
            }
            const auto index =
                static_cast<std::uint32_t>(records.size() - 1);
            const std::size_t firstDraw = batchDraws_.size();
            if (batch.kind == OverlayBatchKind::Arrows) {
                batchDraws_.push_back(
                    {index, 1, count * 2 * kConeSides * 3});
            }
            if (numbers) {
                batchDraws_.push_back({index, 3,
                                       count * gpu.shape[2] * 6,
                                       false, 0, batch.screen});
            } else if (batch.kind == OverlayBatchKind::Markers) {
                batchDraws_.push_back({index, 2, count * 6});
            } else {
                batchDraws_.push_back(
                    {index, 0, count * segments * 6});
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
            } else if (!batch.count.Empty()) {
                // Drawn by its count: its draws' arguments written on
                // the GPU (`countArgsMain`).
                CountJob job;
                job.batch = index;
                job.args = argsTotal;
                job.count = count;
                argsTotal += 2;
                for (std::size_t d = firstDraw;
                     d < batchDraws_.size(); ++d) {
                    BatchDraw& draw = batchDraws_[d];
                    const auto k =
                        static_cast<std::uint32_t>(d - firstDraw);
                    job.vertices[k] =
                        count > 0 ? draw.vertices / count : 0;
                    draw.counted = true;
                    draw.args = job.args + k;
                }
                countJobs_.push_back(job);
            }
        }
    }
    // The pose jobs' rows follow the prims'.
    for (const PoseJob& job : poseJobs_) {
        records[job.batch].extras[3] =
            static_cast<std::uint32_t>(anchorPaths_.size()) +
            job.row + 1;
    }
    placed_ = std::move(placed);
    if (records.empty()) {
        return;
    }
    const bool onGpu =
        std::ranges::any_of(placed_, [](const Placed& p) {
            return IsGpu(p.Source().Location().kind);
        });
    if (onGpu && compute_ == nullptr) {
        batchDraws_.clear();
        throw std::invalid_argument(
            "Overlay arrays on a GPU need compute interop");
    }
    // Shared with the compute device to take its copies; grown with
    // room to spare, as batches are added every frame. Arrays mapped
    // into it (`MapBatchArray`) that move go to a buffer of their
    // own, so no copy overwrites another's source; the old buffer
    // lives on in their views.
    const VkDeviceSize bytes = VkDeviceSize{total} * sizeof(float);
    const BufferMemory memory = compute_ != nullptr
                                    ? BufferMemory::Shared
                                    : BufferMemory::Device;
    const bool mappedMove =
        batchDataBase_ != nullptr &&
        std::ranges::any_of(placed_, [&](const Placed& p) {
            const auto* data =
                static_cast<const std::byte*>(p.Source().Data());
            return data >= batchDataBase_ &&
                   data < batchDataBase_ + batchData_->Size() &&
                   data != batchDataBase_ +
                               VkDeviceSize{p.first} * sizeof(float);
        });
    if (batchData_->Size() < bytes || mappedMove ||
        (compute_ != nullptr && batchData_->Memory() != memory)) {
        vkDeviceWaitIdle(context_.Device());
        batchData_ = std::make_shared<GpuBuffer>(
            context_, std::max<VkDeviceSize>(2 * bytes, 4096),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memory);
        batchDataBase_ = nullptr;
    }
    for (const Placed& p : placed_) {
        const Tensor& source = p.Source();
        if (IsGpu(source.Location().kind)) {
            continue;
        }
        uploads_.Write(
            *batchData_,
            std::span(static_cast<const std::byte*>(source.Data()),
                      static_cast<std::size_t>(source.Elements()) *
                          sizeof(float)),
            VkDeviceSize{p.first} * sizeof(float));
    }
    CopyGpuArrays();
    const std::span<const std::byte> infos =
        std::as_bytes(std::span(records));
    if (batchInfos_->Size() < infos.size()) {
        vkDeviceWaitIdle(context_.Device());
        batchInfos_ = std::make_unique<GpuBuffer>(
            context_, std::max<VkDeviceSize>(2 * infos.size(), 4096),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    }
    uploads_.Write(*batchInfos_, infos);
    if (sortJobs_.empty() && countJobs_.empty()) {
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

void OverlayPass::CopyGpuArrays() {
    bool copied = false;
    for (const Placed& p : placed_) {
        const Tensor& source = p.Source();
        if (!IsGpu(source.Location().kind) || compute_ == nullptr) {
            continue;
        }
        const VkDeviceSize offset =
            VkDeviceSize{p.first} * sizeof(float);
        // Written each frame, if only in place.
        copied = true;
        // Mapped in place: already there.
        if (batchDataBase_ != nullptr &&
            static_cast<const std::byte*>(source.Data()) ==
                batchDataBase_ + offset) {
            continue;
        }
        compute_->CopyToBuffer(source, *batchData_, offset, p.stream);
    }
    // The worlds and counts may have changed with the arrays.
    if (copied && (!sortJobs_.empty() || !countJobs_.empty())) {
        sortPending_ = true;
    }
}

Tensor OverlayPass::MapBatchArray(const Overlay& overlay,
                                  std::size_t batch,
                                  std::string_view array,
                                  std::uintptr_t stream) {
    if (compute_ == nullptr) {
        throw std::invalid_argument(
            "Mapping overlay arrays needs compute interop");
    }
    const auto found =
        std::ranges::find_if(placed_, [&](const Placed& p) {
            return p.overlay == &overlay && p.batch == batch &&
                   p.name == array && p.array != nullptr;
        });
    if (found == placed_.end()) {
        throw std::invalid_argument(
            "Overlay batch " + std::to_string(batch) + "'s '" +
            std::string(array) +
            "' is not drawn yet: render a frame with it first");
    }
    if (batchDataBase_ == nullptr) {
        batchDataBase_ = static_cast<const std::byte*>(
            compute_->View(batchData_, DType::Float32, {1}).Data());
    }
    const Tensor& source = *found->array;
    // Nothing writes it while frames already submitted may read it.
    compute_->WaitForVulkan(stream);
    Tensor view =
        compute_->View(batchData_, source.Type(), source.Shape(),
                       VkDeviceSize{found->first} * sizeof(float));
    if (view.Data() != source.Data()) {
        compute_->Copy(source, view, stream);
    }
    return view;
}

void OverlayPass::PoseAnchors(const glm::dvec3& origin) {
    gpuAnchors_ = 0;
    anchorsPending_ = false;
    anchorOrigin_ = origin;
    if (anchorPaths_.empty() && poseRows_ == 0) {
        return;
    }
    if (!anchorPaths_.empty() && !resolve_) {
        throw std::invalid_argument(
            "Overlay items posed under prims need a renderer that "
            "poses them");
    }
    ERNIE_ZONE("Overlay anchors");
    // Row-major, relative to the frame's origin, as the meshes'
    // anchor pass writes them (instance_anchors.slang); the pose
    // jobs' rows after, written on the GPU.
    std::vector<glm::vec4> matrices;
    matrices.reserve(4 * (anchorPaths_.size() + poseRows_));
    std::vector<glm::vec4> records;
    const glm::dmat4 relative =
        glm::translate(glm::dmat4(1.0), -origin);
    for (std::size_t a = 0; a < anchorPaths_.size(); ++a) {
        const OverlayAnchorPose pose = resolve_(anchorPaths_[a]);
        const glm::mat4 frame(relative * pose.frame);
        if (pose.slot != 0xFFFFFFFFu && meshes_ != nullptr) {
            MeshDrawable::AppendAnchorRecord(
                records, frame, glm::mat4(pose.below), pose.slot,
                static_cast<std::uint32_t>(a), pose.ops);
            ++gpuAnchors_;
        }
        const glm::mat4 world = frame * glm::mat4(pose.below);
        for (int r = 0; r < 4; ++r) {
            matrices.emplace_back(world[0][r], world[1][r],
                                  world[2][r], world[3][r]);
        }
    }
    matrices.resize(4 * (anchorPaths_.size() + poseRows_),
                    glm::vec4(0.0f));
    Upload(anchorMatrices_, std::as_bytes(std::span(matrices)));
    if (gpuAnchors_ > 0) {
        Upload(anchorRecords_, std::as_bytes(std::span(records)));
    }
    anchorsPending_ = gpuAnchors_ > 0 || !poseJobs_.empty();
}

void OverlayPass::RecordAnchorPoses(VkCommandBuffer commands) const {
    ERNIE_GPU_ZONE(context_, commands, "Overlay anchors");
    if (!poseJobs_.empty()) {
        const DescriptorResource resources[] = {
            DescriptorResource::Of(*batchData_),
            DescriptorResource::Of(*batchInfos_),
            DescriptorResource::Of(*sortData_),
            DescriptorResource::Of(*drawArgs_),
            DescriptorResource::Of(*anchorMatrices_)};
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          posePipeline_->Handle());
        sortLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                         resources);
        const glm::dmat4 relative =
            glm::translate(glm::dmat4(1.0), -anchorOrigin_);
        const auto first =
            static_cast<std::uint32_t>(anchorPaths_.size());
        for (const PoseJob& job : poseJobs_) {
            // The frame's top three rows, relative to the origin.
            const glm::mat4 frame(relative * job.frame);
            SortConstants constants{
                {job.batch, first + job.row, job.count, job.poses},
                {job.indices, 0, 0, 0},
                {}};
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 4; ++c) {
                    constants.frame[r][c] = frame[c][r];
                }
            }
            sortLayout_.PushConstants(commands, &constants,
                                      sizeof(constants));
            vkCmdDispatch(commands, (job.count + 63) / 64, 1, 1);
        }
    }
    if (gpuAnchors_ > 0) {
        meshes_->RecordAnchors(commands, *anchorRecords_, gpuAnchors_,
                               *anchorMatrices_,
                               AnchorOutput::ViewPoses);
    }
    VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memory.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    memory.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    vkCmdPipelineBarrier2(commands, &dependency);
    anchorsPending_ = false;
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
        DescriptorResource::Of(*drawArgs_),
        DescriptorResource::Of(*anchorMatrices_)};
    // Each step for every batch, then a barrier before the next.
    for (std::size_t step = 0;
         !sortJobs_.empty() && step < sortPipelines_.size(); ++step) {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          sortPipelines_[step]->Handle());
        sortLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                         resources);
        for (const SortJob& job : sortJobs_) {
            const SortConstants constants{
                {job.batch, job.slots, job.first, job.args},
                {job.vertices[0], job.vertices[1], 0, 0},
                {}};
            sortLayout_.PushConstants(commands, &constants,
                                      sizeof(constants));
            const std::uint32_t threads = step == 0   ? job.slots
                                          : step == 2 ? 1
                                                      : job.count;
            vkCmdDispatch(commands, (threads + 63) / 64, 1, 1);
        }
        barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                    VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                    VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                VK_ACCESS_2_SHADER_READ_BIT |
                    VK_ACCESS_2_SHADER_WRITE_BIT |
                    VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    }
    // The draws of batches drawn by their counts.
    if (!countJobs_.empty()) {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          countPipeline_->Handle());
        sortLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                         resources);
        for (const CountJob& job : countJobs_) {
            const SortConstants constants{
                {job.batch, 0, 0, job.args},
                {job.vertices[0], job.vertices[1], 0, 0},
                {}};
            sortLayout_.PushConstants(commands, &constants,
                                      sizeof(constants));
            vkCmdDispatch(commands, 1, 1, 1);
        }
        barrier(VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    }
    sortPending_ = false;
}

void OverlayPass::Record(VkCommandBuffer commands,
                         const Target& target, Layer layer) const {
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
    // and the first of a frame poses its anchors under instances,
    // outside the rendering.
    if (sortPending_) {
        RecordSort(commands);
    }
    if (anchorsPending_) {
        RecordAnchorPoses(commands);
    }
    const auto drawn = [layer](bool screen) {
        return layer == Layer::All ||
               (layer == Layer::Screen) == screen;
    };
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
        DescriptorResource::Of(*sortData_),
        DescriptorResource::Of(*anchorMatrices_),
        DescriptorResource::Of(*glyphs_)};
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
    // The host items: of a view of one world, the shared groups and
    // its world's.
    constexpr std::array<std::uint32_t, 3> kVertices{6, 3, 6};
    std::size_t bound = 3;
    for (const Group& group : groups_) {
        if (!drawn(group.screen) ||
            (target.world >= 0 && group.world >= 0 &&
             group.world != target.world)) {
            continue;
        }
        if (bound != group.primitive) {
            vkCmdBindPipeline(
                commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                found->second[group.primitive]->Handle());
            bound = group.primitive;
        }
        constants.batch[0] = group.first;
        layout_.PushConstants(commands, &constants,
                              sizeof(constants));
        vkCmdDraw(commands, group.count * kVertices[group.primitive],
                  1, 0, 0);
    }
    // Then the batches, each expanded from its arrays as it draws:
    // arrows' heads before their shafts. A view of one world draws a
    // sorted batch's shared items and its world's, in world order.
    const bool ofWorld =
        target.world >= 0 &&
        static_cast<std::uint32_t>(target.world) < worlds_;
    for (const BatchDraw& draw : batchDraws_) {
        if (!drawn(draw.screen)) {
            continue;
        }
        const bool sorted = draw.sorted && ofWorld;
        constants.batch[0] = draw.batch;
        constants.batch[1] = sorted ? 1u : 0u;
        constants.batch[3] = worlds_ + 2;  // the batch's slots
        vkCmdBindPipeline(
            commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
            found->second[3 + draw.primitive]->Handle());
        if (!sorted) {
            constants.batch[1] = 0;
            layout_.PushConstants(commands, &constants,
                                  sizeof(constants));
            // Drawn by its count, or every row (the vertex shaders
            // leaving out those past a count).
            if (draw.counted) {
                vkCmdDrawIndirect(commands, drawArgs_->Handle(),
                                  VkDeviceSize{draw.args} *
                                      sizeof(VkDrawIndirectCommand),
                                  1, sizeof(VkDrawIndirectCommand));
            } else {
                vkCmdDraw(commands, draw.vertices, 1, 0, 0);
            }
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
           sortData_->Size() + drawArgs_->Size() +
           anchorMatrices_->Size() + anchorRecords_->Size() +
           glyphs_->Size();
}

}  // namespace ernie
