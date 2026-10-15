// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/overlay_pass.h"

#include "ernie/device/gpu_profile.h"
#include "ernie/kernel/embedded_shaders.h"
#include "ernie/raster/drawable.h"
#include "ernie/raster/mesh_drawable.h"
#include "ernie/util/profile.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

// Every item's `extra` (overlay.slang): a segment's end colour, the
// item's id, kExtra* flags.
struct GpuSegment {
    float a[4];
    float b[4];
    std::uint32_t info[4];
    std::uint32_t extra[4];
};
static_assert(sizeof(GpuSegment) == 64,
              "GpuSegment must match overlay.slang");

struct GpuTriangle {
    float p[3][4];
    std::uint32_t info[4];
    std::uint32_t extra[4];
};
static_assert(sizeof(GpuTriangle) == 80,
              "GpuTriangle must match overlay.slang");

struct GpuSprite {
    float anchor[4];
    float rect[4];
    std::uint32_t info[4];
    std::uint32_t extra[4];
};
static_assert(sizeof(GpuSprite) == 64,
              "GpuSprite must match overlay.slang");

// overlay.slang's kExtra* flags and kInfoMetric.
constexpr std::uint32_t kExtraEndColor = 1;
constexpr std::uint32_t kExtraHalo = 2;
constexpr std::uint32_t kInfoMetric = 1u << 25;
/// overlay.slang's kPickBatch, and the format picks draw into.
constexpr std::uint32_t kPickBatch = 0x80000000u;
constexpr VkFormat kPickFormat = VK_FORMAT_R32G32B32A32_UINT;

struct OverlayConstants {
    std::int32_t view[4];
    float viewport[4];
    std::uint32_t batch[4];
    std::uint32_t layer[4];
};
static_assert(sizeof(OverlayConstants) == 64,
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
    {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
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
constexpr std::uint32_t kBatchEndColors = 4096;
constexpr std::uint32_t kCircleSegments = 48;
constexpr std::uint32_t kConeSides = 8;
/// Segments of a capsule (overlay.slang's `capsuleSegment`): two
/// rings, four sides and four half circles over its caps.
constexpr std::uint32_t kCapsuleSegments = 2 * 24 + 4 + 4 * 12;
/// Characters of the longest number a `Numbers` item prints but its
/// decimals: a sign, ten digits and the point.
constexpr std::uint32_t kNumberChars = 12;
/// overlay.slang's `glyphs`: the words of each glyph, then of the
/// whole font's.
constexpr std::size_t kGlyphWords = 8;
constexpr std::size_t kFontWords = 4;

/// Segments each item of a batch of `kind` draws.
std::uint32_t SegmentsPerItem(OverlayBatchKind kind) {
    switch (kind) {
    case OverlayBatchKind::Lines:
    case OverlayBatchKind::Arrows:
        return 1;
    case OverlayBatchKind::Markers:
    case OverlayBatchKind::Numbers:
    case OverlayBatchKind::Texts:
    case OverlayBatchKind::Image:
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
    {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
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
          MarkerShape shape, std::uint32_t anchor = 0,
          std::uint8_t glyph = 0, bool screen = false) {
    info[0] = PackColor(item.color);
    info[1] = static_cast<std::uint32_t>(item.depth) |
              (static_cast<std::uint32_t>(shape) << 8) |
              (std::uint32_t{glyph} << 16) |
              (screen ? 1u << 24 : 0u) |
              (item.metric ? kInfoMetric : 0u);
    info[2] = static_cast<std::uint32_t>(item.world);
    info[3] = anchor;
}

/// overlay.slang's `extra` of an item: its id, and a segment's end
/// colour.
void Extra(std::uint32_t (&extra)[4], const OverlayItem& item,
           const std::optional<glm::vec4>& end = {}) {
    extra[0] = end ? PackColor(*end) : 0u;
    extra[1] = item.id;
    extra[2] = end ? kExtraEndColor : 0u;
    extra[3] = 0;
}

/// Segment `a` to `b` as pieces `emit(a, b, t0, t1)` (`t` the
/// fraction along it), split where it passes `origin` (the frame's,
/// near the eye) so that each piece's ends lie within `kReach` times
/// its nearest point's distance from it (at least a metre). Float32,
/// which the GPU draws in, keeps a point to a part in ten million of
/// that distance, so a line as long as an orbit passing the eye
/// would otherwise be drawn kilometres from where it is.
template <typename Emit>
void SplitNearOrigin(const glm::dvec3& a, const glm::dvec3& b,
                     const glm::dvec3& origin, Emit&& emit) {
    constexpr double kReach = 100.0;
    const glm::dvec3 along = b - a;
    const double length2 = glm::dot(along, along);
    const auto point = [&](double t) { return a + t * along; };
    const auto distance = [&](double t) {
        return glm::length(point(t) - origin);
    };
    // Pieces [t0, t1] still to emit, in order.
    std::vector<std::pair<double, double>> pending{{0.0, 1.0}};
    std::vector<std::pair<double, double>> done;
    while (!pending.empty()) {
        const auto [t0, t1] = pending.back();
        pending.pop_back();
        // The piece's point nearest the origin.
        const double nearest =
            length2 > 0.0
                ? std::clamp(glm::dot(origin - a, along) / length2,
                             t0, t1)
                : t0;
        const double closest = std::max(distance(nearest), 1.0);
        const double farthest = std::max(distance(t0), distance(t1));
        const double span = (t1 - t0) * std::sqrt(length2);
        if (farthest <= kReach * closest || span <= 1e-3 * closest) {
            done.emplace_back(t0, t1);
            continue;
        }
        if (nearest > t0 && nearest < t1) {
            pending.emplace_back(nearest, t1);
            pending.emplace_back(t0, nearest);
            continue;
        }
        // Nearest at an end: a piece from it as long as half the
        // reach, the rest split again.
        const double step =
            std::min(0.5 * kReach * closest / span, 0.5) * (t1 - t0);
        if (nearest == t0) {
            pending.emplace_back(t0 + step, t1);
            pending.emplace_back(t0, t0 + step);
        } else {
            pending.emplace_back(t1 - step, t1);
            pending.emplace_back(t0, t1 - step);
        }
    }
    for (const auto& [t0, t1] : done) {
        emit(point(t0), point(t1), t0, t1);
    }
}

void Relative(float (&out)[4], const glm::dvec3& p,
              const glm::dvec3& origin, float w = 0.0f) {
    const glm::vec3 relative(p - origin);
    out[0] = relative.x;
    out[1] = relative.y;
    out[2] = relative.z;
    out[3] = w;
}

/// Bytes as words, four to one, little-endian, the last padded with
/// 0s.
void AppendBytes(std::vector<std::uint32_t>& words,
                 std::span<const std::uint8_t> bytes) {
    const std::size_t first = words.size();
    words.resize(first + (bytes.size() + 3) / 4, 0u);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        words[first + i / 4] |= std::uint32_t{bytes[i]}
                                << (8 * (i % 4));
    }
}

/// The font as overlay.slang's `glyphs` reads it: per Latin-1 code
/// its field's box, its advance, its field's first texel and size;
/// the text's height; the kerning; the fields.
std::vector<std::uint32_t> GlyphWords() {
    const OverlayFont& font = OverlayTextFont();
    std::vector<std::uint32_t> words;
    for (const OverlayGlyph& glyph : font.glyphs) {
        for (int k = 0; k < 4; ++k) {
            words.push_back(
                std::bit_cast<std::uint32_t>(glyph.box[k]));
        }
        words.push_back(std::bit_cast<std::uint32_t>(glyph.advance));
        words.push_back(glyph.texel);
        words.push_back(glyph.width |
                        (std::uint32_t{glyph.height} << 16));
        words.push_back(0u);
    }
    static_assert(kGlyphWords == 8);
    words.push_back(std::bit_cast<std::uint32_t>(font.textHeight));
    words.resize(words.size() + kFontWords - 1, 0u);
    AppendBytes(words,
                std::span(reinterpret_cast<const std::uint8_t*>(
                              font.kerning.data()),
                          font.kerning.size()));
    AppendBytes(words, font.field);
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
    emptyItems_ = std::make_unique<GpuBuffer>(
        context, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        BufferMemory::Device);
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
    anchorWorlds_ = std::make_unique<GpuBuffer>(
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
    const std::array<std::vector<std::uint32_t>, 8> vertices{
        compiler_.CompileSpirv(modules, "segmentVertex"),
        compiler_.CompileSpirv(modules, "triangleVertex"),
        compiler_.CompileSpirv(modules, "spriteVertex"),
        compiler_.CompileSpirv(modules, "batchSegmentVertex"),
        compiler_.CompileSpirv(modules, "batchTriangleVertex"),
        compiler_.CompileSpirv(modules, "batchSpriteVertex"),
        compiler_.CompileSpirv(modules, "batchNumberVertex"),
        compiler_.CompileSpirv(modules, "batchImageVertex")};
    for (const VkFormat format : {kLdrFormat, presentFormat_}) {
        if (pipelines_.contains({format, pushbroom})) {
            continue;
        }
        std::array<std::unique_ptr<GraphicsPipeline>, 8>& pipelines =
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
    // Each overlay's host items again only if it changed (or the
    // origin did, which they are relative to): a HUD redrawn every
    // frame leaves the others' as they are.
    const bool moved = origin != preparedOrigin_;
    std::vector<HostLayer> layers;
    for (const auto& [overlay, version] : wanted) {
        const auto old = std::ranges::find_if(
            layers_, [&](const HostLayer& layer) {
                return layer.overlay == overlay;
            });
        if (old != layers_.end() && old->version == version &&
            !moved) {
            layers.push_back(std::move(*old));
        } else {
            HostLayer& layer = layers.emplace_back();
            layer.overlay = overlay;
            layer.version = version;
            if (old != layers_.end()) {
                layer.buffers = std::move(old->buffers);
            }
            BuildLayer(layer, origin);
        }
    }
    // Layers dropped (hidden or removed) free buffers earlier frames
    // may still read.
    if (std::ranges::any_of(layers_, [](const HostLayer& layer) {
            return layer.buffers[0] != nullptr;
        })) {
        vkDeviceWaitIdle(context_.Device());
    }
    layers_ = std::move(layers);
    prepared_ = std::move(wanted);
    preparedOrigin_ = origin;
    worlds_ = worlds;
    // The anchor table: each layer's anchors, a run from its base,
    // then the batches'.
    anchorPaths_.clear();
    anchorIndices_.clear();
    hasScreen_ = false;
    for (HostLayer& layer : layers_) {
        layer.anchorBase =
            static_cast<std::uint32_t>(anchorPaths_.size());
        anchorPaths_.insert(anchorPaths_.end(),
                            layer.overlay->Anchors().begin(),
                            layer.overlay->Anchors().end());
        hasScreen_ =
            hasScreen_ ||
            std::ranges::any_of(layer.groups, [](const Group& group) {
                return group.screen;
            });
    }
    PrepareBatches(origin);
}

void OverlayPass::BuildLayer(HostLayer& layer,
                             const glm::dvec3& origin) {
    const Overlay& overlay = *layer.overlay;
    std::vector<std::pair<GroupKey, GpuSegment>> segments;
    std::vector<std::pair<GroupKey, GpuTriangle>> triangles;
    std::vector<std::pair<GroupKey, GpuSprite>> sprites;
    // Items posed under anchors are in their anchors' space, which
    // count from the layer's base (`HostLayer::anchorBase`).
    const auto originOf = [&](const OverlayItem& item) {
        return item.anchor == 0 ? origin : glm::dvec3(0.0);
    };
    // An item in its hover colour while its id is hovered.
    const auto shown = [&](const OverlayItem& item) {
        OverlayItem drawn = item;
        if (item.id != 0 && item.id == overlay.Hovered() &&
            item.hover) {
            drawn.color = *item.hover;
        }
        return drawn;
    };
    for (const OverlaySegment& segment : overlay.Segments()) {
        const glm::dvec3 from = originOf(segment.item);
        const OverlayItem item = shown(segment.item);
        SplitNearOrigin(
            segment.a, segment.b, from,
            [&](const glm::dvec3& a, const glm::dvec3& b, double t0,
                double t1) {
                GpuSegment gpu;
                Relative(gpu.a, a, from, segment.width);
                Relative(gpu.b, b, from, segment.dash);
                OverlayItem piece = item;
                std::optional<glm::vec4> end;
                if (segment.endColor) {
                    piece.color =
                        glm::mix(item.color, *segment.endColor,
                                 static_cast<float>(t0));
                    end = glm::mix(item.color, *segment.endColor,
                                   static_cast<float>(t1));
                }
                Info(gpu.info, piece, MarkerShape::Square,
                     segment.item.anchor);
                Extra(gpu.extra, segment.item, end);
                segments.emplace_back(
                    GroupKey{false, segment.item.world}, gpu);
            });
    }
    for (const OverlayTriangle& triangle : overlay.Fills()) {
        GpuTriangle gpu;
        const glm::dvec3 from = originOf(triangle.item);
        for (int k = 0; k < 3; ++k) {
            Relative(gpu.p[k],
                     triangle.points[static_cast<std::size_t>(k)],
                     from);
        }
        Info(gpu.info, shown(triangle.item), MarkerShape::Square,
             triangle.item.anchor);
        Extra(gpu.extra, triangle.item);
        triangles.emplace_back(GroupKey{false, triangle.item.world},
                               gpu);
    }
    for (const OverlaySprite& sprite : overlay.Sprites()) {
        GpuSprite gpu;
        Relative(gpu.anchor, sprite.anchor,
                 sprite.screen ? glm::dvec3(0.0)
                               : originOf(sprite.item));
        std::memcpy(gpu.rect, &sprite.rect[0], sizeof(gpu.rect));
        Info(gpu.info, shown(sprite.item), sprite.shape,
             sprite.screen ? 0u : sprite.item.anchor, sprite.glyph,
             sprite.screen);
        Extra(gpu.extra, sprite.item);
        if (sprite.halo > 0.0f) {
            gpu.extra[0] = PackColor(sprite.haloColor);
            gpu.extra[2] |= kExtraHalo;
            gpu.extra[3] = std::bit_cast<std::uint32_t>(sprite.halo);
        }
        sprites.emplace_back(
            GroupKey{sprite.screen, sprite.item.world}, gpu);
    }
    // Triangles first, so lines and markers on them stay visible.
    layer.groups.clear();
    const std::vector<GpuTriangle> sortedTriangles =
        Grouped(triangles, 1, layer.groups);
    const std::vector<GpuSegment> sortedSegments =
        Grouped(segments, 0, layer.groups);
    const std::vector<GpuSprite> sortedSprites =
        Grouped(sprites, 2, layer.groups);
    for (std::unique_ptr<GpuBuffer>& buffer : layer.buffers) {
        if (buffer == nullptr) {
            buffer = std::make_unique<GpuBuffer>(
                context_, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Device);
        }
    }
    Upload(layer.buffers[0],
           std::as_bytes(std::span(sortedSegments)));
    Upload(layer.buffers[1],
           std::as_bytes(std::span(sortedTriangles)));
    Upload(layer.buffers[2], std::as_bytes(std::span(sortedSprites)));
}

void OverlayPass::PrepareBatches(const glm::dvec3& origin) {
    batchDraws_.clear();
    batchOwners_.clear();
    sortJobs_.clear();
    countJobs_.clear();
    poseJobs_.clear();
    poseRows_ = 0;
    sortsByAnchors_ = false;
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
        // In words of 4 bytes: an image's bytes four to a word.
        p.first = total;
        total += static_cast<std::uint32_t>(
            (static_cast<std::size_t>(array.Elements()) *
                 SizeOf(array.Type()) +
             3) /
            4);
        placed.push_back(std::move(p));
        return placed.back().first;
    };
    for (const auto& [overlay, version] : prepared_) {
        for (std::size_t b = 0; b < overlay->Batches().size(); ++b) {
            const OverlayBatch& batch = overlay->Batches()[b];
            const auto count =
                static_cast<std::uint32_t>(batch.Count());
            GpuBatch& gpu = records.emplace_back();
            batchOwners_.emplace_back(overlay, b);
            const std::uintptr_t stream = batch.stream;
            const auto array = [&](std::string_view name,
                                   const Tensor& values) {
                return place({overlay, b, name, &values, {}, stream});
            };
            const bool numbers =
                batch.kind == OverlayBatchKind::Numbers;
            // Numbers and texts draw as glyphs (`batchNumberVertex`).
            const bool text =
                numbers || batch.kind == OverlayBatchKind::Texts;
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
                     : 0) |
                // Lines' and arrows' colours a row per end.
                ((batch.kind == OverlayBatchKind::Lines ||
                  batch.kind == OverlayBatchKind::Arrows) &&
                         !batch.colors.Empty() &&
                         static_cast<std::uint32_t>(
                             batch.colors.Elements()) ==
                             2 * count * (alpha ? 4u : 3u)
                     ? kBatchEndColors
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
                const std::uint32_t poses =
                    array("poses", batch.poses);
                const std::uint32_t worlds =
                    batch.poseWorlds.Empty()
                        ? 0
                        : array("pose_worlds", batch.poseWorlds) + 1;
                poseJobs_.push_back(
                    {static_cast<std::uint32_t>(records.size() - 1),
                     poseRows_, count, poses, indices,
                     batch.poseFrame, worlds});
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
            OverlayItem item{batch.style.color, batch.style.depth,
                             batch.style.world};
            item.metric = batch.style.metric;
            Info(gpu.style, item, batch.shape);
            gpu.style[3] = PackColor(batch.style.background);
            // Arrays in anchors' space are not moved to the frame's
            // origin.
            Relative(gpu.origin, batch.origin,
                     batch.anchors.empty() && !posed && !batch.screen
                         ? origin
                         : glm::dvec3(0.0),
                     batch.kind == OverlayBatchKind::Markers || text
                         ? batch.style.size
                         : batch.style.width);
            gpu.params[0] = static_cast<float>(batch.head);
            gpu.params[1] = static_cast<float>(batch.length);
            // Text's alignment, or lines' dashes.
            gpu.params[2] =
                text ? batch.format.align.x : batch.style.dash;
            gpu.params[3] = batch.format.align.y;
            std::uint32_t chars = 0;
            if (text) {
                gpu.place[0] = batch.placement.at.x;
                gpu.place[1] = batch.placement.at.y;
                gpu.place[2] = batch.placement.offset.x;
                gpu.place[3] = batch.placement.offset.y;
                // Its characters' halo in place of their box.
                gpu.params[0] = batch.style.halo;
                if (batch.screen) {
                    const glm::vec2 align = batch.placement.Align();
                    gpu.params[2] = align.x;
                    gpu.params[3] = align.y;
                }
            }
            if (batch.kind == OverlayBatchKind::Image) {
                // Its texels, a word each but for 3 channels, its
                // size and channels; on the screen, its place and
                // size.
                gpu.text[0] = array("image", batch.image);
                gpu.text[1] = static_cast<std::uint32_t>(
                    batch.image.Shape()[1]);
                gpu.text[2] = static_cast<std::uint32_t>(
                    batch.image.Shape()[0]);
                gpu.text[3] = static_cast<std::uint32_t>(
                    batch.image.Shape()[2]);
                gpu.place[0] = batch.placement.at.x;
                gpu.place[1] = batch.placement.at.y;
                gpu.place[2] = batch.placement.offset.x;
                gpu.place[3] = batch.placement.offset.y;
                gpu.params[0] = batch.imageSize.x;
                gpu.params[1] = batch.imageSize.y;
                const glm::vec2 align = batch.placement.Align();
                gpu.params[2] = align.x;
                gpu.params[3] = align.y;
            }
            if (batch.kind == OverlayBatchKind::Texts) {
                // Each row of codes, its length the most characters.
                gpu.text[0] = array("chars", batch.chars);
                chars = static_cast<std::uint32_t>(
                    batch.chars.Shape()[1]);
                gpu.text[1] = chars;
                gpu.shape[2] = chars + 1;
            }
            if (numbers) {
                // The text's characters, prefix then suffix, as
                // floats among the arrays.
                const std::vector<std::uint8_t> prefix =
                    OverlayCodes(batch.format.prefix);
                const std::vector<std::uint8_t> suffix =
                    OverlayCodes(batch.format.suffix);
                std::vector<float> affixes;
                for (const std::uint8_t c : prefix) {
                    affixes.push_back(static_cast<float>(c));
                }
                for (const std::uint8_t c : suffix) {
                    affixes.push_back(static_cast<float>(c));
                }
                gpu.text[0] =
                    affixes.empty()
                        ? 0
                        : place({overlay, b, "", nullptr,
                                 Tensor::Copy<float>(
                                     affixes,
                                     {static_cast<std::int64_t>(
                                         affixes.size())}),
                                 stream});
                gpu.text[1] =
                    static_cast<std::uint32_t>(prefix.size());
                gpu.text[2] =
                    static_cast<std::uint32_t>(suffix.size());
                gpu.text[3] =
                    static_cast<std::uint32_t>(batch.format.decimals);
                chars = static_cast<std::uint32_t>(affixes.size()) +
                        kNumberChars + gpu.text[3];
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
            if (text) {
                batchDraws_.push_back({index, 3,
                                       count * gpu.shape[2] * 6,
                                       false, 0, batch.screen});
            } else if (batch.kind == OverlayBatchKind::Image) {
                batchDraws_.push_back(
                    {index, 4, count * 6, false, 0, batch.screen});
            } else if (batch.kind == OverlayBatchKind::Markers) {
                batchDraws_.push_back({index, 2, count * 6});
            } else {
                batchDraws_.push_back(
                    {index, 0, count * segments * 6});
            }
            // With worlds, their own or their anchors', and batched
            // views, sorted by world: slots for the shared items,
            // each world's, and worlds no view shows
            // (overlay_sort.slang).
            const bool anchorWorlds =
                batch.worlds.Empty() && batch.style.world < 0 &&
                (posed || !batch.anchors.empty());
            sortsByAnchors_ =
                sortsByAnchors_ || (worlds_ > 0 && anchorWorlds);
            if (worlds_ > 0 &&
                (!batch.worlds.Empty() || anchorWorlds)) {
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
                          SizeOf(source.Type())),
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
    std::vector<std::int32_t> worlds;
    worlds.reserve(anchorPaths_.size() + poseRows_);
    std::vector<glm::vec4> records;
    const glm::dmat4 relative =
        glm::translate(glm::dmat4(1.0), -origin);
    for (std::size_t a = 0; a < anchorPaths_.size(); ++a) {
        const OverlayAnchorPose pose = resolve_(anchorPaths_[a]);
        worlds.push_back(pose.world);
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
    worlds.resize(anchorPaths_.size() + poseRows_, -1);
    Upload(anchorMatrices_, std::as_bytes(std::span(matrices)));
    Upload(anchorWorlds_, std::as_bytes(std::span(worlds)));
    if (gpuAnchors_ > 0) {
        Upload(anchorRecords_, std::as_bytes(std::span(records)));
    }
    anchorsPending_ = gpuAnchors_ > 0 || !poseJobs_.empty();
    // Items may follow their anchors into other worlds.
    if (sortsByAnchors_) {
        sortPending_ = true;
    }
}

void OverlayPass::RecordAnchorPoses(VkCommandBuffer commands) const {
    ERNIE_GPU_ZONE(context_, commands, "Overlay anchors");
    if (!poseJobs_.empty()) {
        const DescriptorResource resources[] = {
            DescriptorResource::Of(*batchData_),
            DescriptorResource::Of(*batchInfos_),
            DescriptorResource::Of(*sortData_),
            DescriptorResource::Of(*drawArgs_),
            DescriptorResource::Of(*anchorMatrices_),
            DescriptorResource::Of(*anchorWorlds_)};
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
                {job.indices, job.worlds, 0, 0},
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
    // For the draws, and the sort reading the worlds.
    memory.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
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
        DescriptorResource::Of(*anchorMatrices_),
        DescriptorResource::Of(*anchorWorlds_)};
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
    // Poses first: sorting by world reads the worlds they write.
    if (anchorsPending_) {
        RecordAnchorPoses(commands);
    }
    if (sortPending_) {
        RecordSort(commands);
    }
    RenderingTarget rendering;
    rendering.color = target.color;
    rendering.extent = target.extent;
    rendering.loadColor = true;
    BeginRendering(commands, rendering);
    DrawItems(commands, target, layer, found->second, nullptr);
    vkCmdEndRendering(commands);
}

void OverlayPass::DrawItems(
    VkCommandBuffer commands, const Target& target, Layer layer,
    const std::array<std::unique_ptr<GraphicsPipeline>, 8>& pipelines,
    const glm::uvec2* pick) const {
    const auto drawn = [layer](bool screen) {
        return layer == Layer::All ||
               (layer == Layer::Screen) == screen;
    };
    // The bindings with a layer's items (or none).
    const auto bind = [&](const HostLayer* items) {
        const auto of = [&](std::size_t k) {
            return DescriptorResource::Of(
                items != nullptr ? *items->buffers[k] : *emptyItems_);
        };
        const DescriptorResource resources[] = {
            target.uniforms,
            DescriptorResource::Of(target.depth),
            of(0),
            of(1),
            of(2),
            DescriptorResource::Of(*batchData_),
            DescriptorResource::Of(*batchInfos_),
            DescriptorResource::Of(*sortData_),
            DescriptorResource::Of(*anchorMatrices_),
            DescriptorResource::Of(*glyphs_),
            DescriptorResource::Of(*anchorWorlds_)};
        layout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                     resources);
    };
    const bool srgb = target.format == VK_FORMAT_R8G8B8A8_SRGB ||
                      target.format == VK_FORMAT_B8G8R8A8_SRGB;
    OverlayConstants constants{
        {target.world, static_cast<std::int32_t>(target.depthLayer),
         srgb ? 1 : 0, target.orthographic ? 1 : 0},
        {static_cast<float>(target.extent.width),
         static_cast<float>(target.extent.height), target.nearPlane,
         target.farPlane},
        {0, 0, 0, 0},
        {0, pick != nullptr ? pick->x : 0u,
         pick != nullptr ? pick->y : 0u, 0}};
    // Each layer's host items: of a view of one world, the shared
    // groups and its world's, their anchors from the layer's base.
    constexpr std::array<std::uint32_t, 3> kVertices{6, 3, 6};
    for (std::size_t l = 0; l < layers_.size(); ++l) {
        const HostLayer& items = layers_[l];
        if (items.groups.empty()) {
            continue;
        }
        bind(&items);
        constants.layer[0] = items.anchorBase;
        constants.layer[3] = static_cast<std::uint32_t>(l) + 1;
        std::size_t bound = 3;
        for (const Group& group : items.groups) {
            if (!drawn(group.screen) ||
                (target.world >= 0 && group.world >= 0 &&
                 group.world != target.world)) {
                continue;
            }
            if (bound != group.primitive) {
                vkCmdBindPipeline(
                    commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipelines[group.primitive]->Handle());
                bound = group.primitive;
            }
            constants.batch[0] = group.first;
            layout_.PushConstants(commands, &constants,
                                  sizeof(constants));
            vkCmdDraw(commands,
                      group.count * kVertices[group.primitive], 1, 0,
                      0);
        }
    }
    // Batch items' anchors are rows of the table themselves.
    bind(nullptr);
    constants.layer[0] = 0;
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
        constants.layer[3] = kPickBatch | draw.batch;
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelines[3 + draw.primitive]->Handle());
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
}

void OverlayPass::RecordPick(VkCommandBuffer commands,
                             const Target& target, std::uint32_t x,
                             std::uint32_t y,
                             const GpuBuffer& readback) const {
    // The pipelines writing what drew each item, made on first use.
    if (pickPipelines_[0] == nullptr) {
        const std::vector<ShaderModuleSource> modules{
            ProjectionModule(target.pushbroom),
            {"overlay", shaders::kOverlaySlang}};
        const std::vector<std::uint32_t> fragment =
            compiler_.CompileSpirv(modules, "overlayPickFragment");
        constexpr const char* kEntries[] = {
            "segmentVertex",       "triangleVertex",
            "spriteVertex",        "batchSegmentVertex",
            "batchTriangleVertex", "batchSpriteVertex",
            "batchNumberVertex",   "batchImageVertex"};
        for (std::size_t k = 0; k < pickPipelines_.size(); ++k) {
            // Kept alive here: the desc only holds a span of it.
            const std::vector<std::uint32_t> vertex =
                compiler_.CompileSpirv(modules, kEntries[k]);
            GraphicsPipelineDesc desc;
            desc.vertexSpirv = vertex;
            desc.fragmentSpirv = fragment;
            desc.colorFormat = kPickFormat;
            desc.depthFormat = VK_FORMAT_UNDEFINED;
            pickPipelines_[k] = std::make_unique<GraphicsPipeline>(
                context_, layout_, desc);
        }
        pickImage_ = std::make_unique<GpuImage>(
            context_, ImageDesc{1, 1, 1, 1, kPickFormat,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT});
    }
    pickImage_->Transition(commands,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    RenderingTarget rendering;
    rendering.color = pickImage_->View();
    rendering.extent = {1, 1};
    BeginRendering(commands, rendering);
    // The view's viewport moved so pixel (x, y) lands on the target's
    // only one, flipped as BeginRendering's.
    const auto width = static_cast<float>(target.extent.width);
    const auto height = static_cast<float>(target.extent.height);
    const VkViewport viewport{-static_cast<float>(x),
                              height - static_cast<float>(y),
                              width,
                              -height,
                              0.0f,
                              1.0f};
    vkCmdSetViewport(commands, 0, 1, &viewport);
    const glm::uvec2 pixel(x, y);
    DrawItems(commands, target, Layer::All, pickPipelines_, &pixel);
    vkCmdEndRendering(commands);
    pickImage_->Transition(commands,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {1, 1, 1};
    vkCmdCopyImageToBuffer(commands, pickImage_->Handle(),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.Handle(), 1, &region);
}

std::optional<OverlayPass::Picked> OverlayPass::DecodePick(
    std::span<const std::uint32_t, 4> words) const {
    if (words[2] == 0) {
        return std::nullopt;
    }
    Picked picked;
    picked.item = words[1];
    if ((words[0] & kPickBatch) != 0) {
        // A batch's record: its overlay and its index there.
        const std::uint32_t batch = words[0] & ~kPickBatch;
        if (batch >= batchOwners_.size()) {
            return std::nullopt;
        }
        picked.overlay = batchOwners_[batch].first;
        picked.batch =
            static_cast<std::int32_t>(batchOwners_[batch].second);
        return picked;
    }
    if (words[0] == 0 || words[0] > layers_.size()) {
        return std::nullopt;
    }
    picked.overlay = layers_[words[0] - 1].overlay;
    return picked;
}

std::uint64_t OverlayPass::GpuBytes() const {
    std::uint64_t bytes = 0;
    for (const HostLayer& layer : layers_) {
        for (const std::unique_ptr<GpuBuffer>& buffer :
             layer.buffers) {
            bytes += buffer->Size();
        }
    }
    return bytes + batchData_->Size() + batchInfos_->Size() +
           sortData_->Size() + drawArgs_->Size() +
           anchorMatrices_->Size() + anchorRecords_->Size() +
           anchorWorlds_->Size() + glyphs_->Size();
}

}  // namespace ernie
