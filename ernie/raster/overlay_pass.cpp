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
};
static_assert(sizeof(OverlayConstants) == 32,
              "OverlayConstants must match overlay.slang");

constexpr VkShaderStageFlags kGraphics =
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

constexpr DescriptorBinding kBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kGraphics},
    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
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
      layout_(context, kBindings, sizeof(OverlayConstants)) {
    CreatePipelines(false);
    // Placeholders, so the bindings always have a buffer.
    for (std::unique_ptr<GpuBuffer>& buffer : buffers_) {
        buffer = std::make_unique<GpuBuffer>(
            context, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            BufferMemory::Device);
    }
}

void OverlayPass::CreatePipelines(bool pushbroom) {
    const std::vector<ShaderModuleSource> modules{
        ProjectionModule(pushbroom),
        {"overlay", shaders::kOverlaySlang}};
    const std::vector<std::uint32_t> fragment =
        compiler_.CompileSpirv(modules, "overlayFragment");
    const std::array<std::vector<std::uint32_t>, 3> vertices{
        compiler_.CompileSpirv(modules, "segmentVertex"),
        compiler_.CompileSpirv(modules, "triangleVertex"),
        compiler_.CompileSpirv(modules, "spriteVertex")};
    for (const VkFormat format : {kLdrFormat, presentFormat_}) {
        if (pipelines_.contains({format, pushbroom})) {
            continue;
        }
        std::array<std::unique_ptr<GraphicsPipeline>, 3>& pipelines =
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
                          const glm::dvec3& origin, bool pushbroom) {
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
    if (wanted == prepared_ && origin == preparedOrigin_) {
        return;
    }
    ERNIE_ZONE("Overlay prepare");
    prepared_ = std::move(wanted);
    preparedOrigin_ = origin;
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
    RenderingTarget rendering;
    rendering.color = target.color;
    rendering.extent = target.extent;
    rendering.loadColor = true;
    BeginRendering(commands, rendering);
    const DescriptorResource resources[] = {
        target.uniforms, DescriptorResource::Of(target.depth),
        DescriptorResource::Of(*buffers_[0]),
        DescriptorResource::Of(*buffers_[1]),
        DescriptorResource::Of(*buffers_[2])};
    layout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                 resources);
    const bool srgb = target.format == VK_FORMAT_R8G8B8A8_SRGB ||
                      target.format == VK_FORMAT_B8G8R8A8_SRGB;
    const OverlayConstants constants{
        {target.world, static_cast<std::int32_t>(target.depthLayer),
         srgb ? 1 : 0, target.orthographic ? 1 : 0},
        {static_cast<float>(target.extent.width),
         static_cast<float>(target.extent.height), target.nearPlane,
         target.farPlane}};
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
    vkCmdEndRendering(commands);
}

std::uint64_t OverlayPass::GpuBytes() const {
    std::uint64_t bytes = 0;
    for (const std::unique_ptr<GpuBuffer>& buffer : buffers_) {
        bytes += buffer->Size();
    }
    return bytes;
}

}  // namespace ernie
