// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/points_drawable.h"

#include "ernie/device/gpu_profile.h"
#include "ernie/geometry/frustum.h"
#include "ernie/kernel/embedded_shaders.h"
#include "ernie/raster/mesh_drawable.h"
#include "ernie/raster/minimal_targets.h"
#include "ernie/util/profile.h"

#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ernie {

namespace {

constexpr std::uint32_t kHasWidths = 1;
constexpr std::uint32_t kHasColors = 2;

/// One draw as points.slang reads it.
struct GpuPointsDraw {
    float model[16];
    float previous[16];
    float tint[4];
    std::uint32_t info[4];
    std::uint32_t ids[4];
    /// x: its own temperature in kelvin, 0 for its material's or with
    /// thermal emission off; yzw unused.
    float thermal[4];
};
static_assert(sizeof(GpuPointsDraw) == 192,
              "GpuPointsDraw must match PointsDraw in points.slang");

/// frame.slang's per-draw push constants.
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

/// Frame (0 and 1), material (5) and points (10, its buffers'
/// addresses) bindings; see points.slang.
constexpr DescriptorBinding kBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
};

/// The motion pass's: the points passes' and 14, the view's
/// MotionUniforms (frame.slang).
constexpr DescriptorBinding kMotionBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {14, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
};

/// Culling's: the cloud's buffers' addresses, then the visible chunks
/// and indirect draws it writes (points.slang's `cullMain`).
constexpr DescriptorBinding kCullBindings[] = {
    {10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {19, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {20, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

/// points.slang's PointsResources: a cloud's passes' buffers'
/// addresses.
struct PointsResources {
    VkDeviceAddress positions;
    VkDeviceAddress widths;
    VkDeviceAddress colors;
    VkDeviceAddress draws;
    /// The visible chunks culling listed, or the draws when it lists
    /// none (draws of every point read nothing there).
    VkDeviceAddress visibleChunks;
    VkDeviceAddress cullJobs;
    VkDeviceAddress cullViews;
    VkDeviceAddress cullLights;
};
static_assert(sizeof(PointsResources) == 64,
              "PointsResources must match points.slang");

/// Points per chunk culling tests together (points.slang's `kChunk`).
constexpr std::uint32_t kChunk = 64;

/// Bounds of the box [`low`, `high`] after `transform`.
Bounds3 TransformBox(const glm::mat4& transform, const glm::vec3& low,
                     const glm::vec3& high) {
    glm::vec3 outLow(std::numeric_limits<float>::max());
    glm::vec3 outHigh(std::numeric_limits<float>::lowest());
    for (int i = 0; i < 8; ++i) {
        const glm::vec3 corner((i & 1) ? high.x : low.x,
                               (i & 2) ? high.y : low.y,
                               (i & 4) ? high.z : low.z);
        const glm::vec3 p(transform * glm::vec4(corner, 1.0f));
        outLow = glm::min(outLow, p);
        outHigh = glm::max(outHigh, p);
    }
    return {outLow, outHigh};
}

/// Bounds of spheres at `points` of diameter `widths` (or `width` for
/// all).
Bounds3 SphereBounds(std::span<const float> points,
                     std::span<const float> widths, float width) {
    glm::vec3 low(std::numeric_limits<float>::max());
    glm::vec3 high(std::numeric_limits<float>::lowest());
    for (std::size_t i = 0; 3 * i + 2 < points.size(); ++i) {
        const glm::vec3 p(points[3 * i], points[3 * i + 1],
                          points[3 * i + 2]);
        const float r = 0.5f * (widths.empty() ? width : widths[i]);
        low = glm::min(low, p - r);
        high = glm::max(high, p + r);
    }
    return points.empty() ? Bounds3{glm::vec3(0.0f), glm::vec3(0.0f)}
                          : Bounds3{low, high};
}

glm::mat4 Relative(glm::dmat4 world, const glm::dvec3& origin) {
    world[3] -= glm::dvec4(origin, 0.0);
    return glm::mat4(world);
}

float MaxScale(const glm::mat4& m) {
    return std::max({glm::length(glm::vec3(m[0])),
                     glm::length(glm::vec3(m[1])),
                     glm::length(glm::vec3(m[2]))});
}

std::vector<ShaderModuleSource>
PointsModules(bool pushbroom = false) {
    return {
        {"brdf", std::string(BrdfSlang())},
        ProjectionModule(pushbroom),
        {"bindless", shaders::kBindlessSlang},
        {"frame", shaders::kFrameSlang},
        {"gbuffer", shaders::kGBufferSlang},
        {"surface", shaders::kSurfaceSlang},
        {"points", shaders::kPointsSlang}};
}

}  // namespace

PointsDrawable::PointsDrawable(RasterRenderer& renderer)
    : renderer_(renderer), context_(renderer.Context()),
      layout_(renderer.Context(), kBindings, sizeof(DrawConstants)),
      motionLayout_(renderer.Context(), kMotionBindings,
                    sizeof(DrawConstants)),
      cullLayout_(renderer.Context(), kCullBindings,
                  sizeof(DrawConstants)) {
    const ShaderCompiler& compiler = renderer_.Compiler();
    const std::vector<ShaderModuleSource> modules = PointsModules();
    // The descriptions view the code, which must outlive them.
    const std::vector<std::uint32_t> shadowVertex =
        compiler.CompileSpirv(modules, "shadowVertex");
    const std::vector<std::uint32_t> shadowFragment =
        compiler.CompileSpirv(modules, "shadowFragment");
    CreateGBufferPipeline(false);
    GraphicsPipelineDesc shadow;
    shadow.vertexSpirv = shadowVertex;
    shadow.fragmentSpirv = shadowFragment;
    shadow.depthFormat = VK_FORMAT_D32_SFLOAT;
    shadow.depthBias = true;
    shadow.depthClamp = true;
    shadowPipeline_ =
        std::make_unique<GraphicsPipeline>(context_, layout_, shadow);
    cullPipeline_ = std::make_unique<ComputePipeline>(
        context_, cullLayout_,
        compiler.CompileSpirv(modules, "cullMain"));
    draws_ = std::make_unique<GpuBuffer>(
        context_, sizeof(GpuPointsDraw),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
}

PointsDrawable::~PointsDrawable() {
    vkDeviceWaitIdle(context_.Device());
}

void PointsDrawable::CreateGBufferPipeline(bool pushbroom) {
    const ShaderCompiler& compiler = renderer_.Compiler();
    const std::vector<ShaderModuleSource> modules =
        PointsModules(pushbroom);
    // The descriptions view the code, which must outlive them.
    const std::vector<std::uint32_t> vertex =
        compiler.CompileSpirv(modules, "vertexMain");
    const std::vector<std::uint32_t> fragment =
        compiler.CompileSpirv(modules, "fragmentMain");
    GraphicsPipelineDesc gbuffer;
    gbuffer.vertexSpirv = vertex;
    gbuffer.fragmentSpirv = fragment;
    gbuffer.colorFormats.assign(
        RenderTargets::kGBufferFormats.begin(),
        RenderTargets::kGBufferFormats.end());
    gbuffer.depthFormat = RenderTargets::kDepthFormat;
    gbuffer.depthCompare = RenderTargets::kDepthCompare;
    gbufferPipelines_[pushbroom ? 1 : 0] =
        std::make_unique<GraphicsPipeline>(context_, layout_,
                                           gbuffer);
}

void PointsDrawable::CreateMinimalPipelines() {
    const ShaderCompiler& compiler = renderer_.Compiler();
    const std::vector<ShaderModuleSource> modules =
        PointsModules(false);
    const std::vector<std::uint32_t> vertex =
        compiler.CompileSpirv(modules, "vertexMain");
    const char* const entries[] = {"minimalFragmentMain",
                                   "idsFragmentMain"};
    for (std::size_t i = 0; i < minimalPipelines_.size(); ++i) {
        const std::vector<std::uint32_t> fragment =
            compiler.CompileSpirv(modules, entries[i]);
        GraphicsPipelineDesc minimal;
        minimal.vertexSpirv = vertex;
        minimal.fragmentSpirv = fragment;
        minimal.colorFormats.assign(
            MinimalTargets::kColorFormats.begin(),
            MinimalTargets::kColorFormats.end());
        minimal.depthFormat = MinimalTargets::kDepthFormat;
        minimal.depthCompare = RenderTargets::kDepthCompare;
        minimalPipelines_[i] = std::make_unique<GraphicsPipeline>(
            context_, layout_, minimal);
    }
}

void PointsDrawable::Check(const Cloud& cloud) const {
    if (!(cloud.data.width > 0.0f) && cloud.data.widths.empty()) {
        throw std::invalid_argument("Points need a positive width");
    }
    if (cloud.onDevice) {
        return;
    }
    const PointsData& data = cloud.data;
    if (data.points.size() % 3 != 0) {
        throw std::invalid_argument("Points must have 3 values each");
    }
    const std::size_t count = data.points.size() / 3;
    if (!data.widths.empty() && data.widths.size() != count) {
        throw std::invalid_argument("Points need one width each");
    }
    if (!data.colors.empty() && data.colors.size() != 3 * count) {
        throw std::invalid_argument("Points need one colour each");
    }
}

std::uint32_t PointsDrawable::Add(PointsData data) {
    Cloud cloud;
    cloud.data = std::move(data);
    Check(cloud);
    cloud.count = cloud.data.points.size() / 3;
    std::tie(cloud.low, cloud.high) = SphereBounds(
        cloud.data.points, cloud.data.widths, cloud.data.width);
    Allocate(cloud);
    const std::uint32_t handle = nextHandle_++;
    clouds_.emplace(handle, std::move(cloud));
    return handle;
}

void PointsDrawable::Set(std::uint32_t points, PointsData data) {
    const auto it = clouds_.find(points);
    if (it == clouds_.end()) {
        throw std::invalid_argument("Unknown points " +
                                    std::to_string(points));
    }
    Cloud next;
    next.data = std::move(data);
    Check(next);
    Cloud& cloud = it->second;
    const std::size_t count = next.data.points.size() / 3;
    const bool layout =
        count != cloud.count || cloud.onDevice ||
        next.data.widths.empty() != cloud.data.widths.empty() ||
        next.data.colors.empty() != cloud.data.colors.empty();
    cloud.data = std::move(next.data);
    cloud.device = {};
    cloud.onDevice = false;
    cloud.count = count;
    std::tie(cloud.low, cloud.high) = SphereBounds(
        cloud.data.points, cloud.data.widths, cloud.data.width);
    if (layout) {
        Allocate(cloud);
    } else {
        cloud.dirty = true;
    }
}

void PointsDrawable::Set(std::uint32_t points,
                         const DevicePointsData& data) {
    const auto it = clouds_.find(points);
    if (it == clouds_.end()) {
        throw std::invalid_argument("Unknown points " +
                                    std::to_string(points));
    }
    if (compute_ == nullptr) {
        throw std::invalid_argument(
            "Points on a GPU need compute interop");
    }
    const auto check = [&](const Tensor& tensor,
                           std::int64_t components,
                           const char* what) {
        if (tensor.Location() != compute_->Location() ||
            tensor.Type() != DType::Float32 ||
            !tensor.IsContiguous() ||
            tensor.Elements() % components != 0) {
            throw std::invalid_argument(
                std::string(what) +
                " must be a contiguous float32 tensor on compute "
                "device " +
                std::to_string(compute_->Device()));
        }
    };
    check(data.points, 3, "Points");
    const auto count =
        static_cast<std::size_t>(data.points.Elements() / 3);
    if (!data.widths.Empty()) {
        check(data.widths, 1, "Point widths");
        if (static_cast<std::size_t>(data.widths.Elements()) !=
            count) {
            throw std::invalid_argument("Points need one width each");
        }
    } else if (!(data.width > 0.0f)) {
        throw std::invalid_argument("Points need a positive width");
    }
    if (!data.colors.Empty()) {
        check(data.colors, 3, "Point colours");
        if (static_cast<std::size_t>(data.colors.Elements()) !=
            3 * count) {
            throw std::invalid_argument(
                "Points need one colour each");
        }
    }
    Cloud& cloud = it->second;
    const bool hadWidths = cloud.onDevice
                               ? !cloud.device.widths.Empty()
                               : !cloud.data.widths.empty();
    const bool hadColors = cloud.onDevice
                               ? !cloud.device.colors.Empty()
                               : !cloud.data.colors.empty();
    const bool layout = count != cloud.count ||
                        (!cloud.onDevice && cloud.mapped.Empty()) ||
                        data.widths.Empty() == hadWidths ||
                        data.colors.Empty() == hadColors;
    // The view `MapPositions` gave out, handed back: the positions
    // are in place already.
    cloud.inPlace = !layout && !cloud.mapped.Empty() &&
                    data.points.Data() == cloud.mapped.Data();
    cloud.device = data;
    cloud.onDevice = true;
    cloud.count = count;
    cloud.data = {};
    cloud.data.width = data.width;
    if (data.extent) {
        std::tie(cloud.low, cloud.high) = *data.extent;
    } else {
        const Tensor host = compute_->Download(data.points);
        const std::span<const float> hostPoints(host.As<float>(),
                                                3 * count);
        std::vector<float> hostWidths;
        if (!data.widths.Empty()) {
            const Tensor w = compute_->Download(data.widths);
            hostWidths.assign(w.As<float>(), w.As<float>() + count);
        }
        std::tie(cloud.low, cloud.high) =
            SphereBounds(hostPoints, hostWidths, data.width);
    }
    if (layout) {
        Allocate(cloud);
    } else {
        cloud.dirty = true;
    }
}

Tensor PointsDrawable::MapPositions(std::uint32_t points,
                                    std::uintptr_t stream) {
    const auto it = clouds_.find(points);
    if (it == clouds_.end()) {
        throw std::invalid_argument("Unknown points " +
                                    std::to_string(points));
    }
    if (compute_ == nullptr) {
        throw std::invalid_argument(
            "Mapping points needs compute interop");
    }
    Cloud& cloud = it->second;
    if (cloud.positions->Memory() != BufferMemory::Shared) {
        // Made before compute interop: into memory the compute device
        // sees, both halves holding the points now.
        Allocate(cloud);
        if (!cloud.onDevice) {
            const std::span<const std::byte> bytes =
                std::as_bytes(std::span(cloud.data.points));
            cloud.positions->Write(bytes);
            cloud.positions->Write(bytes, bytes.size());
            if (!cloud.data.widths.empty()) {
                cloud.widths->Write(
                    std::as_bytes(std::span(cloud.data.widths)));
            }
            if (!cloud.data.colors.empty()) {
                cloud.colors->Write(
                    std::as_bytes(std::span(cloud.data.colors)));
            }
            cloud.dirty = false;
        }
    }
    const auto count = static_cast<std::int64_t>(cloud.count);
    const auto view = [&] {
        return compute_->View(cloud.positions, DType::Float32,
                              {count, 3},
                              VkDeviceSize{cloud.set} * cloud.count *
                                  3 * sizeof(float));
    };
    // Nothing writes the half frames still in flight may read until
    // they finish.
    compute_->WaitForVulkan(stream);
    const Tensor current = view();
    if (!cloud.written) {
        cloud.set ^= 1u;
        cloud.written = true;
        cloud.moving = true;
    }
    cloud.mapped = view();
    if (cloud.mapped.Data() != current.Data()) {
        compute_->Copy(current, cloud.mapped, stream);
    }
    return cloud.mapped;
}

void PointsDrawable::Remove(std::uint32_t points) {
    if (clouds_.erase(points) == 0) {
        throw std::invalid_argument("Unknown points " +
                                    std::to_string(points));
    }
    std::erase_if(items_, [points](const PointsItem& item) {
        return item.points == points;
    });
    previous_.resize(items_.size());
}

void PointsDrawable::Allocate(Cloud& cloud) {
    // Earlier frames may still read the buffers it replaces.
    if (cloud.positions != nullptr) {
        vkDeviceWaitIdle(context_.Device());
    }
    const BufferMemory memory = compute_ != nullptr
                                    ? BufferMemory::Shared
                                    : BufferMemory::Device;
    const auto make = [&](std::size_t floats) {
        return std::make_shared<GpuBuffer>(
            context_,
            std::max<VkDeviceSize>(floats * sizeof(float), 16),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memory);
    };
    cloud.positions = make(6 * cloud.count);
    cloud.widths = make(cloud.count);
    cloud.colors = make(3 * cloud.count);
    // Both halves of the positions hold the points; they stand still
    // until written again.
    cloud.set = 0;
    cloud.moving = false;
    cloud.written = true;
    cloud.dirty = true;
}

void PointsDrawable::Write(Cloud& cloud) {
    if (!cloud.written) {
        cloud.set ^= 1u;
        cloud.written = true;
        cloud.moving = true;
    }
    const VkDeviceSize half =
        VkDeviceSize{cloud.set} * cloud.count * 3 * sizeof(float);
    if (cloud.onDevice) {
        const DevicePointsData& device = cloud.device;
        if (cloud.inPlace) {
            // Written there through `MapPositions`: the frame waits
            // for their stream.
            compute_->SignalVulkan(device.stream);
            cloud.inPlace = false;
        } else {
            compute_->CopyToBuffer(device.points, *cloud.positions,
                                   half, device.stream);
        }
        if (!device.widths.Empty()) {
            compute_->CopyToBuffer(device.widths, *cloud.widths, 0,
                                   device.stream);
        }
        if (!device.colors.Empty()) {
            compute_->CopyToBuffer(device.colors, *cloud.colors, 0,
                                   device.stream);
        }
    } else {
        FrameUploads& uploads = renderer_.Uploads();
        const PointsData& data = cloud.data;
        uploads.Write(*cloud.positions,
                      std::as_bytes(std::span(data.points)), half);
        if (!cloud.moving) {
            // A fresh cloud: the other half too, so it stands still
            // whichever half is current.
            uploads.Write(*cloud.positions,
                          std::as_bytes(std::span(data.points)),
                          VkDeviceSize{1u - cloud.set} * cloud.count *
                              3 * sizeof(float));
        }
        if (!data.widths.empty()) {
            uploads.Write(*cloud.widths,
                          std::as_bytes(std::span(data.widths)));
        }
        if (!data.colors.empty()) {
            uploads.Write(*cloud.colors,
                          std::as_bytes(std::span(data.colors)));
        }
    }
    cloud.dirty = false;
}

void PointsDrawable::SetItems(std::vector<PointsItem> items) {
    for (const PointsItem& item : items) {
        if (!clouds_.contains(item.points)) {
            throw std::invalid_argument("Unknown points " +
                                        std::to_string(item.points));
        }
        if (!renderer_.HasMaterial(item.material)) {
            throw std::invalid_argument(
                "Unknown material " + std::to_string(item.material));
        }
        if (item.anchorInstancer != 0 && anchors_ == nullptr) {
            throw std::invalid_argument(
                "Points posed under instances need the meshes' "
                "drawable");
        }
    }
    // As `MeshDrawable::SetItems`: a draw of the same prim moves from
    // its transform.
    std::map<std::uint32_t, std::size_t> old;
    for (std::size_t i = 0; i < items_.size(); ++i) {
        old.try_emplace(items_[i].primId, i);
    }
    std::vector<glm::dmat4> previous;
    for (const PointsItem& item : items) {
        const auto found = old.find(item.primId);
        if (found == old.end()) {
            previous.push_back(item.transform);
            continue;
        }
        const glm::dmat4& was = items_[found->second].transform;
        previous.push_back(
            was == item.transform ? previous_[found->second] : was);
    }
    items_ = std::move(items);
    previous_ = std::move(previous);
}

void PointsDrawable::SettleMotion() {
    for (std::size_t i = 0; i < items_.size(); ++i) {
        previous_[i] = items_[i].transform;
    }
    for (auto& [handle, cloud] : clouds_) {
        cloud.moving = false;
        cloud.written = false;
        // As `MeshDrawable::SettleMotion`: a view mapped for this
        // frame expires.
        if (cloud.onDevice) {
            cloud.mapped = {};
        }
    }
}

std::uint64_t PointsDrawable::PointCount() const {
    std::uint64_t count = 0;
    for (const PointsItem& item : items_) {
        count += clouds_.at(item.points).count;
    }
    return count;
}

Bounds3 PointsDrawable::ItemBounds(std::size_t item) const {
    const Cloud& cloud = clouds_.at(items_[item].points);
    // Under an instance, its bounds wherever the instance goes.
    const bool anchored = items_[item].anchorInstancer != 0;
    const glm::vec3 low =
        anchored ? items_[item].anchorBounds.first : cloud.low;
    const glm::vec3 high =
        anchored ? items_[item].anchorBounds.second : cloud.high;
    Bounds3 box = TransformBox(
        Relative(items_[item].transform, origin_), low, high);
    if (previous_[item] != items_[item].transform || cloud.moving) {
        // Where it was too; a moving cloud's own last bounds are not
        // kept, so its box at the last transform stands in for them.
        const Bounds3 was = TransformBox(
            Relative(previous_[item], origin_), low, high);
        box = {glm::min(box.first, was.first),
               glm::max(box.second, was.second)};
    }
    return box;
}

std::optional<Bounds3> PointsDrawable::Bounds() const {
    std::optional<Bounds3> bounds;
    for (std::size_t i = 0; i < items_.size(); ++i) {
        if (clouds_.at(items_[i].points).count == 0) {
            continue;
        }
        const Bounds3 box = ItemBounds(i);
        bounds = bounds
                     ? Bounds3{glm::min(bounds->first, box.first),
                               glm::max(bounds->second, box.second)}
                     : box;
    }
    return bounds;
}

std::map<std::int32_t, Bounds3> PointsDrawable::WorldsBounds() const {
    std::map<std::int32_t, Bounds3> bounds;
    for (std::size_t i = 0; i < items_.size(); ++i) {
        if (clouds_.at(items_[i].points).count == 0) {
            continue;
        }
        const Bounds3 box = ItemBounds(i);
        const auto [it, added] =
            bounds.try_emplace(items_[i].world, box);
        if (!added) {
            it->second =
                Bounds3{glm::min(it->second.first, box.first),
                        glm::max(it->second.second, box.second)};
        }
    }
    return bounds;
}

void PointsDrawable::Prepare(
    const FrameParams& params,
    std::span<const ShadowCascade> cascades) {
    ERNIE_ZONE("Points prepare");
    // Nothing per view without points to draw: batches of thousands
    // of views pay nothing.
    viewItems_.clear();
    viewJobs_.clear();
    shadowJobs_.clear();
    jobs_.clear();
    itemJobs_.clear();
    anchored_ = 0;
    if (items_.empty()) {
        return;
    }
    pushbroom_ = params.camera.pushbroom;
    minimal_ = params.minimal;
    if (gbufferPipelines_[pushbroom_ ? 1 : 0] == nullptr) {
        CreateGBufferPipeline(pushbroom_);
    }
    if (minimal_ != MinimalMode::Off &&
        minimalPipelines_[0] == nullptr) {
        CreateMinimalPipelines();
    }
    for (auto& [handle, cloud] : clouds_) {
        if (cloud.dirty && cloud.count > 0) {
            Write(cloud);
        }
    }
    std::vector<GpuPointsDraw> draws(items_.size());
    // Items under instances: their models are posed on the GPU
    // (`RecordPrepass`), with the instances' slots in this frame's
    // halves (the meshes prepare first).
    std::vector<glm::vec4> records;
    for (std::size_t i = 0; i < items_.size(); ++i) {
        const PointsItem& item = items_[i];
        const Cloud& cloud = clouds_.at(item.points);
        GpuPointsDraw& draw = draws[i];
        const bool anchored = item.anchorInstancer != 0;
        const glm::mat4 model = Relative(
            anchored ? item.transform * item.below : item.transform,
            origin_);
        const glm::mat4 previous =
            anchored ? model
                     : glm::mat4(Relative(previous_[i], origin_));
        if (anchored) {
            MeshDrawable::AppendAnchorRecord(
                records, Relative(item.transform, origin_),
                glm::mat4(item.below),
                anchors_->AnchorSlot(item.anchorInstancer,
                                     item.anchorRow),
                static_cast<std::uint32_t>(i),
                anchors_->AnchorOps(item.anchorInstancer));
        }
        std::memcpy(draw.model, glm::value_ptr(model),
                    sizeof(draw.model));
        std::memcpy(draw.previous, glm::value_ptr(previous),
                    sizeof(draw.previous));
        draw.tint[0] = item.tint.r;
        draw.tint[1] = item.tint.g;
        draw.tint[2] = item.tint.b;
        draw.tint[3] =
            cloud.onDevice ? cloud.device.width : cloud.data.width;
        const auto count = static_cast<std::uint32_t>(cloud.count);
        draw.info[0] = cloud.set * count;
        draw.info[1] =
            (cloud.moving ? 1u - cloud.set : cloud.set) * count;
        const bool widths = cloud.onDevice
                                ? !cloud.device.widths.Empty()
                                : !cloud.data.widths.empty();
        const bool colors = cloud.onDevice
                                ? !cloud.device.colors.Empty()
                                : !cloud.data.colors.empty();
        draw.info[2] =
            (widths ? kHasWidths : 0u) | (colors ? kHasColors : 0u);
        const float scale = MaxScale(model);
        std::memcpy(&draw.info[3], &scale, sizeof(scale));
        draw.ids[0] = item.material;
        draw.ids[1] = item.primId;
        draw.ids[2] = item.linkGroup;
        draw.ids[3] = count;
        draw.thermal[0] = params.thermal.enable ? item.temperature : 0.0f;
    }
    const VkDeviceSize bytes = std::max<VkDeviceSize>(
        draws.size() * sizeof(GpuPointsDraw), 16);
    if (draws_->Size() < bytes) {
        vkDeviceWaitIdle(context_.Device());
        draws_ = std::make_unique<GpuBuffer>(
            context_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            BufferMemory::Device);
    }
    if (!draws.empty()) {
        renderer_.Uploads().Write(*draws_,
                                  std::as_bytes(std::span(draws)));
    }
    anchored_ = static_cast<std::uint32_t>(records.size() / 9);
    if (anchored_ > 0) {
        const VkDeviceSize recordBytes =
            records.size() * sizeof(glm::vec4);
        if (anchorRecords_ == nullptr ||
            anchorRecords_->Size() < recordBytes) {
            vkDeviceWaitIdle(context_.Device());
            anchorRecords_ = std::make_unique<GpuBuffer>(
                context_, recordBytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Device);
        }
        renderer_.Uploads().Write(*anchorRecords_,
                                  std::as_bytes(std::span(records)));
    }
    // With GPU culling, each still cloud's draw in each view and
    // shadow view becomes a job testing its chunks there, the views'
    // matrices read where the GPU keeps them; a cloud moving across
    // the shutter, or a view swept or seen at sub-frames, draws every
    // point.
    const bool gpu = params.culling == Culling::Gpu &&
                     !params.camera.pushbroom &&
                     renderer_.Subframes() == 1;
    const auto still = [&](std::size_t i) {
        return previous_[i] == items_[i].transform &&
               !clouds_.at(items_[i].points).moving;
    };
    viewItems_.assign(params.Views(), {});
    viewJobs_.assign(params.Views(), {});
    shadowJobs_.assign(cascades.size(),
                       std::vector<std::int32_t>(items_.size(), -1));
    std::uint32_t visible = 0;
    const auto addJob = [&](std::uint32_t kind, std::uint32_t index,
                            std::size_t item) {
        jobs_.emplace_back(kind, index, visible, 0u);
        const auto count = static_cast<std::uint32_t>(
            clouds_.at(items_[item].points).count);
        visible += (count + kChunk - 1) / kChunk;
        return static_cast<std::int32_t>(jobs_.size() - 1);
    };
    // Each view draws the items of its worlds whose bounds meet its
    // frustum; jobs test them on the GPU instead, an item's together.
    std::vector<Frustum> frusta;
    for (std::size_t v = 0; v < params.Views(); ++v) {
        frusta.push_back(
            Frustum::FromMatrix(params.View(v).ViewProjection()));
    }
    for (std::size_t i = 0; i < items_.size(); ++i) {
        const PointsItem& item = items_[i];
        const std::size_t count = clouds_.at(item.points).count;
        const bool culled = gpu && still(i) && count > 0;
        const auto first = static_cast<std::uint32_t>(jobs_.size());
        const Bounds3 box = ItemBounds(i);
        for (std::size_t v = 0; v < params.Views(); ++v) {
            const std::int32_t world = params.ViewWorld(v);
            if (count == 0 || (item.world != -1 && world != -1 &&
                               item.world != world)) {
                continue;
            }
            if (culled) {
                viewItems_[v].push_back(
                    static_cast<std::uint32_t>(i));
                viewJobs_[v].push_back(
                    addJob(0, static_cast<std::uint32_t>(v), i));
            } else if (params.culling == Culling::Off ||
                       params.posesOnGpu ||
                       frusta[v].Intersects(box.first, box.second)) {
                viewItems_[v].push_back(
                    static_cast<std::uint32_t>(i));
                viewJobs_[v].push_back(-1);
            }
        }
        for (std::size_t c = 0; c < cascades.size() && culled; ++c) {
            const ShadowCascade& cascade = cascades[c];
            if (cascade.moving ||
                (cascade.world && item.world != -1 &&
                 item.world != *cascade.world) ||
                ((cascade.shadowMask >> (item.linkGroup & 31u)) &
                 1u) == 0) {
                continue;
            }
            shadowJobs_[c][i] = addJob(
                1,
                cascade.gpuFitted ? cascade.gpuMatrix
                                  : static_cast<std::uint32_t>(c),
                i);
        }
        itemJobs_.emplace_back(
            first, static_cast<std::uint32_t>(jobs_.size()) - first);
    }
    if (jobs_.empty()) {
        WriteResources();
        return;
    }
    // Each job's indirect draw: six vertices per point, its instances
    // counted up by the culling.
    std::vector<VkDrawIndirectCommand> commands(jobs_.size(),
                                                {6, 0, 0, 0});
    const auto reserve = [&](std::unique_ptr<GpuBuffer>& buffer,
                             VkDeviceSize bytes,
                             VkBufferUsageFlags usage) {
        bytes = std::max<VkDeviceSize>(bytes, 16);
        if (buffer == nullptr || buffer->Size() < bytes) {
            if (buffer != nullptr) {
                vkDeviceWaitIdle(context_.Device());
            }
            buffer = std::make_unique<GpuBuffer>(
                context_, bytes, usage, BufferMemory::Device);
        }
    };
    reserve(cullJobs_, jobs_.size() * sizeof(glm::uvec4),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    reserve(cullCommands_,
            commands.size() * sizeof(VkDrawIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    reserve(visibleChunks_,
            VkDeviceSize{visible} * sizeof(std::uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    renderer_.Uploads().Write(*cullJobs_,
                              std::as_bytes(std::span(jobs_)));
    renderer_.Uploads().Write(*cullCommands_,
                              std::as_bytes(std::span(commands)));
    WriteResources();
}

void PointsDrawable::RecordCull(VkCommandBuffer commands) const {
    if (jobs_.empty()) {
        return;
    }
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      cullPipeline_->Handle());
    for (std::size_t i = 0; i < items_.size(); ++i) {
        const auto [first, jobs] = itemJobs_[i];
        if (jobs == 0) {
            continue;
        }
        const Cloud& cloud = clouds_.at(items_[i].points);
        const DescriptorResource resources[] = {
            DescriptorResource::Of(*cloud.resources),
            DescriptorResource::Of(*visibleChunks_),
            DescriptorResource::Of(*cullCommands_)};
        cullLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                         resources);
        const auto count = static_cast<std::uint32_t>(cloud.count);
        const std::uint32_t chunks = (count + kChunk - 1) / kChunk;
        // Past the groups every device dispatches along x, in runs.
        for (std::uint32_t done = 0; done < chunks; done += 65535) {
            DrawConstants constants{};
            constants.first = static_cast<std::uint32_t>(i);
            constants.instanceOffset = count;
            constants.indexOffset = first;
            constants.records = static_cast<std::uint32_t>(
                RasterRenderer::ViewUniformsStride() / 16);
            constants.view = done;
            cullLayout_.PushConstants(commands, &constants,
                                      sizeof(constants));
            vkCmdDispatch(commands, std::min(chunks - done, 65535u),
                          jobs, 1);
        }
    }
    // The renderer's barrier follows for the vertex stages; the
    // indirect draws read the counts too.
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                           VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT |
                            VK_ACCESS_2_SHADER_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commands, &dependency);
}

std::vector<std::uint32_t> PointsDrawable::CulledPointCounts() const {
    if (jobs_.empty()) {
        return {};
    }
    std::vector<VkDrawIndirectCommand> drawn(jobs_.size());
    cullCommands_->Read(std::as_writable_bytes(std::span(drawn)));
    std::vector<std::uint32_t> counts;
    for (const VkDrawIndirectCommand& command : drawn) {
        counts.push_back(command.instanceCount);
    }
    return counts;
}

std::uint32_t PointsDrawable::InstanceOffset(std::int32_t job) const {
    return job < 0 ? 0u : 1u + jobs_[static_cast<std::size_t>(job)].z;
}

void PointsDrawable::Draw(VkCommandBuffer commands,
                          std::uint32_t item,
                          std::int32_t job) const {
    if (job < 0) {
        vkCmdDraw(commands, 6,
                  static_cast<std::uint32_t>(
                      clouds_.at(items_[item].points).count),
                  0, 0);
        return;
    }
    vkCmdDrawIndirect(commands, cullCommands_->Handle(),
                      static_cast<VkDeviceSize>(job) *
                          sizeof(VkDrawIndirectCommand),
                      1, sizeof(VkDrawIndirectCommand));
}

void PointsDrawable::RecordPrepass(VkCommandBuffer commands) const {
    if (anchored_ == 0) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Anchored points");
    anchors_->RecordAnchors(commands, *anchorRecords_, anchored_,
                            *draws_, AnchorOutput::PointsDraws);
}

std::vector<DescriptorResource>
PointsDrawable::Resources(const RasterFrame& frame,
                          std::size_t item) const {
    std::vector<DescriptorResource> resources = {frame.resources,
                                                 frame.shadows};
    renderer_.MaterialOrDefault(items_[item].material)
        .AppendResources(resources);
    resources.push_back(DescriptorResource::Of(
        *clouds_.at(items_[item].points).resources));
    return resources;
}

void PointsDrawable::WriteResources() {
    const auto address = [](const auto& buffer) {
        return buffer != nullptr ? buffer->Address()
                                 : VkDeviceAddress{0};
    };
    for (auto& [handle, cloud] : clouds_) {
        const PointsResources resources{
            address(cloud.positions),
            address(cloud.widths),
            address(cloud.colors),
            address(draws_),
            visibleChunks_ != nullptr ? address(visibleChunks_)
                                      : address(draws_),
            address(cullJobs_),
            renderer_.ViewUniforms().Address(),
            renderer_.LightViewMatrices().Address()};
        if (cloud.resources == nullptr) {
            cloud.resources = std::make_unique<GpuBuffer>(
                context_, sizeof(resources),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                BufferMemory::Device);
        }
        renderer_.Uploads().Write(
            *cloud.resources,
            std::as_bytes(std::span(&resources, 1)));
    }
}

void PointsDrawable::RecordMotion(
    VkCommandBuffer commands, const RasterFrame& frame,
    std::size_t view, const DescriptorResource& motion) const {
    if (view >= viewItems_.size()) {
        return;
    }
    bool bound = false;
    for (const std::uint32_t i : viewItems_[view]) {
        const Cloud& cloud = clouds_.at(items_[i].points);
        if (previous_[i] == items_[i].transform && !cloud.moving) {
            continue;
        }
        if (motionPipeline_ == nullptr) {
            const std::vector<ShaderModuleSource> modules =
                PointsModules();
            const std::vector<std::uint32_t> vertex =
                renderer_.Compiler().CompileSpirv(modules,
                                                  "vertexMain");
            const std::vector<std::uint32_t> fragment =
                renderer_.Compiler().CompileSpirv(modules,
                                                  "motionFragment");
            GraphicsPipelineDesc desc;
            desc.vertexSpirv = vertex;
            desc.fragmentSpirv = fragment;
            desc.colorFormats = {VK_FORMAT_R32G32B32A32_SFLOAT};
            desc.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
            desc.depthFormat = RenderTargets::kDepthFormat;
            // The same depths as the colour pass's: only the surfaces
            // it kept.
            desc.depthCompare = RenderTargets::kDepthCompareEqual;
            desc.depthWrite = false;
            motionPipeline_ = std::make_unique<GraphicsPipeline>(
                context_, motionLayout_, desc);
        }
        if (!bound) {
            vkCmdBindPipeline(commands,
                              VK_PIPELINE_BIND_POINT_GRAPHICS,
                              motionPipeline_->Handle());
            bound = true;
        }
        std::vector<DescriptorResource> resources =
            Resources(frame, i);
        resources.push_back(motion);
        motionLayout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           resources);
        DrawConstants constants{};
        constants.first = i;
        constants.view = frame.view;
        motionLayout_.PushConstants(commands, &constants,
                                    sizeof(constants));
        vkCmdDraw(commands, 6,
                  static_cast<std::uint32_t>(cloud.count), 0, 0);
    }
}

void PointsDrawable::RecordShadow(
    VkCommandBuffer commands, const RasterFrame& frame,
    std::size_t cascade, const glm::mat4& lightViewProjection,
    const ShadowCasters& casters) const {
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      shadowPipeline_->Handle());
    for (std::size_t i = 0; i < items_.size(); ++i) {
        const std::size_t count = clouds_.at(items_[i].points).count;
        if (count == 0 || !casters.Draws(items_[i].world) ||
            !casters.Casts(items_[i].linkGroup)) {
            continue;
        }
        const std::int32_t job = cascade < shadowJobs_.size()
                                     ? shadowJobs_[cascade][i]
                                     : -1;
        const std::vector<DescriptorResource> resources =
            Resources(frame, i);
        layout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                     resources);
        DrawConstants constants{};
        std::memcpy(constants.lightViewProj,
                    glm::value_ptr(lightViewProjection),
                    sizeof(constants.lightViewProj));
        constants.first = static_cast<std::uint32_t>(i);
        constants.instanceOffset = InstanceOffset(job);
        constants.shadowFromFrame = frame.shadowFromFrame ? 1u : 0u;
        constants.view = frame.view;
        constants.shadowLayer = frame.shadowLayer;
        layout_.PushConstants(commands, &constants,
                              sizeof(constants));
        Draw(commands, static_cast<std::uint32_t>(i), job);
    }
}

void PointsDrawable::RecordColor(VkCommandBuffer commands,
                                 const RasterFrame& frame,
                                 std::size_t view,
                                 DrawPhase phase) const {
    if (phase == DrawPhase::Occludees || view >= viewItems_.size()) {
        return;
    }
    const GraphicsPipeline& pipeline =
        minimal_ != MinimalMode::Off
            ? *minimalPipelines_[minimal_ == MinimalMode::None ? 1
                                                               : 0]
            : *gbufferPipelines_[pushbroom_ ? 1 : 0];
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.Handle());
    for (std::size_t k = 0; k < viewItems_[view].size(); ++k) {
        const std::uint32_t i = viewItems_[view][k];
        const std::int32_t job = viewJobs_[view][k];
        const std::vector<DescriptorResource> resources =
            Resources(frame, i);
        layout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                     resources);
        DrawConstants constants{};
        constants.first = i;
        constants.instanceOffset = InstanceOffset(job);
        constants.view = frame.view;
        layout_.PushConstants(commands, &constants,
                              sizeof(constants));
        Draw(commands, i, job);
    }
}

}  // namespace ernie
