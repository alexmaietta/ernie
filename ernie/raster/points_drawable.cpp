#include "ernie/raster/points_drawable.h"

#include "ernie/device/gpu_profile.h"
#include "ernie/geometry/frustum.h"
#include "ernie/kernel/embedded_shaders.h"
#include "ernie/raster/mesh_drawable.h"
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
};
static_assert(sizeof(GpuPointsDraw) == 176, "GpuPointsDraw must match PointsDraw in points.slang");

/// frame.slang's per-draw push constants.
struct DrawConstants {
    float lightViewProj[16];
    std::uint32_t first;
    std::uint32_t instanceOffset;
    std::uint32_t indexOffset;
    std::uint32_t records;
    std::uint32_t shadowFromFrame;
    std::uint32_t padding[3];
};
static_assert(sizeof(DrawConstants) == 96, "DrawConstants must match frame.slang");

constexpr VkShaderStageFlags kGraphics = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

/// Frame (0 to 4), material (5 to 9) and points (10 to 13) bindings; see points.slang.
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
    {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
};

/// The motion pass's: the points passes' and 14, the view's MotionUniforms (frame.slang).
constexpr DescriptorBinding kMotionBindings[] = {
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
    {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {14, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
};

/// Bounds of the box [`low`, `high`] after `transform`.
Bounds3 TransformBox(const glm::mat4& transform, const glm::vec3& low, const glm::vec3& high) {
    glm::vec3 outLow(std::numeric_limits<float>::max());
    glm::vec3 outHigh(std::numeric_limits<float>::lowest());
    for (int i = 0; i < 8; ++i) {
        const glm::vec3 corner((i & 1) ? high.x : low.x, (i & 2) ? high.y : low.y,
                               (i & 4) ? high.z : low.z);
        const glm::vec3 p(transform * glm::vec4(corner, 1.0f));
        outLow = glm::min(outLow, p);
        outHigh = glm::max(outHigh, p);
    }
    return {outLow, outHigh};
}

/// Bounds of spheres at `points` of diameter `widths` (or `width` for all).
Bounds3 SphereBounds(std::span<const float> points, std::span<const float> widths, float width) {
    glm::vec3 low(std::numeric_limits<float>::max());
    glm::vec3 high(std::numeric_limits<float>::lowest());
    for (std::size_t i = 0; 3 * i + 2 < points.size(); ++i) {
        const glm::vec3 p(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
        const float r = 0.5f * (widths.empty() ? width : widths[i]);
        low = glm::min(low, p - r);
        high = glm::max(high, p + r);
    }
    return points.empty() ? Bounds3{glm::vec3(0.0f), glm::vec3(0.0f)} : Bounds3{low, high};
}

glm::mat4 Relative(glm::dmat4 world, const glm::dvec3& origin) {
    world[3] -= glm::dvec4(origin, 0.0);
    return glm::mat4(world);
}

float MaxScale(const glm::mat4& m) {
    return std::max(
        {glm::length(glm::vec3(m[0])), glm::length(glm::vec3(m[1])), glm::length(glm::vec3(m[2]))});
}

std::vector<ShaderModuleSource> PointsModules(bool pushbroom = false) {
    return {{"brdf", std::string(BrdfSlang())},
            ProjectionModule(pushbroom),
            {"frame", shaders::kFrameSlang},
            {"surface", shaders::kSurfaceSlang},
            {"points", shaders::kPointsSlang}};
}

}  // namespace

PointsDrawable::PointsDrawable(RasterRenderer& renderer)
    : renderer_(renderer), context_(renderer.Context()),
      layout_(renderer.Context(), kBindings, sizeof(DrawConstants)),
      motionLayout_(renderer.Context(), kMotionBindings, sizeof(DrawConstants)) {
    const ShaderCompiler& compiler = renderer_.Compiler();
    const std::vector<ShaderModuleSource> modules = PointsModules();
    // The descriptions view the code, which must outlive them.
    const std::vector<std::uint32_t> shadowVertex = compiler.CompileSpirv(modules, "shadowVertex");
    const std::vector<std::uint32_t> shadowFragment =
        compiler.CompileSpirv(modules, "shadowFragment");
    CreateGBufferPipeline(false);
    GraphicsPipelineDesc shadow;
    shadow.vertexSpirv = shadowVertex;
    shadow.fragmentSpirv = shadowFragment;
    shadow.depthFormat = VK_FORMAT_D32_SFLOAT;
    shadow.depthBiasSlope = 1.5f;
    shadowPipeline_ = std::make_unique<GraphicsPipeline>(context_, layout_, shadow);
    draws_ = std::make_unique<GpuBuffer>(context_, sizeof(GpuPointsDraw),
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
}

PointsDrawable::~PointsDrawable() {
    vkDeviceWaitIdle(context_.Device());
}

void PointsDrawable::CreateGBufferPipeline(bool pushbroom) {
    const ShaderCompiler& compiler = renderer_.Compiler();
    const std::vector<ShaderModuleSource> modules = PointsModules(pushbroom);
    // The descriptions view the code, which must outlive them.
    const std::vector<std::uint32_t> vertex = compiler.CompileSpirv(modules, "vertexMain");
    const std::vector<std::uint32_t> fragment = compiler.CompileSpirv(modules, "fragmentMain");
    GraphicsPipelineDesc gbuffer;
    gbuffer.vertexSpirv = vertex;
    gbuffer.fragmentSpirv = fragment;
    gbuffer.colorFormats.assign(RenderTargets::kGBufferFormats.begin(),
                                RenderTargets::kGBufferFormats.end());
    gbuffer.depthFormat = RenderTargets::kDepthFormat;
    gbufferPipelines_[pushbroom ? 1 : 0] =
        std::make_unique<GraphicsPipeline>(context_, layout_, gbuffer);
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
    std::tie(cloud.low, cloud.high) =
        SphereBounds(cloud.data.points, cloud.data.widths, cloud.data.width);
    Allocate(cloud);
    const std::uint32_t handle = nextHandle_++;
    clouds_.emplace(handle, std::move(cloud));
    return handle;
}

void PointsDrawable::Set(std::uint32_t points, PointsData data) {
    const auto it = clouds_.find(points);
    if (it == clouds_.end()) {
        throw std::invalid_argument("Unknown points " + std::to_string(points));
    }
    Cloud next;
    next.data = std::move(data);
    Check(next);
    Cloud& cloud = it->second;
    const std::size_t count = next.data.points.size() / 3;
    const bool layout = count != cloud.count || cloud.onDevice ||
                        next.data.widths.empty() != cloud.data.widths.empty() ||
                        next.data.colors.empty() != cloud.data.colors.empty();
    cloud.data = std::move(next.data);
    cloud.device = {};
    cloud.onDevice = false;
    cloud.count = count;
    std::tie(cloud.low, cloud.high) =
        SphereBounds(cloud.data.points, cloud.data.widths, cloud.data.width);
    if (layout) {
        Allocate(cloud);
    } else {
        cloud.dirty = true;
    }
}

void PointsDrawable::Set(std::uint32_t points, const DevicePointsData& data) {
    const auto it = clouds_.find(points);
    if (it == clouds_.end()) {
        throw std::invalid_argument("Unknown points " + std::to_string(points));
    }
    if (compute_ == nullptr) {
        throw std::invalid_argument("Points on a GPU need compute interop");
    }
    const auto check = [&](const Tensor& tensor, std::int64_t components, const char* what) {
        if (tensor.Location() != compute_->Location() || tensor.Type() != DType::Float32 ||
            !tensor.IsContiguous() || tensor.Elements() % components != 0) {
            throw std::invalid_argument(std::string(what) +
                                        " must be a contiguous float32 tensor on compute device " +
                                        std::to_string(compute_->Device()));
        }
    };
    check(data.points, 3, "Points");
    const auto count = static_cast<std::size_t>(data.points.Elements() / 3);
    if (!data.widths.Empty()) {
        check(data.widths, 1, "Point widths");
        if (static_cast<std::size_t>(data.widths.Elements()) != count) {
            throw std::invalid_argument("Points need one width each");
        }
    } else if (!(data.width > 0.0f)) {
        throw std::invalid_argument("Points need a positive width");
    }
    if (!data.colors.Empty()) {
        check(data.colors, 3, "Point colours");
        if (static_cast<std::size_t>(data.colors.Elements()) != 3 * count) {
            throw std::invalid_argument("Points need one colour each");
        }
    }
    Cloud& cloud = it->second;
    const bool hadWidths =
        cloud.onDevice ? !cloud.device.widths.Empty() : !cloud.data.widths.empty();
    const bool hadColors =
        cloud.onDevice ? !cloud.device.colors.Empty() : !cloud.data.colors.empty();
    const bool layout = count != cloud.count ||
                        (!cloud.onDevice && cloud.mapped.Empty()) ||
                        data.widths.Empty() == hadWidths || data.colors.Empty() == hadColors;
    // The view `MapPositions` gave out, handed back: the positions are in place already.
    cloud.inPlace = !layout && !cloud.mapped.Empty() && data.points.Data() == cloud.mapped.Data();
    cloud.device = data;
    cloud.onDevice = true;
    cloud.count = count;
    cloud.data = {};
    cloud.data.width = data.width;
    if (data.extent) {
        std::tie(cloud.low, cloud.high) = *data.extent;
    } else {
        const Tensor host = compute_->Download(data.points);
        const std::span<const float> hostPoints(host.As<float>(), 3 * count);
        std::vector<float> hostWidths;
        if (!data.widths.Empty()) {
            const Tensor w = compute_->Download(data.widths);
            hostWidths.assign(w.As<float>(), w.As<float>() + count);
        }
        std::tie(cloud.low, cloud.high) = SphereBounds(hostPoints, hostWidths, data.width);
    }
    if (layout) {
        Allocate(cloud);
    } else {
        cloud.dirty = true;
    }
}

Tensor PointsDrawable::MapPositions(std::uint32_t points, std::uintptr_t stream) {
    const auto it = clouds_.find(points);
    if (it == clouds_.end()) {
        throw std::invalid_argument("Unknown points " + std::to_string(points));
    }
    if (compute_ == nullptr) {
        throw std::invalid_argument("Mapping points needs compute interop");
    }
    Cloud& cloud = it->second;
    if (cloud.positions->Memory() != BufferMemory::Shared) {
        // Made before compute interop: into memory the compute device sees, both halves holding
        // the points now.
        Allocate(cloud);
        if (!cloud.onDevice) {
            const std::span<const std::byte> bytes = std::as_bytes(std::span(cloud.data.points));
            cloud.positions->Write(bytes);
            cloud.positions->Write(bytes, bytes.size());
            if (!cloud.data.widths.empty()) {
                cloud.widths->Write(std::as_bytes(std::span(cloud.data.widths)));
            }
            if (!cloud.data.colors.empty()) {
                cloud.colors->Write(std::as_bytes(std::span(cloud.data.colors)));
            }
            cloud.dirty = false;
        }
    }
    const auto count = static_cast<std::int64_t>(cloud.count);
    const auto view = [&] {
        return compute_->View(cloud.positions, DType::Float32, {count, 3},
                              VkDeviceSize{cloud.set} * cloud.count * 3 * sizeof(float));
    };
    // Nothing writes the half frames still in flight may read until they finish.
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
        throw std::invalid_argument("Unknown points " + std::to_string(points));
    }
    std::erase_if(items_, [points](const PointsItem& item) { return item.points == points; });
    previous_.resize(items_.size());
}

void PointsDrawable::Allocate(Cloud& cloud) {
    // Earlier frames may still read the buffers it replaces.
    if (cloud.positions != nullptr) {
        vkDeviceWaitIdle(context_.Device());
    }
    const BufferMemory memory = compute_ != nullptr ? BufferMemory::Shared : BufferMemory::Device;
    const auto make = [&](std::size_t floats) {
        return std::make_shared<GpuBuffer>(context_,
                                           std::max<VkDeviceSize>(floats * sizeof(float), 16),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memory);
    };
    cloud.positions = make(6 * cloud.count);
    cloud.widths = make(cloud.count);
    cloud.colors = make(3 * cloud.count);
    // Both halves of the positions hold the points; they stand still until written again.
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
    const VkDeviceSize half = VkDeviceSize{cloud.set} * cloud.count * 3 * sizeof(float);
    if (cloud.onDevice) {
        const DevicePointsData& device = cloud.device;
        if (cloud.inPlace) {
            // Written there through `MapPositions`: the frame waits for their stream.
            compute_->SignalVulkan(device.stream);
            cloud.inPlace = false;
        } else {
            compute_->CopyToBuffer(device.points, *cloud.positions, half, device.stream);
        }
        if (!device.widths.Empty()) {
            compute_->CopyToBuffer(device.widths, *cloud.widths, 0, device.stream);
        }
        if (!device.colors.Empty()) {
            compute_->CopyToBuffer(device.colors, *cloud.colors, 0, device.stream);
        }
    } else {
        FrameUploads& uploads = renderer_.Uploads();
        const PointsData& data = cloud.data;
        uploads.Write(*cloud.positions, std::as_bytes(std::span(data.points)), half);
        if (!cloud.moving) {
            // A fresh cloud: the other half too, so it stands still whichever half is current.
            uploads.Write(*cloud.positions, std::as_bytes(std::span(data.points)),
                          VkDeviceSize{1u - cloud.set} * cloud.count * 3 * sizeof(float));
        }
        if (!data.widths.empty()) {
            uploads.Write(*cloud.widths, std::as_bytes(std::span(data.widths)));
        }
        if (!data.colors.empty()) {
            uploads.Write(*cloud.colors, std::as_bytes(std::span(data.colors)));
        }
    }
    cloud.dirty = false;
}

void PointsDrawable::SetItems(std::vector<PointsItem> items) {
    for (const PointsItem& item : items) {
        if (!clouds_.contains(item.points)) {
            throw std::invalid_argument("Unknown points " + std::to_string(item.points));
        }
        if (!renderer_.HasMaterial(item.material)) {
            throw std::invalid_argument("Unknown material " + std::to_string(item.material));
        }
        if (item.anchorInstancer != 0 && anchors_ == nullptr) {
            throw std::invalid_argument("Points posed under instances need the meshes' drawable");
        }
    }
    // As `MeshDrawable::SetItems`: a draw of the same prim moves from its transform.
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
        previous.push_back(was == item.transform ? previous_[found->second] : was);
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
        // As `MeshDrawable::SettleMotion`: a view mapped for this frame expires.
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
    const glm::vec3 low = anchored ? items_[item].anchorBounds.first : cloud.low;
    const glm::vec3 high = anchored ? items_[item].anchorBounds.second : cloud.high;
    Bounds3 box = TransformBox(Relative(items_[item].transform, origin_), low, high);
    if (previous_[item] != items_[item].transform || cloud.moving) {
        // Where it was too; a moving cloud's own last bounds are not kept, so its box at the
        // last transform stands in for them.
        const Bounds3 was = TransformBox(Relative(previous_[item], origin_), low, high);
        box = {glm::min(box.first, was.first), glm::max(box.second, was.second)};
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
        bounds = bounds ? Bounds3{glm::min(bounds->first, box.first),
                                  glm::max(bounds->second, box.second)}
                        : box;
    }
    return bounds;
}

std::optional<Bounds3> PointsDrawable::WorldBounds(std::int32_t world) const {
    std::optional<Bounds3> bounds;
    for (std::size_t i = 0; i < items_.size(); ++i) {
        if (items_[i].world != world || clouds_.at(items_[i].points).count == 0) {
            continue;
        }
        const Bounds3 box = ItemBounds(i);
        bounds = bounds ? Bounds3{glm::min(bounds->first, box.first),
                                  glm::max(bounds->second, box.second)}
                        : box;
    }
    return bounds;
}

void PointsDrawable::Prepare(const FrameParams& params, std::span<const ShadowCascade>) {
    ERNIE_ZONE("Points prepare");
    // Nothing per view without points to draw: batches of thousands of views pay nothing.
    viewItems_.clear();
    anchored_ = 0;
    if (items_.empty()) {
        return;
    }
    pushbroom_ = params.camera.pushbroom;
    if (gbufferPipelines_[pushbroom_ ? 1 : 0] == nullptr) {
        CreateGBufferPipeline(pushbroom_);
    }
    for (auto& [handle, cloud] : clouds_) {
        if (cloud.dirty && cloud.count > 0) {
            Write(cloud);
        }
    }
    std::vector<GpuPointsDraw> draws(items_.size());
    // Items under instances: their models are posed on the GPU (`RecordPrepass`), with the
    // instances' slots in this frame's halves (the meshes prepare first).
    std::vector<glm::vec4> records;
    for (std::size_t i = 0; i < items_.size(); ++i) {
        const PointsItem& item = items_[i];
        const Cloud& cloud = clouds_.at(item.points);
        GpuPointsDraw& draw = draws[i];
        const bool anchored = item.anchorInstancer != 0;
        const glm::mat4 model = Relative(anchored ? item.transform * item.below : item.transform,
                                         origin_);
        const glm::mat4 previous =
            anchored ? model : glm::mat4(Relative(previous_[i], origin_));
        if (anchored) {
            MeshDrawable::AppendAnchorRecord(records, Relative(item.transform, origin_),
                                             glm::mat4(item.below),
                                             anchors_->AnchorSlot(item.anchorInstancer,
                                                                  item.anchorRow),
                                             static_cast<std::uint32_t>(i));
        }
        std::memcpy(draw.model, glm::value_ptr(model), sizeof(draw.model));
        std::memcpy(draw.previous, glm::value_ptr(previous), sizeof(draw.previous));
        draw.tint[0] = item.tint.r;
        draw.tint[1] = item.tint.g;
        draw.tint[2] = item.tint.b;
        draw.tint[3] = cloud.onDevice ? cloud.device.width : cloud.data.width;
        const auto count = static_cast<std::uint32_t>(cloud.count);
        draw.info[0] = cloud.set * count;
        draw.info[1] = (cloud.moving ? 1u - cloud.set : cloud.set) * count;
        const bool widths = cloud.onDevice ? !cloud.device.widths.Empty() : !cloud.data.widths.empty();
        const bool colors = cloud.onDevice ? !cloud.device.colors.Empty() : !cloud.data.colors.empty();
        draw.info[2] = (widths ? kHasWidths : 0u) | (colors ? kHasColors : 0u);
        const float scale = MaxScale(model);
        std::memcpy(&draw.info[3], &scale, sizeof(scale));
        draw.ids[0] = item.material;
        draw.ids[1] = item.primId;
    }
    const VkDeviceSize bytes = std::max<VkDeviceSize>(draws.size() * sizeof(GpuPointsDraw), 16);
    if (draws_->Size() < bytes) {
        vkDeviceWaitIdle(context_.Device());
        draws_ = std::make_unique<GpuBuffer>(context_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                             BufferMemory::Device);
    }
    if (!draws.empty()) {
        renderer_.Uploads().Write(*draws_, std::as_bytes(std::span(draws)));
    }
    anchored_ = static_cast<std::uint32_t>(records.size() / 9);
    if (anchored_ > 0) {
        const VkDeviceSize recordBytes = records.size() * sizeof(glm::vec4);
        if (anchorRecords_ == nullptr || anchorRecords_->Size() < recordBytes) {
            vkDeviceWaitIdle(context_.Device());
            anchorRecords_ = std::make_unique<GpuBuffer>(
                context_, recordBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
        }
        renderer_.Uploads().Write(*anchorRecords_, std::as_bytes(std::span(records)));
    }
    // Each view draws the items of its worlds whose bounds meet its frustum.
    viewItems_.assign(params.Views(), {});
    for (std::size_t v = 0; v < params.Views(); ++v) {
        const std::int32_t world = params.ViewWorld(v);
        const Frustum frustum = Frustum::FromMatrix(params.View(v).ViewProjection());
        for (std::size_t i = 0; i < items_.size(); ++i) {
            const PointsItem& item = items_[i];
            if (clouds_.at(item.points).count == 0 ||
                (item.world != -1 && world != -1 && item.world != world)) {
                continue;
            }
            const Bounds3 box = ItemBounds(i);
            if (params.posesOnGpu || frustum.Intersects(box.first, box.second)) {
                viewItems_[v].push_back(static_cast<std::uint32_t>(i));
            }
        }
    }
}

void PointsDrawable::RecordPrepass(VkCommandBuffer commands) const {
    if (anchored_ == 0) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Anchored points");
    anchors_->RecordAnchors(commands, *anchorRecords_, anchored_, *draws_,
                            AnchorOutput::PointsDraws);
}

std::vector<DescriptorResource> PointsDrawable::Resources(const RasterFrame& frame,
                                                          std::size_t item) const {
    std::vector<DescriptorResource> resources = {frame.uniforms, frame.shadows, frame.shadowMap,
                                                 frame.shadowSampler, frame.clampSampler};
    renderer_.MaterialOrDefault(items_[item].material).AppendResources(resources);
    const Cloud& cloud = clouds_.at(items_[item].points);
    for (const GpuBuffer* buffer :
         {cloud.positions.get(), cloud.widths.get(), cloud.colors.get(), draws_.get()}) {
        resources.push_back(DescriptorResource::Of(*buffer));
    }
    return resources;
}

void PointsDrawable::RecordMotion(VkCommandBuffer commands, const RasterFrame& frame,
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
            const std::vector<ShaderModuleSource> modules = PointsModules();
            const std::vector<std::uint32_t> vertex =
                renderer_.Compiler().CompileSpirv(modules, "vertexMain");
            const std::vector<std::uint32_t> fragment =
                renderer_.Compiler().CompileSpirv(modules, "motionFragment");
            GraphicsPipelineDesc desc;
            desc.vertexSpirv = vertex;
            desc.fragmentSpirv = fragment;
            desc.colorFormats = {VK_FORMAT_R32G32B32A32_SFLOAT};
            desc.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
            desc.depthFormat = RenderTargets::kDepthFormat;
            // The same depths as the colour pass's: only the surfaces it kept.
            desc.depthCompare = VK_COMPARE_OP_LESS_OR_EQUAL;
            desc.depthWrite = false;
            motionPipeline_ = std::make_unique<GraphicsPipeline>(context_, motionLayout_, desc);
        }
        if (!bound) {
            vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, motionPipeline_->Handle());
            bound = true;
        }
        std::vector<DescriptorResource> resources = Resources(frame, i);
        resources.push_back(motion);
        motionLayout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, resources);
        DrawConstants constants{};
        constants.first = i;
        motionLayout_.PushConstants(commands, &constants, sizeof(constants));
        vkCmdDraw(commands, 6, static_cast<std::uint32_t>(cloud.count), 0, 0);
    }
}

void PointsDrawable::RecordShadow(VkCommandBuffer commands, const RasterFrame& frame,
                                  std::size_t, const glm::mat4& lightViewProjection,
                                  const ShadowCasters& casters) const {
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline_->Handle());
    for (std::size_t i = 0; i < items_.size(); ++i) {
        const std::size_t count = clouds_.at(items_[i].points).count;
        if (count == 0 || !casters.Draws(items_[i].world)) {
            continue;
        }
        layout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, Resources(frame, i));
        DrawConstants constants{};
        std::memcpy(constants.lightViewProj, glm::value_ptr(lightViewProjection),
                    sizeof(constants.lightViewProj));
        constants.first = static_cast<std::uint32_t>(i);
        constants.shadowFromFrame = frame.shadowFromFrame ? 1u : 0u;
        layout_.PushConstants(commands, &constants, sizeof(constants));
        vkCmdDraw(commands, 6, static_cast<std::uint32_t>(count), 0, 0);
    }
}

void PointsDrawable::RecordColor(VkCommandBuffer commands, const RasterFrame& frame,
                                 std::size_t view, DrawPhase phase) const {
    if (phase == DrawPhase::Occludees || view >= viewItems_.size()) {
        return;
    }
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      gbufferPipelines_[pushbroom_ ? 1 : 0]->Handle());
    for (const std::uint32_t i : viewItems_[view]) {
        layout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, Resources(frame, i));
        DrawConstants constants{};
        constants.first = i;
        layout_.PushConstants(commands, &constants, sizeof(constants));
        vkCmdDraw(commands, 6, static_cast<std::uint32_t>(clouds_.at(items_[i].points).count), 0,
                  0);
    }
}

}  // namespace ernie
