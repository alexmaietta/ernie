#include "ernie/raster/mesh_drawable.h"

#include "ernie/geometry/frustum.h"
#include "ernie/kernel/embedded_shaders.h"
#include "ernie/util/profile.h"

#include <glm/gtc/type_ptr.hpp>
#include <meshoptimizer.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace ernie {

namespace {

constexpr std::uint32_t kHasNormals = 1;
constexpr std::uint32_t kHasColors = 2;
constexpr std::uint32_t kInstanced = 4;
constexpr std::uint32_t kHasOrientations = 8;
constexpr std::uint32_t kHasScales = 16;

/// One draw as `mesh.slang` reads it.
struct GpuDraw {
    float model[16];
    float prototype[16];
    float tint[4];
    std::uint32_t offsets[4];
    std::uint32_t points[4];
    float previous[16];  ///< `model` at the last frame, for motion blur.
};
static_assert(sizeof(GpuDraw) == 240, "GpuDraw must match MeshDraw in mesh.slang");

struct DrawConstants {
    float lightViewProj[16];
    std::uint32_t first;
    std::uint32_t instanceOffset;
    std::uint32_t indexOffset;
    std::uint32_t records;
};
static_assert(sizeof(DrawConstants) == 80, "DrawConstants must match frame.slang");

constexpr VkShaderStageFlags kGraphics = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

/// Per instanced draw, what cull.slang needs to place its instances' bounding spheres.
struct GpuCullItem {
    float model[16];
    float sphere[4];  ///< Prototype's bounding sphere after its own transform: centre, radius.
    /// Largest scale of `model`; flags (as bits): orientations 1, scales 2; largest scale of
    /// the prototype's transform.
    float info[4];
    std::uint32_t levels[4];  ///< First level of detail in the level table, level count.
};
static_assert(sizeof(GpuCullItem) == 112, "GpuCullItem must match CullItem in cull.slang");

static_assert(sizeof(MeshDrawable::GpuCullJob) == 32,
              "GpuCullJob must match CullJob in cull.slang");

constexpr VkShaderStageFlags kCompute = VK_SHADER_STAGE_COMPUTE_BIT;

/// cull.slang's bindings: culling data, jobs, planes, instance ids and arrays, and the outputs.
constexpr DescriptorBinding kCullBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {10, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
};

/// cull.slang's modes.
constexpr std::uint32_t kCullFrustum = 0;
constexpr std::uint32_t kCullOccluders = 1;
constexpr std::uint32_t kCullTwoPassFirst = 2;
constexpr std::uint32_t kCullTwoPassSecond = 3;

/// cull.slang's float4s per view: 6 planes, the view-projection's columns, pixels per unit and the
/// error, the depth pyramid.
constexpr std::size_t kCullViewStride = 12;

/// Largest length of the first three columns of `m`: how much it scales lengths at most.
float MaxScale(const glm::mat4& m) {
    return std::max(
        {glm::length(glm::vec3(m[0])), glm::length(glm::vec3(m[1])), glm::length(glm::vec3(m[2]))});
}

/// Bit flags of `GpuCullItem::info[1]`.
constexpr std::uint32_t kCullOrientations = 1;
constexpr std::uint32_t kCullScales = 2;

/// Grows `buffer` to hold `bytes`, waiting for the device first if it replaces one in use.
void Reserve(const VulkanContext& context, std::unique_ptr<GpuBuffer>& buffer, VkDeviceSize bytes,
             VkBufferUsageFlags usage, BufferMemory memory) {
    bytes = std::max<VkDeviceSize>(bytes, 16);
    if (buffer != nullptr && buffer->Size() >= bytes) {
        return;
    }
    if (buffer != nullptr) {
        vkDeviceWaitIdle(context.Device());
    }
    buffer = std::make_unique<GpuBuffer>(context, bytes, usage, memory);
}

/// Frame (0 to 4), material (5 to 9) and mesh (10 to 19) bindings; see mesh.slang.
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
    {14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {17, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {19, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
};

/// The mesh passes, which write the renderer's G-buffer.
std::vector<ShaderModuleSource> MeshModules() {
    return {{"brdf", std::string(BrdfSlang())},
            {"frame", shaders::kFrameSlang},
            {"surface", shaders::kSurfaceSlang},
            {"mesh", shaders::kMeshSlang}};
}

/// A host-visible storage buffer holding `bytes`, at least 16 bytes so it can always be bound.
std::unique_ptr<GpuBuffer> StorageBuffer(const VulkanContext& context,
                                         std::span<const std::byte> bytes) {
    auto buffer =
        std::make_unique<GpuBuffer>(context, std::max<VkDeviceSize>(bytes.size(), 16),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Upload);
    if (!bytes.empty()) {
        buffer->Write(bytes);
    }
    return buffer;
}

template <typename T>
std::unique_ptr<GpuBuffer> StorageBuffer(const VulkanContext& context, const std::vector<T>& data) {
    return StorageBuffer(context, std::as_bytes(std::span(data)));
}

void CheckTriples(const std::vector<float>& values, std::size_t count, const char* what) {
    if (!values.empty() && values.size() != 3 * count) {
        throw std::invalid_argument(std::string(what) + " must have 3 values per point");
    }
}

void CheckInstances(const InstanceData& data) {
    if (data.positions.size() % 3 != 0) {
        throw std::invalid_argument("Instance positions must have 3 values per instance");
    }
    const std::size_t count = data.positions.size() / 3;
    if (!data.orientations.empty() && data.orientations.size() != 4 * count) {
        throw std::invalid_argument("Instance orientations must have 4 values per instance");
    }
    if (!data.scales.empty() && data.scales.size() != 3 * count) {
        throw std::invalid_argument("Instance scales must have 3 values per instance");
    }
}

std::pair<glm::vec3, glm::vec3> PointBounds(const std::vector<float>& points) {
    glm::vec3 low(std::numeric_limits<float>::max());
    glm::vec3 high(std::numeric_limits<float>::lowest());
    for (std::size_t i = 0; i + 2 < points.size(); i += 3) {
        const glm::vec3 p(points[i], points[i + 1], points[i + 2]);
        low = glm::min(low, p);
        high = glm::max(high, p);
    }
    return {low, high};
}

/// Bounds of the box [`low`, `high`] after `transform`.
std::pair<glm::vec3, glm::vec3> TransformBox(const glm::mat4& transform, const glm::vec3& low,
                                             const glm::vec3& high) {
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

/// Simplified levels of detail of `data`'s triangles after the full mesh, each of about half the
/// triangles of the one before and simplified from the full mesh, appended to `indices`; none
/// for small meshes or once simplification stalls. Errors are in mesh units.
///
/// References:
/// - M. Garland and P. S. Heckbert, "Surface Simplification Using Quadric Error Metrics",
///   SIGGRAPH '97, 209-216, 1997. https://doi.org/10.1145/258734.258849
/// - A. Kapoulkine, "meshoptimizer", `meshopt_simplify`. https://github.com/zeux/meshoptimizer
template <typename Level>
std::vector<Level> SimplifiedLevels(const MeshData& data, std::size_t maxLevels,
                                    std::size_t minTriangles, std::vector<std::uint32_t>& indices) {
    std::vector<Level> levels;
    const std::size_t count = data.indices.size();
    const std::size_t vertices = data.points.size() / 3;
    if (count / 3 < minTriangles || vertices == 0) {
        return levels;
    }
    const float scale = meshopt_simplifyScale(data.points.data(), vertices, 3 * sizeof(float));
    std::size_t previous = count;
    std::vector<std::uint32_t> simplified(count);
    while (levels.size() + 1 < maxLevels) {
        const std::size_t target = previous / 2 / 3 * 3;
        if (target < 3 * 8) {
            break;
        }
        float error = 0.0f;
        const std::size_t kept =
            meshopt_simplify(simplified.data(), data.indices.data(), count, data.points.data(),
                             vertices, 3 * sizeof(float), target, 1.0f, 0, &error);
        if (kept == 0 || kept > previous * 9 / 10) {
            break;  // simplification stalled: no level worth its memory
        }
        levels.push_back({static_cast<std::uint32_t>(count + indices.size()),
                          static_cast<std::uint32_t>(kept), error * scale});
        indices.insert(indices.end(), simplified.begin(),
                       simplified.begin() + static_cast<std::ptrdiff_t>(kept));
        previous = kept;
    }
    return levels;
}

}  // namespace

MeshDrawable::MeshDrawable(RasterRenderer& renderer)
    : renderer_(renderer), context_(renderer.Context()),
      layout_(renderer.Context(), kBindings, sizeof(DrawConstants)),
      cullLayout_(renderer.Context(), kCullBindings, 32) {
    // Stands in for a depth pyramid in culling without one.
    noHzb_ =
        std::make_unique<GpuImage>(context_, ImageDesc{1, 1, 1, 1, VK_FORMAT_R32_SFLOAT,
                                                       VK_IMAGE_USAGE_SAMPLED_BIT, false, true});
    context_.Submit([this](VkCommandBuffer commands) {
        noHzb_->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    });
    const std::vector<ShaderModuleSource> cull{{"cull", shaders::kCullSlang}};
    cullPipeline_ = std::make_unique<ComputePipeline>(
        context_, cullLayout_, renderer_.Compiler().CompileSpirv(cull, "cullMain"));
    CreateGBufferPipeline();
    const std::vector<std::uint32_t> vertex =
        renderer_.Compiler().CompileSpirv(MeshModules(), "meshShadowVertex");
    GraphicsPipelineDesc shadow;
    shadow.vertexSpirv = vertex;
    shadow.depthFormat = VK_FORMAT_D32_SFLOAT;
    shadow.depthBiasSlope = 1.5f;
    shadowPipeline_ = std::make_unique<GraphicsPipeline>(context_, layout_, shadow);
    RebuildGeometry();
    RebuildInstances();
    draws_ = StorageBuffer(context_, std::span<const std::byte>());
    instanceIds_ = StorageBuffer(context_, std::span<const std::byte>());
    drawRecords_ = StorageBuffer(context_, std::span<const std::byte>());
}

MeshDrawable::~MeshDrawable() {
    vkDeviceWaitIdle(context_.Device());
}

void MeshDrawable::CreateGBufferPipeline() {
    const std::vector<ShaderModuleSource> modules = MeshModules();
    const std::vector<std::uint32_t> vertex =
        renderer_.Compiler().CompileSpirv(modules, "meshVertexMain");
    const std::vector<std::uint32_t> fragment =
        renderer_.Compiler().CompileSpirv(modules, "meshFragmentMain");
    GraphicsPipelineDesc desc;
    desc.vertexSpirv = vertex;
    desc.fragmentSpirv = fragment;
    desc.colorFormats.assign(RenderTargets::kGBufferFormats.begin(),
                             RenderTargets::kGBufferFormats.end());
    desc.depthFormat = RenderTargets::kDepthFormat;
    gbufferPipeline_ = std::make_unique<GraphicsPipeline>(context_, layout_, desc);
}

std::uint32_t MeshDrawable::AddMesh(MeshData data) {
    ERNIE_ZONE("Mesh add (LOD simplify)");
    if (data.points.size() % 3 != 0 || data.indices.size() % 3 != 0) {
        throw std::invalid_argument("Mesh points and indices must come in threes");
    }
    const std::size_t count = data.points.size() / 3;
    CheckTriples(data.normals, count, "Mesh normals");
    CheckTriples(data.colors, count, "Mesh colours");
    for (const std::uint32_t index : data.indices) {
        if (index >= count) {
            throw std::invalid_argument("Mesh index out of range");
        }
    }
    Mesh mesh;
    std::tie(mesh.low, mesh.high) = PointBounds(data.points);
    mesh.levels.push_back({0, static_cast<std::uint32_t>(data.indices.size()), 0.0f});
    for (const MeshLevel& level :
         SimplifiedLevels<MeshLevel>(data, kMaxLevels, kMinLodTriangles, mesh.lodIndices)) {
        mesh.levels.push_back(level);
    }
    mesh.data = std::move(data);
    const std::uint32_t handle = nextHandle_++;
    meshes_.emplace(handle, std::move(mesh));
    geometryDirty_ = true;
    return handle;
}

void MeshDrawable::SetPoints(std::uint32_t mesh, std::span<const float> points) {
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " + std::to_string(mesh));
    }
    if (points.size() != it->second.data.points.size()) {
        throw std::invalid_argument("SetPoints must keep the number of points");
    }
    it->second.data.points.assign(points.begin(), points.end());
    std::tie(it->second.low, it->second.high) = PointBounds(it->second.data.points);
    it->second.devicePoints = Tensor();
    it->second.pointsDirty = true;
}

void MeshDrawable::SetPoints(std::uint32_t mesh, const Tensor& points,
                             std::optional<std::pair<glm::vec3, glm::vec3>> bounds,
                             std::uintptr_t stream) {
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " + std::to_string(mesh));
    }
    CheckDeviceTensor(points, 3, "Mesh points");
    Mesh& target = it->second;
    if (static_cast<std::size_t>(points.Elements()) != target.data.points.size()) {
        throw std::invalid_argument("SetPoints must keep the number of points");
    }
    if (bounds) {
        std::tie(target.low, target.high) = *bounds;
    } else {
        const Tensor host = compute_->Download(points);
        target.data.points.assign(host.As<float>(), host.As<float>() + host.Elements());
        std::tie(target.low, target.high) = PointBounds(target.data.points);
    }
    target.devicePoints = points;
    target.deviceStream = stream;
    target.pointsDirty = true;
}

void MeshDrawable::CopyDevicePoints(Mesh& mesh) {
    compute_->CopyToBuffer(mesh.devicePoints, *points_,
                           VkDeviceSize{mesh.firstVertex} * 3 * sizeof(float), mesh.deviceStream);
}

std::shared_ptr<GpuBuffer> MeshDrawable::Pool(std::span<const std::byte> bytes) const {
    if (compute_ == nullptr) {
        return StorageBuffer(context_, bytes);
    }
    auto buffer =
        std::make_shared<GpuBuffer>(context_, std::max<VkDeviceSize>(bytes.size(), 16),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Shared);
    if (!bytes.empty()) {
        buffer->Write(bytes);
    }
    return buffer;
}

void MeshDrawable::RemoveMesh(std::uint32_t mesh) {
    if (meshes_.erase(mesh) == 0) {
        throw std::invalid_argument("Unknown mesh " + std::to_string(mesh));
    }
    std::erase_if(items_, [mesh](const MeshItem& item) { return item.mesh == mesh; });
    itemPipelines_.clear();
    geometryDirty_ = true;
}

std::uint32_t MeshDrawable::AddInstancer(InstanceData data) {
    CheckInstances(data);
    const std::uint32_t handle = nextHandle_++;
    Instancer instancer;
    instancer.count = data.positions.size() / 3;
    instancer.data = std::move(data);
    instancers_.emplace(handle, std::move(instancer));
    instancesDirty_ = true;
    return handle;
}

void MeshDrawable::SetInstances(std::uint32_t instancer, InstanceData data) {
    const auto it = instancers_.find(instancer);
    if (it == instancers_.end()) {
        throw std::invalid_argument("Unknown instancer " + std::to_string(instancer));
    }
    CheckInstances(data);
    Instancer& target = it->second;
    const std::size_t count = data.positions.size() / 3;
    instancesDirty_ = instancesDirty_ || count != target.count;
    target.count = count;
    target.data = std::move(data);
    target.device = {};
    target.onDevice = false;
    target.dirty = true;
}

void MeshDrawable::CheckDeviceTensor(const Tensor& tensor, std::size_t components,
                                     const char* what) const {
    if (compute_ == nullptr) {
        throw std::invalid_argument("GPU inputs need MeshDrawable::SetComputeInterop");
    }
    const bool shaped =
        tensor.Rank() == 1 ||
        (tensor.Rank() == 2 && tensor.Shape()[1] == static_cast<std::int64_t>(components));
    if (tensor.Location() != compute_->Location() || tensor.Type() != DType::Float32 ||
        !tensor.IsContiguous() || !shaped ||
        static_cast<std::size_t>(tensor.Elements()) % components != 0) {
        throw std::invalid_argument(std::string(what) + " must be a contiguous float32 tensor of " +
                                    std::to_string(components) + "-vectors on the compute device " +
                                    std::to_string(compute_->Device()));
    }
}

void MeshDrawable::SetInstances(std::uint32_t instancer, const DeviceInstanceData& data) {
    const auto it = instancers_.find(instancer);
    if (it == instancers_.end()) {
        throw std::invalid_argument("Unknown instancer " + std::to_string(instancer));
    }
    CheckDeviceTensor(data.positions, 3, "Instance positions");
    const auto count = static_cast<std::size_t>(data.positions.Elements()) / 3;
    if (!data.orientations.Empty()) {
        CheckDeviceTensor(data.orientations, 4, "Instance orientations");
        if (static_cast<std::size_t>(data.orientations.Elements()) != 4 * count) {
            throw std::invalid_argument("Instance orientations must have 4 values per instance");
        }
    }
    if (!data.scales.Empty()) {
        CheckDeviceTensor(data.scales, 3, "Instance scales");
        if (static_cast<std::size_t>(data.scales.Elements()) != 3 * count) {
            throw std::invalid_argument("Instance scales must have 3 values per instance");
        }
    }
    Instancer& target = it->second;
    instancesDirty_ = instancesDirty_ || count != target.count;
    target.count = count;
    target.device = data;
    target.onDevice = true;
    target.dirty = true;
    target.data = {};
    if (!data.extent) {
        const Tensor positions = compute_->Download(data.positions);
        target.data.positions.assign(positions.As<float>(), positions.As<float>() + 3 * count);
        if (!data.scales.Empty()) {
            const Tensor scales = compute_->Download(data.scales);
            target.data.scales.assign(scales.As<float>(), scales.As<float>() + 3 * count);
        }
    }
}

void MeshDrawable::SetComputeInterop(ComputeInterop* compute) {
    compute_ = compute;
    geometryDirty_ = true;
    instancesDirty_ = true;
}

void MeshDrawable::RemoveInstancer(std::uint32_t instancer) {
    if (instancers_.erase(instancer) == 0) {
        throw std::invalid_argument("Unknown instancer " + std::to_string(instancer));
    }
    std::erase_if(items_,
                  [instancer](const MeshItem& item) { return item.instancer == instancer; });
    itemPipelines_.clear();
    instancesDirty_ = true;
}

void MeshDrawable::SetItems(std::vector<MeshItem> items) {
    for (const MeshItem& item : items) {
        if (!meshes_.contains(item.mesh)) {
            throw std::invalid_argument("Unknown mesh " + std::to_string(item.mesh));
        }
        if (!renderer_.HasMaterial(item.material)) {
            throw std::invalid_argument("Unknown material " + std::to_string(item.material));
        }
        if (item.instancer) {
            const auto it = instancers_.find(*item.instancer);
            if (it == instancers_.end()) {
                throw std::invalid_argument("Unknown instancer " + std::to_string(*item.instancer));
            }
            const std::size_t count = it->second.count;
            for (const std::uint32_t instance : item.instances) {
                if (instance >= count) {
                    throw std::invalid_argument("Instance index out of range");
                }
            }
        }
        if (!item.instanceWorlds.empty() &&
            (item.instanceWorlds.size() != item.instances.size() ||
             !std::is_sorted(item.instanceWorlds.begin(), item.instanceWorlds.end()))) {
            throw std::invalid_argument("Instance worlds must match the instances, sorted");
        }
    }
    std::vector<std::vector<WorldRange>> ranges;
    for (const MeshItem& item : items) {
        std::vector<WorldRange> runs;
        if (item.instanceWorlds.empty()) {
            const auto count =
                static_cast<std::uint32_t>(item.instancer ? item.instances.size() : std::size_t{1});
            runs.push_back({item.world, 0, count});
        }
        for (std::uint32_t i = 0; i < item.instanceWorlds.size(); ++i) {
            if (runs.empty() || runs.back().world != item.instanceWorlds[i]) {
                runs.push_back({item.instanceWorlds[i], i, 0});
            }
            ++runs.back().count;
        }
        ranges.push_back(std::move(runs));
    }
    // Each draw's transform at the last frame: a draw of the same prim keeps its own while its
    // transform stays, else moves from it; a new draw stands still.
    std::map<std::uint32_t, std::size_t> old;
    for (std::size_t i = 0; i < items_.size(); ++i) {
        old.try_emplace(items_[i].primId, i);
    }
    std::vector<glm::dmat4> previous;
    for (const MeshItem& item : items) {
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
    itemRanges_ = std::move(ranges);
    itemPipelines_.clear();
    drawsDirty_ = true;
    UpdateTransforms();
}

void MeshDrawable::SettleMotion() {
    for (std::size_t i = 0; i < items_.size(); ++i) {
        if (previous_[i] != items_[i].transform) {
            previous_[i] = items_[i].transform;
            drawsDirty_ = true;
        }
    }
    if (drawsDirty_) {
        UpdateTransforms();
    }
}

void MeshDrawable::SetOrigin(const glm::dvec3& origin) {
    if (origin != origin_) {
        origin_ = origin;
        UpdateTransforms();
        drawsDirty_ = true;
    }
}

void MeshDrawable::UpdateTransforms() {
    const auto relative = [&](glm::dmat4 world) {
        world[3] -= glm::dvec4(origin_, 0.0);
        return glm::mat4(world);
    };
    transforms_.clear();
    previousTransforms_.clear();
    for (std::size_t i = 0; i < items_.size(); ++i) {
        transforms_.push_back(relative(items_[i].transform));
        previousTransforms_.push_back(relative(previous_[i]));
    }
}

void MeshDrawable::RebuildGeometry() {
    ERNIE_ZONE("Mesh rebuild geometry");
    std::vector<float> points;
    std::vector<float> normals;
    std::vector<float> colors;
    std::vector<std::uint32_t> indices;
    for (auto& [handle, mesh] : meshes_) {
        mesh.firstIndex = static_cast<std::uint32_t>(indices.size());
        mesh.firstVertex = static_cast<std::uint32_t>(points.size() / 3);
        const std::size_t count = mesh.data.points.size();
        points.insert(points.end(), mesh.data.points.begin(), mesh.data.points.end());
        if (mesh.data.normals.empty()) {
            normals.resize(normals.size() + count, 0.0f);
        } else {
            normals.insert(normals.end(), mesh.data.normals.begin(), mesh.data.normals.end());
        }
        if (mesh.data.colors.empty()) {
            colors.resize(colors.size() + count, 0.0f);
        } else {
            colors.insert(colors.end(), mesh.data.colors.begin(), mesh.data.colors.end());
        }
        indices.insert(indices.end(), mesh.data.indices.begin(), mesh.data.indices.end());
        indices.insert(indices.end(), mesh.lodIndices.begin(), mesh.lodIndices.end());
        mesh.pointsDirty = false;
    }
    vkDeviceWaitIdle(context_.Device());
    points_ = Pool(std::as_bytes(std::span(points)));
    normals_ = StorageBuffer(context_, normals);
    colors_ = StorageBuffer(context_, colors);
    indices_ = StorageBuffer(context_, indices);
    for (auto& [handle, mesh] : meshes_) {
        if (!mesh.devicePoints.Empty() && compute_ != nullptr) {
            CopyDevicePoints(mesh);
        }
    }
    geometryDirty_ = false;
}

void MeshDrawable::RebuildInstances() {
    ERNIE_ZONE("Mesh rebuild instances");
    std::vector<float> positions;
    std::vector<float> orientations;
    std::vector<float> scales;
    for (auto& [handle, instancer] : instancers_) {
        instancer.first = static_cast<std::uint32_t>(positions.size() / 3);
        const std::size_t count = instancer.count;
        const InstanceData& data = instancer.data;
        const bool host = !instancer.onDevice;
        if (host && data.positions.size() == 3 * count) {
            positions.insert(positions.end(), data.positions.begin(), data.positions.end());
        } else {
            positions.resize(positions.size() + 3 * count, 0.0f);
        }
        if (host && data.orientations.size() == 4 * count) {
            orientations.insert(orientations.end(), data.orientations.begin(),
                                data.orientations.end());
        } else {
            orientations.resize(orientations.size() + 4 * count, 0.0f);
        }
        if (host && data.scales.size() == 3 * count) {
            scales.insert(scales.end(), data.scales.begin(), data.scales.end());
        } else {
            scales.resize(scales.size() + 3 * count, 1.0f);
        }
    }
    vkDeviceWaitIdle(context_.Device());
    positions_ = Pool(std::as_bytes(std::span(positions)));
    orientations_ = Pool(std::as_bytes(std::span(orientations)));
    scales_ = Pool(std::as_bytes(std::span(scales)));
    for (auto& [handle, instancer] : instancers_) {
        if (instancer.onDevice) {
            WriteInstancer(instancer);
        }
        instancer.dirty = false;
    }
    instancesDirty_ = false;
}

void MeshDrawable::WriteInstancer(const Instancer& instancer) {
    const VkDeviceSize first = instancer.first;
    if (instancer.onDevice) {
        const DeviceInstanceData& device = instancer.device;
        compute_->CopyToBuffer(device.positions, *positions_, first * 3 * sizeof(float),
                               device.stream);
        if (!device.orientations.Empty()) {
            compute_->CopyToBuffer(device.orientations, *orientations_, first * 4 * sizeof(float),
                                   device.stream);
        }
        if (!device.scales.Empty()) {
            compute_->CopyToBuffer(device.scales, *scales_, first * 3 * sizeof(float),
                                   device.stream);
        }
        return;
    }
    const InstanceData& data = instancer.data;
    FrameUploads& uploads = renderer_.Uploads();
    uploads.Write(*positions_, std::as_bytes(std::span(data.positions)), first * 3 * sizeof(float));
    if (!data.orientations.empty()) {
        uploads.Write(*orientations_, std::as_bytes(std::span(data.orientations)),
                      first * 4 * sizeof(float));
    }
    if (!data.scales.empty()) {
        uploads.Write(*scales_, std::as_bytes(std::span(data.scales)), first * 3 * sizeof(float));
    }
}

void MeshDrawable::Prepare(const FrameParams& params, std::span<const ShadowCascade> cascades) {
    ERNIE_ZONE("Mesh prepare");
    viewWorlds_.clear();
    for (std::size_t v = 0; v < params.Views(); ++v) {
        viewWorlds_.push_back(params.ViewWorld(v));
    }
    // The per-draw data changes only with the draws, meshes or instancers, not every frame.
    bool drawsChanged = drawsDirty_ || geometryDirty_ || instancesDirty_ ||
                        renderer_.MaterialSetVersion() != drawsMaterials_;
    for (const auto& [handle, instancer] : instancers_) {
        drawsChanged = drawsChanged || instancer.dirty;
    }
    if (geometryDirty_) {
        RebuildGeometry();
    } else {
        for (auto& [handle, mesh] : meshes_) {
            if (!mesh.pointsDirty) {
                continue;
            }
            if (!mesh.devicePoints.Empty()) {
                CopyDevicePoints(mesh);
            } else {
                renderer_.Uploads().Write(*points_, std::as_bytes(std::span(mesh.data.points)),
                                          VkDeviceSize{mesh.firstVertex} * 3 * sizeof(float));
            }
            mesh.pointsDirty = false;
        }
    }
    if (instancesDirty_) {
        RebuildInstances();
    } else {
        for (auto& [handle, instancer] : instancers_) {
            if (instancer.dirty) {
                WriteInstancer(instancer);
                instancer.dirty = false;
            }
        }
    }

    if (drawsChanged) {
        WriteDraws();
        drawsDirty_ = false;
        drawsMaterials_ = renderer_.MaterialSetVersion();
    }
    Cull(params, cascades);
}

void MeshDrawable::WriteDraws() {
    ERNIE_ZONE("Mesh write draws");
    itemIdOffsets_.clear();
    std::vector<GpuDraw> draws(items_.size());
    std::vector<std::uint32_t> instanceIds;
    itemPipelines_.clear();
    for (std::size_t i = 0; i < items_.size(); ++i) {
        const MeshItem& item = items_[i];
        const Mesh& mesh = meshes_.at(item.mesh);
        itemPipelines_.push_back(gbufferPipeline_.get());
        GpuDraw& draw = draws[i];
        std::memcpy(draw.model, glm::value_ptr(transforms_[i]), sizeof(draw.model));
        std::memcpy(draw.previous, glm::value_ptr(previousTransforms_[i]), sizeof(draw.previous));
        std::memcpy(draw.prototype, glm::value_ptr(item.prototype), sizeof(draw.prototype));
        draw.tint[0] = item.tint.r;
        draw.tint[1] = item.tint.g;
        draw.tint[2] = item.tint.b;
        draw.offsets[0] = mesh.firstIndex;
        draw.offsets[1] = mesh.firstVertex;
        draw.offsets[2] = static_cast<std::uint32_t>(instanceIds.size());
        itemIdOffsets_.push_back(draw.offsets[2]);
        std::uint32_t flags = 0;
        flags |= mesh.data.normals.empty() ? 0u : kHasNormals;
        flags |= item.vertexColors && !mesh.data.colors.empty() ? kHasColors : 0u;
        if (item.instancer) {
            const Instancer& instancer = instancers_.at(*item.instancer);
            flags |= kInstanced;
            const bool orientations = instancer.onDevice ? !instancer.device.orientations.Empty()
                                                         : !instancer.data.orientations.empty();
            const bool scales = instancer.onDevice ? !instancer.device.scales.Empty()
                                                   : !instancer.data.scales.empty();
            flags |= orientations ? kHasOrientations : 0u;
            flags |= scales ? kHasScales : 0u;
            const std::uint32_t first = instancer.first;
            draw.points[3] = first;
            for (const std::uint32_t instance : item.instances) {
                instanceIds.push_back(first + instance);
            }
        }
        draw.offsets[3] = flags;
        draw.points[0] = mesh.firstVertex;
        draw.points[1] =
            renderer_.HasMaterial(item.material) ? item.material : RasterRenderer::kDefaultMaterial;
        draw.points[2] = item.primId;
    }
    const auto bytes = [](const auto& data) {
        return std::max<VkDeviceSize>(data.size() * sizeof(data[0]), 16);
    };
    if (draws_->Size() < bytes(draws)) {
        vkDeviceWaitIdle(context_.Device());
        draws_ = StorageBuffer(context_, draws);
    } else if (!draws.empty()) {
        renderer_.Uploads().Write(*draws_, std::as_bytes(std::span(draws)));
    }
    if (instanceIds_->Size() < bytes(instanceIds)) {
        vkDeviceWaitIdle(context_.Device());
        instanceIds_ = StorageBuffer(context_, instanceIds);
    } else if (!instanceIds.empty()) {
        renderer_.Uploads().Write(*instanceIds_, std::as_bytes(std::span(instanceIds)));
    }
}

std::optional<Bounds3> MeshDrawable::Bounds() const {
    std::optional<Bounds3> bounds;
    const auto add = [&bounds](const std::pair<glm::vec3, glm::vec3>& box) {
        bounds = bounds ? std::pair(glm::min(bounds->first, box.first),
                                    glm::max(bounds->second, box.second))
                        : box;
    };
    for (std::size_t n = 0; n < items_.size(); ++n) {
        const MeshItem& item = items_[n];
        const glm::mat4& transform = transforms_[n];
        const Mesh& mesh = meshes_.at(item.mesh);
        if (mesh.data.points.empty()) {
            continue;
        }
        if (!item.instancer) {
            add(TransformBox(transform, mesh.low, mesh.high));
            continue;
        }
        const Instancer& instancer = instancers_.at(*item.instancer);
        const std::optional<Bounds3>& extent = instancer.device.extent;
        if (instancer.onDevice && extent) {
            add(TransformBox(transform, extent->first, extent->second));
            continue;
        }
        const auto [low, high] = TransformBox(item.prototype, mesh.low, mesh.high);
        const float radius = glm::length(glm::max(glm::abs(low), glm::abs(high)));
        const InstanceData& data = instancer.data;
        for (const std::uint32_t i : item.instances) {
            float scale = 1.0f;
            if (!data.scales.empty()) {
                scale = std::max({std::abs(data.scales[3 * i]), std::abs(data.scales[3 * i + 1]),
                                  std::abs(data.scales[3 * i + 2])});
            }
            const glm::vec3 p(data.positions[3 * i], data.positions[3 * i + 1],
                              data.positions[3 * i + 2]);
            add(TransformBox(transform, p - glm::vec3(radius * scale),
                             p + glm::vec3(radius * scale)));
        }
    }
    return bounds;
}

std::optional<Bounds3> MeshDrawable::WorldBounds(std::int32_t world) const {
    std::optional<Bounds3> bounds;
    for (std::size_t i = 0; i < items_.size() && i < itemRanges_.size(); ++i) {
        for (const WorldRange& range : itemRanges_[i]) {
            if (range.world != world || range.count == 0) {
                continue;
            }
            if (const std::optional<Bounds3> box = RangeBounds(i, range)) {
                bounds = bounds ? Bounds3(glm::min(bounds->first, box->first),
                                          glm::max(bounds->second, box->second))
                                : *box;
            }
        }
    }
    return bounds;
}

std::uint64_t MeshDrawable::TriangleCount() const {
    std::uint64_t triangles = 0;
    for (const MeshItem& item : items_) {
        const std::uint64_t perMesh = meshes_.at(item.mesh).data.indices.size() / 3;
        triangles += perMesh * (item.instancer ? item.instances.size() : 1u);
    }
    return triangles;
}

std::vector<DescriptorResource> MeshDrawable::Resources(const RasterFrame& frame, std::size_t item,
                                                        bool culled) const {
    const MeshItem& draw = items_[item];
    std::vector<DescriptorResource> resources = {frame.uniforms, frame.shadows, frame.shadowMap,
                                                 frame.shadowSampler, frame.clampSampler};
    renderer_.MaterialOrDefault(draw.material).AppendResources(resources);
    const std::array<const GpuBuffer*, 10> buffers{
        points_.get(),       normals_.get(), indices_.get(),
        draws_.get(),        colors_.get(),  positions_.get(),
        orientations_.get(), scales_.get(),  culled ? visibleIds_.get() : instanceIds_.get(),
        drawRecords_.get()};
    for (const GpuBuffer* buffer : buffers) {
        resources.push_back(DescriptorResource::Of(*buffer));
    }
    return resources;
}

std::optional<Bounds3> MeshDrawable::RangeBounds(std::size_t item, const WorldRange& range) const {
    const MeshItem& draw = items_[item];
    const Mesh& mesh = meshes_.at(draw.mesh);
    if (mesh.data.points.empty() && mesh.devicePoints.Empty()) {
        return std::nullopt;
    }
    // The draw's box where it is and, moving, where it was at the last frame: sub-frames of a
    // motion-blurred frame see it anywhere between.
    const auto transformBox = [&](const glm::vec3& low, const glm::vec3& high) {
        Bounds3 box = TransformBox(transforms_[item], low, high);
        if (previousTransforms_[item] != transforms_[item]) {
            const Bounds3 was = TransformBox(previousTransforms_[item], low, high);
            box = {glm::min(box.first, was.first), glm::max(box.second, was.second)};
        }
        return box;
    };
    if (!draw.instancer) {
        return transformBox(mesh.low, mesh.high);
    }
    const Instancer& instancer = instancers_.at(*draw.instancer);
    if (instancer.onDevice && instancer.device.extent) {
        return transformBox(instancer.device.extent->first, instancer.device.extent->second);
    }
    const InstanceData& data = instancer.data;
    if (data.positions.size() != 3 * instancer.count) {
        return std::nullopt;  // nothing on the host to bound them with: always drawn
    }
    // The range's instances, each a sphere around its position enclosing the posed prototype.
    const auto [low, high] = TransformBox(draw.prototype, mesh.low, mesh.high);
    const float radius = glm::length(glm::max(glm::abs(low), glm::abs(high)));
    glm::vec3 boxLow(std::numeric_limits<float>::max());
    glm::vec3 boxHigh(std::numeric_limits<float>::lowest());
    for (std::uint32_t k = range.first; k < range.first + range.count; ++k) {
        const std::uint32_t i = draw.instances[k];
        float scale = 1.0f;
        if (!data.scales.empty()) {
            scale = std::max({std::abs(data.scales[3 * i]), std::abs(data.scales[3 * i + 1]),
                              std::abs(data.scales[3 * i + 2])});
        }
        const glm::vec3 p(data.positions[3 * i], data.positions[3 * i + 1],
                          data.positions[3 * i + 2]);
        boxLow = glm::min(boxLow, p - glm::vec3(radius * scale));
        boxHigh = glm::max(boxHigh, p + glm::vec3(radius * scale));
    }
    return transformBox(boxLow, boxHigh);
}

void MeshDrawable::Cull(const FrameParams& params, std::span<const ShadowCascade> cascades) {
    ERNIE_ZONE("Mesh cull lists");
    culling_ = params.culling;
    posesOnGpu_ = params.posesOnGpu;
    occlusion_ = culling_ == Culling::Gpu ? params.occlusion : Occlusion::Off;
    // Each draw range's world bounds, then the ranges each view and shadow view can see: all of
    // them without culling, else those whose bounds meet the frustum. With GPU culling, every
    // instanced range left becomes a job that tests its instances one by one (cull.slang).
    std::vector<std::vector<std::optional<Bounds3>>> bounds(items_.size());
    for (std::size_t i = 0; i < items_.size(); ++i) {
        for (const WorldRange& range : itemRanges_[i]) {
            bounds[i].push_back(range.count == 0 || culling_ == Culling::Off
                                    ? std::nullopt
                                    : RangeBounds(i, range));
        }
    }
    std::vector<GpuCullJob> jobs;
    std::vector<glm::vec4> views;
    std::uint32_t visible = 0;
    std::uint32_t draws = 0;
    std::vector<glm::uvec4> records;  // per indirect draw: draw, instance offset, index offset
    cullDrawCounts_.clear();
    viewJobStarts_.clear();
    // Each level's draw of a run of ids starting at `ids`: the draw, its ids' offset past the
    // draw's own and its indices.
    const auto addRecords = [&](std::uint32_t item, std::uint32_t ids, std::uint32_t count) {
        const std::vector<MeshLevel>& levels = meshes_.at(items_[item].mesh).levels;
        for (std::uint32_t l = 0; l < levels.size(); ++l) {
            records.emplace_back(item, ids + l * count - itemIdOffsets_[item], levels[l].first, 0u);
        }
    };
    // The depth pyramid views draw against (`RenderTargets::Hzb`): level 0 the largest
    // power-of-two size within the image.
    const auto pyramidWidth = static_cast<float>(std::bit_floor(params.camera.width));
    const auto pyramidHeight = static_cast<float>(std::bit_floor(params.camera.height));
    const auto pyramidMips = static_cast<float>(std::bit_width(
        std::max(std::bit_floor(params.camera.width), std::bit_floor(params.camera.height))));
    // The draw ranges a view or shadow view may see, in item order: every one, or for one world
    // its own and (unless it skips them) the shared ones, so a view of one of many worlds looks
    // at its world's ranges only.
    using RangeIndex = std::pair<std::size_t, std::size_t>;
    std::vector<RangeIndex> everyRange;
    std::unordered_map<std::int32_t, std::vector<RangeIndex>> rangesOfWorld;
    for (std::size_t i = 0; i < items_.size(); ++i) {
        for (std::size_t r = 0; r < itemRanges_[i].size(); ++r) {
            everyRange.emplace_back(i, r);
            rangesOfWorld[itemRanges_[i][r].world].emplace_back(i, r);
        }
    }
    std::map<std::pair<std::int32_t, bool>, std::vector<RangeIndex>> seenRanges;
    const std::vector<RangeIndex> none;
    const auto candidates = [&](const ShadowCasters* seen) -> const std::vector<RangeIndex>& {
        if (seen == nullptr || seen->world == ShadowCasters::kAllWorlds) {
            return everyRange;
        }
        const std::pair key{seen->world, seen->shared};
        auto [it, added] = seenRanges.try_emplace(key);
        if (added) {
            const auto shared = rangesOfWorld.find(-1);
            const auto own = rangesOfWorld.find(seen->world);
            const std::vector<RangeIndex>& a =
                seen->shared && shared != rangesOfWorld.end() ? shared->second : none;
            const std::vector<RangeIndex>& b =
                seen->world >= 0 && own != rangesOfWorld.end() ? own->second : none;
            std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(it->second));
        }
        return it->second;
    };
    // `pixelsPerUnit`: how many pixels (or shadow map texels) a world unit spans at w = 1.
    // Without `posed`, the view's pose is only an estimate (`FrameParams::posesOnGpu`): it tests
    // no bounds, and its cull view's planes and matrix come from the GPU (`RecordViewPoses`).
    const auto list = [&](const glm::mat4& viewProjection, float pixelsPerUnit, bool pyramid,
                          const ShadowCasters* seen, bool posed) {
        const Frustum frustum = Frustum::FromMatrix(viewProjection);
        const auto slot = static_cast<std::uint32_t>(views.size() / kCullViewStride);
        for (const glm::vec4& plane : frustum.planes) {
            views.push_back(plane / std::max(glm::length(glm::vec3(plane)), 1e-30f));
        }
        for (int column = 0; column < 4; ++column) {
            views.push_back(viewProjection[column]);
        }
        views.emplace_back(pixelsPerUnit, params.meshLodError, 0.0f, 0.0f);
        views.emplace_back(pyramid ? pyramidWidth : 0.0f, pyramid ? pyramidHeight : 0.0f,
                           pyramid ? pyramidMips : 0.0f, 0.0f);
        std::vector<VisibleRange> ranges;
        for (const auto [i, r] : candidates(seen)) {
            const WorldRange& range = itemRanges_[i][r];
            if (range.count == 0 || (seen != nullptr && !seen->Draws(range.world))) {
                continue;
            }
            const std::optional<Bounds3>& box = bounds[i][r];
            if (posed && box && !frustum.Intersects(box->first, box->second)) {
                continue;
            }
            VisibleRange entry{static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(r)};
            if (culling_ == Culling::Gpu && items_[i].instancer) {
                entry.job = static_cast<std::int32_t>(jobs.size());
                GpuCullJob job{};
                job.range[0] = entry.item;
                job.range[1] = itemIdOffsets_[i] + range.first;
                job.range[2] = range.count;
                job.range[3] = slot;
                // A run of the range's count of ids and an indirect draw per level.
                const auto levels =
                    static_cast<std::uint32_t>(meshes_.at(items_[i].mesh).levels.size());
                job.output[0] = visible;
                job.output[1] = draws;
                jobs.push_back(job);
                addRecords(entry.item, visible, range.count);
                cullDrawCounts_.push_back(levels);
                visible += range.count * levels;
                draws += levels;
            }
            ranges.push_back(entry);
        }
        return ranges;
    };
    viewDraws_.clear();
    for (std::size_t v = 0; v < params.Views(); ++v) {
        viewJobStarts_.push_back(static_cast<std::uint32_t>(jobs.size()));
        // A view sees what every world shares and its own world, or every world for -1.
        const ShadowCasters seen{true, params.ViewWorld(v)};
        const glm::mat4 viewProjection = params.View(v).ViewProjection();
        // A unit at w = 1 spans the projection's y scale (the length of the matrix's y row) over
        // half the image's height.
        const float perUnit = glm::length(glm::vec3(viewProjection[0][1], viewProjection[1][1],
                                                    viewProjection[2][1])) *
                              0.5f * static_cast<float>(params.camera.height);
        viewDraws_.push_back(
            list(viewProjection, perUnit, occlusion_ != Occlusion::Off, &seen, !params.posesOnGpu));
    }
    viewJobStarts_.push_back(static_cast<std::uint32_t>(jobs.size()));
    // Two-pass occlusion culling draws each view job twice; its second draws follow all the
    // first ones, in the same order, so they batch as the first ones do.
    if (occlusion_ == Occlusion::TwoPass) {
        for (std::uint32_t j = 0; j < viewJobStarts_.back(); ++j) {
            GpuCullJob& job = jobs[j];
            job.output[2] = visible;
            job.output[3] = draws;
            addRecords(job.range[0], visible, job.range[2]);
            visible += job.range[2] * cullDrawCounts_[j];
            draws += cullDrawCounts_[j];
        }
    }
    shadowDraws_.clear();
    for (const ShadowCascade& cascade : cascades) {
        // Shadow maps: one texel spans `texelWorldSize` (at w = 1 for perspective views).
        shadowDraws_.push_back(list(cascade.lightViewProjection,
                                    1.0f / std::max(cascade.texelWorldSize, 1e-30f), false, nullptr,
                                    true));
    }
    // A world's shadow pass draws its world's casters and the shared ones, not every world's.
    shadowWorldDraws_.assign(shadowDraws_.size(), {});
    for (std::size_t c = 0; c < shadowDraws_.size(); ++c) {
        for (const VisibleRange& visible : shadowDraws_[c]) {
            shadowWorldDraws_[c][itemRanges_[visible.item][visible.range].world].push_back(visible);
        }
    }
    WriteCullJobs(jobs, views, visible, draws);
    if (!records.empty()) {
        Reserve(context_, drawRecords_, records.size() * sizeof(glm::uvec4),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
        renderer_.Uploads().Write(*drawRecords_, std::as_bytes(std::span(records)));
    }
    // One visibility bit per (view, instance id) for two-pass occlusion culling, forgotten when
    // the views or the instances change.
    std::uint32_t instanceIds = 0;
    for (const auto& [handle, instancer] : instancers_) {
        instanceIds += static_cast<std::uint32_t>(instancer.count);
    }
    if (occlusion_ == Occlusion::TwoPass) {
        const std::pair key{static_cast<std::uint32_t>(params.Views()), instanceIds};
        if (key != visibilityKey_) {
            visibilityKey_ = key;
            const VkDeviceSize words = (VkDeviceSize{key.first} * key.second + 31) / 32;
            Reserve(context_, visibility_, words * sizeof(std::uint32_t),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
            resetVisibility_ = true;
        }
    }
    visibilityStride_ = instanceIds;
}

void MeshDrawable::WriteCullJobs(const std::vector<GpuCullJob>& jobs,
                                 const std::vector<glm::vec4>& views, std::uint32_t visible,
                                 std::uint32_t draws) {
    ERNIE_ZONE("Mesh write cull jobs");
    cullJobs_.assign(jobs.begin(), jobs.end());
    if (jobs.empty()) {
        return;
    }
    std::vector<GpuCullItem> items(items_.size());
    // Each instanced mesh's levels, once: first index (relative), count, error bits, unused.
    std::vector<glm::uvec4> levels;
    std::map<std::uint32_t, std::uint32_t> levelsOf;
    for (std::size_t i = 0; i < items_.size(); ++i) {
        const MeshItem& item = items_[i];
        if (!item.instancer) {
            continue;
        }
        const Mesh& mesh = meshes_.at(item.mesh);
        const auto [first, added] =
            levelsOf.try_emplace(item.mesh, static_cast<std::uint32_t>(levels.size()));
        if (added) {
            for (const MeshLevel& level : mesh.levels) {
                std::uint32_t errorBits = 0;
                std::memcpy(&errorBits, &level.error, sizeof(errorBits));
                levels.emplace_back(level.first, level.count, errorBits, 0u);
            }
        }
        const auto [low, high] = TransformBox(item.prototype, mesh.low, mesh.high);
        const glm::vec3 centre = 0.5f * (low + high);
        GpuCullItem& entry = items[i];
        std::memcpy(entry.model, glm::value_ptr(transforms_[i]), sizeof(entry.model));
        entry.sphere[0] = centre.x;
        entry.sphere[1] = centre.y;
        entry.sphere[2] = centre.z;
        entry.sphere[3] = 0.5f * glm::length(high - low);
        entry.info[0] = MaxScale(transforms_[i]);
        entry.info[2] = MaxScale(item.prototype);
        entry.levels[0] = first->second;
        entry.levels[1] = static_cast<std::uint32_t>(mesh.levels.size());
        const Instancer& instancer = instancers_.at(*item.instancer);
        std::uint32_t flags = 0;
        if (instancer.onDevice ? !instancer.device.orientations.Empty()
                               : !instancer.data.orientations.empty()) {
            flags |= kCullOrientations;
        }
        if (instancer.onDevice ? !instancer.device.scales.Empty()
                               : !instancer.data.scales.empty()) {
            flags |= kCullScales;
        }
        std::memcpy(&entry.info[1], &flags, sizeof(flags));
    }
    const auto upload = [&](std::unique_ptr<GpuBuffer>& buffer, std::span<const std::byte> bytes) {
        Reserve(context_, buffer, bytes.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Device);
        renderer_.Uploads().Write(*buffer, bytes);
    };
    upload(cullItemBuffer_, std::as_bytes(std::span(items)));
    upload(cullJobBuffer_, std::as_bytes(std::span(jobs)));
    upload(cullViewBuffer_, std::as_bytes(std::span(views)));
    upload(cullLevelBuffer_, std::as_bytes(std::span(levels)));
    Reserve(context_, visibleIds_, VkDeviceSize{visible} * sizeof(std::uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    Reserve(context_, cullCommands_, VkDeviceSize{draws} * sizeof(VkDrawIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            BufferMemory::Device);
}

void MeshDrawable::Dispatch(VkCommandBuffer commands, VkImageView hzb, std::uint32_t firstJob,
                            std::uint32_t jobs, std::uint32_t mode, std::uint32_t firstView) const {
    if (jobs == 0) {
        return;
    }
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, cullPipeline_->Handle());
    const DescriptorResource resources[] = {
        DescriptorResource::Of(*cullItemBuffer_),
        DescriptorResource::Of(*cullJobBuffer_),
        DescriptorResource::Of(*cullViewBuffer_),
        DescriptorResource::Of(*instanceIds_),
        DescriptorResource::Of(*positions_),
        DescriptorResource::Of(*orientations_),
        DescriptorResource::Of(*scales_),
        DescriptorResource::Of(*visibleIds_),
        DescriptorResource::Of(*cullCommands_),
        DescriptorResource::Of(*cullLevelBuffer_),
        DescriptorResource::Of(hzb, VK_IMAGE_LAYOUT_GENERAL),
        DescriptorResource::Of(visibility_ != nullptr ? *visibility_ : *cullLevelBuffer_)};
    cullLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE, resources);
    // One workgroup per job, spread over y beyond the x limit every device supports.
    const std::uint32_t groupsX = std::min<std::uint32_t>(jobs, 65535);
    const std::uint32_t groupsY = (jobs + groupsX - 1) / groupsX;
    const std::uint32_t constants[8] = {jobs, groupsX,   firstJob,          0,
                                        mode, firstView, visibilityStride_, 0};
    cullLayout_.PushConstants(commands, constants, sizeof(constants));
    vkCmdDispatch(commands, groupsX, groupsY, 1);
}

void MeshDrawable::RecordViewPoses(VkCommandBuffer commands) const {
    if (!posesOnGpu_ || cullJobs_.empty()) {
        return;
    }
    // Views take the first cull view slots (`Cull`): their planes, then their matrix's columns.
    RasterRenderer::ViewFields fields;
    fields.planes = 0;
    fields.viewProjection = 6 * sizeof(glm::vec4);
    renderer_.PatchViews(commands, *cullViewBuffer_, 0, kCullViewStride * sizeof(glm::vec4),
                         fields);
}

void MeshDrawable::RecordCull(VkCommandBuffer commands) const {
    if (cullJobs_.empty()) {
        return;
    }
    if (resetVisibility_) {
        vkCmdFillBuffer(commands, visibility_->Handle(), 0, VK_WHOLE_SIZE, 0);
        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commands, &dependency);
        resetVisibility_ = false;
    }
    // Views' jobs come first, then shadow views'. Shadow views and views without occlusion
    // culling cull their frusta now; with occlusion culling views wait for their depth pyramids
    // (`RecordOcclusionCull`), but two-pass first draws what they saw last frame.
    const std::uint32_t viewJobs = viewJobStarts_.back();
    const auto total = static_cast<std::uint32_t>(cullJobs_.size());
    const VkImageView none = noHzb_->View();
    if (occlusion_ == Occlusion::Off) {
        Dispatch(commands, none, 0, total, kCullFrustum, 0);
        return;
    }
    if (occlusion_ == Occlusion::TwoPass) {
        Dispatch(commands, none, 0, viewJobs, kCullTwoPassFirst, 0);
    }
    Dispatch(commands, none, viewJobs, total - viewJobs, kCullFrustum, 0);
}

void MeshDrawable::RecordOcclusionCull(VkCommandBuffer commands, const GpuImage& hzb,
                                       std::uint32_t firstView, std::uint32_t views) const {
    if (occlusion_ == Occlusion::Off || firstView + views >= viewJobStarts_.size()) {
        return;
    }
    const std::uint32_t first = viewJobStarts_[firstView];
    const std::uint32_t last = viewJobStarts_[firstView + views];
    Dispatch(commands, hzb.View(), first, last - first,
             occlusion_ == Occlusion::Occluders ? kCullOccluders : kCullTwoPassSecond, firstView);
}

std::vector<std::vector<std::uint32_t>> MeshDrawable::CulledLevelCounts() const {
    std::vector<std::vector<std::uint32_t>> counts;
    if (cullJobs_.empty()) {
        return counts;
    }
    std::vector<VkDrawIndirectCommand> drawn(cullCommands_->Size() / sizeof(VkDrawIndirectCommand));
    cullCommands_->Read(std::as_writable_bytes(std::span(drawn)));
    const std::uint32_t viewJobs = viewJobStarts_.empty() ? 0 : viewJobStarts_.back();
    for (std::size_t j = 0; j < cullJobs_.size(); ++j) {
        std::vector<std::uint32_t>& job = counts.emplace_back();
        const bool second = occlusion_ == Occlusion::TwoPass && j < viewJobs;
        for (std::uint32_t l = 0; l < cullDrawCounts_[j]; ++l) {
            // Two-pass draws a view's instances in either pass: count both.
            job.push_back(drawn[cullJobs_[j].output[1] + l].instanceCount +
                          (second ? drawn[cullJobs_[j].output[3] + l].instanceCount : 0u));
        }
    }
    return counts;
}

std::vector<std::uint32_t> MeshDrawable::CulledInstanceCounts() const {
    std::vector<std::uint32_t> counts;
    for (const std::vector<std::uint32_t>& levels : CulledLevelCounts()) {
        std::uint32_t sum = 0;
        for (const std::uint32_t count : levels) {
            sum += count;
        }
        counts.push_back(sum);
    }
    return counts;
}

std::size_t MeshDrawable::LevelCount(std::uint32_t mesh) const {
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " + std::to_string(mesh));
    }
    return it->second.levels.size();
}

std::size_t MeshDrawable::DrawCount(std::size_t view) const {
    return view < viewDraws_.size() ? viewDraws_[view].size() : 0;
}

std::size_t MeshDrawable::Record(VkCommandBuffer commands, const RasterFrame& frame,
                                 std::span<const VisibleRange> ranges, const glm::mat4& light,
                                 const ShadowCasters* casters, bool shadow, bool direct,
                                 int culledSet) const {
    std::size_t calls = 0;
    const GraphicsPipeline* bound = nullptr;
    std::uint32_t pushed = std::numeric_limits<std::uint32_t>::max();
    bool pushedCulled = false;
    const auto bind = [&](std::uint32_t item, bool culled) {
        const GraphicsPipeline* pipeline = shadow ? shadowPipeline_.get() : itemPipelines_[item];
        if (pipeline != bound) {
            bound = pipeline;
            vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, bound->Handle());
        }
        if (item != pushed || culled != pushedCulled) {
            pushed = item;
            pushedCulled = culled;
            layout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, Resources(frame, item, culled));
        }
    };
    DrawConstants constants{};
    std::memcpy(constants.lightViewProj, glm::value_ptr(light), sizeof(constants.lightViewProj));

    // GPU-culled ranges draw indirectly, each level a command with a per-draw record (mesh.slang
    // reads it by draw index). Consecutive commands that bind the same resources, which shadow
    // passes always do and colour passes do within a material, go out in one multi-draw call.
    struct Batch {
        std::uint32_t item = 0;  ///< The draw whose resources the batch binds.
        std::uint32_t first = 0;
        std::uint32_t count = 0;
    };
    std::optional<Batch> batch;
    const auto materialOf = [&](std::uint32_t item) {
        const std::uint32_t material = items_[item].material;
        return renderer_.HasMaterial(material) ? material : RasterRenderer::kDefaultMaterial;
    };
    const auto flush = [&] {
        if (!batch) {
            return;
        }
        bind(batch->item, true);
        const auto stride = static_cast<std::uint32_t>(sizeof(VkDrawIndirectCommand));
        if (context_.SupportsMultiDrawIndirect()) {
            constants.records = batch->first + 1;
            layout_.PushConstants(commands, &constants, sizeof(constants));
            vkCmdDrawIndirect(commands, cullCommands_->Handle(),
                              VkDeviceSize{batch->first} * stride, batch->count, stride);
            ++calls;
        } else {
            for (std::uint32_t k = 0; k < batch->count; ++k) {
                constants.records = batch->first + k + 1;
                layout_.PushConstants(commands, &constants, sizeof(constants));
                vkCmdDrawIndirect(commands, cullCommands_->Handle(),
                                  VkDeviceSize{batch->first + k} * stride, 1, stride);
                ++calls;
            }
        }
        constants.records = 0;
        batch.reset();
    };

    for (const VisibleRange& visible : ranges) {
        const WorldRange& range = itemRanges_[visible.item][visible.range];
        if (casters != nullptr && !casters->Draws(range.world)) {
            continue;
        }
        const MeshItem& draw = items_[visible.item];
        const auto vertices = static_cast<std::uint32_t>(meshes_.at(draw.mesh).data.indices.size());
        if (vertices == 0) {
            continue;
        }
        if (visible.job >= 0) {
            if (culledSet < 0) {
                continue;
            }
            const GpuCullJob& job = cullJobs_[static_cast<std::size_t>(visible.job)];
            const std::uint32_t levels = cullDrawCounts_[static_cast<std::size_t>(visible.job)];
            const std::uint32_t firstDraw = culledSet == 0 ? job.output[1] : job.output[3];
            const bool joins =
                batch && batch->first + batch->count == firstDraw &&
                (shadow || (materialOf(batch->item) == materialOf(visible.item) &&
                            itemPipelines_[batch->item] == itemPipelines_[visible.item]));
            if (!joins) {
                flush();
                batch = Batch{visible.item, firstDraw, 0};
            }
            batch->count += levels;
            continue;
        }
        if (!direct) {
            continue;
        }
        flush();
        bind(visible.item, false);
        constants.first = visible.item;
        constants.instanceOffset = range.first;
        constants.indexOffset = 0;
        layout_.PushConstants(commands, &constants, sizeof(constants));
        vkCmdDraw(commands, vertices, range.count, 0, 0);
        ++calls;
    }
    flush();
    return calls;
}

void MeshDrawable::RecordShadow(VkCommandBuffer commands, const RasterFrame& frame,
                                std::size_t cascade, const glm::mat4& lightViewProjection,
                                const ShadowCasters& casters) const {
    if (cascade >= shadowDraws_.size()) {
        return;
    }
    if (casters.world == ShadowCasters::kAllWorlds) {
        Record(commands, frame, shadowDraws_[cascade], lightViewProjection, &casters, true, true,
               0);
        return;
    }
    const auto& byWorld = shadowWorldDraws_[cascade];
    for (const std::int32_t world : {std::int32_t{-1}, casters.world}) {
        const auto draws = byWorld.find(world);
        if (draws != byWorld.end() && casters.Draws(world)) {
            Record(commands, frame, draws->second, lightViewProjection, &casters, true, true, 0);
        }
        if (casters.world < 0) {
            break;  // the shared casters only
        }
    }
}

void MeshDrawable::RecordColor(VkCommandBuffer commands, const RasterFrame& frame, std::size_t view,
                               DrawPhase phase) const {
    if (view >= viewDraws_.size()) {
        return;
    }
    // Which draws each phase records: meshes drawn directly occlude; GPU-culled instances draw
    // with their (first) draws, or with occlusion culling after the depth pyramid, two-pass
    // drawing what was visible last frame first and the rest after.
    bool direct = true;
    int culledSet = 0;
    if (phase == DrawPhase::Occluders) {
        culledSet = occlusion_ == Occlusion::TwoPass ? 0 : -1;
    } else if (phase == DrawPhase::Occludees) {
        direct = false;
        culledSet = occlusion_ == Occlusion::TwoPass ? 1 : 0;
    }
    const std::size_t calls = Record(commands, frame, viewDraws_[view], glm::mat4(1.0f), nullptr,
                                     false, direct, culledSet);
    viewCalls_.resize(viewDraws_.size());
    viewCalls_[view] = (phase == DrawPhase::Occludees ? viewCalls_[view] : 0) + calls;
}

std::size_t MeshDrawable::DrawCalls(std::size_t view) const {
    return view < viewCalls_.size() ? viewCalls_[view] : 0;
}

}  // namespace ernie
