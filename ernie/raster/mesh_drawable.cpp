// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/mesh_drawable.h"

#include "ernie/device/gpu_profile.h"
#include "ernie/geometry/frustum.h"
#include "ernie/kernel/embedded_shaders.h"
#include "ernie/raster/minimal_targets.h"
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
#include <unordered_set>
#include <utility>

namespace ernie {

namespace {

constexpr std::uint32_t kHasNormals = 1;
constexpr std::uint32_t kHasColors = 2;
constexpr std::uint32_t kInstanced = 4;
constexpr std::uint32_t kHasOrientations = 8;
constexpr std::uint32_t kHasScales = 16;
constexpr std::uint32_t kHasUvs = 32;
constexpr std::uint32_t kHasInstanceColors = 64;
/// The frame has thermal emission on (`MeshDrawable::thermal_`).
constexpr std::uint32_t kThermal = 128;
/// The mesh has temperatures per point (`SetTemperatures`).
constexpr std::uint32_t kHasTemperatures = 256;
/// The mesh has a thermal network's nodes and their temperatures
/// (`SetNodeIds`, `SetNodeTemperatures`).
constexpr std::uint32_t kHasNodes = 512;
constexpr std::uint32_t kHasTangents = 1024;
/// Instanced: temperatures per instance in place of the draw's own,
/// and whether each instance is drawn.
constexpr std::uint32_t kHasInstanceTemperatures = 2048;
constexpr std::uint32_t kHasInstanceVisibility = 4096;
/// Normals per triangle corner (face-varying or uniform), in place of
/// per point.
constexpr std::uint32_t kHasCornerNormals = 8192;
/// Colours per triangle corner (face-varying or uniform), in place of
/// per point.
constexpr std::uint32_t kHasCornerColors = 16384;
/// The draw's light link group (`LinkGroups`), in the flags' high
/// bits.
constexpr std::uint32_t kLinkGroupShift = 24;

/// One draw as `mesh.slang` reads it.
struct GpuDraw {
    float model[16];
    float prototype[16];
    float tint[4];
    std::uint32_t offsets[4];
    std::uint32_t points[4];
    float previous[16];  ///< `model` at the last frame, for motion
                         ///< blur.
    /// The instance arrays' and points' offsets in their
    /// double-buffered pools, now and at the last frame.
    std::uint32_t motion[4];
    std::uint32_t uv[4];  ///< The mesh's first corner in the texture
                          ///< coordinate pool, its slot of active
                          ///< counts plus 1 (0 for none) and the
                          ///< points it holds; unused.
    /// The mesh's first corner in the corner normal pool and in the
    /// corner colour pool; unused.
    std::uint32_t corners[4];
};
static_assert(sizeof(GpuDraw) == 288,
              "GpuDraw must match MeshDraw in mesh.slang");

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

/// Per instanced draw, what cull.slang needs to place its instances'
/// bounding spheres.
struct GpuCullItem {
    float model[16];
    float sphere[4];  ///< Prototype's bounding sphere after its own
                      ///< transform: centre, radius.
    /// Largest scale of `model`; flags (as bits): orientations 1,
    /// scales 2; largest scale of the prototype's transform.
    float info[4];
    std::uint32_t levels[4];  ///< First level of detail in the level
                              ///< table, level count.
    float previousModel[16];  ///< `model` at the last frame, with
                              ///< flag `kCullMoved`.
};
static_assert(sizeof(GpuCullItem) == 176,
              "GpuCullItem must match CullItem in cull.slang");

static_assert(sizeof(MeshDrawable::GpuCullJob) == 32,
              "GpuCullJob must match CullJob in cull.slang");

constexpr VkShaderStageFlags kCompute = VK_SHADER_STAGE_COMPUTE_BIT;

/// cull.slang's bindings: what it writes (visible ids, indirect
/// draws, visibility bits) and the depth pyramids; it reads the rest
/// by address (`CullConstants`).
constexpr DescriptorBinding kCullBindings[] = {
    {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {10, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
};

/// cull.slang's CullConstants: the dispatch's jobs and mode, then the
/// buffers it reads by address.
struct CullConstants {
    std::uint32_t size[4];
    std::uint32_t mode[4];
    VkDeviceAddress items;
    VkDeviceAddress jobs;
    VkDeviceAddress views;
    VkDeviceAddress instanceIds;
    VkDeviceAddress positions;
    VkDeviceAddress orientations;
    VkDeviceAddress scales;
    VkDeviceAddress lods;
    VkDeviceAddress activeCounts;
    VkDeviceAddress visible;  ///< Instances' visibility, if any.
};
static_assert(sizeof(CullConstants) == 112,
              "CullConstants must match cull.slang");

/// active_draws.slang's bindings: counts in use and the draws.
constexpr DescriptorBinding kActiveDrawsBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
};

/// skinning.slang's push constants, which reach every buffer by
/// address.
struct SkinConstants {
    std::uint32_t info[4];
    std::uint32_t samples[4];
    VkDeviceAddress rest = 0;
    VkDeviceAddress influenceStarts = 0;
    VkDeviceAddress influenceJoints = 0;
    VkDeviceAddress influenceWeights = 0;
    VkDeviceAddress shapeStarts = 0;
    VkDeviceAddress shapeOffsets = 0;
    VkDeviceAddress shapeIndices = 0;
    VkDeviceAddress shapeWeights = 0;
    VkDeviceAddress pose = 0;
    VkDeviceAddress output = 0;
};
static_assert(sizeof(SkinConstants) == 112,
              "SkinConstants must match skinning.slang");

struct InstanceBlendConstants {
    std::uint32_t info[4];
    std::uint32_t blend[4];
    VkDeviceAddress positions = 0;
    VkDeviceAddress orientations = 0;
    VkDeviceAddress scales = 0;
    VkDeviceAddress outPositions = 0;
    VkDeviceAddress outOrientations = 0;
    VkDeviceAddress outScales = 0;
};

struct RefineConstants {
    std::uint32_t info[4];
    VkDeviceAddress control = 0;
    VkDeviceAddress starts = 0;
    VkDeviceAddress indices = 0;
    VkDeviceAddress weights = 0;
    VkDeviceAddress output = 0;
};

/// mesh_normals.slang's bindings: points, indices, adjacency and the
/// normals it writes; the topology passes' fixed-point sums and the
/// points and triangles in use.
constexpr DescriptorBinding kNormalsBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
};

/// instance_compose.slang's bindings: raw arrays, ops, instance
/// lists, parents, and the pools.
constexpr DescriptorBinding kComposeBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
};

/// instance_bounds.slang's bindings: positions, scales, bounds and
/// the meshes' counts in use.
constexpr DescriptorBinding kBoundsBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
};

/// instance_bounds.slang's BoundsJob.
struct GpuBoundsJob {
    std::uint32_t first;
    std::uint32_t count;
    std::uint32_t flags;  ///< 1 scaled, 2 a mesh's points.
    std::uint32_t out;
    std::uint32_t countSlot;
    std::uint32_t unused[3];
};

/// instance_anchors.slang's bindings: records, the pools, and what is
/// written.
constexpr DescriptorBinding kAnchorBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
};

/// light_views.slang's `cullViewsMain` bindings: the fitted matrices,
/// the cull views' matrices and slots, and the cull views.
constexpr DescriptorBinding kLightCullBindings[] = {
    {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
};

/// The row width an instance op of `kind` takes, or both widths it
/// accepts (scale: 3 or 1).
std::pair<std::int64_t, std::int64_t> OpWidths(InstanceOpKind kind) {
    switch (kind) {
    case InstanceOpKind::Translate:
        return {3, 3};
    case InstanceOpKind::Scale:
        return {3, 1};
    case InstanceOpKind::Orient:
        return {4, 4};
    case InstanceOpKind::RotateX:
    case InstanceOpKind::RotateY:
    case InstanceOpKind::RotateZ:
        return {1, 1};
    case InstanceOpKind::Transform:
        return {16, 16};
    default:
        return {3, 3};  // Euler angles
    }
}

/// Per point of a mesh of `count` points, the triangles of `indices`
/// around it: offsets (count + 1) into the triangle indices that
/// Per point of a mesh, its smooth tangent along which its texture
/// coordinates' u grows (unnormalized): the sum over its triangles of
/// each one's unit dP/du weighted by its area, from `uvs` per corner
/// of `indices`. Zero where no triangle around a point has u varying.
///
/// Reference: E. Lengyel, "Computing Tangent Space Basis Vectors for
/// an Arbitrary Mesh", Terathon Software, 2001.
std::vector<float> TangentsOf(std::span<const float> points,
                              std::span<const std::uint32_t> indices,
                              std::span<const float> uvs) {
    std::vector<float> tangents(points.size(), 0.0f);
    const auto point = [&](std::uint32_t i) {
        return glm::vec3(points[3 * i], points[3 * i + 1],
                         points[3 * i + 2]);
    };
    for (std::size_t first = 0;
         first + 2 < indices.size() && 2 * first + 5 < uvs.size();
         first += 3) {
        const glm::vec3 p0 = point(indices[first]);
        const glm::vec3 e1 = point(indices[first + 1]) - p0;
        const glm::vec3 e2 = point(indices[first + 2]) - p0;
        const glm::vec2 uv0(uvs[2 * first], uvs[2 * first + 1]);
        const glm::vec2 d1 =
            glm::vec2(uvs[2 * first + 2], uvs[2 * first + 3]) - uv0;
        const glm::vec2 d2 =
            glm::vec2(uvs[2 * first + 4], uvs[2 * first + 5]) - uv0;
        const float det = d1.x * d2.y - d2.x * d1.y;
        const glm::vec3 dpdu =
            (e1 * d2.y - e2 * d1.y) / (det != 0.0f ? det : 1.0f);
        const float length = glm::length(dpdu);
        if (det == 0.0f || !(length > 0.0f)) {
            continue;
        }
        const glm::vec3 weighted =
            dpdu / length * glm::length(glm::cross(e1, e2));
        for (std::size_t c = 0; c < 3; ++c) {
            const std::size_t at =
                3 * std::size_t{indices[first + c]};
            if (at + 2 < tangents.size()) {
                tangents[at] += weighted.x;
                tangents[at + 1] += weighted.y;
                tangents[at + 2] += weighted.z;
            }
        }
    }
    return tangents;
}

/// follow them.
std::vector<std::uint32_t>
TrianglesAround(std::span<const std::uint32_t> indices,
                std::size_t count) {
    std::vector<std::uint32_t> offsets(count + 1, 0);
    for (const std::uint32_t index : indices) {
        ++offsets[index + 1];
    }
    for (std::size_t i = 0; i < count; ++i) {
        offsets[i + 1] += offsets[i];
    }
    std::vector<std::uint32_t> adjacency(offsets.begin(),
                                         offsets.end());
    adjacency.resize(count + 1 + indices.size());
    std::vector<std::uint32_t> next(offsets.begin(),
                                    offsets.end() - 1);
    for (std::size_t corner = 0; corner < indices.size(); ++corner) {
        adjacency[count + 1 + next[indices[corner]]++] =
            static_cast<std::uint32_t>(corner / 3);
    }
    return adjacency;
}

/// cull.slang's modes.
constexpr std::uint32_t kCullFrustum = 0;
constexpr std::uint32_t kCullOccluders = 1;
constexpr std::uint32_t kCullTwoPassFirst = 2;
constexpr std::uint32_t kCullTwoPassSecond = 3;

/// cull.slang's float4s per view: 6 planes, the view-projection's
/// columns, pixels per unit and the error, the depth pyramid.
constexpr std::size_t kCullViewStride = 12;

/// Largest length of the first three columns of `m`: how much it
/// scales lengths at most.
float MaxScale(const glm::mat4& m) {
    return std::max({glm::length(glm::vec3(m[0])),
                     glm::length(glm::vec3(m[1])),
                     glm::length(glm::vec3(m[2]))});
}

/// Bit flags of `GpuCullItem::info[1]`.
constexpr std::uint32_t kCullOrientations = 1;
constexpr std::uint32_t kCullScales = 2;
constexpr std::uint32_t kCullMoved =
    4;  ///< The instancer's transform moved since the last frame.
/// A draw without an instancer: its one instance is its transform's.
constexpr std::uint32_t kCullNoInstances = 8;
/// Its instancer hides some instances (`instanceVisible_`): never kept.
constexpr std::uint32_t kCullVisibility = 16;

/// Grows `buffer` to hold `bytes`, waiting for the device first if it
/// replaces one in use.
void Reserve(const VulkanContext& context,
             std::unique_ptr<GpuBuffer>& buffer, VkDeviceSize bytes,
             VkBufferUsageFlags usage, BufferMemory memory) {
    bytes = std::max<VkDeviceSize>(bytes, 16);
    if (buffer != nullptr && buffer->Size() >= bytes) {
        return;
    }
    if (buffer != nullptr) {
        vkDeviceWaitIdle(context.Device());
    }
    buffer =
        std::make_unique<GpuBuffer>(context, bytes, usage, memory);
}

/// Frame (0 and 1), material (5) and mesh (10, its buffers'
/// addresses; 22 to 26, its material's textures) bindings; see
/// mesh.slang.
constexpr DescriptorBinding kBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
};

/// The motion pass's: the mesh passes' and 20, the view's
/// MotionUniforms (frame.slang), last as the pass appends it.
constexpr DescriptorBinding kMotionBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {20, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
};

/// mesh.slang's MeshResources: the mesh passes' buffers' addresses.
struct MeshResources {
    VkDeviceAddress points;
    VkDeviceAddress normals;
    VkDeviceAddress indices;
    VkDeviceAddress draws;
    VkDeviceAddress colors;
    VkDeviceAddress positions;
    VkDeviceAddress orientations;
    VkDeviceAddress scales;
    /// Every instance's ids, or for culled draws those culling left.
    VkDeviceAddress instanceIds;
    VkDeviceAddress drawRecords;
    VkDeviceAddress uvs;
    VkDeviceAddress instanceColors;
    VkDeviceAddress activeCounts;
    /// Meshes' temperatures per point
    /// (`MeshDrawable::temperatures_`).
    VkDeviceAddress temperatures;
    /// Their thermal networks' nodes per point and the nodes'
    /// temperatures (`MeshDrawable::nodeIds_`, `nodeTemperatures_`).
    VkDeviceAddress nodeIds;
    VkDeviceAddress nodeTemperatures;
    /// Smooth tangents per point of the meshes an anisotropic
    /// material shades (`MeshDrawable::tangents_`).
    VkDeviceAddress tangents;
    /// Instances' temperatures and visibility
    /// (`MeshDrawable::instanceTemperatures_`, `instanceVisible_`).
    VkDeviceAddress instanceTemperatures;
    VkDeviceAddress instanceVisible;
    /// Normals per corner (`MeshDrawable::cornerNormals_`).
    VkDeviceAddress cornerNormals;
    /// Colours per corner (`MeshDrawable::cornerColors_`).
    VkDeviceAddress cornerColors;
};
static_assert(sizeof(MeshResources) == 168,
              "MeshResources must match mesh.slang");

/// The mesh passes, which write the renderer's G-buffer.
std::vector<ShaderModuleSource> MeshModules(bool pushbroom = false) {
    return {{"brdf", std::string(BrdfSlang())},
            ProjectionModule(pushbroom),
            {"bindless", shaders::kBindlessSlang},
            {"frame", shaders::kFrameSlang},
            {"gbuffer", shaders::kGBufferSlang},
            {"surface", shaders::kSurfaceSlang},
            {"mesh", shaders::kMeshSlang}};
}

/// A host-visible storage buffer holding `bytes`, at least 16 bytes
/// so it can always be bound.
std::unique_ptr<GpuBuffer>
StorageBuffer(const VulkanContext& context,
              std::span<const std::byte> bytes) {
    auto buffer = std::make_unique<GpuBuffer>(
        context, std::max<VkDeviceSize>(bytes.size(), 16),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Upload);
    if (!bytes.empty()) {
        buffer->Write(bytes);
    }
    return buffer;
}

template <typename T>
std::unique_ptr<GpuBuffer> StorageBuffer(const VulkanContext& context,
                                         const std::vector<T>& data) {
    return StorageBuffer(context, std::as_bytes(std::span(data)));
}

void CheckTriples(const std::vector<float>& values, std::size_t count,
                  const char* what) {
    if (!values.empty() && values.size() != 3 * count) {
        throw std::invalid_argument(std::string(what) +
                                    " must have 3 values per point");
    }
}

void CheckInstances(const InstanceData& data) {
    if (data.positions.size() % 3 != 0) {
        throw std::invalid_argument(
            "Instance positions must have 3 values per instance");
    }
    const std::size_t count = data.positions.size() / 3;
    if (!data.orientations.empty() &&
        data.orientations.size() != 4 * count) {
        throw std::invalid_argument(
            "Instance orientations must have 4 values per instance");
    }
    if (!data.scales.empty() && data.scales.size() != 3 * count) {
        throw std::invalid_argument(
            "Instance scales must have 3 values per instance");
    }
    if (!data.colors.empty() && data.colors.size() != 3 * count) {
        throw std::invalid_argument(
            "Instance colours must have 3 values per instance");
    }
    if (!data.temperatures.empty() && data.temperatures.size() != count) {
        throw std::invalid_argument(
            "Instance temperatures must have 1 value per instance");
    }
    if (!data.visible.empty() && data.visible.size() != count) {
        throw std::invalid_argument(
            "Instance visibility must have 1 value per instance");
    }
}

std::pair<glm::vec3, glm::vec3>
PointBounds(const std::vector<float>& points) {
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
std::pair<glm::vec3, glm::vec3>
TransformBox(const glm::mat4& transform, const glm::vec3& low,
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

/// What a draw takes of its item that may change without its layout:
/// its transforms (now and at the last frame), its prototype's, its
/// tint and its temperature (when drawn thermal).
void PoseDraw(GpuDraw& draw, const glm::mat4& model,
              const glm::mat4& previous, const glm::mat4& prototype,
              const glm::vec3& tint, float temperature) {
    std::memcpy(draw.model, glm::value_ptr(model), sizeof(draw.model));
    std::memcpy(draw.previous, glm::value_ptr(previous),
                sizeof(draw.previous));
    std::memcpy(draw.prototype, glm::value_ptr(prototype),
                sizeof(draw.prototype));
    draw.tint[0] = tint.r;
    draw.tint[1] = tint.g;
    draw.tint[2] = tint.b;
    draw.tint[3] = temperature;
}

/// The same of a cull item: its transforms, its prototype's box
/// [`low`, `high`] as a sphere, and whether it moved; its other flags
/// (`info[1]`) kept.
void PoseCullItem(GpuCullItem& entry, const glm::mat4& model,
                  const glm::mat4& previous, const glm::mat4& prototype,
                  const glm::vec3& low, const glm::vec3& high) {
    const auto [boxLow, boxHigh] = TransformBox(prototype, low, high);
    const glm::vec3 centre = 0.5f * (boxLow + boxHigh);
    std::memcpy(entry.model, glm::value_ptr(model), sizeof(entry.model));
    entry.sphere[0] = centre.x;
    entry.sphere[1] = centre.y;
    entry.sphere[2] = centre.z;
    entry.sphere[3] = 0.5f * glm::length(boxHigh - boxLow);
    entry.info[0] = MaxScale(model);
    entry.info[2] = MaxScale(prototype);
    std::uint32_t flags = 0;
    std::memcpy(&flags, &entry.info[1], sizeof(flags));
    flags &= ~kCullMoved;
    if (previous != model) {
        flags |= kCullMoved;
        std::memcpy(entry.previousModel, glm::value_ptr(previous),
                    sizeof(entry.previousModel));
    }
    std::memcpy(&entry.info[1], &flags, sizeof(flags));
}

/// Simplified levels of detail of `data`'s triangles after the full
/// mesh, each of about half the triangles of the one before and
/// simplified from the full mesh, appended to `indices`; none for
/// small meshes or once simplification stalls. Errors are in mesh
/// units.
///
/// References:
/// - M. Garland and P. S. Heckbert, "Surface Simplification Using
///   Quadric Error Metrics", SIGGRAPH '97, 209-216, 1997.
///   https://doi.org/10.1145/258734.258849
/// - A. Kapoulkine, "meshoptimizer", `meshopt_simplify`.
///   https://github.com/zeux/meshoptimizer
///
/// `locked` holds a `meshopt_SimplifyVertex_*` flag per point (or is
/// empty): points on texture coordinate seams stay, so the seams do.
template <typename Level>
std::vector<Level>
SimplifiedLevels(const MeshData& data, std::size_t maxLevels,
                 std::size_t minTriangles,
                 std::vector<std::uint32_t>& indices,
                 std::span<const unsigned char> locked = {}) {
    std::vector<Level> levels;
    const std::size_t count = data.indices.size();
    const std::size_t vertices = data.points.size() / 3;
    if (count / 3 < minTriangles || vertices == 0) {
        return levels;
    }
    const float scale = meshopt_simplifyScale(
        data.points.data(), vertices, 3 * sizeof(float));
    std::size_t previous = count;
    std::vector<std::uint32_t> simplified(count);
    while (levels.size() + 1 < maxLevels) {
        const std::size_t target = previous / 2 / 3 * 3;
        if (target < 3 * 8) {
            break;
        }
        float error = 0.0f;
        const std::size_t kept = meshopt_simplifyWithAttributes(
            simplified.data(), data.indices.data(), count,
            data.points.data(), vertices, 3 * sizeof(float), nullptr,
            0, nullptr, 0, locked.empty() ? nullptr : locked.data(),
            target, 1.0f, 0, &error);
        if (kept == 0 || kept > previous * 9 / 10) {
            break;  // simplification stalled: no level worth its
                    // memory
        }
        levels.push_back(
            {static_cast<std::uint32_t>(count + indices.size()),
             static_cast<std::uint32_t>(kept), error * scale});
        indices.insert(indices.end(), simplified.begin(),
                       simplified.begin() +
                           static_cast<std::ptrdiff_t>(kept));
        previous = kept;
    }
    return levels;
}

/// Per point of `data`, the distinct texture coordinates of the
/// triangle corners at it.
std::vector<std::vector<glm::vec2>> PointUvs(const MeshData& data) {
    std::vector<std::vector<glm::vec2>> uvs(data.points.size() / 3);
    for (std::size_t c = 0; c < data.indices.size(); ++c) {
        const glm::vec2 uv(data.uvs[2 * c], data.uvs[2 * c + 1]);
        std::vector<glm::vec2>& at = uvs[data.indices[c]];
        if (std::find(at.begin(), at.end(), uv) == at.end()) {
            at.push_back(uv);
        }
    }
    return uvs;
}

/// Texture coordinates for the corners of simplified triangles
/// `lodIndices`, whose corners are points of the full mesh: a point's
/// own, or on a seam (several), the one nearest the triangle's other
/// corners', so each triangle stays on its side of the seam.
std::vector<float>
LodUvs(const std::vector<std::vector<glm::vec2>>& pointUvs,
       std::span<const std::uint32_t> lodIndices) {
    std::vector<float> uvs;
    uvs.reserve(2 * lodIndices.size());
    for (std::size_t t = 0; t + 2 < lodIndices.size(); t += 3) {
        glm::vec2 reference(0.0f);
        float weight = 0.0f;
        for (std::size_t k = 0; k < 3; ++k) {
            const std::vector<glm::vec2>& at =
                pointUvs[lodIndices[t + k]];
            if (at.size() == 1) {
                reference += at[0];
                weight += 1.0f;
            }
        }
        if (weight == 0.0f) {
            reference = pointUvs[lodIndices[t]].empty()
                            ? glm::vec2(0.0f)
                            : pointUvs[lodIndices[t]][0];
        } else {
            reference /= weight;
        }
        for (std::size_t k = 0; k < 3; ++k) {
            const std::vector<glm::vec2>& at =
                pointUvs[lodIndices[t + k]];
            glm::vec2 best(0.0f);
            float distance = std::numeric_limits<float>::infinity();
            for (const glm::vec2& uv : at) {
                if (glm::distance(uv, reference) < distance) {
                    distance = glm::distance(uv, reference);
                    best = uv;
                }
            }
            uvs.insert(uvs.end(), {best.x, best.y});
        }
    }
    return uvs;
}

/// Per point of `data`, the distinct values of `corners` (three per
/// triangle corner: normals, colours) at it.
std::vector<std::vector<glm::vec3>>
PointCornerValues(const MeshData& data,
                  const std::vector<float>& corners) {
    std::vector<std::vector<glm::vec3>> values(data.points.size() /
                                               3);
    for (std::size_t c = 0; c < data.indices.size(); ++c) {
        const glm::vec3 v(corners[3 * c], corners[3 * c + 1],
                          corners[3 * c + 2]);
        std::vector<glm::vec3>& at = values[data.indices[c]];
        if (std::find(at.begin(), at.end(), v) == at.end()) {
            at.push_back(v);
        }
    }
    return values;
}

/// Colours for the corners of simplified triangles `lodIndices`,
/// whose corners are points of the full mesh: a point's own, or where
/// faces of several colours meet, the one nearest the triangle's
/// other corners', as `LodUvs` picks texture coordinates.
std::vector<float> LodCornerColors(
    const std::vector<std::vector<glm::vec3>>& pointColors,
    std::span<const std::uint32_t> lodIndices) {
    std::vector<float> colors;
    colors.reserve(3 * lodIndices.size());
    for (std::size_t t = 0; t + 2 < lodIndices.size(); t += 3) {
        glm::vec3 reference(0.0f);
        float weight = 0.0f;
        for (std::size_t k = 0; k < 3; ++k) {
            const std::vector<glm::vec3>& at =
                pointColors[lodIndices[t + k]];
            if (at.size() == 1) {
                reference += at[0];
                weight += 1.0f;
            }
        }
        if (weight > 0.0f) {
            reference /= weight;
        } else if (!pointColors[lodIndices[t]].empty()) {
            reference = pointColors[lodIndices[t]][0];
        }
        for (std::size_t k = 0; k < 3; ++k) {
            glm::vec3 best(0.0f);
            float distance = std::numeric_limits<float>::infinity();
            for (const glm::vec3& c :
                 pointColors[lodIndices[t + k]]) {
                if (glm::distance(c, reference) < distance) {
                    distance = glm::distance(c, reference);
                    best = c;
                }
            }
            colors.insert(colors.end(), {best.x, best.y, best.z});
        }
    }
    return colors;
}

/// Normals for the corners of simplified triangles `lodIndices`,
/// whose corners are `points` of the full mesh: a point's own, or on
/// a hard edge (several), the one nearest the simplified triangle's
/// own normal, so each triangle keeps its side of the edge.
std::vector<float> LodCornerNormals(
    const std::vector<std::vector<glm::vec3>>& pointNormals,
    std::span<const float> points,
    std::span<const std::uint32_t> lodIndices) {
    const auto point = [&](std::uint32_t i) {
        return glm::vec3(points[3 * i], points[3 * i + 1],
                         points[3 * i + 2]);
    };
    std::vector<float> normals;
    normals.reserve(3 * lodIndices.size());
    for (std::size_t t = 0; t + 2 < lodIndices.size(); t += 3) {
        const glm::vec3 a = point(lodIndices[t]);
        const glm::vec3 face =
            glm::cross(point(lodIndices[t + 1]) - a,
                       point(lodIndices[t + 2]) - a);
        for (std::size_t k = 0; k < 3; ++k) {
            const std::vector<glm::vec3>& at =
                pointNormals[lodIndices[t + k]];
            glm::vec3 best(0.0f);
            float nearest = -std::numeric_limits<float>::infinity();
            for (const glm::vec3& n : at) {
                const float length = glm::length(n);
                const float facing =
                    length > 0.0f ? glm::dot(n, face) / length : 0.0f;
                if (facing > nearest) {
                    nearest = facing;
                    best = n;
                }
            }
            normals.insert(normals.end(), {best.x, best.y, best.z});
        }
    }
    return normals;
}

}  // namespace

MeshDrawable::MeshDrawable(RasterRenderer& renderer)
    : renderer_(renderer), context_(renderer.Context()),
      layout_(renderer.Context(), kBindings, sizeof(DrawConstants)),
      motionLayout_(renderer.Context(), kMotionBindings,
                    sizeof(DrawConstants)),
      normalsLayout_(renderer.Context(), kNormalsBindings, 32),
      composeLayout_(renderer.Context(), kComposeBindings, 32),
      boundsLayout_(renderer.Context(), kBoundsBindings, 16),
      activeDrawsLayout_(renderer.Context(), kActiveDrawsBindings,
                         16),
      anchorLayout_(renderer.Context(), kAnchorBindings, 16),
      lightCullLayout_(renderer.Context(), kLightCullBindings, 64),
      skinLayout_(renderer.Context(), {}, sizeof(SkinConstants)),
      cullLayout_(renderer.Context(), kCullBindings,
                  sizeof(CullConstants)) {
    const std::vector<ShaderModuleSource> skinning{
        {"skinning", shaders::kSkinningSlang}};
    skinPipeline_ = std::make_unique<ComputePipeline>(
        context_, skinLayout_,
        renderer_.Compiler().CompileSpirv(skinning, "skinMain"));
    refinePipeline_ = std::make_unique<ComputePipeline>(
        context_, skinLayout_,
        renderer_.Compiler().CompileSpirv(skinning, "refineMain"));
    instanceBlendPipeline_ = std::make_unique<ComputePipeline>(
        context_, skinLayout_,
        renderer_.Compiler().CompileSpirv(skinning,
                                          "blendInstancesMain"));
    const std::vector<ShaderModuleSource> normals{
        {"mesh_normals", shaders::kMeshNormalsSlang}};
    normalsPipeline_ = std::make_unique<ComputePipeline>(
        context_, normalsLayout_,
        renderer_.Compiler().CompileSpirv(normals, "normalsMain"));
    constexpr std::array<const char*, 4> kTopologyPasses = {
        "topologyClearMain", "topologyLargestMain", "topologyAddMain",
        "topologyNormalizeMain"};
    for (std::size_t pass = 0; pass < kTopologyPasses.size();
         ++pass) {
        topologyPipelines_[pass] = std::make_unique<ComputePipeline>(
            context_, normalsLayout_,
            renderer_.Compiler().CompileSpirv(normals,
                                              kTopologyPasses[pass]));
    }
    const std::vector<ShaderModuleSource> compose{
        {"instance_compose", shaders::kInstanceComposeSlang}};
    composePipeline_ = std::make_unique<ComputePipeline>(
        context_, composeLayout_,
        renderer_.Compiler().CompileSpirv(compose, "composeMain"));
    const std::vector<ShaderModuleSource> bounds{
        {"instance_bounds", shaders::kInstanceBoundsSlang}};
    boundsClearPipeline_ = std::make_unique<ComputePipeline>(
        context_, boundsLayout_,
        renderer_.Compiler().CompileSpirv(bounds, "boundsClearMain"));
    boundsPipeline_ = std::make_unique<ComputePipeline>(
        context_, boundsLayout_,
        renderer_.Compiler().CompileSpirv(bounds, "boundsMain"));
    const std::vector<ShaderModuleSource> activeDraws{
        {"active_draws", shaders::kActiveDrawsSlang}};
    activeDrawsPipeline_ = std::make_unique<ComputePipeline>(
        context_, activeDrawsLayout_,
        renderer_.Compiler().CompileSpirv(activeDraws,
                                          "activeDrawsMain"));
    const std::vector<ShaderModuleSource> anchors{
        {"instance_anchors", shaders::kInstanceAnchorsSlang}};
    anchorPipeline_ = std::make_unique<ComputePipeline>(
        context_, anchorLayout_,
        renderer_.Compiler().CompileSpirv(anchors, "anchorsMain"));
    const std::vector<ShaderModuleSource> lightViews{
        {"light_views", shaders::kLightViewsSlang}};
    lightCullPipeline_ = std::make_unique<ComputePipeline>(
        context_, lightCullLayout_,
        renderer_.Compiler().CompileSpirv(lightViews,
                                          "cullViewsMain"));
    // Stands in for a depth pyramid in culling without one.
    noHzb_ = std::make_unique<GpuImage>(
        context_, ImageDesc{1, 1, 1, 1, VK_FORMAT_R32_SFLOAT,
                            VK_IMAGE_USAGE_SAMPLED_BIT, false, true});
    context_.Submit([this](VkCommandBuffer commands) {
        noHzb_->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    });
    const std::vector<ShaderModuleSource> cull{
        ProjectionModule(true), {"cull", shaders::kCullSlang}};
    cullPipeline_ = std::make_unique<ComputePipeline>(
        context_, cullLayout_,
        renderer_.Compiler().CompileSpirv(cull, "cullMain"));
    CreateGBufferPipeline(false);
    const std::vector<std::uint32_t> vertex =
        renderer_.Compiler().CompileSpirv(MeshModules(),
                                          "meshShadowVertex");
    GraphicsPipelineDesc shadow;
    shadow.vertexSpirv = vertex;
    shadow.depthFormat = VK_FORMAT_D32_SFLOAT;
    shadow.depthBias = true;
    shadow.depthClamp = true;
    shadowPipeline_ =
        std::make_unique<GraphicsPipeline>(context_, layout_, shadow);
    RebuildGeometry();
    RebuildInstances();
    draws_ = StorageBuffer(context_, std::span<const std::byte>());
    instanceIds_ =
        StorageBuffer(context_, std::span<const std::byte>());
    drawRecords_ =
        StorageBuffer(context_, std::span<const std::byte>());
    activeCounts_ =
        StorageBuffer(context_, std::span<const std::byte>());
}

MeshDrawable::~MeshDrawable() {
    vkDeviceWaitIdle(context_.Device());
}

void MeshDrawable::CreateGBufferPipeline(bool pushbroom) {
    const std::vector<ShaderModuleSource> modules =
        MeshModules(pushbroom);
    const std::vector<std::uint32_t> vertex =
        renderer_.Compiler().CompileSpirv(modules, "meshVertexMain");
    const std::vector<std::uint32_t> fragment =
        renderer_.Compiler().CompileSpirv(modules,
                                          "meshFragmentMain");
    GraphicsPipelineDesc desc;
    desc.vertexSpirv = vertex;
    desc.fragmentSpirv = fragment;
    desc.colorFormats.assign(RenderTargets::kGBufferFormats.begin(),
                             RenderTargets::kGBufferFormats.end());
    desc.depthFormat = RenderTargets::kDepthFormat;
    desc.depthCompare = RenderTargets::kDepthCompare;
    gbufferPipelines_[pushbroom ? 1 : 0] =
        std::make_unique<GraphicsPipeline>(context_, layout_, desc);
}

void MeshDrawable::CreateCutoutPipelines(bool pushbroom) {
    std::unique_ptr<GraphicsPipeline>& gbuffer =
        cutoutPipelines_[pushbroom ? 1 : 0];
    if (gbuffer == nullptr) {
        const std::vector<ShaderModuleSource> modules =
            MeshModules(pushbroom);
        const std::vector<std::uint32_t> vertex =
            renderer_.Compiler().CompileSpirv(modules,
                                              "meshVertexMain");
        const std::vector<std::uint32_t> fragment =
            renderer_.Compiler().CompileSpirv(
                modules, "meshCutoutFragmentMain");
        GraphicsPipelineDesc desc;
        desc.vertexSpirv = vertex;
        desc.fragmentSpirv = fragment;
        desc.colorFormats.assign(
            RenderTargets::kGBufferFormats.begin(),
            RenderTargets::kGBufferFormats.end());
        desc.depthFormat = RenderTargets::kDepthFormat;
        desc.depthCompare = RenderTargets::kDepthCompare;
        gbuffer = std::make_unique<GraphicsPipeline>(context_,
                                                     layout_, desc);
    }
    if (shadowCutoutPipeline_ == nullptr) {
        const std::vector<ShaderModuleSource> modules = MeshModules();
        const std::vector<std::uint32_t> vertex =
            renderer_.Compiler().CompileSpirv(
                modules, "meshShadowCutoutVertex");
        const std::vector<std::uint32_t> fragment =
            renderer_.Compiler().CompileSpirv(
                modules, "meshShadowCutoutFragment");
        GraphicsPipelineDesc shadow;
        shadow.vertexSpirv = vertex;
        shadow.fragmentSpirv = fragment;
        shadow.depthFormat = VK_FORMAT_D32_SFLOAT;
        shadow.depthBias = true;
        shadow.depthClamp = true;
        shadowCutoutPipeline_ = std::make_unique<GraphicsPipeline>(
            context_, layout_, shadow);
    }
}

void MeshDrawable::CreateMinimalPipelines() {
    if (minimalPipelines_[0] != nullptr) {
        return;
    }
    const std::vector<ShaderModuleSource> modules =
        MeshModules(false);
    const std::vector<std::uint32_t> vertex =
        renderer_.Compiler().CompileSpirv(modules, "meshVertexMain");
    const char* const entries[] = {
        "meshMinimalFragmentMain", "meshMinimalCutoutFragmentMain",
        "meshIdsFragmentMain", "meshIdsCutoutFragmentMain"};
    for (std::size_t i = 0; i < minimalPipelines_.size(); ++i) {
        // Kept alive until the pipeline is made: the desc only spans
        // it.
        const std::vector<std::uint32_t> fragment =
            renderer_.Compiler().CompileSpirv(modules, entries[i]);
        GraphicsPipelineDesc desc;
        desc.vertexSpirv = vertex;
        desc.fragmentSpirv = fragment;
        desc.colorFormats.assign(
            MinimalTargets::kColorFormats.begin(),
            MinimalTargets::kColorFormats.end());
        desc.depthFormat = MinimalTargets::kDepthFormat;
        desc.depthCompare = RenderTargets::kDepthCompare;
        minimalPipelines_[i] = std::make_unique<GraphicsPipeline>(
            context_, layout_, desc);
    }
}

std::uint32_t MeshDrawable::CullLevels(std::size_t item) const {
    if (!items_[item].instancer) {
        return 1;
    }
    return static_cast<std::uint32_t>(
        meshes_.at(items_[item].mesh).levels.size());
}

bool MeshDrawable::Transparent(std::size_t item) const {
    return renderer_.TransparentMaterial(items_[item].material);
}

bool MeshDrawable::Cutout(std::size_t item) const {
    return renderer_.MaterialOrDefault(items_[item].material)
        .Settings()
        .Cutout();
}

std::uint32_t MeshDrawable::AddMesh(MeshData data) {
    ERNIE_ZONE("Mesh add (LOD simplify)");
    if (data.points.size() % 3 != 0 || data.indices.size() % 3 != 0) {
        throw std::invalid_argument(
            "Mesh points and indices must come in threes");
    }
    const std::size_t count = data.points.size() / 3;
    CheckTriples(data.normals, count, "Mesh normals");
    CheckTriples(data.colors, count, "Mesh colours");
    for (const std::uint32_t index : data.indices) {
        if (index >= count) {
            throw std::invalid_argument("Mesh index out of range");
        }
    }
    if (!data.uvs.empty() &&
        data.uvs.size() != 2 * data.indices.size()) {
        throw std::invalid_argument(
            "Mesh texture coordinates must come one pair per corner");
    }
    for (const std::uint32_t index : data.lodIndices) {
        if (index >= count) {
            throw std::invalid_argument(
                "Mesh level index out of range");
        }
    }
    for (const MeshData::Level& level : data.levels) {
        if (level.count % 3 != 0 ||
            std::size_t{level.first} + level.count >
                data.lodIndices.size()) {
            throw std::invalid_argument(
                "Mesh levels must be triangles of lodIndices");
        }
    }
    if (!data.levels.empty() && !data.uvs.empty() &&
        data.lodUvs.size() != 2 * data.lodIndices.size()) {
        throw std::invalid_argument(
            "Mesh level texture coordinates must come one pair per "
            "corner");
    }
    if (!data.cornerNormals.empty() &&
        (data.cornerNormals.size() != 3 * data.indices.size() ||
         (!data.levels.empty() && data.lodCornerNormals.size() !=
                                      3 * data.lodIndices.size()))) {
        throw std::invalid_argument(
            "Mesh corner normals must come one per corner");
    }
    if (!data.cornerColors.empty() &&
        (data.cornerColors.size() != 3 * data.indices.size() ||
         (!data.levels.empty() && data.lodCornerColors.size() !=
                                      3 * data.lodIndices.size()))) {
        throw std::invalid_argument(
            "Mesh corner colours must come one per corner");
    }
    Mesh mesh;
    std::tie(mesh.low, mesh.high) = PointBounds(data.points);
    mesh.data = std::move(data);
    // Without normals, smooth ones as Hydra gives: each point's the
    // area-weighted sum of its triangles', on the GPU
    // (`mesh_normals.slang`) once and again whenever its points move.
    // Device triangles (`SetIndices`) leave them 0, which draws each
    // triangle's own.
    if (mesh.data.normals.empty() &&
        mesh.data.cornerNormals.empty() &&
        !mesh.data.indices.empty()) {
        mesh.data.normals.assign(mesh.data.points.size(), 0.0f);
        mesh.recomputeNormals = true;
        mesh.generatedNormals = true;
        mesh.adjacency = TrianglesAround(mesh.data.indices, count);
    }
    // Points all zero hold the place of device ones (`MeshDataOf`):
    // levels simplified from them would be degenerate, so they wait
    // for the first device points (`SetPoints`).
    mesh.levelsPending =
        !mesh.data.indices.empty() &&
        std::ranges::all_of(mesh.data.points,
                            [](float p) { return p == 0.0f; });
    if (mesh.levelsPending) {
        mesh.levels.push_back(
            {0, static_cast<std::uint32_t>(mesh.data.indices.size()),
             0.0f});
    } else {
        BuildLevels(mesh);
    }
    const std::uint32_t handle = nextHandle_++;
    meshes_.emplace(handle, std::move(mesh));
    geometryDirty_ = true;
    return handle;
}

void MeshDrawable::ForgetTriangleNormals(Mesh& mesh) {
    if (mesh.generatedNormals) {
        mesh.topologyNormals = true;
    }
    mesh.data.cornerNormals.clear();
    mesh.lodCornerNormals.clear();
    mesh.data.cornerColors.clear();
    mesh.lodCornerColors.clear();
}

void MeshDrawable::BuildLevels(Mesh& mesh) const {
    ERNIE_ZONE("Mesh LOD simplify");
    const MeshData& data = mesh.data;
    mesh.levels.assign(
        1,
        {0, static_cast<std::uint32_t>(data.indices.size()), 0.0f});
    mesh.lodIndices.clear();
    mesh.lodUvs.clear();
    mesh.lodCornerNormals.clear();
    mesh.lodCornerColors.clear();
    // Levels the mesh comes with (a subdivision surface's coarser
    // tessellations), taken as they are.
    if (!data.levels.empty()) {
        std::size_t end = 0;
        for (const MeshData::Level& level : data.levels) {
            if (data.indices.size() / 3 < kMinLodTriangles ||
                mesh.levels.size() >= kMaxLevels ||
                level.count < 3 * 8) {
                break;
            }
            mesh.levels.push_back(
                {static_cast<std::uint32_t>(data.indices.size()) +
                     level.first,
                 level.count, level.error});
            end =
                std::max<std::size_t>(end, level.first + level.count);
        }
        mesh.lodIndices.assign(data.lodIndices.begin(),
                               data.lodIndices.begin() +
                                   static_cast<std::ptrdiff_t>(end));
        if (!data.uvs.empty()) {
            mesh.lodUvs.assign(
                data.lodUvs.begin(),
                data.lodUvs.begin() +
                    static_cast<std::ptrdiff_t>(2 * end));
        }
        if (!data.cornerNormals.empty()) {
            mesh.lodCornerNormals.assign(
                data.lodCornerNormals.begin(),
                data.lodCornerNormals.begin() +
                    static_cast<std::ptrdiff_t>(3 * end));
        }
        if (!data.cornerColors.empty()) {
            mesh.lodCornerColors.assign(
                data.lodCornerColors.begin(),
                data.lodCornerColors.begin() +
                    static_cast<std::ptrdiff_t>(3 * end));
        }
        mesh.data.levels = {};
        mesh.data.lodIndices = {};
        mesh.data.lodUvs = {};
        mesh.data.lodCornerNormals = {};
        mesh.data.lodCornerColors = {};
        return;
    }
    // Points on a seam, a hard edge or a colour boundary (several
    // texture coordinates, normals or colours at them) stay as they
    // are, so the levels keep them.
    std::vector<std::vector<glm::vec2>> pointUvs;
    std::vector<std::vector<glm::vec3>> pointNormals;
    std::vector<std::vector<glm::vec3>> pointColors;
    std::vector<unsigned char> locked;
    if (!data.uvs.empty() || !data.cornerNormals.empty() ||
        !data.cornerColors.empty()) {
        locked.assign(data.points.size() / 3, 0);
    }
    if (!data.cornerColors.empty()) {
        pointColors = PointCornerValues(data, data.cornerColors);
        for (std::size_t p = 0; p < pointColors.size(); ++p) {
            if (pointColors[p].size() > 1) {
                locked[p] = meshopt_SimplifyVertex_Lock;
            }
        }
    }
    if (!data.uvs.empty()) {
        pointUvs = PointUvs(data);
        for (std::size_t p = 0; p < pointUvs.size(); ++p) {
            if (pointUvs[p].size() > 1) {
                locked[p] = meshopt_SimplifyVertex_Lock;
            }
        }
    }
    if (!data.cornerNormals.empty()) {
        pointNormals = PointCornerValues(data, data.cornerNormals);
        for (std::size_t p = 0; p < pointNormals.size(); ++p) {
            if (pointNormals[p].size() > 1) {
                locked[p] = meshopt_SimplifyVertex_Lock;
            }
        }
    }
    for (const MeshLevel& level : SimplifiedLevels<MeshLevel>(
             data, kMaxLevels, kMinLodTriangles, mesh.lodIndices,
             locked)) {
        mesh.levels.push_back(level);
    }
    if (!data.uvs.empty()) {
        mesh.lodUvs = LodUvs(pointUvs, mesh.lodIndices);
    }
    if (!data.cornerNormals.empty()) {
        mesh.lodCornerNormals = LodCornerNormals(
            pointNormals, data.points, mesh.lodIndices);
    }
    if (!data.cornerColors.empty()) {
        mesh.lodCornerColors =
            LodCornerColors(pointColors, mesh.lodIndices);
    }
}

void MeshDrawable::SetPoints(std::uint32_t mesh,
                             std::span<const float> points) {
    ForgetBounds();
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " +
                                    std::to_string(mesh));
    }
    if (points.size() != it->second.data.points.size()) {
        throw std::invalid_argument(
            "SetPoints must keep the number of points");
    }
    it->second.data.points.assign(points.begin(), points.end());
    std::tie(it->second.low, it->second.high) =
        PointBounds(it->second.data.points);
    it->second.devicePoints = Tensor();
    it->second.gpuPoints.reset();
    it->second.lastGpuPoints.reset();
    it->second.pointsDirty = true;
    activeMeshes_.insert(mesh);
    if (it->second.levelsPending &&
        it->second.deviceIndices.Empty()) {
        it->second.levelsPending = false;
        BuildLevels(it->second);
        geometryDirty_ = true;
    }
}

void MeshDrawable::SetPoints(
    std::uint32_t mesh, const Tensor& points,
    std::optional<std::pair<glm::vec3, glm::vec3>> bounds,
    std::uintptr_t stream) {
    ForgetBounds();
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " +
                                    std::to_string(mesh));
    }
    CheckDeviceTensor(points, 3, "Mesh points");
    Mesh& target = it->second;
    if (static_cast<std::size_t>(points.Elements()) !=
        target.data.points.size()) {
        throw std::invalid_argument(
            "SetPoints must keep the number of points");
    }
    // Levels of detail waiting for the first points need them on the
    // host, once.
    const bool levels =
        target.levelsPending && target.deviceIndices.Empty();
    // Without bounds, the points are downloaded only until the GPU
    // has measured them (`ReadBounds`).
    const bool measured = !bounds && target.gpuPoints.has_value();
    if ((!bounds && !measured) || levels) {
        const Tensor host = compute_->Download(points);
        target.data.points.assign(host.As<float>(),
                                  host.As<float>() + host.Elements());
    }
    target.pointsBounded = bounds.has_value();
    if (bounds) {
        std::tie(target.low, target.high) = *bounds;
        target.gpuPoints.reset();
        target.lastGpuPoints.reset();
    } else if (!measured) {
        std::tie(target.low, target.high) =
            PointBounds(target.data.points);
    }
    if (levels) {
        target.levelsPending = false;
        BuildLevels(target);
        geometryDirty_ = true;
    }
    // The view `MapPoints` gave out, handed back: the points are in
    // the pool already.
    target.pointsInPlace =
        !target.mappedPoints.Empty() &&
        points.Data() == target.mappedPoints.Data() &&
        !geometryDirty_;
    target.devicePoints = points;
    target.deviceStream = stream;
    target.pointsDirty = true;
    activeMeshes_.insert(mesh);
    gpuPointMeshes_.insert(mesh);
    // Normals it has follow the points on the GPU, unless they come
    // from the device too.
    // Not with device triangles: the host's are placeholders.
    const bool recompute = !target.data.normals.empty() &&
                           target.deviceNormals.Empty() &&
                           target.deviceIndices.Empty() &&
                           !target.data.indices.empty();
    if (recompute && !target.recomputeNormals) {
        target.recomputeNormals = true;
        target.adjacency = TrianglesAround(
            target.data.indices, target.data.points.size() / 3);
        geometryDirty_ = true;
    }
}

void MeshDrawable::SetNormals(std::uint32_t mesh,
                              const Tensor& normals,
                              std::uintptr_t stream) {
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " +
                                    std::to_string(mesh));
    }
    CheckDeviceTensor(normals, 3, "Mesh normals");
    Mesh& target = it->second;
    if (static_cast<std::size_t>(normals.Elements()) !=
        target.data.points.size()) {
        throw std::invalid_argument(
            "Mesh normals need one per point");
    }
    if (target.data.normals.empty()) {
        // Shaded with normals from now on: the draws' flags change.
        target.data.normals.assign(target.data.points.size(), 0.0f);
        geometryDirty_ = true;
    }
    if (target.recomputeNormals) {
        target.recomputeNormals = false;
        target.adjacency.clear();
        geometryDirty_ = true;
    }
    target.generatedNormals = false;
    if (target.topologyNormals) {
        target.topologyNormals = false;
        geometryDirty_ = true;
    }
    // Normals per point from now on, in place of any per corner.
    if (!target.data.cornerNormals.empty()) {
        target.data.cornerNormals.clear();
        target.lodCornerNormals.clear();
        geometryDirty_ = true;
    }
    target.deviceNormals = normals;
    target.normalsStream = stream;
    target.normalsDirty = true;
    activeMeshes_.insert(mesh);
}

void MeshDrawable::SetColors(std::uint32_t mesh, const Tensor& colors,
                             std::uintptr_t stream) {
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " +
                                    std::to_string(mesh));
    }
    CheckDeviceTensor(colors, 3, "Mesh colours");
    Mesh& target = it->second;
    if (target.data.colors.empty()) {
        throw std::invalid_argument(
            "Device colours need a mesh added with colours");
    }
    if (static_cast<std::size_t>(colors.Elements()) !=
        target.data.points.size()) {
        throw std::invalid_argument(
            "Mesh colours need one per point");
    }
    target.deviceColors = colors;
    target.colorsStream = stream;
    target.colorsDirty = true;
    activeMeshes_.insert(mesh);
}

void MeshDrawable::SetTemperatures(std::uint32_t mesh,
                                   std::span<const float> kelvin) {
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " +
                                    std::to_string(mesh));
    }
    Mesh& target = it->second;
    if (!kelvin.empty() &&
        kelvin.size() != target.data.points.size() / 3) {
        throw std::invalid_argument(
            "Mesh temperatures need one per point");
    }
    const bool has = !kelvin.empty();
    drawsDirty_ = drawsDirty_ || has != target.hasTemperatures;
    target.hasTemperatures = has;
    target.temperatures.assign(kelvin.begin(), kelvin.end());
    target.deviceTemperatures = Tensor();
    target.temperaturesDirty = has;
    activeMeshes_.insert(mesh);
}

void MeshDrawable::SetTemperatures(std::uint32_t mesh,
                                   const Tensor& kelvin,
                                   std::uintptr_t stream) {
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " +
                                    std::to_string(mesh));
    }
    CheckDeviceTensor(kelvin, 1, "Mesh temperatures");
    Mesh& target = it->second;
    if (static_cast<std::size_t>(kelvin.Elements()) !=
        target.data.points.size() / 3) {
        throw std::invalid_argument(
            "Mesh temperatures need one per point");
    }
    drawsDirty_ = drawsDirty_ || !target.hasTemperatures;
    target.hasTemperatures = true;
    target.temperatures.clear();
    target.deviceTemperatures = kelvin;
    target.temperaturesStream = stream;
    target.temperaturesDirty = true;
    activeMeshes_.insert(mesh);
}

void MeshDrawable::SetNodeIds(std::uint32_t mesh,
                              std::span<const std::uint32_t> ids) {
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " +
                                    std::to_string(mesh));
    }
    Mesh& target = it->second;
    if (!ids.empty() && ids.size() != target.data.points.size() / 3) {
        throw std::invalid_argument(
            "Mesh node ids need one per point");
    }
    target.nodeIds.assign(ids.begin(), ids.end());
    target.nodeCount =
        ids.empty() ? 0u
                    : *std::max_element(ids.begin(), ids.end()) + 1;
    // The pools take the ids and make room for the nodes.
    geometryDirty_ = true;
    drawsDirty_ = true;
}

void MeshDrawable::SetNodeTemperatures(
    std::uint32_t mesh, std::span<const float> kelvin) {
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " +
                                    std::to_string(mesh));
    }
    Mesh& target = it->second;
    if (!kelvin.empty() && kelvin.size() < target.nodeCount) {
        throw std::invalid_argument(
            "Mesh node temperatures need one per node, " +
            std::to_string(target.nodeCount));
    }
    const bool has = !kelvin.empty();
    drawsDirty_ = drawsDirty_ || has != target.hasNodeTemperatures;
    target.hasNodeTemperatures = has;
    target.nodeTemperatures.assign(kelvin.begin(),
                                   kelvin.begin() +
                                       (has ? target.nodeCount : 0));
    target.deviceNodeTemperatures = Tensor();
    target.nodeTemperaturesDirty = has;
    activeMeshes_.insert(mesh);
}

void MeshDrawable::SetNodeTemperatures(std::uint32_t mesh,
                                       const Tensor& kelvin,
                                       std::uintptr_t stream) {
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " +
                                    std::to_string(mesh));
    }
    CheckDeviceTensor(kelvin, 1, "Mesh node temperatures");
    Mesh& target = it->second;
    if (static_cast<std::size_t>(kelvin.Elements()) <
        target.nodeCount) {
        throw std::invalid_argument(
            "Mesh node temperatures need one per node, " +
            std::to_string(target.nodeCount));
    }
    drawsDirty_ = drawsDirty_ || !target.hasNodeTemperatures;
    target.hasNodeTemperatures = true;
    target.nodeTemperatures.clear();
    target.deviceNodeTemperatures = kelvin.Slice(0, target.nodeCount);
    target.nodeTemperaturesStream = stream;
    target.nodeTemperaturesDirty = true;
    activeMeshes_.insert(mesh);
}

void MeshDrawable::SetIndices(std::uint32_t mesh,
                              const Tensor& indices,
                              std::uintptr_t stream) {
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " +
                                    std::to_string(mesh));
    }
    if (compute_ == nullptr) {
        throw std::invalid_argument(
            "GPU inputs need MeshDrawable::SetComputeInterop");
    }
    Mesh& target = it->second;
    if (indices.Location() != compute_->Location() ||
        (indices.Type() != DType::Int32 &&
         indices.Type() != DType::UInt32) ||
        !indices.IsContiguous()) {
        throw std::invalid_argument(
            "Mesh indices must be a contiguous int32 or uint32 "
            "tensor "
            "on the compute device " +
            std::to_string(compute_->Device()));
    }
    if (static_cast<std::size_t>(indices.Elements()) !=
        target.data.indices.size()) {
        throw std::invalid_argument(
            "SetIndices must keep the number of triangles");
    }
    // Levels simplified from the host's placeholder zeros mean
    // nothing; nor do triangles around the points from them.
    target.levelsPending = false;
    if (target.levels.size() > 1 || target.recomputeNormals ||
        !target.data.cornerNormals.empty() ||
        !target.data.cornerColors.empty()) {
        target.levels.resize(1);
        target.lodIndices.clear();
        target.lodUvs.clear();
        target.recomputeNormals = false;
        target.adjacency.clear();
        ForgetTriangleNormals(target);
        geometryDirty_ = true;
    }
    // The view `MapIndices` gave out, handed back: the triangles are
    // in the pool already.
    target.indicesInPlace =
        !target.mappedIndices.Empty() &&
        indices.Data() == target.mappedIndices.Data() &&
        !geometryDirty_;
    target.deviceIndices = indices;
    target.indicesStream = stream;
    target.indicesDirty = true;
    activeMeshes_.insert(mesh);
}

Tensor MeshDrawable::MapIndices(std::uint32_t mesh,
                                std::uintptr_t stream) {
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " +
                                    std::to_string(mesh));
    }
    if (it->second.deviceIndices.Empty()) {
        throw std::invalid_argument(
            "Only triangle indices written on the GPU can be mapped");
    }
    EnsurePools();
    Mesh& target = it->second;
    // One copy of the indices: nothing writes them until the frames
    // that may read them finish. Shaped and typed as written, and
    // starting as them.
    compute_->WaitForVulkan(stream);
    target.mappedIndices = compute_->View(
        indices_, target.deviceIndices.Type(),
        target.deviceIndices.Shape(),
        VkDeviceSize{target.firstIndex} * sizeof(std::uint32_t));
    if (target.mappedIndices.Data() != target.deviceIndices.Data()) {
        compute_->Copy(target.deviceIndices, target.mappedIndices,
                       stream);
    }
    return target.mappedIndices;
}

void MeshDrawable::SetActiveCounts(std::uint32_t mesh,
                                   const Tensor& counts,
                                   std::uintptr_t stream) {
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " +
                                    std::to_string(mesh));
    }
    Mesh& target = it->second;
    if (!counts.Empty()) {
        if (counts.Type() != DType::Int32 || counts.Rank() != 1 ||
            counts.Elements() < 2 || !counts.IsContiguous()) {
            throw std::invalid_argument(
                "Active counts must be a contiguous int32 (n >= 2,) "
                "tensor: points, then triangle indices, then "
                "anything");
        }
        if (IsGpu(counts.Location().kind) &&
            (compute_ == nullptr ||
             counts.Location() != compute_->Location())) {
            throw std::invalid_argument(
                "Active counts on a GPU must be on the compute "
                "device, with MeshDrawable::SetComputeInterop");
        }
    }
    // Gaining or losing them takes or frees a slot.
    geometryDirty_ = geometryDirty_ ||
                     counts.Empty() != target.activeCounts.Empty();
    // They count the full mesh's triangles: no levels of detail.
    if (!counts.Empty()) {
        target.levelsPending = false;
        if (target.levels.size() > 1) {
            target.levels.resize(1);
            target.lodIndices.clear();
            target.lodUvs.clear();
            geometryDirty_ = true;
        }
    }
    // The first two: a simulator's own counts may carry more, such as
    // an overflow flag.
    target.activeCounts =
        counts.Empty() ? counts : counts.Slice(0, 2);
    target.countsStream = stream;
    target.countsDirty = true;
    activeMeshes_.insert(mesh);
    if (!geometryDirty_ && !counts.Empty() &&
        !IsGpu(counts.Location().kind)) {
        renderer_.Uploads().Write(
            *activeCounts_,
            std::as_bytes(std::span(counts.As<std::int32_t>(), 2)),
            VkDeviceSize{target.countSlot - 1} * 2 *
                sizeof(std::int32_t));
    }
}

void MeshDrawable::SetWriter(std::uint32_t mesh, MeshWriter writer,
                             std::optional<Bounds3> bounds) {
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " +
                                    std::to_string(mesh));
    }
    Mesh& target = it->second;
    // Gaining or losing one takes or frees a slot of counts.
    geometryDirty_ = geometryDirty_ || !writer != !target.writer;
    target.writer = std::move(writer);
    if (target.writer) {
        gpuPointMeshes_.insert(mesh);
    }
    target.gpuPoints.reset();
    target.lastGpuPoints.reset();
    target.pointsBounded = target.writer && bounds.has_value();
    if (target.pointsBounded) {
        std::tie(target.low, target.high) = *bounds;
    }
    if (target.writer) {
        writers_.insert(mesh);
        // Levels simplified from the placeholders mean nothing.
        target.levelsPending = false;
        if (target.levels.size() > 1 || target.recomputeNormals ||
            !target.data.cornerNormals.empty() ||
            !target.data.cornerColors.empty()) {
            target.levels.resize(1);
            target.lodIndices.clear();
            target.lodUvs.clear();
            target.recomputeNormals = false;
            target.adjacency.clear();
            ForgetTriangleNormals(target);
            geometryDirty_ = true;
        }
    } else {
        writers_.erase(mesh);
    }
    ForgetBounds();
}

void MeshDrawable::SetSkin(std::uint32_t mesh, Skin skin) {
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " +
                                    std::to_string(mesh));
    }
    Mesh& target = it->second;
    const std::size_t points = target.data.points.size() / 3;
    // Device samples are copied in on the device below.
    const bool deviceRest = !skin.deviceRest.empty();
    if (deviceRest) {
        for (const Tensor& sample : skin.deviceRest) {
            CheckDeviceTensor(sample, 3, "Skin rest samples");
            if (sample.Elements() !=
                skin.deviceRest.front().Elements()) {
                throw std::invalid_argument(
                    "A skin's rest samples need one size");
            }
        }
        skin.restSamples = skin.deviceRest.size();
        skin.restPoints.assign(
            skin.restSamples *
                static_cast<std::size_t>(
                    skin.deviceRest.front().Elements()),
            0.0f);
    }
    if (skin.restSamples == 0 ||
        skin.restPoints.size() % (3 * skin.restSamples) != 0) {
        throw std::invalid_argument("A skin's rest samples disagree");
    }
    const std::size_t controls =
        skin.restPoints.size() / 3 / skin.restSamples;
    const bool refined = !skin.stencils.starts.empty();
    if (refined ? skin.stencils.starts.size() != points + 1
                : controls != points) {
        throw std::invalid_argument(
            "A skin needs a rest point per mesh point, or a stencil "
            "per mesh "
            "point");
    }
    const std::size_t ranges =
        skin.constantInfluences ? 2 : controls + 1;
    if (skin.influenceStarts.size() != ranges ||
        skin.influenceJoints.size() != skin.influenceWeights.size() ||
        skin.influenceStarts.back() > skin.influenceJoints.size() ||
        (!skin.shapeStarts.empty() &&
         (skin.shapeStarts.size() != controls + 1 ||
          skin.shapeOffsets.size() != 3 * skin.shapeIndices.size() ||
          skin.shapeStarts.back() > skin.shapeIndices.size()))) {
        throw std::invalid_argument("A skin's arrays disagree");
    }
    for (const std::uint32_t joint : skin.influenceJoints) {
        if (joint >= skin.joints) {
            throw std::invalid_argument(
                "A skin's joint is out of range");
        }
    }
    for (const std::uint32_t shape : skin.shapeIndices) {
        if (shape >= skin.subShapes) {
            throw std::invalid_argument(
                "A skin's sub-shape is out of range");
        }
    }
    for (const std::uint32_t index : skin.stencils.indices) {
        if (index >= controls) {
            throw std::invalid_argument(
                "A skin's stencil is out of range");
        }
    }
    // Every array in one buffer, each 16-byte aligned.
    std::vector<std::byte> bytes;
    const auto append = [&](std::size_t slot, auto span) {
        bytes.resize((bytes.size() + 15) / 16 * 16);
        target.skinOffsets[slot] = bytes.size();
        const auto raw = std::as_bytes(span);
        bytes.insert(bytes.end(), raw.begin(), raw.end());
    };
    append(0, std::span<const float>(skin.restPoints));
    append(1, std::span<const std::uint32_t>(skin.influenceStarts));
    append(2, std::span<const std::uint32_t>(skin.influenceJoints));
    append(3, std::span<const float>(skin.influenceWeights));
    append(4, std::span<const std::uint32_t>(skin.shapeStarts));
    append(5, std::span<const float>(skin.shapeOffsets));
    append(6, std::span<const std::uint32_t>(skin.shapeIndices));
    append(7, std::span<const std::uint32_t>(skin.stencils.starts));
    append(8, std::span<const std::uint32_t>(skin.stencils.indices));
    append(9, std::span<const float>(skin.stencils.weights));
    target.skinData = std::make_unique<GpuBuffer>(
        context_, std::max<VkDeviceSize>(bytes.size(), 16),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        deviceRest ? BufferMemory::Shared : BufferMemory::Device);
    target.skinData->Write(bytes);
    for (std::size_t s = 0; deviceRest && s < skin.deviceRest.size();
         ++s) {
        compute_->CopyToBuffer(skin.deviceRest[s], *target.skinData,
                               target.skinOffsets[0] +
                                   s * 3 * controls * sizeof(float),
                               skin.restStream);
    }
    skin.deviceRest.clear();
    skin.restPoints.clear();
    skin.restPoints.resize(3 * controls * skin.restSamples);
    const std::size_t rows =
        24 + (skin.dualQuaternion ? 20 : 12) * skin.joints;
    target.skinPose = std::make_unique<GpuBuffer>(
        context_, (rows + skin.subShapes + 4) * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        BufferMemory::Device);
    target.skinControl =
        refined ? std::make_unique<GpuBuffer>(
                      context_,
                      std::max<std::size_t>(3 * controls, 4) *
                          sizeof(float),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      BufferMemory::Device)
                : nullptr;
    target.skinPoseData.clear();
    target.skinPosed = false;
    target.skin = std::make_unique<Skin>(std::move(skin));
    // Its normals follow its triangles as it moves.
    if (target.data.normals.empty()) {
        target.data.normals.assign(target.data.points.size(), 0.0f);
    }
    if (!target.recomputeNormals && !target.data.indices.empty()) {
        target.recomputeNormals = true;
        target.adjacency =
            TrianglesAround(target.data.indices, points);
    }
    geometryDirty_ = true;
}

void MeshDrawable::SetSkinPose(std::uint32_t mesh, SkinPose pose) {
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end() || !it->second.skin) {
        throw std::invalid_argument("Mesh " + std::to_string(mesh) +
                                    " is not skinned");
    }
    Mesh& target = it->second;
    const Skin& skin = *target.skin;
    const std::size_t rows =
        24 + (skin.dualQuaternion ? 20 : 12) * skin.joints;
    if (pose.rows.size() != rows ||
        pose.subShapeWeights.size() != skin.subShapes ||
        pose.lower >= skin.restSamples ||
        pose.upper >= skin.restSamples) {
        throw std::invalid_argument(
            "A skin pose needs rows for every joint, a "
            "weight for every sub-shape and two of its "
            "rest samples");
    }
    target.skinSamples = {pose.lower, pose.upper,
                          std::bit_cast<std::uint32_t>(pose.alpha)};
    target.skinPoseData = std::move(pose.rows);
    target.skinPoseData.insert(target.skinPoseData.end(),
                               pose.subShapeWeights.begin(),
                               pose.subShapeWeights.end());
    target.skinPosed = true;
    std::tie(target.low, target.high) = pose.bounds;
    target.pointsBounded = true;
    target.pointsDirty = true;
    activeMeshes_.insert(mesh);
    ForgetBounds();
}

void MeshDrawable::SetInstanceSamples(std::uint32_t instancer,
                                      InstanceSamples samples) {
    const auto it = instancers_.find(instancer);
    if (it == instancers_.end()) {
        throw std::invalid_argument("Unknown instancer " +
                                    std::to_string(instancer));
    }
    Instancer& target = it->second;
    const std::size_t n = target.count;
    struct Array {
        std::size_t samples;
        const std::vector<float>* host;
        const std::vector<Tensor>* device;
        std::size_t width;
        const char* what;
    };
    const std::array<Array, 3> arrays{{
        {samples.positionSamples, &samples.positions,
         &samples.devicePositions, 3, "Instance position samples"},
        {samples.orientationSamples, &samples.orientations,
         &samples.deviceOrientations, 4,
         "Instance orientation samples"},
        {samples.scaleSamples, &samples.scales, &samples.deviceScales,
         3, "Instance scale samples"},
    }};
    bool device = false;
    VkDeviceSize size = 0;
    for (std::size_t k = 0; k < arrays.size(); ++k) {
        const Array& a = arrays[k];
        target.sampleOffsets[k] = size;
        target.sampleCounts[k] =
            static_cast<std::uint32_t>(a.samples);
        if (a.samples == 0) {
            continue;
        }
        if (!a.device->empty()) {
            device = true;
            if (a.device->size() != a.samples) {
                throw std::invalid_argument(
                    std::string(a.what) +
                    " need a tensor per sample");
            }
            for (const Tensor& tensor : *a.device) {
                CheckDeviceTensor(tensor, a.width, a.what);
                if (static_cast<std::size_t>(tensor.Elements()) !=
                    n * a.width) {
                    throw std::invalid_argument(
                        std::string(a.what) +
                        " need a row per instance");
                }
            }
        } else if (a.host->size() != a.samples * n * a.width) {
            throw std::invalid_argument(
                std::string(a.what) +
                " need a row per instance and sample");
        }
        size +=
            (a.samples * n * a.width * sizeof(float) + 15) / 16 * 16;
    }
    target.sampleData = std::make_unique<GpuBuffer>(
        context_, std::max<VkDeviceSize>(size, 16),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        device ? BufferMemory::Shared : BufferMemory::Device);
    for (std::size_t k = 0; k < arrays.size(); ++k) {
        const Array& a = arrays[k];
        if (a.samples == 0) {
            continue;
        }
        if (a.device->empty()) {
            target.sampleData->Write(
                std::as_bytes(std::span(*a.host)),
                target.sampleOffsets[k]);
            continue;
        }
        // Copied on the device, once.
        for (std::size_t s = 0; s < a.samples; ++s) {
            compute_->CopyToBuffer((*a.device)[s], *target.sampleData,
                                   target.sampleOffsets[k] +
                                       s * n * a.width *
                                           sizeof(float),
                                   samples.stream);
        }
    }
    target.sampled = true;
    target.dirty = true;
}

void MeshDrawable::SetInstanceBlend(std::uint32_t instancer,
                                    std::uint32_t lower,
                                    std::uint32_t upper, float alpha,
                                    const Bounds3& positions,
                                    float scale) {
    const auto it = instancers_.find(instancer);
    if (it == instancers_.end() || !it->second.sampled) {
        throw std::invalid_argument("Instancer " +
                                    std::to_string(instancer) +
                                    " has no samples");
    }
    Instancer& target = it->second;
    target.blend = {lower, upper,
                    std::bit_cast<std::uint32_t>(alpha)};
    target.sampledPositions = positions;
    target.sampledScale = scale;
    target.dirty = true;
    ForgetBounds();
}

bool MeshDrawable::MeasuresPoints(const Mesh& mesh) {
    return (mesh.writer || !mesh.devicePoints.Empty()) &&
           !mesh.pointsBounded && !mesh.data.points.empty();
}

void MeshDrawable::RecordWriters(VkCommandBuffer commands) const {
    if (writers_.empty()) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Mesh writers");
    const auto barrier = [commands](VkAccessFlags2 srcAccess) {
        VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        memory.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        memory.srcAccessMask = srcAccess;
        memory.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        memory.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT |
                               VK_ACCESS_2_MEMORY_WRITE_BIT;
        VkDependencyInfo dependency{
            VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &memory;
        vkCmdPipelineBarrier2(commands, &dependency);
    };
    // After the frames before, which may still read the meshes, and
    // this frame's uploads.
    barrier(VK_ACCESS_2_MEMORY_READ_BIT |
            VK_ACCESS_2_MEMORY_WRITE_BIT);
    for (const std::uint32_t handle : writers_) {
        const Mesh& mesh = meshes_.at(handle);
        MeshWriteTarget target;
        target.points =
            points_->Address() +
            VkDeviceSize{PointBase(mesh) + mesh.firstVertex} * 3 *
                sizeof(float);
        // Its own normals; those made for it follow its triangles.
        if (!mesh.data.normals.empty() && !mesh.generatedNormals) {
            target.normals =
                normals_->Address() +
                VkDeviceSize{mesh.firstVertex} * 3 * sizeof(float);
        }
        target.indices =
            indices_->Address() +
            VkDeviceSize{mesh.firstIndex} * sizeof(std::uint32_t);
        target.counts = activeCounts_->Address() +
                        VkDeviceSize{mesh.countSlot - 1} * 2 *
                            sizeof(std::int32_t);
        target.pointCapacity =
            static_cast<std::uint32_t>(mesh.data.points.size() / 3);
        target.triangleCapacity =
            static_cast<std::uint32_t>(mesh.data.indices.size() / 3);
        mesh.writer(commands, target);
    }
    // Before anything reads them: bounds, culling and drawing.
    barrier(VK_ACCESS_2_MEMORY_WRITE_BIT);
}

void MeshDrawable::RecordActiveDraws(VkCommandBuffer commands) const {
    if (countSlots_ == 0) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Active draws");
    const auto barrier = [commands](VkPipelineStageFlags2 dstStage,
                                    VkAccessFlags2 dstAccess) {
        VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        memory.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        memory.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        memory.dstStageMask = dstStage;
        memory.dstAccessMask = dstAccess;
        VkDependencyInfo dependency{
            VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &memory;
        vkCmdPipelineBarrier2(commands, &dependency);
    };
    // After the counts are uploaded, copied in or written, and the
    // last frame's draws read.
    barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT |
                VK_ACCESS_2_SHADER_WRITE_BIT);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      activeDrawsPipeline_->Handle());
    const DescriptorResource resources[] = {
        DescriptorResource::Of(*activeCounts_),
        DescriptorResource::Of(*activeDraws_)};
    activeDrawsLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                            resources);
    const std::uint32_t constants[4] = {countSlots_, 0, 0, 0};
    activeDrawsLayout_.PushConstants(commands, constants,
                                     sizeof(constants));
    vkCmdDispatch(commands, (countSlots_ + 63) / 64, 1, 1);
    barrier(VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
}

void MeshDrawable::CopyDevicePoints(Mesh& mesh) {
    compute_->CopyToBuffer(
        mesh.devicePoints, *points_,
        VkDeviceSize{PointBase(mesh) + mesh.firstVertex} * 3 *
            sizeof(float),
        mesh.deviceStream);
}

void MeshDrawable::FlipPoints(Mesh& mesh) {
    if (!mesh.pointsWritten) {
        mesh.pointSet ^= 1u;
        mesh.pointsWritten = true;
        mesh.pointsMoving = true;
        drawsDirty_ = true;
    }
}

std::uint32_t MeshDrawable::PointBase(const Mesh& mesh,
                                      bool previous) const {
    const std::uint32_t set = previous && mesh.pointsMoving
                                  ? 1u - mesh.pointSet
                                  : mesh.pointSet;
    return set * pointTotal_;
}

std::uint32_t MeshDrawable::InstanceBase(const Instancer& instancer,
                                         bool previous) const {
    const std::uint32_t set = previous && instancer.moving
                                  ? 1u - instancer.set
                                  : instancer.set;
    return set * instanceTotal_;
}

Tensor MeshDrawable::PoolView(const std::shared_ptr<GpuBuffer>& pool,
                              std::size_t offset,
                              std::vector<std::int64_t> shape) const {
    return compute_->View(pool, DType::Float32, std::move(shape),
                          static_cast<VkDeviceSize>(offset) *
                              sizeof(float));
}

void MeshDrawable::EnsurePools() {
    if (compute_ == nullptr) {
        throw std::invalid_argument(
            "Mapping arrays needs compute interop");
    }
    if (geometryDirty_) {
        RebuildGeometry();
    }
    if (instancesDirty_) {
        RebuildInstances();
    }
}

Tensor MeshDrawable::MapPoints(std::uint32_t mesh,
                               std::uintptr_t stream) {
    const auto it = meshes_.find(mesh);
    if (it == meshes_.end()) {
        throw std::invalid_argument("Unknown mesh " +
                                    std::to_string(mesh));
    }
    EnsurePools();
    Mesh& target = it->second;
    const auto count =
        static_cast<std::int64_t>(target.data.points.size() / 3);
    // Nothing writes the half frames still in flight may read until
    // they finish.
    compute_->WaitForVulkan(stream);
    const Tensor current = PoolView(
        points_,
        3 * std::size_t{PointBase(target) + target.firstVertex},
        {count, 3});
    FlipPoints(target);
    activeMeshes_.insert(mesh);
    target.mappedPoints = PoolView(
        points_,
        3 * std::size_t{PointBase(target) + target.firstVertex},
        {count, 3});
    if (target.mappedPoints.Data() != current.Data()) {
        compute_->Copy(current, target.mappedPoints, stream);
    }
    return target.mappedPoints;
}

Tensor MeshDrawable::MapInstances(std::uint32_t instancer,
                                  std::size_t array,
                                  std::uintptr_t stream) {
    ForgetBounds();
    const auto it = instancers_.find(instancer);
    if (it == instancers_.end()) {
        throw std::invalid_argument("Unknown instancer " +
                                    std::to_string(instancer));
    }
    if (array > 3) {
        throw std::invalid_argument(
            "Instance arrays are 0 positions, 1 orientations, 2 "
            "scales, 3 colours");
    }
    if (it->second.composed) {
        throw std::invalid_argument(
            "Instances composed on the GPU cannot be mapped");
    }
    EnsurePools();
    Instancer& target = it->second;
    const std::array<const std::shared_ptr<GpuBuffer>*, 4> pools{
        &positions_, &orientations_, &scales_, &instanceColors_};
    const std::array<std::size_t, 4> components{3, 4, 3, 3};
    const auto count = static_cast<std::int64_t>(target.count);
    const auto view = [&](std::size_t a) {
        return PoolView(
            *pools[a],
            components[a] *
                std::size_t{InstanceBase(target) + target.first},
            {count, static_cast<std::int64_t>(components[a])});
    };
    compute_->WaitForVulkan(stream);
    if (!target.written) {
        // Every array moves to the other half together, as they are
        // now.
        std::array<Tensor, 4> current{view(0), view(1), view(2),
                                      view(3)};
        FlipInstancer(target);
        for (std::size_t a = 0; a < current.size(); ++a) {
            compute_->Copy(current[a], view(a), stream);
        }
    }
    target.mapped[array] = view(array);
    return target.mapped[array];
}

Tensor MeshDrawable::MapComposedOp(std::uint32_t instancer,
                                   std::size_t op,
                                   std::uintptr_t stream) {
    ForgetBounds();
    const auto it = instancers_.find(instancer);
    if (it == instancers_.end()) {
        throw std::invalid_argument("Unknown instancer " +
                                    std::to_string(instancer));
    }
    if (compute_ == nullptr) {
        throw std::invalid_argument(
            "Mapping arrays needs compute interop");
    }
    const Instancer& target = it->second;
    if (!target.composed) {
        throw std::invalid_argument("Instancer " +
                                    std::to_string(instancer) +
                                    " is not composed on the GPU");
    }
    const std::vector<DeviceInstanceOp>& ops =
        target.composedData.ops;
    if (op >= ops.size()) {
        throw std::invalid_argument(
            "Instancer " + std::to_string(instancer) + " has " +
            std::to_string(ops.size()) + " transform ops");
    }
    if (composeLayoutDirty_) {
        // Laid out now, as `Prepare` would, so the rows' place in the
        // raw arrays is known.
        RebuildCompose();
        drawsDirty_ = true;
    }
    VkDeviceSize offset = target.rawFirst;
    for (std::size_t k = 0; k < op; ++k) {
        offset += static_cast<VkDeviceSize>(ops[k].rows.Elements());
    }
    const Tensor view =
        compute_->View(composeRaw_, DType::Float32,
                       ops[op].rows.Shape(), offset * sizeof(float));
    // Nothing writes the rows until the frames already submitted have
    // composed from them.
    compute_->WaitForVulkan(stream);
    if (ops[op].rows.Data() != view.Data()) {
        compute_->Copy(ops[op].rows, view, stream);
    }
    return view;
}

void MeshDrawable::CopyDeviceAttributes(Mesh& mesh, bool all) {
    const VkDeviceSize offset =
        VkDeviceSize{mesh.firstVertex} * 3 * sizeof(float);
    if (!mesh.deviceNormals.Empty() && (all || mesh.normalsDirty)) {
        compute_->CopyToBuffer(mesh.deviceNormals, *normals_, offset,
                               mesh.normalsStream);
    }
    if (!mesh.deviceColors.Empty() && (all || mesh.colorsDirty)) {
        compute_->CopyToBuffer(mesh.deviceColors, *colors_, offset,
                               mesh.colorsStream);
    }
    if (!mesh.deviceTemperatures.Empty() &&
        (all || mesh.temperaturesDirty)) {
        ERNIE_ZONE("Mesh thermal device copy");
        compute_->CopyToBuffer(
            mesh.deviceTemperatures, *temperatures_,
            VkDeviceSize{mesh.firstVertex} * sizeof(float),
            mesh.temperaturesStream);
        mesh.temperaturesDirty = false;
    }
    if (!mesh.deviceNodeTemperatures.Empty() &&
        (all || mesh.nodeTemperaturesDirty)) {
        ERNIE_ZONE("Mesh thermal node device copy");
        compute_->CopyToBuffer(
            mesh.deviceNodeTemperatures, *nodeTemperatures_,
            VkDeviceSize{mesh.firstNode} * sizeof(float),
            mesh.nodeTemperaturesStream);
        mesh.nodeTemperaturesDirty = false;
    }
    // The full mesh's triangles, first of its indices; written in
    // place (`MapIndices`), they are there already, and the frame
    // waits for their stream.
    if (!mesh.deviceIndices.Empty() && (all || mesh.indicesDirty)) {
        if (mesh.indicesInPlace && !all) {
            compute_->SignalVulkan(mesh.indicesStream);
        } else {
            compute_->CopyToBuffer(mesh.deviceIndices, *indices_,
                                   VkDeviceSize{mesh.firstIndex} *
                                       sizeof(std::uint32_t),
                                   mesh.indicesStream);
        }
    }
    if (mesh.countSlot != 0 && !mesh.activeCounts.Empty() &&
        IsGpu(mesh.activeCounts.Location().kind) &&
        (all || mesh.countsDirty)) {
        compute_->CopyToBuffer(mesh.activeCounts, *activeCounts_,
                               VkDeviceSize{mesh.countSlot - 1} * 2 *
                                   sizeof(std::int32_t),
                               mesh.countsStream);
    }
    mesh.normalsDirty = false;
    mesh.colorsDirty = false;
    mesh.indicesDirty = false;
    mesh.indicesInPlace = false;
    mesh.countsDirty = false;
}

std::shared_ptr<GpuBuffer>
MeshDrawable::Pool(std::span<const std::byte> bytes) const {
    ERNIE_ZONE("Mesh pool upload");
    if (compute_ == nullptr) {
        return StorageBuffer(context_, bytes);
    }
    auto buffer = std::make_shared<GpuBuffer>(
        context_, std::max<VkDeviceSize>(bytes.size(), 16),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Shared);
    if (!bytes.empty()) {
        buffer->Write(bytes);
    }
    return buffer;
}

void MeshDrawable::RemoveMesh(std::uint32_t mesh) {
    RemoveMeshes(std::span(&mesh, 1));
}

void MeshDrawable::RemoveMeshes(std::span<const std::uint32_t> meshes) {
    if (meshes.empty()) {
        return;
    }
    for (const std::uint32_t mesh : meshes) {
        if (!meshes_.contains(mesh)) {
            throw std::invalid_argument("Unknown mesh " +
                                        std::to_string(mesh));
        }
    }
    const std::unordered_set<std::uint32_t> removed(meshes.begin(),
                                                    meshes.end());
    for (const std::uint32_t mesh : removed) {
        meshes_.erase(mesh);
        activeMeshes_.erase(mesh);
        gpuPointMeshes_.erase(mesh);
        writers_.erase(mesh);
    }
    std::erase_if(items_, [&removed](const MeshItem& item) {
        return removed.contains(item.mesh);
    });
    ForgetBounds();
    itemPipelines_.clear();
    geometryDirty_ = true;
}

bool MeshDrawable::HasInstanceColors(const Instancer& instancer) {
    if (instancer.composed) {
        return false;
    }
    return instancer.onDevice ? !instancer.device.colors.Empty()
                              : !instancer.data.colors.empty();
}

bool MeshDrawable::HasInstanceTemperatures(const Instancer& instancer) {
    return !instancer.composed &&
           (instancer.onDevice ? !instancer.device.temperatures.Empty()
                               : !instancer.data.temperatures.empty());
}

bool MeshDrawable::HasInstanceVisibility(const Instancer& instancer) {
    return !instancer.composed &&
           (instancer.onDevice ? !instancer.device.visible.Empty()
                               : !instancer.data.visible.empty());
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

void MeshDrawable::SetInstances(std::uint32_t instancer,
                                InstanceData data) {
    ForgetBounds();
    const auto it = instancers_.find(instancer);
    if (it == instancers_.end()) {
        throw std::invalid_argument("Unknown instancer " +
                                    std::to_string(instancer));
    }
    CheckInstances(data);
    Instancer& target = it->second;
    composeLayoutDirty_ = composeLayoutDirty_ || target.composed;
    target.composed = false;
    const std::size_t count = data.positions.size() / 3;
    instancesDirty_ = instancesDirty_ || count != target.count;
    // Draws read instance colours by a flag.
    drawsDirty_ = drawsDirty_ ||
                  HasInstanceColors(target) != !data.colors.empty();
    target.count = count;
    target.data = std::move(data);
    target.device = {};
    target.onDevice = false;
    target.dirty = true;
}

void MeshDrawable::CheckDeviceTensor(const Tensor& tensor,
                                     std::size_t components,
                                     const char* what) const {
    if (compute_ == nullptr) {
        throw std::invalid_argument(
            "GPU inputs need MeshDrawable::SetComputeInterop");
    }
    const bool shaped =
        tensor.Rank() == 1 ||
        (tensor.Rank() == 2 &&
         tensor.Shape()[1] == static_cast<std::int64_t>(components));
    if (tensor.Location() != compute_->Location() ||
        tensor.Type() != DType::Float32 || !tensor.IsContiguous() ||
        !shaped ||
        static_cast<std::size_t>(tensor.Elements()) % components !=
            0) {
        throw std::invalid_argument(
            std::string(what) +
            " must be a contiguous float32 tensor of " +
            std::to_string(components) +
            "-vectors on the compute device " +
            std::to_string(compute_->Device()));
    }
}

void MeshDrawable::SetInstances(std::uint32_t instancer,
                                const DeviceInstanceData& data) {
    ForgetBounds();
    const auto it = instancers_.find(instancer);
    if (it == instancers_.end()) {
        throw std::invalid_argument("Unknown instancer " +
                                    std::to_string(instancer));
    }
    CheckDeviceTensor(data.positions, 3, "Instance positions");
    const auto count =
        static_cast<std::size_t>(data.positions.Elements()) / 3;
    if (!data.orientations.Empty()) {
        CheckDeviceTensor(data.orientations, 4,
                          "Instance orientations");
        if (static_cast<std::size_t>(data.orientations.Elements()) !=
            4 * count) {
            throw std::invalid_argument("Instance orientations must "
                                        "have 4 values per instance");
        }
        if (data.orientationsWFirst &&
            (data.orientations.Rank() != 2 ||
             data.orientations.Shape()[1] != 4)) {
            throw std::invalid_argument(
                "Instance orientations with w first must be (n, 4)");
        }
    }
    if (!data.scales.Empty()) {
        CheckDeviceTensor(data.scales, 3, "Instance scales");
        if (static_cast<std::size_t>(data.scales.Elements()) !=
            3 * count) {
            throw std::invalid_argument(
                "Instance scales must have 3 values per instance");
        }
    }
    if (!data.colors.Empty()) {
        CheckDeviceTensor(data.colors, 3, "Instance colours");
        if (static_cast<std::size_t>(data.colors.Elements()) !=
            3 * count) {
            throw std::invalid_argument(
                "Instance colours must have 3 values per instance");
        }
    }
    for (const auto& [array, what] :
         {std::pair{&data.temperatures, "Instance temperatures"},
          std::pair{&data.visible, "Instance visibility"}}) {
        if (!array->Empty()) {
            CheckDeviceTensor(*array, 1, what);
            if (static_cast<std::size_t>(array->Elements()) != count) {
                throw std::invalid_argument(
                    std::string(what) + " must have 1 value per instance");
            }
        }
    }
    Instancer& target = it->second;
    composeLayoutDirty_ = composeLayoutDirty_ || target.composed;
    target.composed = false;
    instancesDirty_ = instancesDirty_ || count != target.count;
    // Arrays handed back as the views `MapInstances` gave out are
    // written in place.
    const std::array<const Tensor*, 4> arrays{
        &data.positions, &data.orientations, &data.scales,
        &data.colors};
    for (std::size_t a = 0; a < arrays.size(); ++a) {
        target.inPlace[a] =
            !arrays[a]->Empty() && !target.mapped[a].Empty() &&
            arrays[a]->Data() == target.mapped[a].Data() &&
            !instancesDirty_;
    }
    drawsDirty_ = drawsDirty_ ||
                  HasInstanceColors(target) != !data.colors.Empty();
    // Measurements of other instances, or of host ones, do not bound
    // these.
    if (count != target.count || !target.onDevice) {
        target.gpuPositions.reset();
        target.lastGpuPositions.reset();
    }
    target.count = count;
    target.device = data;
    target.onDevice = true;
    target.dirty = true;
    // Bounds come from the extent, or from the positions measured on
    // the GPU (`RecordBounds`) once they have been: only before that
    // are the arrays copied to the host for them.
    if (data.extent || target.gpuPositions) {
        target.data = {};
        return;
    }
    target.data = {};
    if (!data.extent) {
        const Tensor positions = compute_->Download(data.positions);
        target.data.positions.assign(
            positions.As<float>(), positions.As<float>() + 3 * count);
        if (!data.scales.Empty()) {
            const Tensor scales = compute_->Download(data.scales);
            target.data.scales.assign(scales.As<float>(),
                                      scales.As<float>() + 3 * count);
        }
    }
}

void MeshDrawable::SetInstances(std::uint32_t instancer,
                                const ComposedInstanceData& data) {
    ForgetBounds();
    const auto it = instancers_.find(instancer);
    if (it == instancers_.end()) {
        throw std::invalid_argument("Unknown instancer " +
                                    std::to_string(instancer));
    }
    if (data.ops.empty()) {
        throw std::invalid_argument(
            "Composed instances need at least one transform op");
    }
    const std::int64_t rows = data.ops.front().rows.Rank() > 0
                                  ? data.ops.front().rows.Shape()[0]
                                  : 0;
    for (const DeviceInstanceOp& op : data.ops) {
        const auto [width, alternative] = OpWidths(op.kind);
        const std::int64_t given =
            op.rows.Rank() == 2 ? op.rows.Shape()[1] : -1;
        CheckDeviceTensor(op.rows,
                          static_cast<std::size_t>(
                              std::max<std::int64_t>(given, 1)),
                          "Instance op rows");
        if (given != width && given != alternative) {
            throw std::invalid_argument(
                "Instance op rows must be (n, " +
                std::to_string(width) + ") for this op");
        }
        if (op.rows.Shape()[0] != rows) {
            throw std::invalid_argument(
                "Every instance op needs the same number of rows");
        }
    }
    if (!data.parents.empty() &&
        data.parents.size() != static_cast<std::size_t>(rows)) {
        throw std::invalid_argument("Composed instances need one "
                                    "parent per instance, or none");
    }
    for (const InstanceParent& parent : data.parents) {
        if (parent.instancer != 0 &&
            !instancers_.contains(parent.instancer)) {
            throw std::invalid_argument(
                "Unknown parent instancer " +
                std::to_string(parent.instancer));
        }
    }
    Instancer& target = it->second;
    // The compose layout (raw offsets, ops, levels) follows the ops'
    // shapes and the parents.
    const auto sameLayout = [&] {
        const ComposedInstanceData& old = target.composedData;
        if (!target.composed || old.ops.size() != data.ops.size() ||
            old.parents.size() != data.parents.size()) {
            return false;
        }
        for (std::size_t k = 0; k < data.ops.size(); ++k) {
            if (old.ops[k].kind != data.ops[k].kind ||
                old.ops[k].wFirst != data.ops[k].wFirst ||
                old.ops[k].rows.Shape() != data.ops[k].rows.Shape()) {
                return false;
            }
        }
        for (std::size_t i = 0; i < data.parents.size(); ++i) {
            const InstanceParent& a = old.parents[i];
            const InstanceParent& b = data.parents[i];
            if (a.instancer != b.instancer || a.row != b.row ||
                a.translate != b.translate ||
                a.rotation != b.rotation || a.scale != b.scale) {
                return false;
            }
        }
        return true;
    };
    composeLayoutDirty_ = composeLayoutDirty_ || !sameLayout();
    instancesDirty_ = instancesDirty_ ||
                      static_cast<std::size_t>(rows) != target.count;
    target.count = static_cast<std::size_t>(rows);
    target.composed = true;
    target.composedData = data;
    target.onDevice = true;
    target.device = {};
    target.device.extent = data.extent;
    target.device.stream = data.stream;
    target.data = {};
    target.inPlace = {};
    target.mapped = {};
    target.dirty = true;
}

void MeshDrawable::SetComputeInterop(ComputeInterop* compute) {
    compute_ = compute;
    geometryDirty_ = true;
    instancesDirty_ = true;
}

void MeshDrawable::RemoveInstancer(std::uint32_t instancer) {
    ForgetBounds();
    const auto it = instancers_.find(instancer);
    if (it == instancers_.end()) {
        throw std::invalid_argument("Unknown instancer " +
                                    std::to_string(instancer));
    }
    composeLayoutDirty_ = composeLayoutDirty_ || it->second.composed;
    instancers_.erase(it);
    std::erase_if(items_, [instancer](const MeshItem& item) {
        return item.instancer == instancer;
    });
    itemPipelines_.clear();
    instancesDirty_ = true;
}

namespace {

/// Whether draw `a` keeps `b`'s layout: only moved, retinted or
/// reheated.
bool SameLayout(const MeshItem& a, const MeshItem& b) {
    return a.mesh == b.mesh && a.instancer == b.instancer &&
           a.instances == b.instances && a.world == b.world &&
           a.instanceWorlds == b.instanceWorlds &&
           a.material == b.material && a.primId == b.primId &&
           a.linkGroup == b.linkGroup &&
           a.vertexColors == b.vertexColors;
}

}  // namespace

void MeshDrawable::PatchItem(std::size_t i, MeshItem item) {
    MeshItem& was = items_[i];
    if (item.transform == was.transform &&
        item.prototype == was.prototype && item.tint == was.tint &&
        item.temperature == was.temperature) {
        return;
    }
    // Moved from where it was; retinted or reheated, its bounds stay.
    if (item.transform != was.transform) {
        const auto relative = [&](glm::dmat4 world) {
            world[3] -= glm::dvec4(origin_, 0.0);
            return glm::mat4(world);
        };
        previous_[i] = was.transform;
        previousTransforms_[i] = relative(was.transform);
        transforms_[i] = relative(item.transform);
        itemsMoving_ = true;
        movedItems_.push_back(i);
    }
    if (item.transform != was.transform ||
        item.prototype != was.prototype) {
        ForgetItemBounds(i);
    }
    was = std::move(item);
    posedItems_.push_back(i);
}

bool MeshDrawable::UpdateItems(
    std::span<std::pair<std::size_t, MeshItem>> items) {
    if (drawsDirty_ ||
        drawBytes_.size() != items_.size() * sizeof(GpuDraw)) {
        return false;
    }
    for (const auto& [i, item] : items) {
        if (i >= items_.size() || !SameLayout(item, items_[i])) {
            return false;
        }
    }
    for (auto& [i, item] : items) {
        PatchItem(i, std::move(item));
    }
    return true;
}

void MeshDrawable::SetItems(std::vector<MeshItem> items) {
    ERNIE_ZONE("Mesh set items");
    for (const MeshItem& item : items) {
        if (!meshes_.contains(item.mesh)) {
            throw std::invalid_argument("Unknown mesh " +
                                        std::to_string(item.mesh));
        }
        if (!renderer_.HasMaterial(item.material)) {
            throw std::invalid_argument(
                "Unknown material " + std::to_string(item.material));
        }
        if (item.instancer) {
            const auto it = instancers_.find(*item.instancer);
            if (it == instancers_.end()) {
                throw std::invalid_argument(
                    "Unknown instancer " +
                    std::to_string(*item.instancer));
            }
            const std::size_t count = it->second.count;
            for (const std::uint32_t instance : item.instances) {
                if (instance >= count) {
                    throw std::invalid_argument(
                        "Instance index out of range");
                }
            }
        }
        if (!item.instanceWorlds.empty() &&
            (item.instanceWorlds.size() != item.instances.size() ||
             !std::is_sorted(item.instanceWorlds.begin(),
                             item.instanceWorlds.end()))) {
            throw std::invalid_argument(
                "Instance worlds must match the instances, sorted");
        }
    }
    // The same draws only moved, retinted or reheated (a few prims
    // edited): their entries are patched (`PatchItems`), the layout,
    // its cull lists and the other draws kept.
    if (!drawsDirty_ && items.size() == items_.size() &&
        drawBytes_.size() == items_.size() * sizeof(GpuDraw) &&
        std::ranges::equal(items, items_, SameLayout)) {
        for (std::size_t i = 0; i < items.size(); ++i) {
            PatchItem(i, std::move(items[i]));
        }
        return;
    }
    std::vector<std::vector<WorldRange>> ranges;
    for (const MeshItem& item : items) {
        std::vector<WorldRange> runs;
        if (item.instanceWorlds.empty()) {
            const auto count = static_cast<std::uint32_t>(
                item.instancer ? item.instances.size()
                               : std::size_t{1});
            runs.push_back({item.world, 0, count});
        }
        for (std::uint32_t i = 0; i < item.instanceWorlds.size();
             ++i) {
            if (runs.empty() ||
                runs.back().world != item.instanceWorlds[i]) {
                runs.push_back({item.instanceWorlds[i], i, 0});
            }
            ++runs.back().count;
        }
        ranges.push_back(std::move(runs));
    }
    // Each draw's transform at the last frame: a draw of the same
    // prim keeps its own while its transform stays, else moves from
    // it; a new draw stands still.
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
        previous.push_back(
            was == item.transform ? previous_[found->second] : was);
    }
    items_ = std::move(items);
    previous_ = std::move(previous);
    itemRanges_ = std::move(ranges);
    ++layoutVersion_;
    itemPipelines_.clear();
    drawsDirty_ = true;
    itemsMoving_ = true;
    movedAll_ = true;
    movedItems_.clear();
    UpdateTransforms();
}

void MeshDrawable::SettleMotion() {
    if (itemsMoving_) {
        itemsMoving_ = false;
        // Those that moved stand still: their entries patched
        // (`PatchItems`), and their bounds no longer reach back.
        const auto settle = [this](std::size_t i) {
            if (i < items_.size() &&
                previous_[i] != items_[i].transform) {
                previous_[i] = items_[i].transform;
                previousTransforms_[i] = transforms_[i];
                posedItems_.push_back(i);
                worldsBounds_.reset();
            }
        };
        if (movedAll_) {
            for (std::size_t i = 0; i < items_.size(); ++i) {
                settle(i);
            }
        } else {
            for (const std::size_t i : movedItems_) {
                settle(i);
            }
        }
        movedAll_ = false;
        movedItems_.clear();
    }
    // Points and instance arrays that moved stand still until written
    // again.
    for (auto it = activeMeshes_.begin();
         it != activeMeshes_.end();) {
        Mesh& mesh = meshes_.at(*it);
        drawsDirty_ = drawsDirty_ || mesh.pointsMoving;
        mesh.pointsMoving = false;
        mesh.pointsWritten = false;
        mesh.mappedPoints = {};
        const bool pending = mesh.pointsDirty || mesh.normalsDirty ||
                             mesh.colorsDirty ||
                             mesh.temperaturesDirty ||
                             mesh.nodeTemperaturesDirty;
        it = pending ? std::next(it) : activeMeshes_.erase(it);
    }
    for (auto& [handle, instancer] : instancers_) {
        // Only its draws' bases change (`PatchInstanceBases`).
        instancer.rebase = instancer.rebase || instancer.moving;
        instancer.moving = false;
        instancer.written = false;
        // Views mapped for this frame expire: one handed back later
        // is copied from, as the half behind it is no longer the one
        // the next frame draws.
        instancer.mapped = {};
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
    ForgetBounds();
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
    ++layoutVersion_;
    ForgetBounds();
    std::vector<float> points;
    std::vector<float> normals;
    std::vector<float> colors;
    std::vector<float> temperatures;
    std::vector<std::uint32_t> nodeIds;
    std::vector<float> nodeTemperatures;
    std::vector<std::uint32_t> indices;
    std::vector<std::uint32_t> adjacency;
    std::size_t normalSums = 0;
    topologyMeshes_.clear();
    std::vector<float> uvs;
    std::vector<float> cornerNormals;
    std::vector<float> cornerColors;
    // Smooth tangents, per point of every mesh, where an anisotropic
    // material shades one (none held otherwise).
    const bool tangents =
        std::ranges::any_of(meshes_, [](const auto& entry) {
            return entry.second.wantsTangents &&
                   !entry.second.data.uvs.empty();
        });
    std::vector<float> tangentPool;
    // Points and triangles in use, by slot; device ones start as
    // every one, until copied in.
    std::vector<std::int32_t> counts;
    {
        ERNIE_ZONE("Mesh reserve geometry");
        std::size_t pointFloats = 0;
        std::size_t indexCount = 0;
        std::size_t uvFloats = 0;
        std::size_t nodes = 0;
        for (const auto& [handle, mesh] : meshes_) {
            pointFloats += mesh.data.points.size();
            indexCount +=
                mesh.data.indices.size() + mesh.lodIndices.size();
            uvFloats += mesh.data.uvs.size() + mesh.lodUvs.size();
            nodes += mesh.nodeTemperatures.empty()
                         ? mesh.nodeCount
                         : mesh.nodeTemperatures.size();
        }
        // The points twice: both halves of the pool.
        points.reserve(2 * pointFloats);
        normals.reserve(pointFloats);
        colors.reserve(pointFloats);
        temperatures.reserve(pointFloats / 3);
        nodeIds.reserve(pointFloats / 3);
        nodeTemperatures.reserve(nodes);
        indices.reserve(indexCount);
        uvs.reserve(uvFloats);
    }
    for (auto& [handle, mesh] : meshes_) {
        mesh.countSlot = 0;
        if (mesh.writer) {
            // Nothing in use until written.
            mesh.countSlot =
                static_cast<std::uint32_t>(counts.size() / 2 + 1);
            counts.insert(counts.end(), {0, 0});
        } else if (!mesh.activeCounts.Empty()) {
            mesh.countSlot =
                static_cast<std::uint32_t>(counts.size() / 2 + 1);
            if (IsGpu(mesh.activeCounts.Location().kind)) {
                counts.push_back(static_cast<std::int32_t>(
                    mesh.data.points.size() / 3));
                counts.push_back(static_cast<std::int32_t>(
                    mesh.data.indices.size()));
            } else {
                counts.push_back(
                    mesh.activeCounts.As<std::int32_t>()[0]);
                counts.push_back(
                    mesh.activeCounts.As<std::int32_t>()[1]);
            }
        }
        mesh.firstUv = static_cast<std::uint32_t>(uvs.size() / 2);
        uvs.insert(uvs.end(), mesh.data.uvs.begin(),
                   mesh.data.uvs.end());
        uvs.insert(uvs.end(), mesh.lodUvs.begin(), mesh.lodUvs.end());
        mesh.firstCornerNormal =
            static_cast<std::uint32_t>(cornerNormals.size() / 3);
        cornerNormals.insert(cornerNormals.end(),
                             mesh.data.cornerNormals.begin(),
                             mesh.data.cornerNormals.end());
        cornerNormals.insert(cornerNormals.end(),
                             mesh.lodCornerNormals.begin(),
                             mesh.lodCornerNormals.end());
        mesh.firstCornerColor =
            static_cast<std::uint32_t>(cornerColors.size() / 3);
        cornerColors.insert(cornerColors.end(),
                            mesh.data.cornerColors.begin(),
                            mesh.data.cornerColors.end());
        cornerColors.insert(cornerColors.end(),
                            mesh.lodCornerColors.begin(),
                            mesh.lodCornerColors.end());
        mesh.firstIndex = static_cast<std::uint32_t>(indices.size());
        mesh.firstVertex =
            static_cast<std::uint32_t>(points.size() / 3);
        const std::size_t count = mesh.data.points.size();
        points.insert(points.end(), mesh.data.points.begin(),
                      mesh.data.points.end());
        mesh.tangents =
            tangents && mesh.wantsTangents && !mesh.data.uvs.empty();
        if (mesh.tangents) {
            const std::vector<float> own = TangentsOf(
                mesh.data.points, mesh.data.indices, mesh.data.uvs);
            tangentPool.insert(tangentPool.end(), own.begin(),
                               own.end());
        } else if (tangents) {
            tangentPool.resize(tangentPool.size() + count, 0.0f);
        }
        if (mesh.data.normals.empty()) {
            normals.resize(normals.size() + count, 0.0f);
        } else {
            normals.insert(normals.end(), mesh.data.normals.begin(),
                           mesh.data.normals.end());
        }
        if (mesh.data.colors.empty()) {
            colors.resize(colors.size() + count, 0.0f);
        } else {
            colors.insert(colors.end(), mesh.data.colors.begin(),
                          mesh.data.colors.end());
        }
        // Device temperatures are copied in after.
        if (mesh.temperatures.empty()) {
            temperatures.resize(temperatures.size() + count / 3,
                                0.0f);
        } else {
            temperatures.insert(temperatures.end(),
                                mesh.temperatures.begin(),
                                mesh.temperatures.end());
            mesh.temperaturesDirty = false;
        }
        // Its nodes per point, and room for their temperatures
        // (device ones copied in after).
        if (mesh.nodeIds.empty()) {
            nodeIds.resize(nodeIds.size() + count / 3, 0u);
        } else {
            nodeIds.insert(nodeIds.end(), mesh.nodeIds.begin(),
                           mesh.nodeIds.end());
        }
        mesh.firstNode =
            static_cast<std::uint32_t>(nodeTemperatures.size());
        if (mesh.nodeTemperatures.empty()) {
            nodeTemperatures.resize(
                nodeTemperatures.size() + mesh.nodeCount, 0.0f);
        } else {
            nodeTemperatures.insert(nodeTemperatures.end(),
                                    mesh.nodeTemperatures.begin(),
                                    mesh.nodeTemperatures.end());
            mesh.nodeTemperaturesDirty = false;
        }
        indices.insert(indices.end(), mesh.data.indices.begin(),
                       mesh.data.indices.end());
        indices.insert(indices.end(), mesh.lodIndices.begin(),
                       mesh.lodIndices.end());
        mesh.firstAdjacency =
            static_cast<std::uint32_t>(adjacency.size());
        adjacency.insert(adjacency.end(), mesh.adjacency.begin(),
                         mesh.adjacency.end());
        // Three sums per point and its largest triangle's.
        if (mesh.topologyNormals) {
            mesh.firstNormalSum =
                static_cast<std::uint32_t>(normalSums);
            normalSums += mesh.data.points.size() + 1;
            topologyMeshes_.push_back(handle);
        }
        mesh.pointsDirty = false;
    }
    vkDeviceWaitIdle(context_.Device());
    // Both halves start with the points; each mesh's are in half 0
    // and stand still.
    pointTotal_ = static_cast<std::uint32_t>(points.size() / 3);
    points.insert(points.end(), points.begin(), points.end());
    for (auto& [handle, mesh] : meshes_) {
        mesh.pointSet = 0;
        mesh.pointsMoving = false;
        mesh.pointsWritten = false;
        // Points mapped into the old pool copy across from there.
        mesh.pointsInPlace = false;
        mesh.mappedPoints = Tensor();
        mesh.indicesInPlace = false;
        mesh.mappedIndices = Tensor();
    }
    activeCounts_ = Pool(std::as_bytes(std::span(counts)));
    // Each slot's draw, as many triangles as its mesh holds until the
    // counts are read (`RecordActiveDraws`).
    countSlots_ = static_cast<std::uint32_t>(counts.size() / 2);
    std::vector<std::uint32_t> draws(8 * std::size_t{countSlots_}, 0);
    for (const auto& [handle, mesh] : meshes_) {
        if (mesh.countSlot != 0) {
            const auto triangles = static_cast<std::uint32_t>(
                mesh.data.indices.size() / 3);
            std::uint32_t* draw = &draws[8 * (mesh.countSlot - 1)];
            draw[0] = 3 * triangles;
            draw[1] = 1;
            draw[4] = triangles;
        }
    }
    activeDraws_ = std::make_unique<GpuBuffer>(
        context_, std::max<VkDeviceSize>(draws.size() * 4, 32),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        BufferMemory::Device);
    if (!draws.empty()) {
        activeDraws_->Write(std::as_bytes(std::span(draws)));
    }
    points_ = Pool(std::as_bytes(std::span(points)));
    normals_ = Pool(std::as_bytes(std::span(normals)));
    tangents_ =
        tangents ? StorageBuffer(context_, tangentPool) : nullptr;
    colors_ = Pool(std::as_bytes(std::span(colors)));
    temperatures_ = Pool(std::as_bytes(std::span(temperatures)));
    nodeIds_ = Pool(std::as_bytes(std::span(nodeIds)));
    nodeTemperatures_ =
        Pool(std::as_bytes(std::span(nodeTemperatures)));
    // Shared, as triangles may be copied in from the compute device.
    indices_ = Pool(std::as_bytes(std::span(indices)));
    uvs_ = StorageBuffer(context_, uvs);
    cornerNormals_ = StorageBuffer(context_, cornerNormals);
    cornerColors_ = StorageBuffer(context_, cornerColors);
    adjacency_ = StorageBuffer(context_, adjacency);
    // Cleared on the GPU before each use.
    normalSums_ = std::make_unique<GpuBuffer>(
        context_,
        std::max<VkDeviceSize>(normalSums * sizeof(std::int32_t), 16),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    pendingNormals_.clear();
    pendingSkins_.clear();
    for (auto& [handle, mesh] : meshes_) {
        if (compute_ != nullptr) {
            if (!mesh.devicePoints.Empty()) {
                CopyDevicePoints(mesh);
            }
            CopyDeviceAttributes(mesh, true);
        }
        // Skinned meshes posed again over their rest points.
        if (mesh.skin && !mesh.skinPoseData.empty()) {
            renderer_.Uploads().Write(
                *mesh.skinPose,
                std::as_bytes(std::span(mesh.skinPoseData)));
            pendingSkins_.push_back(handle);
            mesh.pointsDirty = false;
        }
        if (mesh.recomputeNormals) {
            pendingNormals_.push_back(handle);
        }
    }
    geometryDirty_ = false;
}

void MeshDrawable::RebuildInstances() {
    ERNIE_ZONE("Mesh rebuild instances");
    ++layoutVersion_;
    ForgetBounds();
    std::vector<float> positions;
    std::vector<float> orientations;
    std::vector<float> scales;
    std::vector<float> colors;
    std::vector<float> temperatures;
    std::vector<float> visible;
    for (auto& [handle, instancer] : instancers_) {
        instancer.first =
            static_cast<std::uint32_t>(positions.size() / 3);
        const std::size_t count = instancer.count;
        const InstanceData& data = instancer.data;
        const bool host = !instancer.onDevice;
        if (host && data.positions.size() == 3 * count) {
            positions.insert(positions.end(), data.positions.begin(),
                             data.positions.end());
        } else {
            positions.resize(positions.size() + 3 * count, 0.0f);
        }
        if (host && data.orientations.size() == 4 * count) {
            orientations.insert(orientations.end(),
                                data.orientations.begin(),
                                data.orientations.end());
        } else {
            orientations.resize(orientations.size() + 4 * count,
                                0.0f);
        }
        if (host && data.scales.size() == 3 * count) {
            scales.insert(scales.end(), data.scales.begin(),
                          data.scales.end());
        } else {
            scales.resize(scales.size() + 3 * count, 1.0f);
        }
        if (host && data.colors.size() == 3 * count) {
            colors.insert(colors.end(), data.colors.begin(),
                          data.colors.end());
        } else {
            colors.resize(colors.size() + 3 * count, 1.0f);
        }
        if (host && data.temperatures.size() == count) {
            temperatures.insert(temperatures.end(),
                                data.temperatures.begin(),
                                data.temperatures.end());
        } else {
            temperatures.resize(temperatures.size() + count, 0.0f);
        }
        if (host && data.visible.size() == count) {
            visible.insert(visible.end(), data.visible.begin(),
                           data.visible.end());
        } else {
            visible.resize(visible.size() + count, 1.0f);
        }
    }
    vkDeviceWaitIdle(context_.Device());
    // Both halves start with the arrays; each instancer's are in half
    // 0 and stand still.
    instanceTotal_ = static_cast<std::uint32_t>(positions.size() / 3);
    for (std::vector<float>* array :
         {&positions, &orientations, &scales, &colors, &temperatures,
          &visible}) {
        array->insert(array->end(), array->begin(), array->end());
    }
    positions_ = Pool(std::as_bytes(std::span(positions)));
    orientations_ = Pool(std::as_bytes(std::span(orientations)));
    scales_ = Pool(std::as_bytes(std::span(scales)));
    instanceColors_ = Pool(std::as_bytes(std::span(colors)));
    instanceTemperatures_ = Pool(std::as_bytes(std::span(temperatures)));
    instanceVisible_ = Pool(std::as_bytes(std::span(visible)));
    for (auto& [handle, instancer] : instancers_) {
        instancer.set = 0;
        instancer.moving = false;
        instancer.written = false;
        // Arrays mapped into the old pools copy across from there.
        instancer.inPlace = {};
        instancer.mapped = {};
        if (instancer.sampled) {
            pendingBlends_.push_back(handle);
        } else if (instancer.onDevice) {
            WriteInstancer(instancer, false);
        }
        instancer.dirty = false;
    }
    instancesDirty_ = false;
}

void MeshDrawable::RebuildCompose() {
    ERNIE_ZONE("Mesh rebuild compose");
    ForgetBounds();
    std::uint32_t raw = 0;
    std::uint32_t records = 0;
    std::vector<glm::uvec4> ops;
    for (auto& [handle, instancer] : instancers_) {
        if (!instancer.composed) {
            continue;
        }
        instancer.rawFirst = raw;
        instancer.opsFirst = static_cast<std::uint32_t>(ops.size());
        instancer.parentsFirst = records;
        std::uint32_t offset = 0;
        for (const DeviceInstanceOp& op :
             instancer.composedData.ops) {
            const auto width =
                static_cast<std::uint32_t>(op.rows.Shape()[1]);
            ops.emplace_back(static_cast<std::uint32_t>(op.kind),
                             offset, width, op.wFirst ? 1u : 0u);
            offset +=
                static_cast<std::uint32_t>(instancer.count) * width;
        }
        raw += offset;
        records += static_cast<std::uint32_t>(instancer.count);
        instancer.levels.assign(instancer.count, 0);
        // Its arrays move to their new place in the raw pool.
        instancer.dirty = true;
    }

    // Levels: an instance posed under a composed one is one deeper,
    // so its parent is composed first. A chain deeper than the passes
    // allow is a cycle.
    constexpr int kMaxDepth = 64;
    bool changed = true;
    for (int pass = 0; changed; ++pass) {
        if (pass > kMaxDepth) {
            throw std::invalid_argument("Composed instances' parents "
                                        "form a cycle or nest deeper "
                                        "than " +
                                        std::to_string(kMaxDepth));
        }
        changed = false;
        for (auto& [handle, instancer] : instancers_) {
            if (!instancer.composed) {
                continue;
            }
            const std::vector<InstanceParent>& parents =
                instancer.composedData.parents;
            for (std::size_t i = 0; i < parents.size(); ++i) {
                const auto parent =
                    instancers_.find(parents[i].instancer);
                if (parent == instancers_.end() ||
                    !parent->second.composed ||
                    parents[i].row >= parent->second.count) {
                    continue;
                }
                const std::uint32_t level =
                    parent->second.levels[parents[i].row] + 1;
                if (instancer.levels[i] < level) {
                    instancer.levels[i] = level;
                    changed = true;
                }
            }
        }
    }

    // Each instancer's instances per level, the dispatches in level
    // order.
    std::vector<std::uint32_t> lists;
    composeJobs_.clear();
    for (const auto& [handle, instancer] : instancers_) {
        if (!instancer.composed) {
            continue;
        }
        std::map<std::uint32_t, std::vector<std::uint32_t>> byLevel;
        for (std::uint32_t i = 0; i < instancer.count; ++i) {
            byLevel[instancer.levels[i]].push_back(i);
        }
        for (const auto& [level, instances] : byLevel) {
            composeJobs_.push_back(
                {handle, static_cast<std::uint32_t>(lists.size()),
                 static_cast<std::uint32_t>(instances.size()),
                 level});
            lists.insert(lists.end(), instances.begin(),
                         instances.end());
        }
    }
    std::stable_sort(composeJobs_.begin(), composeJobs_.end(),
                     [](const ComposeJob& a, const ComposeJob& b) {
                         return a.level < b.level;
                     });

    const VkDeviceSize rawBytes =
        std::max<VkDeviceSize>(VkDeviceSize{raw} * sizeof(float), 16);
    if (composeRaw_ == nullptr || composeRaw_->Size() < rawBytes) {
        vkDeviceWaitIdle(context_.Device());
        composeRaw_ = std::make_shared<GpuBuffer>(
            context_, rawBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            BufferMemory::Shared);
    }
    const auto upload = [&](std::unique_ptr<GpuBuffer>& buffer,
                            std::span<const std::byte> bytes) {
        Reserve(context_, buffer, bytes.size(),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Device);
        if (!bytes.empty()) {
            renderer_.Uploads().Write(*buffer, bytes);
        }
    };
    upload(composeOps_, std::as_bytes(std::span(ops)));
    upload(composeLists_, std::as_bytes(std::span(lists)));
    Reserve(context_, composeParents_,
            VkDeviceSize{records} * 3 * sizeof(glm::vec4),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    composeLayoutDirty_ = false;
}

void MeshDrawable::PrepareCompose(bool all) {
    ERNIE_ZONE("Mesh prepare compose");
    pendingCompose_.clear();
    if (composeJobs_.empty()) {
        return;
    }
    // Instances posed under ones that move this frame move with them:
    // composed again, into the other half, keeping the last frame's
    // poses for motion blur.
    for (bool changed = true; changed;) {
        changed = false;
        for (auto& [handle, instancer] : instancers_) {
            if (!instancer.composed || instancer.written) {
                continue;
            }
            for (const InstanceParent& parent :
                 instancer.composedData.parents) {
                const auto it = instancers_.find(parent.instancer);
                if (it != instancers_.end() && it->second.written) {
                    FlipInstancer(instancer);
                    changed = true;
                    break;
                }
            }
        }
    }
    for (const ComposeJob& job : composeJobs_) {
        if (all || instancers_.at(job.instancer).written) {
            pendingCompose_.push_back(job);
        }
    }
    if (pendingCompose_.empty()) {
        return;
    }
    // Every instance's parent record, with its parent's slot in this
    // frame's half.
    std::vector<glm::vec4> records;
    for (const auto& [handle, instancer] : instancers_) {
        if (!instancer.composed) {
            continue;
        }
        const std::vector<InstanceParent>& parents =
            instancer.composedData.parents;
        for (std::size_t i = 0; i < instancer.count; ++i) {
            std::uint32_t slot = 0xFFFFFFFFu;
            InstanceParent parent;
            if (!parents.empty()) {
                parent = parents[i];
                const auto it = instancers_.find(parent.instancer);
                if (it != instancers_.end() &&
                    parent.row < it->second.count) {
                    slot = InstanceBase(it->second) +
                           it->second.first + parent.row;
                }
            }
            records.emplace_back(std::bit_cast<float>(slot),
                                 parent.translate);
            records.push_back(parent.rotation);
            records.emplace_back(parent.scale, 0.0f);
        }
    }
    renderer_.Uploads().Write(*composeParents_,
                              std::as_bytes(std::span(records)));
}

void MeshDrawable::RecordCompose(VkCommandBuffer commands) const {
    if (pendingCompose_.empty()) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Instance compose");
    const auto barrier = [commands](VkPipelineStageFlags2 dstStage) {
        VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memory.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        memory.dstStageMask = dstStage;
        memory.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        VkDependencyInfo dependency{
            VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &memory;
        vkCmdPipelineBarrier2(commands, &dependency);
    };
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      composePipeline_->Handle());
    const DescriptorResource resources[] = {
        DescriptorResource::Of(*composeRaw_),
        DescriptorResource::Of(*composeOps_),
        DescriptorResource::Of(*composeLists_),
        DescriptorResource::Of(*composeParents_),
        DescriptorResource::Of(*positions_),
        DescriptorResource::Of(*orientations_),
        DescriptorResource::Of(*scales_)};
    composeLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                        resources);
    std::uint32_t level = pendingCompose_.front().level;
    for (const ComposeJob& job : pendingCompose_) {
        if (job.level != level) {
            // Parents' poses before their children read them.
            barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            level = job.level;
        }
        const Instancer& instancer = instancers_.at(job.instancer);
        const std::uint32_t constants[8] = {
            job.count,
            job.list,
            instancer.rawFirst,
            instancer.opsFirst,
            static_cast<std::uint32_t>(
                instancer.composedData.ops.size()),
            InstanceBase(instancer) + instancer.first,
            instancer.parentsFirst,
            static_cast<std::uint32_t>(instancer.count)};
        composeLayout_.PushConstants(commands, constants,
                                     sizeof(constants));
        vkCmdDispatch(commands, (job.count + 63) / 64, 1, 1);
    }
    barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
}

void MeshDrawable::FlipInstancer(Instancer& instancer) {
    if (!instancer.written) {
        instancer.set ^= 1u;
        instancer.written = true;
        instancer.moving = true;
        instancer.rebase = true;
    }
}

void MeshDrawable::WriteInstancer(Instancer& instancer, bool flip) {
    if (flip) {
        FlipInstancer(instancer);
    }
    const VkDeviceSize first =
        VkDeviceSize{InstanceBase(instancer)} + instancer.first;
    if (instancer.composed) {
        // The ops' raw arrays; the compose pass writes the pools from
        // them (`RecordCompose`). Rows written in place
        // (`MapComposedOp`) are there already; the frame waits for
        // their stream.
        const ComposedInstanceData& data = instancer.composedData;
        VkDeviceSize offset = instancer.rawFirst;
        bool inPlace = false;
        for (const DeviceInstanceOp& op : data.ops) {
            const Tensor raw =
                compute_->View(composeRaw_, DType::Float32, {1},
                               offset * sizeof(float));
            if (op.rows.Data() == raw.Data()) {
                inPlace = true;
            } else {
                compute_->CopyToBuffer(op.rows, *composeRaw_,
                                       offset * sizeof(float),
                                       data.stream);
            }
            offset += static_cast<VkDeviceSize>(op.rows.Elements());
        }
        if (inPlace) {
            compute_->SignalVulkan(data.stream);
        }
        return;
    }
    if (instancer.onDevice) {
        // Arrays written in place (`MapInstances`) are there already;
        // the frame waits for their stream.
        const DeviceInstanceData& device = instancer.device;
        if (std::any_of(instancer.inPlace.begin(),
                        instancer.inPlace.end(),
                        [](bool inPlace) { return inPlace; })) {
            compute_->SignalVulkan(device.stream);
        }
        if (!instancer.inPlace[0]) {
            compute_->CopyToBuffer(device.positions, *positions_,
                                   first * 3 * sizeof(float),
                                   device.stream);
        }
        if (!device.orientations.Empty() && !instancer.inPlace[1]) {
            if (device.orientationsWFirst) {
                // w, x, y, z into the pool's x, y, z, w.
                constexpr std::array<std::size_t, 4> kXyzwOfWxyz{
                    1, 2, 3, 0};
                compute_->CopyColumnsToBuffer(
                    device.orientations, kXyzwOfWxyz, *orientations_,
                    first * 4 * sizeof(float), device.stream);
            } else {
                compute_->CopyToBuffer(
                    device.orientations, *orientations_,
                    first * 4 * sizeof(float), device.stream);
            }
        }
        if (!device.scales.Empty() && !instancer.inPlace[2]) {
            compute_->CopyToBuffer(device.scales, *scales_,
                                   first * 3 * sizeof(float),
                                   device.stream);
        }
        if (!device.colors.Empty() && !instancer.inPlace[3]) {
            compute_->CopyToBuffer(device.colors, *instanceColors_,
                                   first * 3 * sizeof(float),
                                   device.stream);
        }
        if (!device.temperatures.Empty()) {
            compute_->CopyToBuffer(device.temperatures,
                                   *instanceTemperatures_,
                                   first * sizeof(float), device.stream);
        }
        if (!device.visible.Empty()) {
            compute_->CopyToBuffer(device.visible, *instanceVisible_,
                                   first * sizeof(float), device.stream);
        }
        instancer.inPlace = {};
        return;
    }
    const InstanceData& data = instancer.data;
    FrameUploads& uploads = renderer_.Uploads();
    uploads.Write(*positions_,
                  std::as_bytes(std::span(data.positions)),
                  first * 3 * sizeof(float));
    if (!data.orientations.empty()) {
        uploads.Write(*orientations_,
                      std::as_bytes(std::span(data.orientations)),
                      first * 4 * sizeof(float));
    }
    if (!data.scales.empty()) {
        uploads.Write(*scales_, std::as_bytes(std::span(data.scales)),
                      first * 3 * sizeof(float));
    }
    if (!data.colors.empty()) {
        uploads.Write(*instanceColors_,
                      std::as_bytes(std::span(data.colors)),
                      first * 3 * sizeof(float));
    }
    if (!data.temperatures.empty()) {
        uploads.Write(*instanceTemperatures_,
                      std::as_bytes(std::span(data.temperatures)),
                      first * sizeof(float));
    }
    if (!data.visible.empty()) {
        uploads.Write(*instanceVisible_,
                      std::as_bytes(std::span(data.visible)),
                      first * sizeof(float));
    }
}

void MeshDrawable::Prepare(const FrameParams& params,
                           std::span<const ShadowCascade> cascades) {
    ERNIE_ZONE("Mesh prepare");
    // Device instancers' bounds measured frames ago, in place of
    // their positions copied to the host.
    ReadBounds(renderer_.Uploads().Slot());
    // A pushbroom camera's views draw with the pipeline compiled for
    // it.
    const bool pushbroom = params.camera.pushbroom;
    if (gbufferPipelines_[pushbroom ? 1 : 0] == nullptr) {
        CreateGBufferPipeline(pushbroom);
    }
    if (pushbroom != pushbroom_) {
        pushbroom_ = pushbroom;
        std::fill(itemPipelines_.begin(), itemPipelines_.end(),
                  gbufferPipelines_[pushbroom ? 1 : 0].get());
    }
    // Minimal shading draws with its own pipelines, compiled the
    // first time a frame asks.
    minimal_ = params.minimal;
    if (minimal_ != MinimalMode::Off) {
        CreateMinimalPipelines();
    }
    // Cutout materials draw with their own pipelines, compiled the
    // first time one appears.
    // What the items' materials ask (cutout, transparent, tangents),
    // found again only when the items or the materials change.
    const std::uint64_t materials = renderer_.MaterialsVersion();
    const bool scan = scannedLayout_ != layoutVersion_ ||
                      scannedMaterials_ != materials;
    if (scan) {
        ERNIE_ZONE("Mesh scan materials");
        scannedLayout_ = layoutVersion_;
        scannedMaterials_ = materials;
        anyCutout_ = false;
        anyTransparent_ = false;
        itemCutouts_.assign(items_.size(), false);
        for (std::size_t i = 0; i < items_.size(); ++i) {
            itemCutouts_[i] = Cutout(i);
            anyCutout_ = anyCutout_ || itemCutouts_[i];
            anyTransparent_ = anyTransparent_ || Transparent(i);
        }
    }
    if (anyCutout_) {
        CreateCutoutPipelines(pushbroom);
    }
    // Transparent materials draw in their own phase, when shaded.
    drawsTransparent_ = anyTransparent_ && minimal_ == MinimalMode::Off;
    viewWorlds_.clear();
    for (std::size_t v = 0; v < params.Views(); ++v) {
        viewWorlds_.push_back(params.ViewWorld(v));
    }
    if (params.thermal.enable != thermal_) {
        thermal_ = params.thermal.enable;
        drawsDirty_ = true;
    }
    // Meshes an anisotropic material shades take smooth tangents,
    // made with the geometry when that changes.
    if (scan) {
        std::set<std::uint32_t> wanted;
        for (const MeshItem& item : items_) {
            if (renderer_.MaterialReadsTangents(item.material)) {
                wanted.insert(item.mesh);
            }
        }
        for (auto& [handle, mesh] : meshes_) {
            const bool wants = wanted.contains(handle);
            if (wants != mesh.wantsTangents) {
                mesh.wantsTangents = wants;
                geometryDirty_ = true;
            }
        }
    }
    // The per-draw data changes only with the draws, meshes or
    // instancers, not every frame.
    bool drawsChanged =
        drawsDirty_ || geometryDirty_ || instancesDirty_ ||
        renderer_.MaterialSetVersion() != drawsMaterials_ ||
        drawBytes_.size() != items_.size() * sizeof(GpuDraw);
    // Instancers given new arrays alone (a simulation's poses every
    // step) only flip their halves: their draws' bases are patched
    // (`PatchInstanceBases`), not every draw written again.
    std::vector<std::uint32_t> flipped;
    for (const auto& [handle, instancer] : instancers_) {
        if (!instancer.dirty && !instancer.rebase) {
            continue;
        }
        if (instancer.drawnLayout == LayoutOf(instancer)) {
            flipped.push_back(handle);
        } else {
            drawsChanged = true;
        }
    }
    if (geometryDirty_) {
        RebuildGeometry();
    } else {
        // Only meshes whose points move recompute their normals.
        ERNIE_ZONE("Mesh active uploads");
        ERNIE_PLOT("Active meshes",
                   static_cast<std::int64_t>(activeMeshes_.size()));
        pendingNormals_.clear();
        pendingSkins_.clear();
        for (const std::uint32_t handle : activeMeshes_) {
            Mesh& mesh = meshes_.at(handle);
            if (compute_ != nullptr) {
                CopyDeviceAttributes(mesh, false);
            }
            // Host temperatures, only this mesh's range.
            if (mesh.temperaturesDirty &&
                !mesh.temperatures.empty()) {
                ERNIE_ZONE("Mesh thermal upload");
                renderer_.Uploads().Write(
                    *temperatures_,
                    std::as_bytes(std::span(mesh.temperatures)),
                    VkDeviceSize{mesh.firstVertex} * sizeof(float));
                mesh.temperaturesDirty = false;
            }
            if (mesh.nodeTemperaturesDirty &&
                !mesh.nodeTemperatures.empty()) {
                ERNIE_ZONE("Mesh thermal node upload");
                renderer_.Uploads().Write(
                    *nodeTemperatures_,
                    std::as_bytes(std::span(mesh.nodeTemperatures)),
                    VkDeviceSize{mesh.firstNode} * sizeof(float));
                mesh.nodeTemperaturesDirty = false;
            }
            // Written ones are written in place each frame.
            if (!mesh.pointsDirty || mesh.writer) {
                mesh.pointsDirty = false;
                continue;
            }
            // Into the other half, keeping the last frame's points
            // for motion blur.
            FlipPoints(mesh);
            if (mesh.skin) {
                // Posed on the GPU from this frame's pose.
                renderer_.Uploads().Write(
                    *mesh.skinPose,
                    std::as_bytes(std::span(mesh.skinPoseData)));
                pendingSkins_.push_back(handle);
            } else if (mesh.pointsInPlace) {
                // Written there through `MapPoints`: the frame waits
                // for their stream.
                compute_->SignalVulkan(mesh.deviceStream);
                mesh.pointsInPlace = false;
            } else if (!mesh.devicePoints.Empty()) {
                CopyDevicePoints(mesh);
            } else {
                renderer_.Uploads().Write(
                    *points_,
                    std::as_bytes(std::span(mesh.data.points)),
                    VkDeviceSize{PointBase(mesh) + mesh.firstVertex} *
                        3 * sizeof(float));
            }
            if (mesh.recomputeNormals) {
                pendingNormals_.push_back(handle);
            }
            mesh.pointsDirty = false;
        }
    }
    if (composeLayoutDirty_) {
        RebuildCompose();
        drawsChanged = true;
    }
    const bool rebuildInstances = instancesDirty_;
    pendingBlends_.clear();
    if (instancesDirty_) {
        RebuildInstances();
    } else {
        for (auto& [handle, instancer] : instancers_) {
            if (instancer.dirty) {
                // Sampled ones blend on the GPU into the other half.
                if (instancer.sampled) {
                    FlipInstancer(instancer);
                    pendingBlends_.push_back(handle);
                } else {
                    WriteInstancer(instancer, true);
                }
                instancer.dirty = false;
            }
        }
    }
    // Rebuilt pools hold no composed poses yet: compose every
    // instancer.
    PrepareCompose(rebuildInstances);
    // Views posed under instances: their records, relative to the
    // frame's origin, with their anchors' slots in this frame's
    // halves.
    anchorViews_ = 0;
    if (params.posesOnGpu && !anchoredViews_.empty()) {
        const glm::dmat4 toOrigin =
            glm::translate(glm::dmat4(1.0), -params.origin);
        std::vector<glm::vec4> records;
        records.reserve(anchoredViews_.size() * 9);
        for (std::size_t v = 0; v < anchoredViews_.size(); ++v) {
            const AnchoredView& view = anchoredViews_[v];
            AppendAnchorRecord(records,
                               glm::mat4(toOrigin * view.frame),
                               glm::mat4(view.below),
                               AnchorSlot(view.instancer, view.row),
                               static_cast<std::uint32_t>(v),
                               AnchorOps(view.instancer));
        }
        Reserve(context_, anchorRecords_,
                records.size() * sizeof(glm::vec4),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Device);
        renderer_.Uploads().Write(*anchorRecords_,
                                  std::as_bytes(std::span(records)));
        anchorViews_ =
            static_cast<std::uint32_t>(anchoredViews_.size());
    }
    // Lights posed under instances, as the views.
    anchorLights_ = 0;
    if (!anchoredLights_.empty()) {
        const glm::dmat4 toOrigin =
            glm::translate(glm::dmat4(1.0), -params.origin);
        std::vector<glm::vec4> records;
        records.reserve(anchoredLights_.size() * 9);
        for (std::size_t l = 0; l < anchoredLights_.size(); ++l) {
            const AnchoredView& light = anchoredLights_[l];
            AppendAnchorRecord(records,
                               glm::mat4(toOrigin * light.frame),
                               glm::mat4(light.below),
                               AnchorSlot(light.instancer, light.row),
                               static_cast<std::uint32_t>(l),
                               AnchorOps(light.instancer));
        }
        Reserve(context_, anchorLightRecords_,
                records.size() * sizeof(glm::vec4),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Device);
        renderer_.Uploads().Write(*anchorLightRecords_,
                                  std::as_bytes(std::span(records)));
        anchorLights_ =
            static_cast<std::uint32_t>(anchoredLights_.size());
    }

    // Flipped halves move the draws' offsets.
    if (drawsChanged || drawsDirty_) {
        WriteDraws();
        drawsDirty_ = false;
        drawsMaterials_ = renderer_.MaterialSetVersion();
    } else if (!flipped.empty() || !posedItems_.empty()) {
        PatchInstanceBases(flipped);
        PatchItems();
        UploadPatched();
    }
    posedItems_.clear();
    for (auto& [handle, instancer] : instancers_) {
        instancer.rebase = false;
    }
    WriteBoundsJobs();
    Cull(params, cascades);
    // Last, once every buffer the passes read is in place.
    WriteResources();
}

void MeshDrawable::WriteDraws() {
    ERNIE_ZONE("Mesh write draws");
    cullItemsDirty_ = true;
    ++drawsWritten_;
    itemIdOffsets_.clear();
    itemIndices_.clear();
    itemCountSlots_.clear();
    std::vector<GpuDraw> draws(items_.size());
    std::vector<std::uint32_t> instanceIds;
    itemPipelines_.clear();
    itemMaterials_.clear();
    for (std::size_t i = 0; i < items_.size(); ++i) {
        const MeshItem& item = items_[i];
        const Mesh& mesh = meshes_.at(item.mesh);
        itemPipelines_.push_back(
            gbufferPipelines_[pushbroom_ ? 1 : 0].get());
        itemMaterials_.push_back(
            renderer_.HasMaterial(item.material)
                ? item.material
                : RasterRenderer::kDefaultMaterial);
        GpuDraw& draw = draws[i];
        PoseDraw(draw, transforms_[i], previousTransforms_[i],
                 item.prototype, item.tint,
                 thermal_ ? item.temperature : 0.0f);
        draw.offsets[0] = mesh.firstIndex;
        draw.offsets[1] = mesh.firstVertex;
        draw.offsets[2] =
            static_cast<std::uint32_t>(instanceIds.size());
        itemIdOffsets_.push_back(draw.offsets[2]);
        itemIndices_.push_back(
            static_cast<std::uint32_t>(mesh.data.indices.size()));
        itemCountSlots_.push_back(mesh.countSlot);
        std::uint32_t flags = 0;
        flags |= mesh.data.normals.empty() ? 0u : kHasNormals;
        flags |= item.vertexColors && !mesh.data.colors.empty()
                     ? kHasColors
                     : 0u;
        flags |= mesh.data.uvs.empty() ? 0u : kHasUvs;
        flags |=
            mesh.data.cornerNormals.empty() ? 0u : kHasCornerNormals;
        flags |= item.vertexColors && !mesh.data.cornerColors.empty()
                     ? kHasCornerColors
                     : 0u;
        flags |= mesh.tangents ? kHasTangents : 0u;
        flags |= thermal_ ? kThermal : 0u;
        flags |= mesh.hasTemperatures ? kHasTemperatures : 0u;
        flags |= !mesh.nodeIds.empty() && mesh.hasNodeTemperatures
                     ? kHasNodes
                     : 0u;
        draw.corners[0] = mesh.firstCornerNormal;
        draw.corners[1] = mesh.firstCornerColor;
        draw.uv[0] = mesh.firstUv;
        draw.uv[1] = mesh.countSlot;
        draw.uv[2] =
            static_cast<std::uint32_t>(mesh.data.points.size() / 3);
        draw.uv[3] = mesh.firstNode;
        if (item.instancer) {
            const Instancer& instancer =
                instancers_.at(*item.instancer);
            flags |= kInstanced;
            const bool orientations =
                instancer.composed ||
                (instancer.onDevice
                     ? !instancer.device.orientations.Empty()
                     : !instancer.data.orientations.empty());
            const bool scales =
                instancer.composed ||
                (instancer.onDevice ? !instancer.device.scales.Empty()
                                    : !instancer.data.scales.empty());
            flags |= orientations ? kHasOrientations : 0u;
            flags |= scales ? kHasScales : 0u;
            // As a prototype's own colours: not under a bound
            // material.
            flags |= item.vertexColors && HasInstanceColors(instancer)
                         ? kHasInstanceColors
                         : 0u;
            flags |= HasInstanceTemperatures(instancer)
                         ? kHasInstanceTemperatures
                         : 0u;
            flags |= HasInstanceVisibility(instancer) ? kHasInstanceVisibility
                                                      : 0u;
            const std::uint32_t first = instancer.first;
            draw.points[3] = first;
            for (const std::uint32_t instance : item.instances) {
                instanceIds.push_back(first + instance);
            }
            draw.motion[0] = InstanceBase(instancer);
            draw.motion[1] = InstanceBase(instancer, true);
        }
        draw.motion[2] = PointBase(mesh);
        draw.motion[3] = PointBase(mesh, true);
        draw.offsets[3] = flags | (item.linkGroup & 31u)
                                      << kLinkGroupShift;
        draw.points[0] = mesh.firstVertex;
        draw.points[1] = renderer_.HasMaterial(item.material)
                             ? item.material
                             : RasterRenderer::kDefaultMaterial;
        draw.points[2] = item.primId;
    }
    const auto bytes = [](const auto& data) {
        return std::max<VkDeviceSize>(data.size() * sizeof(data[0]),
                                      16);
    };
    if (draws_->Size() < bytes(draws)) {
        vkDeviceWaitIdle(context_.Device());
        draws_ = StorageBuffer(context_, draws);
    } else if (!draws.empty()) {
        renderer_.Uploads().Write(*draws_,
                                  std::as_bytes(std::span(draws)));
    }
    const std::span written = std::as_bytes(std::span(draws));
    drawBytes_.assign(written.begin(), written.end());
    for (auto& [handle, instancer] : instancers_) {
        instancer.drawnLayout = LayoutOf(instancer);
    }
    if (instanceIds_->Size() < bytes(instanceIds)) {
        vkDeviceWaitIdle(context_.Device());
        instanceIds_ = StorageBuffer(context_, instanceIds);
    } else if (!instanceIds.empty()) {
        renderer_.Uploads().Write(
            *instanceIds_, std::as_bytes(std::span(instanceIds)));
    }
}

std::optional<std::pair<Bounds3, float>>
MeshDrawable::MeasuredPositions(const Instancer& instancer) {
    if (!instancer.gpuPositions) {
        return std::nullopt;
    }
    Bounds3 box = *instancer.gpuPositions;
    if (instancer.lastGpuPositions) {
        // Measured frames before this one: ahead by the motion
        // between the last two measurements.
        const Bounds3& last = *instancer.lastGpuPositions;
        box.first =
            glm::min(box.first, 2.0f * box.first - last.first);
        box.second =
            glm::max(box.second, 2.0f * box.second - last.second);
    }
    return std::pair(box, instancer.gpuScale);
}

void MeshDrawable::ReadBounds(std::uint32_t slot) {
    ERNIE_ZONE("Mesh read bounds");
    boundsSlot_ = slot;
    std::vector<std::pair<std::uint32_t, std::size_t>>& jobs =
        boundsJobs_[slot];
    if (!jobs.empty()) {
        // This slot's last frame has finished (`FrameUploads`).
        std::vector<std::uint32_t> words(8 * jobs.size());
        boundsReadback_[slot]->Read(
            std::as_writable_bytes(std::span(words)));
        // instance_bounds.slang's `ordered`, undone.
        const auto value = [](std::uint32_t u) {
            return std::bit_cast<float>(
                (u & 0x80000000u) != 0 ? u & 0x7fffffffu : ~u);
        };
        for (std::size_t j = 0; j < jobs.size(); ++j) {
            const auto it = instancers_.find(jobs[j].first);
            if (it == instancers_.end() || !it->second.onDevice ||
                it->second.count != jobs[j].second) {
                continue;
            }
            const std::uint32_t* w = &words[8 * j];
            Instancer& instancer = it->second;
            instancer.lastGpuPositions = instancer.gpuPositions;
            instancer.gpuPositions = Bounds3(
                glm::vec3(value(w[0]), value(w[1]), value(w[2])),
                glm::vec3(value(w[3]), value(w[4]), value(w[5])));
            instancer.gpuScale = value(w[6]);
        }
        // Meshes' measured points (in use) bound them, ahead by their
        // motion as instancers'.
        for (std::size_t j = 0; j < jobs.size(); ++j) {
            const auto it = meshes_.find(jobs[j].first);
            if (it == meshes_.end() || !MeasuresPoints(it->second) ||
                it->second.data.points.size() / 3 != jobs[j].second) {
                continue;
            }
            const std::uint32_t* w = &words[8 * j];
            Mesh& mesh = it->second;
            const Bounds3 box(
                glm::vec3(value(w[0]), value(w[1]), value(w[2])),
                glm::vec3(value(w[3]), value(w[4]), value(w[5])));
            if (!(box.first.x <= box.second.x)) {
                continue;  // no points in use: the lows still empty
            }
            mesh.lastGpuPoints = mesh.gpuPoints;
            mesh.gpuPoints = box;
            std::tie(mesh.low, mesh.high) = box;
            if (mesh.lastGpuPoints) {
                mesh.low =
                    glm::min(mesh.low, 2.0f * box.first -
                                           mesh.lastGpuPoints->first);
                mesh.high = glm::max(mesh.high,
                                     2.0f * box.second -
                                         mesh.lastGpuPoints->second);
            }
        }
        ForgetBounds();
    }
    // This frame measures every device instancer without an extent,
    // and every mesh's device points without bounds.
    jobs.clear();
    for (const auto& [handle, instancer] : instancers_) {
        if (instancer.onDevice && !instancer.composed &&
            !instancer.device.extent && instancer.count > 0) {
            jobs.emplace_back(handle, instancer.count);
        }
    }
    for (const std::uint32_t handle : gpuPointMeshes_) {
        const Mesh& mesh = meshes_.at(handle);
        if (MeasuresPoints(mesh)) {
            jobs.emplace_back(handle, mesh.data.points.size() / 3);
        }
    }
    const VkDeviceSize bytes =
        VkDeviceSize{jobs.size()} * 8 * sizeof(std::uint32_t);
    if (!jobs.empty() && (boundsReadback_[slot] == nullptr ||
                          boundsReadback_[slot]->Size() < bytes)) {
        boundsReadback_[slot] = std::make_unique<GpuBuffer>(
            context_, std::max<VkDeviceSize>(2 * bytes, 256),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            BufferMemory::Readback);
    }
}

void MeshDrawable::WriteBoundsJobs() {
    ERNIE_ZONE("Mesh write bounds jobs");
    const auto& jobs = boundsJobs_[boundsSlot_];
    if (jobs.empty()) {
        return;
    }
    std::vector<GpuBoundsJob> records(jobs.size());
    std::size_t most = 1;
    for (std::size_t j = 0; j < jobs.size(); ++j) {
        GpuBoundsJob& record = records[j];
        record = {};
        record.out = static_cast<std::uint32_t>(8 * j);
        if (const auto instancer = instancers_.find(jobs[j].first);
            instancer != instancers_.end()) {
            record.first = InstanceBase(instancer->second) +
                           instancer->second.first;
            record.count =
                static_cast<std::uint32_t>(instancer->second.count);
            record.flags =
                instancer->second.device.scales.Empty() ? 0u : 1u;
        } else {
            const Mesh& mesh = meshes_.at(jobs[j].first);
            record.first = PointBase(mesh) + mesh.firstVertex;
            record.count = static_cast<std::uint32_t>(jobs[j].second);
            record.flags = 2u;
            record.countSlot = mesh.countSlot;
        }
        most = std::max<std::size_t>(most, record.count);
    }
    // A row of workgroups per job, as many as its largest needs, at
    // most 1024: past that each thread takes several positions.
    boundsGroups_ = static_cast<std::uint32_t>(
        std::min<std::size_t>((most + 63) / 64, 1024));
    Reserve(context_, boundsJobBuffer_,
            records.size() * sizeof(GpuBoundsJob),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    renderer_.Uploads().Write(*boundsJobBuffer_,
                              std::as_bytes(std::span(records)));
}

void MeshDrawable::RecordBounds(VkCommandBuffer commands) const {
    const auto& jobs = boundsJobs_[boundsSlot_];
    if (jobs.empty()) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Instance bounds");
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
        DescriptorResource::Of(*positions_),
        DescriptorResource::Of(*scales_),
        DescriptorResource::Of(*boundsReadback_[boundsSlot_]),
        DescriptorResource::Of(*activeCounts_),
        DescriptorResource::Of(*points_),
        DescriptorResource::Of(*boundsJobBuffer_)};
    const auto count = static_cast<std::uint32_t>(jobs.size());
    const std::uint32_t constants[4] = {count, boundsGroups_, 0, 0};
    // Every job's words emptied, then every job measured: two
    // dispatches however many jobs.
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      boundsClearPipeline_->Handle());
    boundsLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                       resources);
    boundsLayout_.PushConstants(commands, constants,
                                sizeof(constants));
    vkCmdDispatch(commands, (8 * count + 63) / 64, 1, 1);
    barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT |
                VK_ACCESS_2_SHADER_WRITE_BIT);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      boundsPipeline_->Handle());
    vkCmdDispatch(commands, boundsGroups_, count, 1);
    barrier(VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
}

std::optional<Bounds3> MeshDrawable::Bounds() const {
    if (bounds_) {
        return *bounds_;
    }
    std::optional<Bounds3>& bounds = bounds_.emplace();
    const auto add = [&bounds](
                         const std::pair<glm::vec3, glm::vec3>& box) {
        bounds = bounds
                     ? std::pair(glm::min(bounds->first, box.first),
                                 glm::max(bounds->second, box.second))
                     : box;
    };
    for (std::size_t n = 0; n < items_.size(); ++n) {
        const MeshItem& item = items_[n];
        const glm::mat4& transform = transforms_[n];
        if (!item.instancer) {
            const ItemBox& plain = PlainBox(n);
            if (plain.hostPoints) {
                add(plain.box);
            }
            continue;
        }
        const Mesh& mesh = meshes_.at(item.mesh);
        if (mesh.data.points.empty()) {
            continue;
        }
        const Instancer& instancer = instancers_.at(*item.instancer);
        const std::optional<Bounds3>& extent =
            instancer.device.extent;
        if (instancer.onDevice && extent) {
            add(TransformBox(transform, extent->first,
                             extent->second));
            continue;
        }
        const auto [low, high] =
            TransformBox(item.prototype, mesh.low, mesh.high);
        const float radius =
            glm::length(glm::max(glm::abs(low), glm::abs(high)));
        if (instancer.sampled) {
            const glm::vec3 reach(radius * instancer.sampledScale);
            add(TransformBox(
                transform, instancer.sampledPositions.first - reach,
                instancer.sampledPositions.second + reach));
            continue;
        }
        if (const auto measured = instancer.onDevice
                                      ? MeasuredPositions(instancer)
                                      : std::nullopt) {
            const glm::vec3 reach(radius * measured->second);
            add(TransformBox(transform, measured->first.first - reach,
                             measured->first.second + reach));
            continue;
        }
        const InstanceData& data = instancer.data;
        for (const std::uint32_t i : item.instances) {
            float scale = 1.0f;
            if (!data.scales.empty()) {
                scale = std::max({std::abs(data.scales[3 * i]),
                                  std::abs(data.scales[3 * i + 1]),
                                  std::abs(data.scales[3 * i + 2])});
            }
            const glm::vec3 p(data.positions[3 * i],
                              data.positions[3 * i + 1],
                              data.positions[3 * i + 2]);
            add(TransformBox(transform, p - glm::vec3(radius * scale),
                             p + glm::vec3(radius * scale)));
        }
    }
    return bounds;
}

std::map<std::int32_t, Bounds3> MeshDrawable::WorldsBounds() const {
    if (worldsBounds_) {
        return *worldsBounds_;
    }
    std::map<std::int32_t, Bounds3>& bounds = worldsBounds_.emplace();
    for (std::size_t i = 0;
         i < items_.size() && i < itemRanges_.size(); ++i) {
        for (const WorldRange& range : itemRanges_[i]) {
            if (range.count == 0) {
                continue;
            }
            if (const std::optional<Bounds3> box =
                    RangeBounds(i, range)) {
                const auto [it, added] =
                    bounds.try_emplace(range.world, *box);
                if (!added) {
                    it->second = Bounds3(
                        glm::min(it->second.first, box->first),
                        glm::max(it->second.second, box->second));
                }
            }
        }
    }
    return bounds;
}

std::uint64_t MeshDrawable::TriangleCount() const {
    std::uint64_t triangles = 0;
    for (const MeshItem& item : items_) {
        const std::uint64_t perMesh =
            meshes_.at(item.mesh).data.indices.size() / 3;
        triangles +=
            perMesh * (item.instancer ? item.instances.size() : 1u);
    }
    return triangles;
}

std::vector<DescriptorResource>
MeshDrawable::Resources(const RasterFrame& frame, std::size_t item,
                        bool culled) const {
    const MeshItem& draw = items_[item];
    std::vector<DescriptorResource> resources = {frame.resources,
                                                 frame.shadows};
    renderer_.MaterialOrDefault(draw.material)
        .AppendResources(resources);
    resources.push_back(
        DescriptorResource::Of(*meshResources_[culled ? 1 : 0]));
    return resources;
}

void MeshDrawable::WriteResources() {
    ERNIE_ZONE("Mesh write resources");
    const auto address = [](const auto& buffer) {
        return buffer != nullptr ? buffer->Address()
                                 : VkDeviceAddress{0};
    };
    // Every instance's ids, then those culling left.
    for (int culled = 0; culled < 2; ++culled) {
        const MeshResources resources{
            address(points_),
            address(normals_),
            address(indices_),
            address(draws_),
            address(colors_),
            address(positions_),
            address(orientations_),
            address(scales_),
            address(culled != 0 ? visibleIds_ : instanceIds_),
            address(drawRecords_),
            address(uvs_),
            address(instanceColors_),
            address(activeCounts_),
            address(temperatures_),
            address(nodeIds_),
            address(nodeTemperatures_),
            address(tangents_),
            address(instanceTemperatures_),
            address(instanceVisible_),
            address(cornerNormals_),
            address(cornerColors_)};
        Reserve(context_, meshResources_[culled], sizeof(resources),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                BufferMemory::Device);
        renderer_.Uploads().Write(
            *meshResources_[culled],
            std::as_bytes(std::span(&resources, 1)));
    }
}

const MeshDrawable::ItemBox&
MeshDrawable::PlainBox(std::size_t item) const {
    if (itemBoxes_.size() != items_.size()) {
        itemBoxes_.assign(items_.size(), ItemBox{});
    }
    ItemBox& plain = itemBoxes_[item];
    if (!plain.known) {
        const Mesh& mesh = meshes_.at(items_[item].mesh);
        plain.hostPoints = !mesh.data.points.empty();
        plain.anyPoints =
            plain.hostPoints || !mesh.devicePoints.Empty();
        plain.box = TransformBox(transforms_[item], mesh.low, mesh.high);
        plain.known = true;
    }
    return plain;
}

std::optional<Bounds3>
MeshDrawable::RangeBounds(std::size_t item,
                          const WorldRange& range) const {
    const MeshItem& draw = items_[item];
    // A plain draw's box as it is found once, with where it was at
    // the last frame while it moves.
    if (!draw.instancer) {
        const ItemBox& plain = PlainBox(item);
        if (!plain.anyPoints) {
            return std::nullopt;
        }
        Bounds3 box = plain.box;
        if (previousTransforms_[item] != transforms_[item]) {
            const Mesh& mesh = meshes_.at(draw.mesh);
            const Bounds3 was = TransformBox(previousTransforms_[item],
                                             mesh.low, mesh.high);
            box = {glm::min(box.first, was.first),
                   glm::max(box.second, was.second)};
        }
        return box;
    }
    const Mesh& mesh = meshes_.at(draw.mesh);
    if (mesh.data.points.empty() && mesh.devicePoints.Empty()) {
        return std::nullopt;
    }
    // The draw's box where it is and, moving, where it was at the
    // last frame: sub-frames of a motion-blurred frame see it
    // anywhere between.
    const auto transformBox = [&](const glm::vec3& low,
                                  const glm::vec3& high) {
        Bounds3 box = TransformBox(transforms_[item], low, high);
        if (previousTransforms_[item] != transforms_[item]) {
            const Bounds3 was =
                TransformBox(previousTransforms_[item], low, high);
            box = {glm::min(box.first, was.first),
                   glm::max(box.second, was.second)};
        }
        return box;
    };
    const Instancer& instancer = instancers_.at(*draw.instancer);
    if (instancer.onDevice && instancer.device.extent) {
        return transformBox(instancer.device.extent->first,
                            instancer.device.extent->second);
    }
    // The range's instances, each a sphere around its position
    // enclosing the posed prototype; for a device instancer, the box
    // of all its instances measured on the GPU.
    const auto [low, high] =
        TransformBox(draw.prototype, mesh.low, mesh.high);
    const float radius =
        glm::length(glm::max(glm::abs(low), glm::abs(high)));
    if (const auto measured = instancer.onDevice
                                  ? MeasuredPositions(instancer)
                                  : std::nullopt) {
        const glm::vec3 reach(radius * measured->second);
        return transformBox(measured->first.first - reach,
                            measured->first.second + reach);
    }
    const InstanceData& data = instancer.data;
    if (data.positions.size() != 3 * instancer.count) {
        return std::nullopt;  // nothing on the host to bound them
                              // with: always drawn
    }
    glm::vec3 boxLow(std::numeric_limits<float>::max());
    glm::vec3 boxHigh(std::numeric_limits<float>::lowest());
    for (std::uint32_t k = range.first; k < range.first + range.count;
         ++k) {
        const std::uint32_t i = draw.instances[k];
        float scale = 1.0f;
        if (!data.scales.empty()) {
            scale = std::max({std::abs(data.scales[3 * i]),
                              std::abs(data.scales[3 * i + 1]),
                              std::abs(data.scales[3 * i + 2])});
        }
        const glm::vec3 p(data.positions[3 * i],
                          data.positions[3 * i + 1],
                          data.positions[3 * i + 2]);
        boxLow = glm::min(boxLow, p - glm::vec3(radius * scale));
        boxHigh = glm::max(boxHigh, p + glm::vec3(radius * scale));
    }
    return transformBox(boxLow, boxHigh);
}

void MeshDrawable::Cull(const FrameParams& params,
                        std::span<const ShadowCascade> cascades) {
    ERNIE_ZONE("Mesh cull lists");
    culling_ = params.culling;
    posesOnGpu_ = params.posesOnGpu;
    occlusion_ =
        culling_ == Culling::Gpu ? params.occlusion : Occlusion::Off;
    // Each draw range's world bounds, then the ranges each view and
    // shadow view can see: all of them without culling, else those
    // whose bounds meet the frustum. With GPU culling, every
    // instanced range left becomes a job that tests its instances one
    // by one (cull.slang). The CPU tests a range's bounds only with
    // CPU culling, and only for views whose pose it knows: with the
    // views' poses on the GPU and shadow views the GPU fits, none.
    const bool cpuTests = culling_ == Culling::Cpu;
    const bool tested =
        cpuTests && (!params.posesOnGpu ||
                     std::any_of(cascades.begin(), cascades.end(),
                                 [](const ShadowCascade& c) {
                                     return !c.gpuFitted;
                                 }));
    // With GPU culling every range is a job of the GPU's; under
    // occlusion culling a plain mesh's is an occluder's, tested
    // against the frustum alone and drawn before the depth pyramid
    // is built from it.
    const auto isJob = [&](std::size_t) {
        return culling_ == Culling::Gpu;
    };
    const auto isOccluder = [&](std::size_t item) {
        return occlusion_ != Occlusion::Off &&
               !items_[item].instancer;
    };
    // The views' occluder jobs, kept apart until every view's own
    // jobs are listed: they follow them, frustum-culled with the
    // shadow views' (`RecordCull`), not against the depth pyramid.
    std::vector<GpuCullJob> occluderJobs;
    std::vector<std::uint32_t> occluderDrawCounts;
    std::vector<std::vector<std::optional<Bounds3>>> bounds(
        tested ? items_.size() : 0);
    for (std::size_t i = 0; i < items_.size() && tested; ++i) {
        for (const WorldRange& range : itemRanges_[i]) {
            bounds[i].push_back(range.count == 0 || isJob(i)
                                    ? std::nullopt
                                    : RangeBounds(i, range));
        }
    }
    std::vector<GpuCullJob> jobs;
    std::vector<glm::vec4> views;
    std::uint32_t visible = 0;
    std::uint32_t draws = 0;
    std::vector<glm::uvec4> records;  // per indirect draw: draw,
                                      // instance offset, index offset
    // Each level's draw of a run of ids starting at `ids`: the draw,
    // its ids' offset past the draw's own and its indices.
    const auto addRecords = [&](std::uint32_t item, std::uint32_t ids,
                                std::uint32_t count) {
        const std::vector<MeshLevel>& levels =
            meshes_.at(items_[item].mesh).levels;
        for (std::uint32_t l = 0; l < CullLevels(item); ++l) {
            records.emplace_back(
                item, ids + l * count - itemIdOffsets_[item],
                levels[l].first, 0u);
        }
    };
    // The depth pyramid views draw against (`RenderTargets::Hzb`):
    // level 0 the largest power-of-two size within the image.
    const auto pyramidWidth =
        static_cast<float>(std::bit_floor(params.camera.width));
    const auto pyramidHeight =
        static_cast<float>(std::bit_floor(params.camera.height));
    const auto pyramidMips = static_cast<float>(std::bit_width(
        std::max(std::bit_floor(params.camera.width),
                 std::bit_floor(params.camera.height))));
    // The draw ranges a view or shadow view may see, in item order:
    // every one, or for one world its own and (unless it skips them)
    // the shared ones, so a view of one of many worlds looks at its
    // world's ranges only. Indexed again only when the ranges change.
    if (rangeIndexVersion_ != layoutVersion_) {
        rangeIndexVersion_ = layoutVersion_;
        everyRange_.clear();
        rangesOfWorld_.clear();
        for (std::size_t i = 0; i < items_.size(); ++i) {
            for (std::size_t r = 0; r < itemRanges_[i].size(); ++r) {
                everyRange_.emplace_back(i, r);
                rangesOfWorld_[itemRanges_[i][r].world].emplace_back(
                    i, r);
            }
        }
    }
    const std::vector<RangeIndex>& everyRange = everyRange_;
    const auto& rangesOfWorld = rangesOfWorld_;
    std::map<std::pair<std::int32_t, bool>, std::vector<RangeIndex>>
        seenRanges;
    const std::vector<RangeIndex> none;
    const auto candidates = [&](const ShadowCasters* seen)
        -> const std::vector<RangeIndex>& {
        if (seen == nullptr ||
            seen->world == ShadowCasters::kAllWorlds) {
            return everyRange;
        }
        const std::pair key{seen->world, seen->shared};
        auto [it, added] = seenRanges.try_emplace(key);
        if (added) {
            const auto shared = rangesOfWorld.find(-1);
            const auto own = rangesOfWorld.find(seen->world);
            const std::vector<RangeIndex>& a =
                seen->shared && shared != rangesOfWorld.end()
                    ? shared->second
                    : none;
            const std::vector<RangeIndex>& b =
                seen->world >= 0 && own != rangesOfWorld.end()
                    ? own->second
                    : none;
            std::merge(a.begin(), a.end(), b.begin(), b.end(),
                       std::back_inserter(it->second));
        }
        return it->second;
    };
    // `pixelsPerUnit`: how many pixels (or shadow map texels) a world
    // unit spans at w = 1. Without `posed`, the view's pose is only
    // an estimate (`FrameParams::posesOnGpu`): it tests no bounds,
    // and its cull view's planes and matrix come from the GPU
    // (`RecordViewPoses`). A view's cull block (cull.slang's 12
    // float4s): its planes, matrix, scale and pyramid.
    const auto pushView = [&](const Frustum& frustum,
                              const glm::mat4& viewProjection,
                              float pixelsPerUnit, bool pyramid) {
        for (const glm::vec4& plane : frustum.planes) {
            // An infinite far plane's is (0, 0, 0, near): everything
            // is inside it.
            const float length = glm::length(glm::vec3(plane));
            views.push_back(length > 1e-30f
                                ? plane / length
                                : glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        }
        for (int column = 0; column < 4; ++column) {
            views.push_back(viewProjection[column]);
        }
        views.emplace_back(pixelsPerUnit, params.meshLodError, 0.0f,
                           0.0f);
        views.emplace_back(pyramid ? pyramidWidth : 0.0f,
                           pyramid ? pyramidHeight : 0.0f,
                           pyramid ? pyramidMips : 0.0f, 0.0f);
    };
    const auto list = [&](const glm::mat4& viewProjection,
                          float pixelsPerUnit, bool pyramid,
                          const ShadowCasters* seen, bool posed,
                          std::uint32_t linkMask = ~0u,
                          bool cull = true) {
        // Without culling, planes everything is inside.
        Frustum frustum;
        frustum.planes.fill(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        if (cull) {
            frustum = Frustum::FromMatrix(viewProjection);
        }
        posed = posed && cull;
        const auto slot = static_cast<std::uint32_t>(views.size() /
                                                     kCullViewStride);
        pushView(frustum, viewProjection, pixelsPerUnit, pyramid);
        std::vector<VisibleRange> ranges;
        for (const auto [i, r] : candidates(seen)) {
            const WorldRange& range = itemRanges_[i][r];
            const std::uint32_t group = items_[i].linkGroup & 31u;
            if (range.count == 0 ||
                (seen != nullptr && !seen->Draws(range.world)) ||
                ((linkMask >> group) & 1u) == 0) {
                continue;
            }
            // A job's instances are tested on the GPU, each against
            // the view's planes: nothing to test here.
            const bool job = isJob(i);
            if (posed && culling_ != Culling::Off && !job) {
                const std::optional<Bounds3>& box = bounds[i][r];
                if (box &&
                    !frustum.Intersects(box->first, box->second)) {
                    continue;
                }
            }
            VisibleRange entry{static_cast<std::uint32_t>(i),
                               static_cast<std::uint32_t>(r)};
            // Plain meshes too: a job of their one instance, so every
            // view's draws go out indirectly.
            if (job) {
                GpuCullJob cullJob{};
                cullJob.range[0] = entry.item;
                cullJob.range[1] = itemIdOffsets_[i] + range.first;
                cullJob.range[2] = range.count;
                cullJob.range[3] = slot;
                // A run of the range's count of ids and an indirect
                // draw per level.
                const std::uint32_t levels = CullLevels(i);
                cullJob.output[0] = visible;
                cullJob.output[1] = draws;
                // A view's occluder, numbered once the views' own
                // jobs are all listed (-2 - its index meanwhile).
                if (pyramid && isOccluder(i)) {
                    entry.occluder = true;
                    entry.job = -2 - static_cast<std::int32_t>(
                                         occluderJobs.size());
                    occluderJobs.push_back(cullJob);
                    occluderDrawCounts.push_back(levels);
                } else {
                    entry.job =
                        static_cast<std::int32_t>(jobs.size());
                    jobs.push_back(cullJob);
                    cullDrawCounts_.push_back(levels);
                }
                addRecords(entry.item, visible, range.count);
                visible += range.count * levels;
                draws += levels;
            }
            ranges.push_back(entry);
        }
        return ranges;
    };
    // With the views' poses on the GPU nothing here tests bounds on
    // the CPU: the views' jobs, records and lists stay as they were
    // while what they are made of stays, and only the views' cull
    // blocks are written again.
    std::vector<std::int32_t> viewWorlds(params.Views());
    for (std::size_t v = 0; v < params.Views(); ++v) {
        viewWorlds[v] = params.ViewWorld(v);
    }
    // Host poses too, when no view tests bounds here: then only the
    // views' cull blocks change from frame to frame.
    const bool cacheable =
        (params.posesOnGpu || !cpuTests) && !params.camera.pushbroom;
    const bool reuse = cacheable && viewLayout_.valid &&
                       viewLayout_.layoutVersion == layoutVersion_ &&
                       viewLayout_.idOffsets == itemIdOffsets_ &&
                       viewLayout_.worlds == viewWorlds &&
                       viewLayout_.culling == culling_ &&
                       viewLayout_.occlusion == occlusion_;
    // The shadow views' lists, kept as the views' are while no cascade
    // tests bounds here (the GPU culls, fits it, or it culls nothing)
    // and their worlds, masks and culling stay: only their cull blocks
    // are written again.
    std::vector<std::tuple<std::optional<std::int32_t>, std::uint32_t,
                           bool>>
        cascadeKeys;
    bool shadowsCacheable = cacheable;
    for (const ShadowCascade& cascade : cascades) {
        cascadeKeys.emplace_back(cascade.world, cascade.shadowMask,
                                 cascade.moving);
        shadowsCacheable = shadowsCacheable &&
                           (culling_ == Culling::Gpu ||
                            cascade.gpuFitted || cascade.moving);
    }
    const bool reuseShadows = shadowsCacheable && reuse &&
                              viewLayout_.shadows &&
                              viewLayout_.cascades == cascadeKeys;
    // Both kept: the jobs, records and lists are last frame's, as are
    // the GPU's copies of them, so only the cull blocks go out.
    const bool steady = reuseShadows;
    if (!steady) {
        ++cullLists_;  // the lists made anew: their plans too
    }
    if (reuse) {
        if (!steady) {
            jobs = viewLayout_.jobs;
            records = viewLayout_.records;
            cullDrawCounts_ = viewLayout_.drawCounts;
            viewJobStarts_ = viewLayout_.jobStarts;
        }
        visible = viewLayout_.visible;
        draws = viewLayout_.draws;
        // With poses on the GPU, it writes each view's planes and
        // matrix (`RecordViewPoses`); the rest of a block is the
        // camera's, the same for every view: the first view's,
        // copied. With host poses each view's own.
        for (std::size_t v = 0; v < params.Views(); ++v) {
            if (params.posesOnGpu && v > 0) {
                const std::vector<glm::vec4> block(
                    views.begin(),
                    views.begin() +
                        static_cast<std::ptrdiff_t>(kCullViewStride));
                views.insert(views.end(), block.begin(), block.end());
                continue;
            }
            const glm::mat4 viewProjection =
                params.View(v).ViewProjection();
            const float perUnit =
                glm::length(glm::vec3(viewProjection[0][1],
                                      viewProjection[1][1],
                                      viewProjection[2][1])) *
                0.5f * static_cast<float>(params.camera.height);
            pushView(Frustum::FromMatrix(viewProjection),
                     viewProjection, perUnit,
                     occlusion_ != Occlusion::Off);
        }
    }
    if (!reuse) {
        cullDrawCounts_.clear();
        viewJobStarts_.clear();
        viewDraws_.clear();
        for (std::size_t v = 0; v < params.Views(); ++v) {
            viewJobStarts_.push_back(
                static_cast<std::uint32_t>(jobs.size()));
            // A view sees what every world shares and its own world,
            // or every world for -1.
            const ShadowCasters seen{true, params.ViewWorld(v)};
            const glm::mat4 viewProjection =
                params.View(v).ViewProjection();
            // A unit at w = 1 spans the projection's y scale (the
            // length of the matrix's y row) over half the image's
            // height.
            const float perUnit =
                glm::length(glm::vec3(viewProjection[0][1],
                                      viewProjection[1][1],
                                      viewProjection[2][1])) *
                0.5f * static_cast<float>(params.camera.height);
            const std::size_t slot = views.size();
            viewDraws_.push_back(list(viewProjection, perUnit,
                                      occlusion_ != Occlusion::Off,
                                      &seen, !params.posesOnGpu));
            if (params.camera.pushbroom) {
                // Its depth pyramid is its strip's: instances project
                // through the strip (cull.slang's `project`), within
                // the envelope's planes.
                const PushbroomStrip strip = params.Strip(v);
                const glm::mat4 packed = strip.Packed();
                const glm::vec4 intrinsics = strip.Intrinsics();
                for (int column = 0; column < 4; ++column) {
                    views[slot + 6 +
                          static_cast<std::size_t>(column)] =
                        packed[column];
                }
                views[slot + 10].z = intrinsics.x;
                views[slot + 10].w = intrinsics.y;
                views[slot + 11].w = intrinsics.z;
            }
        }
        viewJobStarts_.push_back(
            static_cast<std::uint32_t>(jobs.size()));
        // Two-pass occlusion culling draws each view job twice; its
        // second draws follow all the first ones, in the same order,
        // so they batch as the first ones do.
        if (occlusion_ == Occlusion::TwoPass) {
            for (std::uint32_t j = 0; j < viewJobStarts_.back();
                 ++j) {
                GpuCullJob& job = jobs[j];
                job.output[2] = visible;
                job.output[3] = draws;
                addRecords(job.range[0], visible, job.range[2]);
                visible += job.range[2] * cullDrawCounts_[j];
                draws += cullDrawCounts_[j];
            }
        }
        // The views' occluders after their own jobs, which the depth
        // pyramid's tests run over (`RecordOcclusionCull`).
        const auto firstOccluder =
            static_cast<std::int32_t>(jobs.size());
        jobs.insert(jobs.end(), occluderJobs.begin(),
                    occluderJobs.end());
        cullDrawCounts_.insert(cullDrawCounts_.end(),
                               occluderDrawCounts.begin(),
                               occluderDrawCounts.end());
        for (std::vector<VisibleRange>& ranges : viewDraws_) {
            for (VisibleRange& range : ranges) {
                if (range.job <= -2) {
                    range.job = firstOccluder + (-2 - range.job);
                }
            }
        }
        viewLayout_.valid = cacheable;
        if (cacheable) {
            viewLayout_.layoutVersion = layoutVersion_;
            viewLayout_.idOffsets = itemIdOffsets_;
            viewLayout_.worlds = viewWorlds;
            viewLayout_.culling = culling_;
            viewLayout_.occlusion = occlusion_;
            viewLayout_.jobs = jobs;
            viewLayout_.records = records;
            viewLayout_.drawCounts = cullDrawCounts_;
            viewLayout_.visible = visible;
            viewLayout_.draws = draws;
            viewLayout_.jobStarts = viewJobStarts_;
        }
    }
    if (!steady) {
        shadowDraws_.clear();
    }
    gpuShadowSlots_.clear();
    const std::size_t jobsBefore = jobs.size();
    const std::size_t recordsBefore = records.size();
    const std::size_t countsBefore = cullDrawCounts_.size();
    if (reuseShadows) {
        visible = viewLayout_.shadowVisible;
        draws = viewLayout_.shadowDrawTotal;
    }
    for (const ShadowCascade& cascade : cascades) {
        if (reuseShadows) {
            if (cascade.gpuFitted && !cascade.moving) {
                gpuShadowSlots_.emplace_back(
                    cascade.gpuMatrix,
                    static_cast<std::uint32_t>(views.size() /
                                               kCullViewStride));
            }
            Frustum frustum;
            frustum.planes.fill(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
            if (!cascade.moving) {
                frustum =
                    Frustum::FromMatrix(cascade.CullViewProjection());
            }
            pushView(frustum, cascade.CullViewProjection(),
                     1.0f / std::max(cascade.texelWorldSize, 1e-30f),
                     false);
            continue;
        }
        // Views the GPU fits cull there only, their blocks patched
        // (`RecordShadowViews`); those moving across the shutter cull
        // nothing.
        if (cascade.gpuFitted && !cascade.moving) {
            gpuShadowSlots_.emplace_back(
                cascade.gpuMatrix,
                static_cast<std::uint32_t>(views.size() /
                                           kCullViewStride));
        }
        // Shadow maps: one texel spans `texelWorldSize` (at w = 1 for
        // perspective views). A map of one world's casters looks at
        // its world's and the shared ones only.
        std::optional<ShadowCasters> seen;
        if (cascade.world) {
            seen = ShadowCasters{true, *cascade.world};
        }
        shadowDraws_.push_back(
            list(cascade.CullViewProjection(),
                 1.0f / std::max(cascade.texelWorldSize, 1e-30f),
                 false, seen ? &*seen : nullptr, !cascade.gpuFitted,
                 cascade.shadowMask, !cascade.moving));
    }
    if (!reuseShadows) {
        viewLayout_.shadows = shadowsCacheable && viewLayout_.valid;
        if (viewLayout_.shadows) {
            const auto from = [](const auto& all, std::size_t first) {
                return std::vector(
                    all.begin() + static_cast<std::ptrdiff_t>(first),
                    all.end());
            };
            viewLayout_.cascades = cascadeKeys;
            viewLayout_.shadowJobs = from(jobs, jobsBefore);
            viewLayout_.shadowRecords = from(records, recordsBefore);
            viewLayout_.shadowDrawCounts =
                from(cullDrawCounts_, countsBefore);
            viewLayout_.shadowVisible = visible;
            viewLayout_.shadowDrawTotal = draws;
            viewLayout_.shadowLists = shadowDraws_;
        }
    }
    if (!gpuShadowSlots_.empty()) {
        Reserve(context_, gpuShadowSlotBuffer_,
                gpuShadowSlots_.size() *
                    sizeof(std::pair<std::uint32_t, std::uint32_t>),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Device);
        std::vector<glm::uvec2> slots;
        for (const auto& [matrix, slot] : gpuShadowSlots_) {
            slots.emplace_back(matrix, slot);
        }
        renderer_.Uploads().Write(*gpuShadowSlotBuffer_,
                                  std::as_bytes(std::span(slots)));
    }
    // A world's shadow pass draws its world's casters and the shared
    // ones, not every world's.
    if (!steady) {
        shadowWorldDraws_.assign(shadowDraws_.size(), {});
        for (std::size_t c = 0; c < shadowDraws_.size(); ++c) {
            for (const VisibleRange& caster : shadowDraws_[c]) {
                shadowWorldDraws_
                    [c][itemRanges_[caster.item][caster.range].world]
                        .push_back(caster);
            }
        }
    }
    WriteCullJobs(steady ? nullptr : &jobs, views, visible, draws);
    if (!steady && !records.empty()) {
        Reserve(context_, drawRecords_,
                records.size() * sizeof(glm::uvec4),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Device);
        renderer_.Uploads().Write(*drawRecords_,
                                  std::as_bytes(std::span(records)));
    }
    // One visibility bit per (view, instance id) for two-pass
    // occlusion culling, forgotten when the views or the instances
    // change.
    std::uint32_t instanceIds = 0;
    for (const auto& [handle, instancer] : instancers_) {
        instanceIds += static_cast<std::uint32_t>(instancer.count);
    }
    if (occlusion_ == Occlusion::TwoPass) {
        const std::pair key{
            static_cast<std::uint32_t>(params.Views()), instanceIds};
        if (key != visibilityKey_) {
            visibilityKey_ = key;
            const VkDeviceSize words =
                (VkDeviceSize{key.first} * key.second + 31) / 32;
            Reserve(context_, visibility_,
                    words * sizeof(std::uint32_t),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    BufferMemory::Device);
            resetVisibility_ = true;
        }
    }
    visibilityStride_ = instanceIds;
}

void MeshDrawable::WriteCullItems() {
    ERNIE_ZONE("Mesh write cull items");
    cullItemsDirty_ = false;
    std::vector<GpuCullItem> items(items_.size());
    // Each instanced mesh's levels, once: first index (relative),
    // count, error bits, unused.
    std::vector<glm::uvec4> levels;
    std::map<std::uint32_t, std::uint32_t> levelsOf;
    for (std::size_t i = 0; i < items_.size(); ++i) {
        const MeshItem& item = items_[i];
        const Mesh& mesh = meshes_.at(item.mesh);
        const auto [first, added] = levelsOf.try_emplace(
            item.mesh, static_cast<std::uint32_t>(levels.size()));
        if (added) {
            for (const MeshLevel& level : mesh.levels) {
                std::uint32_t errorBits = 0;
                std::memcpy(&errorBits, &level.error,
                            sizeof(errorBits));
                levels.emplace_back(level.first, level.count,
                                    errorBits, 0u);
            }
        }
        GpuCullItem& entry = items[i];
        std::memcpy(&entry.info[3], &mesh.countSlot,
                    sizeof(mesh.countSlot));
        entry.levels[0] = first->second;
        entry.levels[1] = CullLevels(i);
        std::uint32_t flags = 0;
        if (!item.instancer) {
            // One instance, its transform's.
            flags |= kCullNoInstances;
        } else {
            const Instancer& instancer =
                instancers_.at(*item.instancer);
            entry.levels[2] = InstanceBase(instancer);
            entry.levels[3] = InstanceBase(instancer, true);
            if (instancer.composed ||
                (instancer.onDevice
                     ? !instancer.device.orientations.Empty()
                     : !instancer.data.orientations.empty())) {
                flags |= kCullOrientations;
            }
            if (instancer.composed ||
                (instancer.onDevice
                     ? !instancer.device.scales.Empty()
                     : !instancer.data.scales.empty())) {
                flags |= kCullScales;
            }
            if (HasInstanceVisibility(instancer)) {
                flags |= kCullVisibility;
            }
        }
        std::memcpy(&entry.info[1], &flags, sizeof(flags));
        PoseCullItem(entry, transforms_[i], previousTransforms_[i],
                     item.prototype, mesh.low, mesh.high);
    }
    const std::span written = std::as_bytes(std::span(items));
    cullItemBytes_.assign(written.begin(), written.end());
    UploadCull(cullItemBuffer_, written);
    UploadCull(cullLevelBuffer_, std::as_bytes(std::span(levels)));
}

std::uint64_t MeshDrawable::LayoutOf(const Instancer& instancer) {
    const bool orientations =
        instancer.composed ||
        (instancer.onDevice ? !instancer.device.orientations.Empty()
                            : !instancer.data.orientations.empty());
    const bool scales =
        instancer.composed ||
        (instancer.onDevice ? !instancer.device.scales.Empty()
                            : !instancer.data.scales.empty());
    return std::uint64_t{instancer.first} << 8 |
           std::uint64_t{orientations} << 0 |
           std::uint64_t{scales} << 1 |
           std::uint64_t{HasInstanceColors(instancer)} << 2 |
           std::uint64_t{instancer.composed} << 3 |
           std::uint64_t{HasInstanceTemperatures(instancer)} << 4 |
           std::uint64_t{HasInstanceVisibility(instancer)} << 5;
}

bool MeshDrawable::CullItemsKept() const {
    return !cullItemsDirty_ &&
           cullItemBytes_.size() == items_.size() * sizeof(GpuCullItem);
}

void MeshDrawable::PatchItems() {
    if (posedItems_.empty()) {
        return;
    }
    ERNIE_ZONE("Mesh patch items");
    std::ranges::sort(posedItems_);
    const auto [last, end] = std::ranges::unique(posedItems_);
    posedItems_.erase(last, end);
    const bool items = CullItemsKept();
    for (const std::size_t i : posedItems_) {
        const MeshItem& item = items_[i];
        GpuDraw draw;
        std::byte* drawAt = drawBytes_.data() + i * sizeof(GpuDraw);
        std::memcpy(&draw, drawAt, sizeof(draw));
        PoseDraw(draw, transforms_[i], previousTransforms_[i],
                 item.prototype, item.tint,
                 thermal_ ? item.temperature : 0.0f);
        std::memcpy(drawAt, &draw, sizeof(draw));
        if (items) {
            const Mesh& mesh = meshes_.at(item.mesh);
            GpuCullItem entry;
            std::byte* entryAt =
                cullItemBytes_.data() + i * sizeof(GpuCullItem);
            std::memcpy(&entry, entryAt, sizeof(entry));
            PoseCullItem(entry, transforms_[i], previousTransforms_[i],
                         item.prototype, mesh.low, mesh.high);
            std::memcpy(entryAt, &entry, sizeof(entry));
        }
    }
}

void MeshDrawable::UploadPatched() {
    renderer_.Uploads().Write(*draws_, drawBytes_);
    if (CullItemsKept()) {
        UploadCull(cullItemBuffer_, cullItemBytes_);
    }
}

void MeshDrawable::PatchInstanceBases(
    std::span<const std::uint32_t> flipped) {
    if (flipped.empty()) {
        return;
    }
    ERNIE_ZONE("Mesh patch instance bases");
    const bool items = CullItemsKept();
    for (std::size_t i = 0; i < items_.size(); ++i) {
        const std::optional<std::uint32_t>& handle = items_[i].instancer;
        if (!handle || std::ranges::find(flipped, *handle) == flipped.end()) {
            continue;
        }
        const Instancer& instancer = instancers_.at(*handle);
        const std::array<std::uint32_t, 2> bases{
            InstanceBase(instancer), InstanceBase(instancer, true)};
        std::memcpy(drawBytes_.data() + i * sizeof(GpuDraw) +
                        offsetof(GpuDraw, motion),
                    bases.data(), sizeof(bases));
        if (items) {
            std::memcpy(cullItemBytes_.data() +
                            i * sizeof(GpuCullItem) +
                            offsetof(GpuCullItem, levels) +
                            2 * sizeof(std::uint32_t),
                        bases.data(), sizeof(bases));
        }
    }
}

void MeshDrawable::UploadCull(std::unique_ptr<GpuBuffer>& buffer,
                              std::span<const std::byte> bytes) {
    Reserve(context_, buffer, bytes.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    renderer_.Uploads().Write(*buffer, bytes);
}

void MeshDrawable::WriteCullJobs(const std::vector<GpuCullJob>* jobs,
                                 const std::vector<glm::vec4>& views,
                                 std::uint32_t visible,
                                 std::uint32_t draws) {
    ERNIE_ZONE("Mesh write cull jobs");
    if (jobs != nullptr) {
        cullJobs_.assign(jobs->begin(), jobs->end());
    }
    if (cullJobs_.empty()) {
        return;
    }
    // The items change only with the draws.
    if (cullItemsDirty_) {
        WriteCullItems();
    }
    if (jobs != nullptr) {
        UploadCull(cullJobBuffer_, std::as_bytes(std::span(*jobs)));
    }
    UploadCull(cullViewBuffer_, std::as_bytes(std::span(views)));
    Reserve(context_, visibleIds_,
            VkDeviceSize{visible} * sizeof(std::uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    Reserve(context_, cullCommands_,
            VkDeviceSize{draws} * sizeof(VkDrawIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            BufferMemory::Device);
}

void MeshDrawable::Dispatch(VkCommandBuffer commands, VkImageView hzb,
                            std::uint32_t firstJob,
                            std::uint32_t jobs, std::uint32_t mode,
                            std::uint32_t firstView) const {
    if (jobs == 0) {
        return;
    }
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      cullPipeline_->Handle());
    const DescriptorResource resources[] = {
        DescriptorResource::Of(*visibleIds_),
        DescriptorResource::Of(*cullCommands_),
        DescriptorResource::Of(hzb, VK_IMAGE_LAYOUT_GENERAL),
        DescriptorResource::Of(visibility_ != nullptr
                                   ? *visibility_
                                   : *cullLevelBuffer_)};
    cullLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                     resources);
    // One workgroup per job, spread over y beyond the x limit every
    // device supports.
    const std::uint32_t groupsX =
        std::min<std::uint32_t>(jobs, 65535);
    const std::uint32_t groupsY = (jobs + groupsX - 1) / groupsX;
    const CullConstants constants{
        {jobs, groupsX, firstJob, 0},
        {mode, firstView, visibilityStride_, 0},
        cullItemBuffer_->Address(),
        cullJobBuffer_->Address(),
        cullViewBuffer_->Address(),
        instanceIds_->Address(),
        positions_->Address(),
        orientations_->Address(),
        scales_->Address(),
        cullLevelBuffer_->Address(),
        activeCounts_->Address(),
        instanceVisible_ != nullptr ? instanceVisible_->Address()
                                    : VkDeviceAddress{0}};
    cullLayout_.PushConstants(commands, &constants,
                              sizeof(constants));
    vkCmdDispatch(commands, groupsX, groupsY, 1);
}

void MeshDrawable::RecordViewPoses(VkCommandBuffer commands) const {
    if (!posesOnGpu_ || cullJobs_.empty()) {
        return;
    }
    // Views take the first cull view slots (`Cull`): their planes,
    // then their matrix's columns.
    RasterRenderer::ViewFields fields;
    fields.planes = 0;
    fields.viewProjection = 6 * sizeof(glm::vec4);
    renderer_.PatchViews(commands, *cullViewBuffer_, 0,
                         kCullViewStride * sizeof(glm::vec4), fields);
}

void MeshDrawable::SetAnchoredViews(std::vector<AnchoredView> views) {
    for (const AnchoredView& view : views) {
        if (view.instancer == 0) {
            continue;
        }
        const auto it = instancers_.find(view.instancer);
        if (it == instancers_.end() || view.row >= it->second.count) {
            throw std::invalid_argument(
                "Anchored view under unknown instance " +
                std::to_string(view.instancer) + ", row " +
                std::to_string(view.row));
        }
    }
    anchoredViews_ = std::move(views);
}

void MeshDrawable::SetAnchoredLights(
    std::vector<AnchoredView> lights) {
    for (const AnchoredView& light : lights) {
        if (light.instancer == 0) {
            continue;
        }
        const auto it = instancers_.find(light.instancer);
        if (it == instancers_.end() ||
            light.row >= it->second.count) {
            throw std::invalid_argument(
                "Anchored light under unknown instance " +
                std::to_string(light.instancer) + ", row " +
                std::to_string(light.row));
        }
    }
    anchoredLights_ = std::move(lights);
}

void MeshDrawable::RecordShadowViews(VkCommandBuffer commands) const {
    if (gpuShadowSlots_.empty() || cullJobs_.empty()) {
        return;  // without cull jobs, no cull views either
    }
    ERNIE_GPU_ZONE(context_, commands, "Shadow cull views");
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      lightCullPipeline_->Handle());
    const DescriptorResource resources[] = {
        DescriptorResource::Of(renderer_.LightViewMatrices()),
        DescriptorResource::Of(*gpuShadowSlotBuffer_),
        DescriptorResource::Of(*cullViewBuffer_)};
    lightCullLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          resources);
    // light_views.slang's LightConstants, its count alone read.
    const std::uint32_t constants[16] = {
        static_cast<std::uint32_t>(gpuShadowSlots_.size())};
    lightCullLayout_.PushConstants(commands, constants,
                                   sizeof(constants));
    vkCmdDispatch(
        commands,
        (static_cast<std::uint32_t>(gpuShadowSlots_.size()) + 63) /
            64,
        1, 1);
}

std::uint32_t MeshDrawable::AnchorSlot(std::uint32_t instancer,
                                       std::uint32_t row) const {
    const auto it = instancers_.find(instancer);
    if (it == instancers_.end() || row >= it->second.count) {
        return 0xFFFFFFFFu;
    }
    return InstanceBase(it->second) + it->second.first + row;
}

std::uint32_t MeshDrawable::AnchorOps(std::uint32_t instancer) const {
    const auto it = instancers_.find(instancer);
    if (it == instancers_.end()) {
        return 0;
    }
    // As the draws' kHasOrientations and kHasScales.
    const Instancer& found = it->second;
    const bool orientations =
        found.composed ||
        (found.onDevice ? !found.device.orientations.Empty()
                        : !found.data.orientations.empty());
    const bool scales = found.composed ||
                        (found.onDevice ? !found.device.scales.Empty()
                                        : !found.data.scales.empty());
    return (orientations ? 1u : 0u) | (scales ? 2u : 0u);
}

void MeshDrawable::AppendAnchorRecord(std::vector<glm::vec4>& records,
                                      const glm::mat4& frame,
                                      const glm::mat4& below,
                                      std::uint32_t slot,
                                      std::uint32_t target,
                                      std::uint32_t ops) {
    for (const glm::mat4* m : {&frame, &below}) {
        for (int row = 0; row < 4; ++row) {
            records.emplace_back((*m)[0][row], (*m)[1][row],
                                 (*m)[2][row], (*m)[3][row]);
        }
    }
    records.emplace_back(std::bit_cast<float>(slot),
                         std::bit_cast<float>(target),
                         std::bit_cast<float>(ops), 0.0f);
}

void MeshDrawable::RecordPrepass(VkCommandBuffer commands) const {
    RecordWriters(commands);
    RecordActiveDraws(commands);
    RecordCompose(commands);
    RecordBounds(commands);
    if (anchorViews_ != 0) {
        ERNIE_GPU_ZONE(context_, commands, "Anchored views");
        RecordAnchors(commands, *anchorRecords_, anchorViews_,
                      renderer_.PoseBuffer(),
                      AnchorOutput::ViewPoses);
    }
    if (anchorLights_ != 0) {
        ERNIE_GPU_ZONE(context_, commands, "Anchored lights");
        RecordAnchors(commands, *anchorLightRecords_, anchorLights_,
                      renderer_.LightPoseBuffer(),
                      AnchorOutput::ViewPoses);
    }
}

void MeshDrawable::RecordAnchors(VkCommandBuffer commands,
                                 const GpuBuffer& records,
                                 std::uint32_t count,
                                 const GpuBuffer& target,
                                 AnchorOutput output) const {
    if (count == 0) {
        return;
    }
    // After the compose pass's poses.
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commands, &dependency);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      anchorPipeline_->Handle());
    const DescriptorResource resources[] = {
        DescriptorResource::Of(records),
        DescriptorResource::Of(*positions_),
        DescriptorResource::Of(*orientations_),
        DescriptorResource::Of(*scales_),
        DescriptorResource::Of(target)};
    anchorLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                       resources);
    const std::uint32_t constants[4] = {
        count, static_cast<std::uint32_t>(output), 0, 0};
    anchorLayout_.PushConstants(commands, constants,
                                sizeof(constants));
    vkCmdDispatch(commands, (count + 63) / 64, 1, 1);
}

void MeshDrawable::RecordSkins(VkCommandBuffer commands) const {
    if (pendingSkins_.empty() && pendingBlends_.empty()) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Mesh skinning");
    const auto barrier = [commands] {
        VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        memory.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        memory.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        memory.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        memory.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT |
                               VK_ACCESS_2_MEMORY_WRITE_BIT;
        VkDependencyInfo dependency{
            VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &memory;
        vkCmdPipelineBarrier2(commands, &dependency);
    };
    // After the poses' uploads and the last frame's reads of the
    // points; the control points, then the refined ones.
    barrier();
    // Sampled instancers' arrays at the time code.
    if (!pendingBlends_.empty()) {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          instanceBlendPipeline_->Handle());
        for (const std::uint32_t handle : pendingBlends_) {
            const Instancer& instancer = instancers_.at(handle);
            const auto count =
                static_cast<std::uint32_t>(instancer.count);
            if (count == 0) {
                continue;
            }
            const VkDeviceSize first =
                VkDeviceSize{InstanceBase(instancer)} +
                instancer.first;
            const VkDeviceAddress data =
                instancer.sampleData->Address();
            InstanceBlendConstants blend;
            blend.info[0] = count;
            blend.info[1] = instancer.sampleCounts[0];
            blend.info[2] = instancer.sampleCounts[1];
            blend.info[3] = instancer.sampleCounts[2];
            blend.blend[0] = instancer.blend[0];
            blend.blend[1] = instancer.blend[1];
            blend.blend[2] = instancer.blend[2];
            blend.blend[3] = 0;
            blend.positions = data + instancer.sampleOffsets[0];
            blend.orientations = data + instancer.sampleOffsets[1];
            blend.scales = data + instancer.sampleOffsets[2];
            blend.outPositions =
                positions_->Address() + first * 3 * sizeof(float);
            blend.outOrientations =
                orientations_->Address() + first * 4 * sizeof(float);
            blend.outScales =
                scales_->Address() + first * 3 * sizeof(float);
            SkinConstants push{};
            static_assert(sizeof(InstanceBlendConstants) <=
                          sizeof(SkinConstants));
            std::memcpy(&push, &blend, sizeof(blend));
            skinLayout_.PushConstants(commands, &push, sizeof(push));
            vkCmdDispatch(commands, (count + 63) / 64, 1, 1);
        }
        barrier();
    }
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      skinPipeline_->Handle());
    bool refining = false;
    for (const std::uint32_t handle : pendingSkins_) {
        const Mesh& mesh = meshes_.at(handle);
        const Skin& skin = *mesh.skin;
        const auto controls = static_cast<std::uint32_t>(
            skin.restPoints.size() / 3 / skin.restSamples);
        const VkDeviceAddress data = mesh.skinData->Address();
        SkinConstants constants;
        constants.info[0] = controls;
        constants.info[1] = skin.dualQuaternion ? 1u : 0u;
        constants.info[2] = skin.constantInfluences ? 1u : 0u;
        constants.info[3] = skin.shapeStarts.empty() ? 0u : 1u;
        constants.samples[0] = mesh.skinSamples[0];
        constants.samples[1] = mesh.skinSamples[1];
        constants.samples[2] = mesh.skinSamples[2];
        constants.samples[3] = 0;
        constants.rest = data + mesh.skinOffsets[0];
        constants.influenceStarts = data + mesh.skinOffsets[1];
        constants.influenceJoints = data + mesh.skinOffsets[2];
        constants.influenceWeights = data + mesh.skinOffsets[3];
        constants.shapeStarts = data + mesh.skinOffsets[4];
        constants.shapeOffsets = data + mesh.skinOffsets[5];
        constants.shapeIndices = data + mesh.skinOffsets[6];
        constants.pose = mesh.skinPose->Address();
        constants.shapeWeights =
            constants.pose +
            (mesh.skinPoseData.size() - skin.subShapes) *
                sizeof(float);
        constants.output = mesh.skinControl
                               ? mesh.skinControl->Address()
                               : points_->Address() +
                                     VkDeviceSize{PointBase(mesh) +
                                                  mesh.firstVertex} *
                                         3 * sizeof(float);
        refining = refining || mesh.skinControl != nullptr;
        skinLayout_.PushConstants(commands, &constants,
                                  sizeof(constants));
        vkCmdDispatch(commands, (controls + 63) / 64, 1, 1);
    }
    barrier();
    if (refining) {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          refinePipeline_->Handle());
        for (const std::uint32_t handle : pendingSkins_) {
            const Mesh& mesh = meshes_.at(handle);
            if (!mesh.skinControl) {
                continue;
            }
            const auto count = static_cast<std::uint32_t>(
                mesh.data.points.size() / 3);
            const VkDeviceAddress data = mesh.skinData->Address();
            SkinConstants push{};
            RefineConstants constants;
            constants.info[0] = count;
            constants.info[1] = constants.info[2] =
                constants.info[3] = 0;
            constants.control = mesh.skinControl->Address();
            constants.starts = data + mesh.skinOffsets[7];
            constants.indices = data + mesh.skinOffsets[8];
            constants.weights = data + mesh.skinOffsets[9];
            constants.output =
                points_->Address() +
                VkDeviceSize{PointBase(mesh) + mesh.firstVertex} * 3 *
                    sizeof(float);
            static_assert(sizeof(RefineConstants) <=
                          sizeof(SkinConstants));
            std::memcpy(&push, &constants, sizeof(constants));
            skinLayout_.PushConstants(commands, &push, sizeof(push));
            vkCmdDispatch(commands, (count + 63) / 64, 1, 1);
        }
        barrier();
    }
}

void MeshDrawable::RecordCull(VkCommandBuffer commands) const {
    // Skinned meshes posed, then normals that follow moving points,
    // before any pass draws them.
    RecordSkins(commands);
    const auto shaderBarrier = [&] {
        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT |
                                VK_ACCESS_2_SHADER_WRITE_BIT;
        VkDependencyInfo dependency{
            VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commands, &dependency);
    };
    if (!pendingNormals_.empty() || !topologyMeshes_.empty()) {
        const DescriptorResource resources[] = {
            DescriptorResource::Of(*points_),
            DescriptorResource::Of(*indices_),
            DescriptorResource::Of(*adjacency_),
            DescriptorResource::Of(*normals_),
            DescriptorResource::Of(*normalSums_),
            DescriptorResource::Of(*activeCounts_)};
        normalsLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                            resources);
    }
    // Normals of meshes whose triangles are written on the GPU, each
    // frame: their sums cleared, scaled by the largest triangle,
    // added to and normalised.
    if (!topologyMeshes_.empty()) {
        ERNIE_GPU_ZONE(context_, commands, "Mesh topology normals");
        for (std::size_t pass = 0; pass < topologyPipelines_.size();
             ++pass) {
            vkCmdBindPipeline(commands,
                              VK_PIPELINE_BIND_POINT_COMPUTE,
                              topologyPipelines_[pass]->Handle());
            for (const std::uint32_t handle : topologyMeshes_) {
                const Mesh& mesh = meshes_.at(handle);
                const auto points = static_cast<std::uint32_t>(
                    mesh.data.points.size() / 3);
                const auto triangles = static_cast<std::uint32_t>(
                    mesh.data.indices.size() / 3);
                const std::uint32_t constants[8] = {
                    mesh.firstVertex, mesh.firstIndex,
                    points,           triangles,
                    PointBase(mesh),  mesh.firstNormalSum,
                    mesh.countSlot,   0};
                normalsLayout_.PushConstants(commands, constants,
                                             sizeof(constants));
                const std::uint32_t threads = pass == 0
                                                  ? 3 * points + 1
                                              : pass == 3 ? points
                                                          : triangles;
                vkCmdDispatch(commands, (threads + 63) / 64, 1, 1);
            }
            shaderBarrier();
        }
    }
    if (!pendingNormals_.empty()) {
        ERNIE_GPU_ZONE(context_, commands, "Mesh normals");
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          normalsPipeline_->Handle());
        for (const std::uint32_t handle : pendingNormals_) {
            const Mesh& mesh = meshes_.at(handle);
            const auto count = static_cast<std::uint32_t>(
                mesh.data.points.size() / 3);
            const std::uint32_t constants[8] = {mesh.firstVertex,
                                                mesh.firstIndex,
                                                count,
                                                mesh.firstAdjacency,
                                                PointBase(mesh),
                                                0,
                                                0,
                                                0};
            normalsLayout_.PushConstants(commands, constants,
                                         sizeof(constants));
            vkCmdDispatch(commands, (count + 63) / 64, 1, 1);
        }
        shaderBarrier();
    }
    if (cullJobs_.empty()) {
        return;
    }
    if (resetVisibility_) {
        vkCmdFillBuffer(commands, visibility_->Handle(), 0,
                        VK_WHOLE_SIZE, 0);
        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT |
                                VK_ACCESS_2_MEMORY_WRITE_BIT;
        VkDependencyInfo dependency{
            VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commands, &dependency);
        resetVisibility_ = false;
    }
    // Views' jobs come first, then shadow views'. Shadow views and
    // views without occlusion culling cull their frusta now; with
    // occlusion culling views wait for their depth pyramids
    // (`RecordOcclusionCull`), but two-pass first draws what they saw
    // last frame.
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
    Dispatch(commands, none, viewJobs, total - viewJobs, kCullFrustum,
             0);
}

void MeshDrawable::RecordOcclusionCull(VkCommandBuffer commands,
                                       const GpuImage& hzb,
                                       std::uint32_t firstView,
                                       std::uint32_t views) const {
    if (occlusion_ == Occlusion::Off ||
        firstView + views >= viewJobStarts_.size()) {
        return;
    }
    const std::uint32_t first = viewJobStarts_[firstView];
    const std::uint32_t last = viewJobStarts_[firstView + views];
    Dispatch(commands, hzb.View(), first, last - first,
             occlusion_ == Occlusion::Occluders ? kCullOccluders
                                                : kCullTwoPassSecond,
             firstView);
}

std::vector<std::vector<std::uint32_t>>
MeshDrawable::CulledLevelCounts() const {
    std::vector<std::vector<std::uint32_t>> counts;
    if (cullJobs_.empty()) {
        return counts;
    }
    std::vector<VkDrawIndirectCommand> drawn(
        cullCommands_->Size() / sizeof(VkDrawIndirectCommand));
    cullCommands_->Read(std::as_writable_bytes(std::span(drawn)));
    const std::uint32_t viewJobs =
        viewJobStarts_.empty() ? 0 : viewJobStarts_.back();
    for (std::size_t j = 0; j < cullJobs_.size(); ++j) {
        std::vector<std::uint32_t>& job = counts.emplace_back();
        const bool second =
            occlusion_ == Occlusion::TwoPass && j < viewJobs;
        for (std::uint32_t l = 0; l < cullDrawCounts_[j]; ++l) {
            // Two-pass draws a view's instances in either pass: count
            // both.
            job.push_back(
                drawn[cullJobs_[j].output[1] + l].instanceCount +
                (second
                     ? drawn[cullJobs_[j].output[3] + l].instanceCount
                     : 0u));
        }
    }
    return counts;
}

std::vector<std::uint32_t>
MeshDrawable::CulledInstanceCounts() const {
    std::vector<std::uint32_t> counts;
    const std::vector<std::vector<std::uint32_t>> jobs =
        CulledLevelCounts();
    for (std::size_t j = 0; j < jobs.size(); ++j) {
        if (!items_[cullJobs_[j].range[0]].instancer) {
            continue;
        }
        const std::vector<std::uint32_t>& levels = jobs[j];
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
        throw std::invalid_argument("Unknown mesh " +
                                    std::to_string(mesh));
    }
    return it->second.levels.size();
}

std::size_t MeshDrawable::DrawCount(std::size_t view) const {
    return view < viewDraws_.size() ? viewDraws_[view].size() : 0;
}

MeshDrawable::PlanVersion MeshDrawable::CurrentPlanVersion() const {
    return {cullLists_, drawsWritten_, layoutVersion_,
            renderer_.MaterialsVersion(),
            shadowCutoutPipeline_ != nullptr &&
                cutoutPipelines_[pushbroom_ ? 1 : 0] != nullptr};
}

std::size_t MeshDrawable::Record(
    VkCommandBuffer commands, const RasterFrame& frame,
    std::span<const VisibleRange> ranges, const glm::mat4& light,
    const ShadowCasters* casters, bool shadow, bool direct,
    int culledSet, const DescriptorResource* motion,
    RecordState* state, std::optional<bool> transparent,
    bool keptList) const {
    std::size_t calls = 0;
    RecordState own;
    RecordState& kept = state != nullptr ? *state : own;
    const GraphicsPipeline*& bound = kept.bound;
    std::uint32_t& pushed = kept.pushedMaterial;
    bool& pushedCulled = kept.pushedCulled;
    const PipelineLayout& layout =
        motion != nullptr ? motionLayout_ : layout_;
    // A colour pass's items, transparent or opaque.
    const auto drawn = [&](std::uint32_t item) {
        return !transparent || Transparent(item) == *transparent;
    };
    // Cutout materials draw with the cutout variants, made by
    // `Prepare` when one appears.
    const auto cutout = [&](std::uint32_t item) {
        return anyCutout_ &&
               (item < itemCutouts_.size() ? itemCutouts_[item]
                                           : Cutout(item)) &&
               shadowCutoutPipeline_ != nullptr &&
               cutoutPipelines_[pushbroom_ ? 1 : 0] != nullptr;
    };
    const auto bind = [&](std::uint32_t item, bool culled) {
        const bool cut = cutout(item);
        const GraphicsPipeline* pipeline =
            motion != nullptr ? motionPipeline_.get()
            : shadow          ? (cut ? shadowCutoutPipeline_.get()
                                     : shadowPipeline_.get())
            : minimal_ != MinimalMode::Off
                ? minimalPipelines_[(minimal_ == MinimalMode::None
                                         ? 2u
                                         : 0u) +
                                    (cut ? 1u : 0u)]
                      .get()
            : cut ? cutoutPipelines_[pushbroom_ ? 1 : 0].get()
                  : itemPipelines_[item];
        if (pipeline != bound) {
            bound = pipeline;
            vkCmdBindPipeline(commands,
                              VK_PIPELINE_BIND_POINT_GRAPHICS,
                              bound->Handle());
        }
        // The resources differ only by material (`Resources`): items
        // sharing one keep them.
        const std::uint32_t material = items_[item].material;
        if (material != pushed || culled != pushedCulled) {
            pushed = material;
            pushedCulled = culled;
            std::vector<DescriptorResource> resources =
                Resources(frame, item, culled);
            if (motion != nullptr) {
                resources.push_back(*motion);
            }
            layout.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        resources);
        }
    };
    DrawConstants constants{};
    std::memcpy(constants.lightViewProj, glm::value_ptr(light),
                sizeof(constants.lightViewProj));
    constants.shadowFromFrame =
        shadow && frame.shadowFromFrame ? 1u : 0u;
    constants.view = frame.view;
    constants.shadowLayer = shadow ? frame.shadowLayer : 0u;

    // GPU-culled ranges draw indirectly, each level a command with a
    // per-draw record (mesh.slang reads it by draw index).
    // Consecutive commands that bind the same resources, which shadow
    // passes always do and colour passes do within a material, go out
    // in one multi-draw call.
    struct Batch {
        std::uint32_t item =
            0;  ///< The draw whose resources the batch binds.
        std::uint32_t first = 0;
        std::uint32_t count = 0;
    };
    std::optional<Batch> batch;
    const auto materialOf = [&](std::uint32_t item) {
        if (item < itemMaterials_.size()) {
            return itemMaterials_[item];
        }
        const std::uint32_t material = items_[item].material;
        return renderer_.HasMaterial(material)
                   ? material
                   : RasterRenderer::kDefaultMaterial;
    };
    const auto flush = [&] {
        if (!batch) {
            return;
        }
        bind(batch->item, true);
        const auto stride =
            static_cast<std::uint32_t>(sizeof(VkDrawIndirectCommand));
        if (context_.SupportsMultiDrawIndirect()) {
            constants.records = batch->first + 1;
            layout.PushConstants(commands, &constants,
                                 sizeof(constants));
            vkCmdDrawIndirect(commands, cullCommands_->Handle(),
                              VkDeviceSize{batch->first} * stride,
                              batch->count, stride);
            ++calls;
        } else {
            for (std::uint32_t k = 0; k < batch->count; ++k) {
                constants.records = batch->first + k + 1;
                layout.PushConstants(commands, &constants,
                                     sizeof(constants));
                vkCmdDrawIndirect(commands, cullCommands_->Handle(),
                                  VkDeviceSize{batch->first + k} *
                                      stride,
                                  1, stride);
                ++calls;
            }
        }
        constants.records = 0;
        batch.reset();
    };

    // What to draw, in order: planned from the ranges, or as planned
    // for a kept list while it and the draws stay.
    const auto plan = [&] {
        std::vector<RecordStep> steps;
        std::optional<RecordStep> open;
        for (std::size_t r = 0; r < ranges.size(); ++r) {
            const VisibleRange& visible = ranges[r];
            const WorldRange& range =
                itemRanges_[visible.item][visible.range];
            if (casters != nullptr && !casters->Draws(range.world)) {
                continue;
            }
            const std::uint32_t vertices = itemIndices_[visible.item];
            if (vertices == 0 || !drawn(visible.item)) {
                continue;
            }
            if (visible.job >= 0) {
                // Occluders draw with the direct draws, their only set;
                // other jobs in their culled set's phase.
                if (visible.occluder ? !direct : culledSet < 0) {
                    continue;
                }
                const int set = visible.occluder ? 0 : culledSet;
                const GpuCullJob& job =
                    cullJobs_[static_cast<std::size_t>(visible.job)];
                const std::uint32_t levels =
                    cullDrawCounts_[static_cast<std::size_t>(
                        visible.job)];
                const std::uint32_t firstDraw =
                    set == 0 ? job.output[1] : job.output[3];
                // Shadow batches span materials, except cutout ones,
                // which bind their own to test their opacity.
                const std::uint32_t at = open ? open->item : visible.item;
                const bool sameMaterial =
                    materialOf(at) == materialOf(visible.item) &&
                    itemPipelines_[at] == itemPipelines_[visible.item];
                const bool joins =
                    open && open->first + open->count == firstDraw &&
                    ((shadow && !cutout(open->item) &&
                      !cutout(visible.item)) ||
                     sameMaterial);
                if (!joins) {
                    if (open) {
                        steps.push_back(*open);
                    }
                    open = RecordStep{visible.item, firstDraw, 0};
                }
                open->count += levels;
                continue;
            }
            if (!direct) {
                continue;
            }
            if (open) {
                steps.push_back(*open);
                open.reset();
            }
            steps.push_back(
                {visible.item, 0, 0, static_cast<std::int64_t>(r)});
        }
        if (open) {
            steps.push_back(*open);
        }
        return steps;
    };
    std::vector<RecordStep> planned;
    const std::vector<RecordStep>* steps = &planned;
    if (keptList && motion == nullptr) {
        const PlanVersion version = CurrentPlanVersion();
        if (version != plansVersion_) {
            plans_.clear();
            plansVersion_ = version;
        }
        const PlanKey key{
            ranges.data(),
            ranges.size(),
            casters != nullptr ? casters->world : 0,
            static_cast<std::int8_t>(
                casters == nullptr ? 0 : casters->shared ? 1 : 2),
            static_cast<std::int8_t>(!transparent ? 0
                                     : *transparent ? 2
                                                    : 1),
            shadow,
            direct,
            culledSet};
        auto it = plans_.find(key);
        if (it == plans_.end()) {
            it = plans_.emplace(key, plan()).first;
        }
        steps = &it->second;
    } else {
        planned = plan();
    }
    for (const RecordStep& step : *steps) {
        if (step.range < 0) {
            batch = Batch{step.item, step.first, step.count};
            flush();
            continue;
        }
        const VisibleRange& visible =
            ranges[static_cast<std::size_t>(step.range)];
        const WorldRange& range =
            itemRanges_[visible.item][visible.range];
        const std::uint32_t vertices = itemIndices_[visible.item];
        bind(visible.item, false);
        constants.first = visible.item;
        constants.instanceOffset = range.first;
        constants.indexOffset = 0;
        layout.PushConstants(commands, &constants, sizeof(constants));
        const std::uint32_t countSlot = itemCountSlots_[visible.item];
        if (countSlot != 0 && range.count == 1) {
            // Only its triangles in use, as many as the GPU counts.
            constexpr std::uint32_t kStride =
                8 * sizeof(std::uint32_t);
            vkCmdDrawIndirect(commands, activeDraws_->Handle(),
                              VkDeviceSize{countSlot - 1} * kStride,
                              1, kStride);
        } else {
            vkCmdDraw(commands, vertices, range.count, 0, 0);
        }
        ++calls;
    }
    flush();
    return calls;
}

bool MeshDrawable::Moving(std::size_t item) const {
    const MeshItem& draw = items_[item];
    if (previousTransforms_[item] != transforms_[item] ||
        meshes_.at(draw.mesh).pointsMoving) {
        return true;
    }
    return draw.instancer && instancers_.at(*draw.instancer).moving;
}

void MeshDrawable::RecordMotion(
    VkCommandBuffer commands, const RasterFrame& frame,
    std::size_t view, const DescriptorResource& motion) const {
    if (view >= viewDraws_.size()) {
        return;
    }
    std::vector<VisibleRange> moving;
    for (const VisibleRange& range : viewDraws_[view]) {
        if (Moving(range.item)) {
            moving.push_back(range);
        }
    }
    if (moving.empty()) {
        return;
    }
    if (motionPipeline_ == nullptr) {
        const std::vector<ShaderModuleSource> modules = MeshModules();
        const std::vector<std::uint32_t> vertex =
            renderer_.Compiler().CompileSpirv(modules,
                                              "meshMotionVertex");
        const std::vector<std::uint32_t> fragment =
            renderer_.Compiler().CompileSpirv(modules,
                                              "meshMotionFragment");
        GraphicsPipelineDesc desc;
        desc.vertexSpirv = vertex;
        desc.fragmentSpirv = fragment;
        desc.colorFormats = {VK_FORMAT_R32G32B32A32_SFLOAT};
        desc.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
        desc.depthFormat = RenderTargets::kDepthFormat;
        // The same depths as the colour pass's: only the surfaces it
        // kept.
        desc.depthCompare = RenderTargets::kDepthCompareEqual;
        desc.depthWrite = false;
        motionPipeline_ = std::make_unique<GraphicsPipeline>(
            context_, motionLayout_, desc);
    }
    // Every draw the colour pass made of them: direct, and GPU-culled
    // from either set.
    Record(commands, frame, moving, glm::mat4(1.0f), nullptr, false,
           true, 0, &motion);
    if (occlusion_ == Occlusion::TwoPass) {
        Record(commands, frame, moving, glm::mat4(1.0f), nullptr,
               false, false, 1, &motion);
    }
}

void MeshDrawable::RecordShadow(VkCommandBuffer commands,
                                const RasterFrame& frame,
                                std::size_t cascade,
                                const glm::mat4& lightViewProjection,
                                const ShadowCasters& casters) const {
    if (cascade >= shadowDraws_.size()) {
        return;
    }
    // Transparent surfaces let the light through, so cast none: a
    // depth map has no partial shadow to give them.
    const std::optional<bool> opaque =
        drawsTransparent_ ? std::optional<bool>(false) : std::nullopt;
    if (casters.world == ShadowCasters::kAllWorlds) {
        Record(commands, frame, shadowDraws_[cascade],
               lightViewProjection, &casters, true, true, 0, nullptr,
               nullptr, opaque, true);
        return;
    }
    const auto& byWorld = shadowWorldDraws_[cascade];
    for (const std::int32_t world :
         {std::int32_t{-1}, casters.world}) {
        const auto draws = byWorld.find(world);
        if (draws != byWorld.end() && casters.Draws(world)) {
            Record(commands, frame, draws->second,
                   lightViewProjection, &casters, true, true, 0,
                   nullptr, nullptr, opaque, true);
        }
        if (casters.world < 0) {
            break;  // the shared casters only
        }
    }
}

void MeshDrawable::RecordColor(VkCommandBuffer commands,
                               const RasterFrame& frame,
                               std::size_t view,
                               DrawPhase phase) const {
    RecordColorState(commands, frame, view, phase, nullptr);
}

void MeshDrawable::RecordColorViews(
    VkCommandBuffer commands, std::span<const RasterFrame> frames,
    std::size_t first, DrawPhase phase) const {
    // The views bind the same resources: what one binds stays bound
    // for the next.
    RecordState state;
    for (std::size_t v = 0; v < frames.size(); ++v) {
        RecordColorState(commands, frames[v], first + v, phase,
                         &state);
    }
}

void MeshDrawable::RecordColorState(VkCommandBuffer commands,
                                    const RasterFrame& frame,
                                    std::size_t view, DrawPhase phase,
                                    RecordState* state) const {
    if (view >= viewDraws_.size()) {
        return;
    }
    // Which draws each phase records: meshes drawn directly occlude;
    // GPU-culled instances draw with their (first) draws, or with
    // occlusion culling after the depth pyramid, two-pass drawing
    // what was visible last frame first and the rest after.
    bool direct = true;
    int culledSet = 0;
    if (phase == DrawPhase::Occluders) {
        culledSet = occlusion_ == Occlusion::TwoPass ? 0 : -1;
    } else if (phase == DrawPhase::Occludees) {
        direct = false;
        culledSet = occlusion_ == Occlusion::TwoPass ? 1 : 0;
    }
    // Transparent items in their own phase alone, every set of them
    // at once: what the opaque phases would draw in turn.
    const std::optional<bool> transparent =
        drawsTransparent_
            ? std::optional<bool>(phase == DrawPhase::Transparent)
            : std::nullopt;
    if (phase == DrawPhase::Transparent) {
        if (!drawsTransparent_) {
            return;
        }
        const bool twoPass = occlusion_ == Occlusion::TwoPass;
        Record(commands, frame, viewDraws_[view], glm::mat4(1.0f),
               nullptr, false, true,
               occlusion_ == Occlusion::Off || twoPass ? 0 : -1,
               nullptr, state, transparent, true);
        if (occlusion_ != Occlusion::Off) {
            Record(commands, frame, viewDraws_[view], glm::mat4(1.0f),
                   nullptr, false, false, twoPass ? 1 : 0, nullptr,
                   state, transparent, true);
        }
        return;
    }
    const std::size_t calls = Record(
        commands, frame, viewDraws_[view], glm::mat4(1.0f), nullptr,
        false, direct, culledSet, nullptr, state, transparent);
    viewCalls_.resize(viewDraws_.size());
    viewCalls_[view] =
        (phase == DrawPhase::Occludees ? viewCalls_[view] : 0) +
        calls;
}

std::size_t MeshDrawable::DrawCalls(std::size_t view) const {
    return view < viewCalls_.size() ? viewCalls_[view] : 0;
}

}  // namespace ernie
