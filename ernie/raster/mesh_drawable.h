// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/compute_interop.h"
#include "ernie/device/pipeline.h"
#include "ernie/geometry/mesh_data.h"
#include "ernie/raster/drawable.h"
#include "ernie/raster/renderer.h"
#include "ernie/session/delegate.h"
#include "ernie/util/tensor.h"

#include <glm/glm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ernie {

/// Per-instance arrays on a CUDA device, as contiguous float32
/// tensors: positions (n, 3), and orientations (n, 4, imaginary part
/// first), scales (n, 3) and colours (n, 3, linear RGB in place of
/// the prototypes'), each empty when absent.
struct DeviceInstanceData {
    Tensor positions;
    Tensor orientations;
    Tensor scales;
    Tensor colors;
    /// Bounds of the instances in the instancer's frame, prototypes
    /// included, as USD's `extent`; without them the positions and
    /// scales are copied to the host to compute bounds.
    std::optional<std::pair<glm::vec3, glm::vec3>> extent;
    /// The CUDA stream the arrays are written on
    /// (`Scene::WriteStream`); copies follow it.
    std::uintptr_t stream = 1;
    /// Whether `orientations` (then (n, 4)) put w first; the copy
    /// reorders them.
    bool orientationsWFirst = false;
};

/// The kinds of transform op instances composed on the GPU take
/// (`ComposedInstanceData`): the Euler rotations in `RotationOrder`'s
/// order.
enum class InstanceOpKind : std::uint32_t {
    Translate = 0,
    Scale = 1,
    Orient = 2,
    RotateX = 3,
    RotateY = 4,
    RotateZ = 5,
    RotateXYZ = 6,
    RotateXZY = 7,
    RotateYXZ = 8,
    RotateYZX = 9,
    RotateZXY = 10,
    RotateZYX = 11,
    Transform = 12,
};

/// One transform op of composed instances: its rows of a contiguous
/// float32 (n, width) array on the compute device, one row per
/// instance: 3 numbers for a translation, scale or Euler angles
/// (degrees), 1 for a uniform scale or a rotation about one axis
/// (degrees), 4 for a quaternion, 16 for a row-major 4 x 4 transform
/// (column vectors).
struct DeviceInstanceOp {
    InstanceOpKind kind = InstanceOpKind::Translate;
    Tensor rows;
    bool wFirst = false;  ///< A quaternion as w, x, y, z.
};

/// What a composed instance is posed under: an instance of another
/// instancer (composed itself, so its pose is written first), then
/// `between`, the transform from that instance's frame to the frame
/// of this instance's ops, as translate, rotation and scale.
struct InstanceParent {
    std::uint32_t instancer = 0;  ///< 0 for none.
    std::uint32_t row = 0;
    glm::vec3 translate{0.0f};
    glm::vec4 rotation{0.0f, 0.0f, 0.0f, 1.0f};  ///< x, y, z, w.
    glm::vec3 scale{1.0f};
};

/// Instances composed on the GPU from transform ops (`SetInstances`):
/// each instance's ops in order, the first outermost, under its
/// parent's pose. Poses are kept as translate, rotate and scale,
/// exact unless a non-uniform scale is applied outside a rotation
/// (shear).
struct ComposedInstanceData {
    std::vector<DeviceInstanceOp> ops;
    std::vector<InstanceParent>
        parents;  ///< One per instance, or empty when none has one.
    /// Bounds of the instances, posed, in the instancer's frame:
    /// required, as the poses never reach the host.
    std::pair<glm::vec3, glm::vec3> extent;
    std::uintptr_t stream =
        1;  ///< The CUDA stream the arrays are written on.
};

/// A view posed under an instance, such as a camera on a link posed
/// on the GPU: its camera to world pose is `frame`, then the
/// instance's pose, then `below`; without an instancer (0), `frame`
/// then `below`.
struct AnchoredView {
    std::uint32_t instancer = 0;
    std::uint32_t row = 0;
    glm::dmat4 frame{1.0};
    glm::dmat4 below{1.0};
};

/// What `MeshDrawable::RecordAnchors` writes per record
/// (instance_anchors.slang): a view's camera to world pose,
/// row-major, into the renderer's GPU poses; or a point cloud's model
/// matrix into its draw (`PointsDrawable`).
enum class AnchorOutput : std::uint32_t {
    ViewPoses = 0,
    PointsDraws = 1,
};

/// One draw of a mesh: either the mesh itself, or the given instances
/// of an instancer with the mesh as their prototype.
struct MeshItem {
    std::uint32_t mesh = 0;
    /// World transform of the mesh, or of the instancer, in double:
    /// the drawable draws it relative to the frame's origin
    /// (`SetOrigin`).
    glm::dmat4 transform{1.0};
    glm::mat4 prototype{1.0f};  ///< Instanced: the prototype's own
                                ///< transform, applied first.
    std::optional<std::uint32_t> instancer;
    std::vector<std::uint32_t>
        instances;  ///< Instanced: indices into the instancer's
                    ///< arrays.
    /// The world the draw belongs to (`tokens::kWorld`): views of a
    /// batch see their own world's draws and those of -1, every
    /// world's.
    std::int32_t world = -1;
    /// Instanced: the world of each of `instances`, which are sorted
    /// by world, in place of `world`; empty when every instance is in
    /// `world`.
    std::vector<std::int32_t> instanceWorlds;
    std::uint32_t material =
        RasterRenderer::kDefaultMaterial;  ///< Id in the renderer's
                                           ///< table.
    glm::vec3 tint{1.0f};  ///< Multiplies the material's albedo where
                           ///< colours are not used.
    std::uint32_t primId =
        1;  ///< Written to the PrimId render var; 0 means background.
    /// Its light link group (`LinkGroups`): which lights light it and
    /// whose shadows it casts.
    std::uint32_t linkGroup = 0;
    bool vertexColors = true;  ///< Use the mesh's vertex colours, if
                               ///< it has any, as the tint.
};

/// Triangle meshes and point-instanced copies of them: a
/// `RasterDrawable`, lit with the renderer's sun and shadows and
/// shaded with the renderer's materials, like the terrain.
///
/// Meshes and instancers are added once and their arrays replaced as
/// they change; the caller lists the draws (`SetItems`), each with a
/// material, and the renderer prepares and records them every frame.
/// Detail textures are projected along the three world axes
/// (triplanar). Geometry of all meshes lives in shared pools of
/// storage buffers that the vertex shader reads (no vertex buffers),
/// rebuilt when meshes are added or removed and rewritten in place
/// when only points move. Instances are composed on the GPU from the
/// raw position, orientation and scale arrays, so updating thousands
/// of instances from a simulation costs a copy of those arrays and
/// nothing per instance on the CPU. Meshes are two-sided; without
/// normals each triangle is lit flat.
///
/// Culling (`FrameParams::culling`) follows each view's frustum and,
/// for shadow views, the light's:
/// - `Culling::Off` draws every draw of the worlds a view sees;
/// - `Culling::Cpu` skips draws (a mesh, or one world's run of an
///   instancer's instances) whose world bounds miss the frustum, once
///   per frame on the CPU;
/// - `Culling::Gpu`, the default, also tests every instance of the
///   instanced draws left on the GPU (cull.slang, `RecordCull`), one
///   workgroup per draw and view, compacting the visible instances in
///   order into an indirect draw; instance arrays written by CUDA are
///   culled without coming to the host. It also picks each instance's
///   level of detail (`LevelCount`), and consecutive indirect draws
///   that bind the same resources go out as one multi-draw call, each
///   draw finding its parameters by draw index (`DrawCalls`).
///
/// References:
/// - Pixar, "UsdGeomPointInstancer",
///   https://openusd.org/release/api/class_usd_geom_point_instancer.html
/// - U. Haar and S. Aaltonen, "GPU-Driven Rendering Pipelines",
///   SIGGRAPH 2015 Advances in Real-Time Rendering in Games course.
///   https://advances.realtimerendering.com/s2015/aaltonenhaar_siggraph2015_combined_final_footer_220dpi.pdf
/// - B. Shacklett et al., "An Extensible, Data-Oriented Architecture
///   for High-Performance, Many-World Simulation", ACM Transactions
///   on Graphics 42(4), 2023 (Madrona's batch renderer).
/// - V. Blanco, "Compute Shader Culling", Vulkan Guide.
///   https://vkguide.dev/docs/gpudriven/compute_culling/
/// - Khronos, Vulkan specification, `vkCmdDrawIndirect` with the
///   `multiDrawIndirect` feature, and `DrawIndex` (`gl_DrawID`,
///   `shaderDrawParameters`).
///   https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawIndirect.html
class MeshDrawable : public RasterDrawable {
  public:
    /// Meshes drawn by `renderer`, which must outlive the drawable.
    /// @throws std::runtime_error if the shadow pipeline cannot be
    /// created.
    explicit MeshDrawable(RasterRenderer& renderer);
    ~MeshDrawable() override;
    MeshDrawable(const MeshDrawable&) = delete;
    MeshDrawable& operator=(const MeshDrawable&) = delete;

    /// Adds a mesh and returns its handle.
    /// @throws std::invalid_argument if an array's size is not a
    /// multiple of 3, normals or colours do not match the points, or
    /// an index is out of range.
    std::uint32_t AddMesh(MeshData data);

    /// Replaces a mesh's points, keeping their number.
    /// @throws std::invalid_argument for an unknown mesh or a
    /// different number of points.
    void SetPoints(std::uint32_t mesh, std::span<const float> points);

    /// Replaces a mesh's points with a contiguous float32 tensor on
    /// the compute device of the same number of points, copied device
    /// to device at the next `Prepare` on compute stream `stream`,
    /// the one the points are written on. `bounds` are the points'
    /// bounds; without them the points are copied to the host to
    /// compute them.
    /// @throws std::invalid_argument for an unknown mesh, a different
    /// number of points, a tensor that is not float32 on the compute
    /// device, or without `SetComputeInterop`.
    void SetPoints(std::uint32_t mesh, const Tensor& points,
                   std::optional<std::pair<glm::vec3, glm::vec3>>
                       bounds = std::nullopt,
                   std::uintptr_t stream = 1);

    /// Replaces a mesh's normals with a contiguous float32 tensor on
    /// the compute device, one normal per point, copied device to
    /// device at the next `Prepare` on stream `stream`. They then
    /// follow the tensor rather than the points.
    /// @throws std::invalid_argument as the device `SetPoints`.
    void SetNormals(std::uint32_t mesh, const Tensor& normals,
                    std::uintptr_t stream = 1);

    /// Replaces a mesh's vertex colours (linear RGB per point) with a
    /// contiguous float32 tensor on the compute device, copied as
    /// `SetNormals`. The mesh must have been added with colours.
    /// @throws std::invalid_argument as the device `SetPoints`, or
    /// for a mesh without colours.
    void SetColors(std::uint32_t mesh, const Tensor& colors,
                   std::uintptr_t stream = 1);

    /// Replaces a mesh's triangles with a contiguous int32 or uint32
    /// tensor on the compute device, three point indices per
    /// triangle, as many as the mesh was added with (zeros holding
    /// their place), copied as `SetNormals`. Indices are not checked:
    /// each must be below the number of points. Such a mesh has no
    /// coarser levels of detail, and host normals it has are not
    /// recomputed as its points move (shaded flat without normals).
    /// @throws std::invalid_argument for an unknown mesh, another
    /// number of indices, a tensor of another type or device, or
    /// without `SetComputeInterop`.
    void SetIndices(std::uint32_t mesh, const Tensor& indices,
                    std::uintptr_t stream = 1);

    /// Maps a mesh's triangle indices for writing in place on the
    /// compute device: a uint32 tensor over the drawable's own index
    /// pool, holding them as they are now. Writes go on `stream`,
    /// which first waits for the frames already submitted; handing
    /// the tensor back with `SetIndices` then draws them without
    /// copying. Map again for each frame.
    /// @throws std::invalid_argument for an unknown mesh, one whose
    /// indices are not on the device, or without `SetComputeInterop`.
    Tensor MapIndices(std::uint32_t mesh, std::uintptr_t stream = 1);

    /// Sets the points and triangle indices a mesh uses of those it
    /// holds (its capacity), as the first two of a contiguous int32
    /// (n >= 2,) tensor on the host or the compute device (the rest
    /// ignored, as a simulator's overflow flag), read on the GPU: the
    /// triangles past the count are not drawn, and only the points in
    /// use bound it. The counts are copied each time they are set. An
    /// empty tensor uses them all.
    /// @throws std::invalid_argument for an unknown mesh, a tensor of
    /// another type or shape, or device counts without
    /// `SetComputeInterop`.
    void SetActiveCounts(std::uint32_t mesh, const Tensor& counts,
                         std::uintptr_t stream = 1);

    /// Has `writer` write the mesh on the GPU each frame
    /// (`RenderDelegate::SetMeshWriter`): recorded at the start of
    /// the prepass, between barriers, into its points (the half
    /// drawn, which no longer alternates), normals, indices and a
    /// slot of counts in use of its own. Without `bounds` the points
    /// in use are measured on the GPU (`RecordBounds`). Null stops
    /// it.
    /// @throws std::invalid_argument for an unknown mesh.
    void SetWriter(std::uint32_t mesh, MeshWriter writer,
                   std::optional<Bounds3> bounds = {});

    /// Maps a mesh's points for writing in place on the compute
    /// device: a float32 (points, 3) tensor over the drawable's own
    /// points pool, the half the next frame draws, holding the points
    /// as they are now (copied on `stream` from the half the last
    /// frame drew, so updates in place see them). Writes go on
    /// `stream`, which first waits for the frames already submitted;
    /// handing the tensor back with the device `SetPoints` then draws
    /// them without copying. Map again for each frame: the halves
    /// alternate so motion blur keeps the last frame's points.
    /// @throws std::invalid_argument for an unknown mesh or without
    /// `SetComputeInterop`.
    Tensor MapPoints(std::uint32_t mesh, std::uintptr_t stream = 1);

    /// Maps instance array `array` (0 positions, 1 orientations, 2
    /// scales, 3 colours) of an instancer for writing in place, as
    /// `MapPoints`;
    /// the device `SetInstances` takes it back. The instancer's other
    /// arrays move to the same half with it.
    /// @throws std::invalid_argument for an unknown instancer or
    /// array, or without `SetComputeInterop`.
    Tensor MapInstances(std::uint32_t instancer, std::size_t array,
                        std::uintptr_t stream = 1);

    /// Maps the rows of transform op `op` of an instancer composed on
    /// the GPU for writing in place: a float32 tensor shaped as the
    /// rows last set, over the raw array the compose pass reads,
    /// holding those rows. Writes go on `stream`, which first waits
    /// for the frames already submitted; the composed `SetInstances`
    /// takes it back without copying. Map again for each frame.
    /// @throws std::invalid_argument for an unknown or uncomposed
    /// instancer, an op past its ops, or without `SetComputeInterop`.
    Tensor MapComposedOp(std::uint32_t instancer, std::size_t op,
                         std::uintptr_t stream = 1);

    /// @throws std::invalid_argument for an unknown mesh.
    void RemoveMesh(std::uint32_t mesh);

    /// Adds an instancer and returns its handle.
    /// @throws std::invalid_argument as `SetInstances`.
    std::uint32_t AddInstancer(InstanceData data);

    /// Replaces an instancer's arrays.
    /// @throws std::invalid_argument for an unknown instancer,
    /// positions whose size is not a multiple of 3, or orientations,
    /// scales or colours that do not match the positions.
    void SetInstances(std::uint32_t instancer, InstanceData data);

    /// @throws std::invalid_argument for an unknown instancer.
    void RemoveInstancer(std::uint32_t instancer);

    /// Replaces an instancer's arrays with tensors on the compute
    /// device, copied device to device at the next `Prepare`; only
    /// the arrays of instancers that changed are copied.
    /// @throws std::invalid_argument for an unknown instancer,
    /// tensors that are not float32 on the compute device or do not
    /// match the positions, or without `SetComputeInterop`.
    void SetInstances(std::uint32_t instancer,
                      const DeviceInstanceData& data);

    /// Makes an instancer's poses composed on the GPU from transform
    /// ops (`ComposedInstanceData`): the ops' arrays are copied
    /// device to device at the next `Prepare`, and the instances
    /// composed level by level, parents before children, before
    /// culling. Instances posed under another instancer's are
    /// composed again whenever that one moves.
    /// @throws std::invalid_argument for an unknown instancer or
    /// parent, no ops, arrays that are not float32 (n, width) on the
    /// compute device or differ in rows, a width that does not fit
    /// the op, parents that do not match the rows, or without
    /// `SetComputeInterop`.
    void SetInstances(std::uint32_t instancer,
                      const ComposedInstanceData& data);

    /// Sets the views the next frames pose on the GPU under instances
    /// (`AnchoredView`), one per view of the frame in order: their
    /// poses, relative to the frame's origin, are written into the
    /// renderer's GPU poses (`RasterRenderer::PoseBuffer`) in
    /// `RecordPrepass`, after the instances are composed, for a frame
    /// with `FrameParams::posesOnGpu`. None, the default, writes
    /// nothing.
    /// @throws std::invalid_argument for an unknown instancer or a
    /// row out of range.
    void SetAnchoredViews(std::vector<AnchoredView> views);

    /// Sets the lights the next frames pose on the GPU under
    /// instances, one per light with `Light::posedOnGpu` in the
    /// frame's order: their poses, relative to the frame's origin,
    /// are written into the renderer's light poses
    /// (`RasterRenderer::LightPoseBuffer`) in `RecordPrepass`, after
    /// the instances are composed.
    /// @throws std::invalid_argument for an unknown instancer or a
    /// row out of range.
    void SetAnchoredLights(std::vector<AnchoredView> lights);

    /// Patches the cull views of the shadow views the GPU fitted
    /// (`ShadowCascade::gpuFitted`) from their matrices
    /// (`RasterRenderer::LightViewMatrices`), after the renderer's
    /// light pass.
    void RecordShadowViews(VkCommandBuffer commands) const override;

    /// The slot of instance `row` of `instancer` in this frame's half
    /// of the instance pools, for records of `RecordAnchors`: valid
    /// from this frame's `Prepare` to the next. ~0 for an unknown
    /// instance.
    std::uint32_t AnchorSlot(std::uint32_t instancer,
                             std::uint32_t row) const;

    /// Appends a record of `RecordAnchors` to `records` (nine
    /// float4): `frame` (relative to the frame's origin), the
    /// instance at `slot` (`AnchorSlot`, ~0 for none), then `below`,
    /// written at index `target`.
    static void AppendAnchorRecord(std::vector<glm::vec4>& records,
                                   const glm::mat4& frame,
                                   const glm::mat4& below,
                                   std::uint32_t slot,
                                   std::uint32_t target);

    /// Records the pass posing `count` records (`AppendAnchorRecord`)
    /// under their instances into `target` as `output`, after this
    /// frame's compose pass (`RecordPrepass`): for drawables posing
    /// what they draw under instances, from their own
    /// `RecordPrepass`.
    void RecordAnchors(VkCommandBuffer commands,
                       const GpuBuffer& records, std::uint32_t count,
                       const GpuBuffer& target,
                       AnchorOutput output) const;

    /// Enables GPU inputs: geometry and instance pools move to memory
    /// shared with `compute` (CUDA or HIP), which must outlive the
    /// drawable. Null turns GPU inputs off again.
    void SetComputeInterop(ComputeInterop* compute);

    /// Sets the draws of the next frames; nothing is drawn until the
    /// next `Prepare`. Draws whose material is removed later use the
    /// default material. Each view draws only the draws and instances
    /// of the worlds it sees (`FrameParams::worlds`), and each shadow
    /// pass those of its `ShadowCasters`.
    /// @throws std::invalid_argument for an unknown mesh, instancer
    /// or material, an instance index out of range, or instance
    /// worlds that do not match the instances or are unsorted. Draws
    /// keep their transforms from the last frame for motion blur,
    /// matched by prim id: a draw whose transform changed moves from
    /// its old one over the frame (`Uniforms::motion`).
    void SetItems(std::vector<MeshItem> items);

    /// Starts a frame's motion: draws that `SetItems` does not move
    /// again, and points and instance arrays not written again,
    /// before the next `Prepare` stand still. Points and instance
    /// arrays written in a frame keep the last frame's in the other
    /// half of their pools, which motion-blurred frames interpolate
    /// from.
    void SettleMotion();

    /// World bounds of the current draws, or none without any.
    void SetOrigin(const glm::dvec3& origin) override;
    const char* ProfileName() const override {
        return "Meshes";
    }
    std::optional<Bounds3> Bounds() const override;

    bool DrawsMinimal() const override {
        return true;
    }

    bool DrawsVirtualShadows() const override {
        return true;
    }

    /// World bounds of the current draws (and runs of instances) in
    /// world `world`.
    std::map<std::int32_t, Bounds3> WorldsBounds() const override;

    /// Uploads changed geometry and the draws of the next frame.
    /// @throws std::invalid_argument if a material's colour pipeline
    /// fails to compile.
    void Prepare(const FrameParams& params,
                 std::span<const ShadowCascade> cascades) override;

    std::size_t MeshCount() const {
        return meshes_.size();
    }

    std::size_t InstancerCount() const {
        return instancers_.size();
    }

    /// The draw ranges view `view` of the last `Prepare` draws: those
    /// of its worlds whose bounds meet its frustum (all of them with
    /// `Culling::Off`).
    std::size_t DrawCount(std::size_t view = 0) const;

    /// The draw calls view `view` recorded last: one per mesh drawn
    /// directly, and one per batch of indirect draws (a multi-draw)
    /// with GPU culling.
    std::size_t DrawCalls(std::size_t view = 0) const;

    /// Records the frame's GPU instance culling (`Culling::Gpu`):
    /// every instanced range a view or shadow view draws is a job
    /// whose visible instances cull.slang compacts into an indirect
    /// draw.
    void RecordViewPoses(VkCommandBuffer commands) const override;
    /// Composes the instances composed on the GPU, then poses the
    /// anchored views under them.
    void RecordPrepass(VkCommandBuffer commands) const override;
    void RecordCull(VkCommandBuffer commands) const override;

    /// Whether the last `Prepare` culls instances by occlusion
    /// (`FrameParams::occlusion` with GPU culling).
    bool CullsOcclusion() const override {
        return occlusion_ != Occlusion::Off;
    }

    /// Culls the instances views `firstView` to `firstView + views`
    /// draw against their depth pyramid `hzb`, between the views' two
    /// phases.
    void RecordOcclusionCull(VkCommandBuffer commands,
                             const GpuImage& hzb,
                             std::uint32_t firstView,
                             std::uint32_t views) const override;

    /// The instances each GPU culling job of an instancer's draw of
    /// the last frame kept, in job order (views, then shadow views;
    /// draws in order); waits for the device. For tests and
    /// statistics.
    std::vector<std::uint32_t> CulledInstanceCounts() const;

    /// The instances each GPU culling job of the last frame drew at
    /// each level of detail, the full mesh first; waits for the
    /// device.
    std::vector<std::vector<std::uint32_t>> CulledLevelCounts() const;

    /// The number of levels of detail of mesh `mesh`, the full mesh
    /// included: meshes of at least `kMinLodTriangles` triangles are
    /// simplified (meshoptimizer) into up to `kMaxLevels` levels of
    /// about half the triangles of the one before.
    /// @throws std::invalid_argument for an unknown mesh.
    std::size_t LevelCount(std::uint32_t mesh) const;

    static constexpr std::size_t kMaxLevels = 8;
    static constexpr std::size_t kMinLodTriangles = 128;

    /// One run of a draw's instances seen from one view; see
    /// cull.slang.
    struct GpuCullJob {
        std::uint32_t range[4];  ///< Draw, first instance id,
                                 ///< instances, frustum (planes / 6).
        /// First visible id and first indirect draw, then of the
        /// second set of draws (two-pass occlusion culling's second
        /// pass).
        std::uint32_t output[4];
    };

    /// Triangles drawn per frame by the current draws, instances
    /// included.
    std::uint64_t TriangleCount() const;

    void RecordShadow(VkCommandBuffer commands,
                      const RasterFrame& frame, std::size_t cascade,
                      const glm::mat4& lightViewProjection,
                      const ShadowCasters& casters) const override;
    void RecordColor(VkCommandBuffer commands,
                     const RasterFrame& frame, std::size_t view,
                     DrawPhase phase = DrawPhase::All) const override;
    void
    RecordColorViews(VkCommandBuffer commands,
                     std::span<const RasterFrame> frames,
                     std::size_t first,
                     DrawPhase phase = DrawPhase::All) const override;

    /// Draws again what view `view` drew of the draws that moved
    /// since the last frame (their transforms, instance arrays or
    /// points), as the colour pass did, with the motion pipeline.
    void
    RecordMotion(VkCommandBuffer commands, const RasterFrame& frame,
                 std::size_t view,
                 const DescriptorResource& motion) const override;

  private:
    /// One level of detail of a mesh: `count` indices from `first`
    /// past the mesh's own first index (0 for the full mesh),
    /// simplified by at most `error` in mesh units.
    struct MeshLevel {
        std::uint32_t first = 0;
        std::uint32_t count = 0;
        float error = 0.0f;
    };

    struct Mesh {
        MeshData data;
        glm::vec3 low{0.0f};
        glm::vec3 high{0.0f};
        std::uint32_t firstIndex = 0;
        std::uint32_t firstVertex = 0;
        bool pointsDirty = false;
        /// Whether its levels of detail wait for its points: those it
        /// was added with were zeros holding the place of device
        /// ones.
        bool levelsPending = false;
        Tensor devicePoints;  ///< CUDA points replacing `data.points`
                              ///< on the GPU, if any.
        std::uintptr_t deviceStream =
            1;  ///< The CUDA stream `devicePoints` are written on.
        Tensor deviceNormals;  ///< Device normals replacing
                               ///< `data.normals`, if any.
        std::uintptr_t normalsStream = 1;
        bool normalsDirty = false;
        Tensor deviceColors;  ///< Device colours replacing
                              ///< `data.colors`, if any.
        std::uintptr_t colorsStream = 1;
        bool colorsDirty = false;
        Tensor deviceIndices;  ///< Device triangle indices replacing
                               ///< `data.indices` (zeros), if any.
        std::uintptr_t indicesStream = 1;
        bool indicesDirty = false;
        /// The view `MapIndices` last gave out, and whether the
        /// indices set are already in it.
        Tensor mappedIndices;
        bool indicesInPlace = false;
        /// Points and indices in use (`SetActiveCounts`), their
        /// stream, and its slot in `activeCounts_` (0 for none, else
        /// the slot plus 1), assigned when the geometry is rebuilt.
        Tensor activeCounts;
        std::uintptr_t countsStream = 1;
        bool countsDirty = false;
        std::uint32_t countSlot = 0;
        /// Of device points without bounds, the bounds of the points
        /// in use measured on the GPU (`RecordBounds`) frames ago,
        /// and the measurement before.
        std::optional<Bounds3> gpuPoints;
        std::optional<Bounds3> lastGpuPoints;
        bool pointsBounded =
            false;  ///< Device points set with bounds.
        /// Writes it on the GPU each frame (`SetWriter`).
        MeshWriter writer;
        /// Whether its normals follow its device points, recomputed
        /// on the GPU when they move: a mesh with normals whose
        /// points, but not normals, come from the device.
        bool recomputeNormals = false;
        /// Per point, the triangles around it (`adjacency_`), for
        /// recomputing normals: offsets (points + 1) then triangle
        /// indices, from `firstAdjacency`.
        std::vector<std::uint32_t> adjacency;
        std::uint32_t firstAdjacency = 0;
        /// Which half of the double-buffered points pool holds its
        /// points now; the other holds the last frame's while
        /// `pointsMoving`. The first write of a frame flips it
        /// (`pointsWritten`), so the last frame's points stay for
        /// motion blur.
        std::uint32_t pointSet = 0;
        bool pointsMoving = false;
        bool pointsWritten = false;
        /// The view `MapPoints` last gave out, and whether the points
        /// written are already in it.
        Tensor mappedPoints;
        bool pointsInPlace = false;
        std::vector<MeshLevel>
            levels;  ///< Its levels of detail, the full mesh first.
        std::vector<std::uint32_t>
            lodIndices;  ///< The coarser levels' indices, one after
                         ///< another.
        /// Texture coordinates of `lodIndices`' corners (after
        /// `data.uvs`, those of the full mesh's), and where both
        /// start in the texture coordinate pool, in corners.
        std::vector<float> lodUvs;
        std::uint32_t firstUv = 0;
    };

    struct Instancer {
        InstanceData data;  ///< Host arrays; for device instancers,
                            ///< host copies for bounds.
        DeviceInstanceData device;
        bool onDevice = false;
        std::size_t count = 0;
        std::uint32_t first = 0;
        bool dirty = false;
        /// Which half of the double-buffered instance pools holds its
        /// arrays now, as the mesh's `pointSet`.
        std::uint32_t set = 0;
        bool moving = false;
        bool written = false;
        /// Per array, the view `MapInstances` last gave out this
        /// frame, and whether the array written is already in it.
        std::array<Tensor, 4> mapped;
        std::array<bool, 4> inPlace{};
        /// Composed on the GPU from transform ops
        /// (`ComposedInstanceData`): the data, its first float in
        /// `composeRaw_`, first op in `composeOps_` and first parent
        /// record in `composeParents_`, and each instance's level
        /// (its depth under composed parents).
        bool composed = false;
        ComposedInstanceData composedData;
        std::uint32_t rawFirst = 0;
        std::uint32_t opsFirst = 0;
        std::uint32_t parentsFirst = 0;
        std::vector<std::uint32_t> levels;
        /// Of a device instancer without an extent, bounds of its
        /// positions and its largest scale measured on the GPU
        /// (`RecordBounds`) frames ago, and the measurement before,
        /// in place of positions copied to the host each frame.
        std::optional<Bounds3> gpuPositions;
        std::optional<Bounds3> lastGpuPositions;
        float gpuScale = 1.0f;
    };

    /// Whether an instancer colours its instances, in place of its
    /// prototypes' colours; composed ones do not.
    static bool HasInstanceColors(const Instancer& instancer);

    /// The bounds of a device instancer's positions from its GPU
    /// measurements, widened by its motion between the last two, and
    /// the largest scale; none until measured.
    static std::optional<std::pair<Bounds3, float>>
    MeasuredPositions(const Instancer& instancer);

    /// Reads the bounds the frames of upload slot `slot` measured,
    /// once they have finished, and chooses the instancers this
    /// frame measures in it.
    void ReadBounds(std::uint32_t slot);
    /// Measures the bounds of the chosen instancers' positions into
    /// the upload slot's readback buffer (instance_bounds.slang).
    void RecordBounds(VkCommandBuffer commands) const;
    /// Writes this frame's bounds jobs (`boundsJobBuffer_`) from
    /// `boundsJobs_`, where the pools hold their positions now.
    void WriteBoundsJobs();
    /// Records the meshes' writers (`SetWriter`).
    void RecordWriters(VkCommandBuffer commands) const;
    /// Writes the draws of the meshes' triangles in use
    /// (`activeDraws_`), which direct draws of them go out as.
    void RecordActiveDraws(VkCommandBuffer commands) const;
    /// Whether the GPU measures the mesh's points (`RecordBounds`):
    /// device or written points without bounds.
    static bool MeasuresPoints(const Mesh& mesh);

    /// The G-buffer pipeline for a frame camera's views, or a
    /// pushbroom's.
    void CreateGBufferPipeline(bool pushbroom);
    void CreateCutoutPipelines(bool pushbroom);
    void CreateMinimalPipelines();
    /// Writes every draw's GPU record and instance ids, and their
    /// pipelines and id offsets.
    void WriteDraws();
    /// Rebuilds `transforms_` from the items and the origin.
    void UpdateTransforms();
    void ForgetBounds() {
        bounds_.reset();
        worldsBounds_.reset();
    }
    void RebuildGeometry();
    /// Simplifies a mesh's coarser levels of detail from its host
    /// data.
    void BuildLevels(Mesh& mesh) const;
    void RebuildInstances();
    /// Lays out the composed instancers: their raw arrays' and
    /// parents' offsets, their ops, and each instance's level, in
    /// instance lists per level (`composeJobs_`).
    void RebuildCompose();
    /// Picks the composed instancers to compose this frame (those
    /// written, and those under them, which move with them) and
    /// writes every instance's parent record.
    void PrepareCompose(bool all);
    /// Records the compose pass (`pendingCompose_`), level by level.
    void RecordCompose(VkCommandBuffer commands) const;
    /// Writes an instancer's arrays into its half of the pools; with
    /// `flip`, into the other half on its first write of a frame,
    /// keeping the last frame's arrays for motion blur.
    void WriteInstancer(Instancer& instancer, bool flip);
    void CopyDevicePoints(Mesh& mesh);
    /// Moves a mesh's points to the other half of the pool on its
    /// first write of a frame.
    void FlipPoints(Mesh& mesh);
    /// Likewise an instancer's arrays.
    void FlipInstancer(Instancer& instancer);
    /// Rebuilds the pools now if they are due, so their buffers exist
    /// to map.
    void EnsurePools();
    /// A float32 tensor of `shape` over `pool` from float `offset`,
    /// on the compute device.
    Tensor PoolView(const std::shared_ptr<GpuBuffer>& pool,
                    std::size_t offset,
                    std::vector<std::int64_t> shape) const;
    /// A mesh's points' offset in the pool, in points: now, or at the
    /// last frame with `previous`.
    std::uint32_t PointBase(const Mesh& mesh,
                            bool previous = false) const;
    /// Likewise an instancer's arrays' offset in the instance pools,
    /// in instances.
    std::uint32_t InstanceBase(const Instancer& instancer,
                               bool previous = false) const;
    /// Copies a mesh's device normals and colours that changed into
    /// the pools.
    void CopyDeviceAttributes(Mesh& mesh, bool all);
    std::shared_ptr<GpuBuffer>
    Pool(std::span<const std::byte> bytes) const;
    void CheckDeviceTensor(const Tensor& tensor,
                           std::size_t components,
                           const char* what) const;
    /// Draw `item`'s bindings; with `culled`, the visible ids of GPU
    /// culling at binding 18.
    std::vector<DescriptorResource>
    Resources(const RasterFrame& frame, std::size_t item,
              bool culled = false) const;
    /// A run of an instanced draw's instances in one world.
    struct WorldRange {
        std::int32_t world = -1;
        std::uint32_t first = 0;
        std::uint32_t count = 0;
    };
    /// One draw range a view or shadow view can see: range `range` of
    /// draw `item`.
    struct VisibleRange {
        std::uint32_t item = 0;
        std::uint32_t range = 0;
        std::int32_t job = -1;  ///< Its GPU culling job, or -1 to
                                ///< draw every instance.
        /// A plain mesh's job under occlusion culling: tested against
        /// the frustum alone and drawn with the occluders, which the
        /// depth pyramid is built from.
        bool occluder = false;
    };
    /// World bounds of draw `item`'s instances in `range`; none to
    /// draw it whatever the view.
    std::optional<Bounds3> RangeBounds(std::size_t item,
                                       const WorldRange& range) const;
    /// Lists the ranges each view and shadow view of the frame can
    /// see (frustum culling).
    void Cull(const FrameParams& params,
              std::span<const ShadowCascade> cascades);
    /// Uploads the frame's culling jobs, their frusta and the draws'
    /// culling data, and sizes the outputs for `visible` instances.
    void WriteCullJobs(const std::vector<GpuCullJob>& jobs,
                       const std::vector<glm::vec4>& views,
                       std::uint32_t visible, std::uint32_t draws);
    /// Uploads the draws' culling data (`cullItemsDirty_`): each
    /// draw's transform, bounds, levels and instances.
    void WriteCullItems();
    /// Uploads `bytes` into `buffer`, grown to hold them.
    void UploadCull(std::unique_ptr<GpuBuffer>& buffer,
                    std::span<const std::byte> bytes);
    /// Records `ranges`, those `casters` draws when given, with the
    /// shadow pipeline or each draw's G-buffer pipeline; GPU-culled
    /// ranges draw indirectly. Records the direct draws among them
    /// with `direct`, and the GPU-culled ones from their first (0) or
    /// second (1) set of draws, or none (-1). Returns the draw calls
    /// it recorded. With `motion`, the view's MotionUniforms: records
    /// with the motion pipeline instead. What `Record` last bound
    /// into a command buffer: kept across the views of
    /// `RecordColorViews`, whose frames bind the same resources.
    struct RecordState {
        const GraphicsPipeline* bound = nullptr;
        std::uint32_t pushedMaterial = 0xFFFFFFFFu;
        bool pushedCulled = false;
    };
    /// With `state`, starts from what it says is bound and leaves it
    /// as bound after.
    std::size_t Record(VkCommandBuffer commands,
                       const RasterFrame& frame,
                       std::span<const VisibleRange> ranges,
                       const glm::mat4& light,
                       const ShadowCasters* casters, bool shadow,
                       bool direct, int culledSet,
                       const DescriptorResource* motion = nullptr,
                       RecordState* state = nullptr) const;
    /// `RecordColor`, from and into `state`.
    void RecordColorState(VkCommandBuffer commands,
                          const RasterFrame& frame, std::size_t view,
                          DrawPhase phase, RecordState* state) const;
    /// Whether draw `item` moved since the last frame: its transform,
    /// instance arrays or points.
    bool Moving(std::size_t item) const;
    /// Records culling jobs `firstJob` to `firstJob + jobs` in
    /// cull.slang's mode `mode`, against the depth pyramid `hzb`
    /// whose layer 0 is view `firstView`.
    void Dispatch(VkCommandBuffer commands, VkImageView hzb,
                  std::uint32_t firstJob, std::uint32_t jobs,
                  std::uint32_t mode, std::uint32_t firstView) const;

    RasterRenderer& renderer_;
    const VulkanContext& context_;
    PipelineLayout layout_;
    PipelineLayout
        motionLayout_;  ///< `layout_` and the MotionUniforms at 20.
    /// The MotionVector var's pipeline, made the first time a frame
    /// holds the var.
    mutable std::unique_ptr<GraphicsPipeline> motionPipeline_;
    std::unique_ptr<GraphicsPipeline> shadowPipeline_;
    /// The G-buffer pipelines for frame cameras and, once one is
    /// seen, pushbrooms.
    std::array<std::unique_ptr<GraphicsPipeline>, 2>
        gbufferPipelines_;
    /// The cutout variants (`MaterialSettings::Cutout`) of the
    /// G-buffer pipelines and the shadow pipeline, made the first
    /// time a frame draws a cutout material.
    std::array<std::unique_ptr<GraphicsPipeline>, 2> cutoutPipelines_;
    std::unique_ptr<GraphicsPipeline> shadowCutoutPipeline_;
    /// A virtual shadow map's levels' (`RasterFrame::virtualShadow`):
    /// plain and cut out, made the first time one draws.
    mutable std::array<std::unique_ptr<GraphicsPipeline>, 2>
        virtualShadowPipelines_;
    /// The virtual shadow pipeline, cut out or not, made if new.
    const GraphicsPipeline* VirtualShadowPipeline(bool cutout) const;
    bool pushbroom_ = false;  ///< Whether the last `Prepare`'s camera
                              ///< is a pushbroom.
    /// The last `Prepare`'s minimal shading, and its pipelines
    /// (`MinimalTargets`): shaded, its cutout variant, then without
    /// colour and its cutout variant.
    MinimalMode minimal_ = MinimalMode::Off;
    std::array<std::unique_ptr<GraphicsPipeline>, 4>
        minimalPipelines_;

    /// Whether item `item`'s material cuts out part of it.
    bool Cutout(std::size_t item) const;
    /// The levels of detail GPU culling picks among for item `item`:
    /// its mesh's for an instancer's draws, the full mesh alone for a
    /// draw of its own.
    std::uint32_t CullLevels(std::size_t item) const;

    std::map<std::uint32_t, Mesh> meshes_;
    std::map<std::uint32_t, Instancer> instancers_;
    std::uint32_t nextHandle_ = 1;
    bool geometryDirty_ = true;
    bool instancesDirty_ = true;
    ComputeInterop* compute_ = nullptr;
    std::vector<MeshItem> items_;
    /// Per draw, its instances by world: one range of every instance
    /// without instance worlds.
    std::vector<std::vector<WorldRange>> itemRanges_;
    std::vector<std::int32_t>
        viewWorlds_;  ///< The world each view of the frame sees.
    std::vector<std::vector<VisibleRange>>
        viewDraws_;  ///< Per view, the ranges it sees.
    std::vector<std::vector<VisibleRange>>
        shadowDraws_;  ///< Per shadow view, likewise.
    /// Per shadow view, its draws by world (-1 shared), for passes of
    /// one world's casters.
    std::vector<
        std::unordered_map<std::int32_t, std::vector<VisibleRange>>>
        shadowWorldDraws_;
    mutable std::vector<std::size_t>
        viewCalls_;  ///< Per view, the draw calls it recorded last.
    std::vector<const GraphicsPipeline*> itemPipelines_;
    /// Per draw, its material as the renderer has it (the default for
    /// one it lacks), as of `WriteDraws`.
    std::vector<std::uint32_t> itemMaterials_;

    /// Points and instance arrays are double-buffered for motion
    /// blur: two halves of `pointTotal_` points or `instanceTotal_`
    /// instances each.
    std::shared_ptr<GpuBuffer> points_;
    std::uint32_t pointTotal_ = 0;
    std::uint32_t instanceTotal_ = 0;
    std::shared_ptr<GpuBuffer> normals_;
    std::shared_ptr<GpuBuffer> colors_;
    std::shared_ptr<GpuBuffer> indices_;
    std::unique_ptr<GpuBuffer>
        uvs_;  ///< Meshes' texture coordinates per corner
               ///< (`Mesh::firstUv`).
    /// Every recomputing mesh's `Mesh::adjacency`, one after another.
    std::unique_ptr<GpuBuffer> adjacency_;
    PipelineLayout normalsLayout_;
    std::unique_ptr<ComputePipeline>
        normalsPipeline_;  ///< mesh_normals.slang.
    PipelineLayout composeLayout_;
    std::unique_ptr<ComputePipeline>
        composePipeline_;  ///< instance_compose.slang.
    PipelineLayout boundsLayout_;
    /// instance_bounds.slang's clear and measure.
    std::unique_ptr<ComputePipeline> boundsClearPipeline_;
    std::unique_ptr<ComputePipeline> boundsPipeline_;
    /// Per upload slot: the readback buffer its frames measure into,
    /// and the instancers and meshes it measured (handle and count),
    /// in order.
    std::array<std::unique_ptr<GpuBuffer>, FrameUploads::kSlots>
        boundsReadback_;
    std::array<std::vector<std::pair<std::uint32_t, std::size_t>>,
               FrameUploads::kSlots>
        boundsJobs_;
    std::uint32_t boundsSlot_ = 0;
    /// This frame's bounds jobs as instance_bounds.slang reads them,
    /// and the workgroups along x each job's row has.
    std::unique_ptr<GpuBuffer> boundsJobBuffer_;
    std::uint32_t boundsGroups_ = 1;
    PipelineLayout activeDrawsLayout_;
    /// active_draws.slang: meshes' indirect draws of their triangles
    /// in use.
    std::unique_ptr<ComputePipeline> activeDrawsPipeline_;
    /// Per slot of `activeCounts_`, 8 uints: the draw of its
    /// triangles in use (VkDrawIndirectCommand), then the triangles
    /// the mesh holds.
    std::unique_ptr<GpuBuffer> activeDraws_;
    std::uint32_t countSlots_ = 0;
    /// Composed instancers' raw op arrays (written by the compute
    /// API), their ops (uint4 each), the instance lists of each level
    /// and instancer, and per instance its parent record.
    std::shared_ptr<GpuBuffer> composeRaw_;
    std::unique_ptr<GpuBuffer> composeOps_;
    std::unique_ptr<GpuBuffer> composeLists_;
    std::unique_ptr<GpuBuffer> composeParents_;
    /// One dispatch of the compose pass: an instancer's instances at
    /// one level, in level order.
    struct ComposeJob {
        std::uint32_t instancer = 0;
        std::uint32_t list = 0;
        std::uint32_t count = 0;
        std::uint32_t level = 0;
    };
    std::vector<ComposeJob> composeJobs_;
    /// The jobs of instancers composed this frame, in level order
    /// (`Prepare`).
    std::vector<ComposeJob> pendingCompose_;
    bool composeLayoutDirty_ = false;
    /// Views posed under instances (`SetAnchoredViews`), their
    /// records for this frame (nine float4 each,
    /// instance_anchors.slang), and the pass writing their poses.
    std::vector<AnchoredView> anchoredViews_;
    std::unique_ptr<GpuBuffer> anchorRecords_;
    std::uint32_t anchorViews_ = 0;
    PipelineLayout anchorLayout_;
    std::unique_ptr<ComputePipeline> anchorPipeline_;
    /// Lights posed under instances (`SetAnchoredLights`) and their
    /// records for this frame.
    std::vector<AnchoredView> anchoredLights_;
    std::unique_ptr<GpuBuffer> anchorLightRecords_;
    std::uint32_t anchorLights_ = 0;
    /// The shadow views the GPU fits, as their matrix
    /// (`ShadowCascade::gpuMatrix`) and cull view slot, and the pass
    /// patching those cull views (light_views.slang).
    std::vector<std::pair<std::uint32_t, std::uint32_t>>
        gpuShadowSlots_;
    std::unique_ptr<GpuBuffer> gpuShadowSlotBuffer_;
    PipelineLayout lightCullLayout_;
    std::unique_ptr<ComputePipeline> lightCullPipeline_;
    /// The meshes whose normals the next frame recomputes, as their
    /// handles.
    std::vector<std::uint32_t> pendingNormals_;
    std::shared_ptr<GpuBuffer> positions_;
    std::shared_ptr<GpuBuffer> orientations_;
    std::shared_ptr<GpuBuffer> scales_;
    /// Linear RGB per instance, laid out as `scales_`; 1 where an
    /// instancer has none.
    std::shared_ptr<GpuBuffer> instanceColors_;
    /// Meshes' points and triangle indices in use (int32 pairs), by
    /// `Mesh::countSlot`; read by the vertex and bounds shaders.
    std::shared_ptr<GpuBuffer> activeCounts_;
    /// Meshes with writers (`SetWriter`).
    std::set<std::uint32_t> writers_;
    std::unique_ptr<GpuBuffer> instanceIds_;
    /// Per indirect draw of GPU culling: its draw, instance offset
    /// and index offset, read by draw index in batched draws
    /// (mesh.slang's binding 19).
    std::unique_ptr<GpuBuffer> drawRecords_;
    std::vector<std::uint32_t>
        itemIdOffsets_;  ///< Per draw, its first entry in
                         ///< `instanceIds_`.
    std::vector<std::uint32_t>
        itemIndices_;  ///< Per draw, its mesh's index count.
    std::vector<std::uint32_t>
        itemCountSlots_;  ///< Per draw, its mesh's `countSlot`.

    Culling culling_ = Culling::Gpu;
    bool drawsDirty_ =
        true;  ///< Whether the draws changed since `WriteDraws`.
    /// Each item's transform relative to `origin_` (`SetOrigin`), in
    /// float for the GPU and the bounds; rebuilt when the items or
    /// the origin change.
    std::vector<glm::mat4> transforms_;
    /// Each item's world transform at the last frame, and relative to
    /// `origin_` as `transforms_`.
    std::vector<glm::dmat4> previous_;
    std::vector<glm::mat4> previousTransforms_;
    glm::dvec3 origin_{0.0};
    /// The renderer's `MaterialSetVersion` at `WriteDraws`, which
    /// resolves material ids.
    std::uint64_t drawsMaterials_ = 0;
    bool posesOnGpu_ = false;  ///< Whether the views' cull planes
                               ///< come from the GPU poses.
    Occlusion occlusion_ = Occlusion::Off;
    std::vector<std::uint32_t>
        viewJobStarts_;  ///< Per view, its first job; then the job
                         ///< count.
    std::unique_ptr<GpuImage> noHzb_;
    /// Two-pass occlusion culling: per (view, instance id), whether
    /// it was visible last frame.
    std::unique_ptr<GpuBuffer> visibility_;
    std::pair<std::uint32_t, std::uint32_t>
        visibilityKey_{};  ///< Views, instance ids.
    std::uint32_t visibilityStride_ = 0;
    mutable bool resetVisibility_ = false;
    PipelineLayout cullLayout_;
    std::unique_ptr<ComputePipeline> cullPipeline_;
    std::vector<GpuCullJob> cullJobs_;
    std::unique_ptr<GpuBuffer> cullItemBuffer_;
    std::unique_ptr<GpuBuffer> cullJobBuffer_;
    std::unique_ptr<GpuBuffer> cullViewBuffer_;
    std::unique_ptr<GpuBuffer> cullLevelBuffer_;
    /// Whether `cullItemBuffer_` and `cullLevelBuffer_` are out of
    /// date: they change only with the draws (`WriteDraws`).
    bool cullItemsDirty_ = true;
    /// `Bounds` and `WorldsBounds` as last computed, until the items,
    /// their transforms, points or instances change
    /// (`ForgetBounds`).
    mutable std::optional<std::optional<Bounds3>> bounds_;
    mutable std::optional<std::map<std::int32_t, Bounds3>>
        worldsBounds_;
    /// Meshes with work for the next frame or to settle after it:
    /// points, normals or colours written or mapped.
    std::set<std::uint32_t> activeMeshes_;
    /// Whether `SetItems` may have moved items `SettleMotion` has yet
    /// to settle.
    bool itemsMoving_ = false;
    /// A draw range: its item and its index in the item's ranges.
    using RangeIndex = std::pair<std::size_t, std::size_t>;
    /// Every draw range in item order, and each world's (-1 the
    /// shared ones'), as of `layoutVersion_` `rangeIndexVersion_`
    /// (`Cull`).
    std::vector<RangeIndex> everyRange_;
    std::unordered_map<std::int32_t, std::vector<RangeIndex>>
        rangesOfWorld_;
    std::uint64_t rangeIndexVersion_ = ~std::uint64_t{0};
    std::vector<std::uint32_t>
        cullDrawCounts_;  ///< Per job, its indirect draws (levels).
    /// Counts changes to what the cull lists are made of: the items,
    /// their ranges, instances and levels (`SetItems`,
    /// `RebuildInstances`, `RebuildGeometry`).
    std::uint64_t layoutVersion_ = 0;
    /// The views' part of the last cull lists, kept while nothing
    /// they depend on changes: with the views' poses on the GPU no
    /// bounds are tested on the CPU, so a batch of thousands of views
    /// makes them once rather than every frame (`Cull`).
    struct ViewCullLayout {
        bool valid = false;
        std::uint64_t layoutVersion = 0;
        std::vector<std::uint32_t> idOffsets;
        std::vector<std::int32_t> worlds;
        Culling culling = Culling::Off;
        Occlusion occlusion = Occlusion::Off;
        std::vector<GpuCullJob> jobs;
        std::vector<glm::uvec4> records;
        std::vector<std::uint32_t> drawCounts;
        std::uint32_t visible = 0;
        std::uint32_t draws = 0;
        std::vector<std::vector<VisibleRange>> viewDraws;
        std::vector<std::uint32_t> jobStarts;
    };
    ViewCullLayout viewLayout_;
    std::unique_ptr<GpuBuffer>
        visibleIds_;  ///< The jobs' visible instance ids, compacted.
    std::unique_ptr<GpuBuffer>
        cullCommands_;  ///< The jobs' indirect draws.
    std::unique_ptr<GpuBuffer> draws_;
};

}  // namespace ernie
