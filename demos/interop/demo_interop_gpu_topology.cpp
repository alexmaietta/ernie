// SPDX-License-Identifier: BSD-3-Clause
// Metaballs polygonised by Vulkan compute shaders inside each frame,
// straight into the renderer's memory
// (RenderDelegate::SetMeshWriter): their points and triangles change
// number as the blobs merge and split, within a capacity, with no
// copy at all.

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/device/buffer.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr std::uint32_t kCells = 48;
constexpr std::uint32_t kGridPoints =
    (kCells + 1) * (kCells + 1) * (kCells + 1);
constexpr std::int64_t kMaxPoints = 1 << 16;
constexpr std::int64_t kMaxTriangles = 1 << 17;
const Path kBlobs("/World/Blobs");

/// Marching tetrahedra over a grid of the blobs' field: each cell as
/// six tetrahedra along its diagonal (Kuhn's), so every edge joins a
/// grid point to one above it and holds one shared point.
///
/// References:
/// - A. Doi and A. Koide, "An efficient method of triangulating
///   equi-valued surfaces by using tetrahedral cells", IEICE
///   Transactions E74(1), 214-224, 1991.
/// - J. F. Blinn, "A generalization of algebraic surface drawing",
///   ACM Transactions on Graphics 1(3), 235-256, 1982.
constexpr const char* kPolygoniseSlang = R"(
static const int kCells = 48;
static const int kSide = kCells + 1;
static const float3 kLow = float3(-1.2, -1.2, 0.0);
static const float kSpacing = 2.4 / kCells;

// The field less 1 at each grid point, each grid edge's point (or
// -1), and the points and triangles emitted.
[[vk::binding(0, 0)]] RWStructuredBuffer<float> values;
[[vk::binding(1, 0)]] RWStructuredBuffer<int> edges;
[[vk::binding(2, 0)]] RWStructuredBuffer<int> counters;

// The mesh's arrays, from its writer.
struct Push {
    float* points;
    float* normals;
    uint* indices;
    int* counts;
    float time;
    int blobs;
    int pointCapacity;
    int triangleCapacity;
}
[[vk::push_constant]] ConstantBuffer<Push> push;

// Blobs drifting on Lissajous paths, meeting and parting.
float4 blob(int i) {
    let phase = push.time * (0.5 + 0.13 * i);
    return float4(0.65 * sin(phase + 1.7 * i), 0.65 * cos(1.3 * phase + 0.9 * i),
                  0.75 + 0.4 * sin(0.8 * phase + 2.3 * i), 0.28);
}

float field(float3 p) {
    float f = 0.0;
    for (int i = 0; i < push.blobs; ++i) {
        let b = blob(i);
        let d = p - b.xyz;
        f += b.w * b.w / (dot(d, d) + 1e-6);
    }
    return f;
}

// The surface's outward normal: down the field's slope.
float3 outward(float3 p) {
    float3 g = float3(0.0);
    for (int i = 0; i < push.blobs; ++i) {
        let b = blob(i);
        let d = p - b.xyz;
        let q = dot(d, d) + 1e-6;
        g += 2.0 * b.w * b.w / (q * q) * d;
    }
    return normalize(g);
}

int gridIndex(int3 p) { return (p.x * kSide + p.y) * kSide + p.z; }
int3 gridOf(int g) { return int3(g / (kSide * kSide), (g / kSide) % kSide, g % kSide); }
float3 gridPoint(int3 p) { return kLow + kSpacing * float3(p); }
float3 pointAt(int v) { return float3(push.points[3 * v], push.points[3 * v + 1], push.points[3 * v + 2]); }
float3 normalAt(int v) { return float3(push.normals[3 * v], push.normals[3 * v + 1], push.normals[3 * v + 2]); }

[shader("compute")]
[numthreads(64, 1, 1)]
void sampleField(uint3 id: SV_DispatchThreadID) {
    if (id.x == 0) {
        counters[0] = 0;
        counters[1] = 0;
    }
    if (id.x < kSide * kSide * kSide) {
        values[id.x] = field(gridPoint(gridOf(id.x))) - 1.0;
    }
}

// A point where the surface cuts each grid edge.
[shader("compute")]
[numthreads(64, 1, 1)]
void cutEdges(uint3 id: SV_DispatchThreadID) {
    let tid = int(id.x);
    if (tid >= 7 * kSide * kSide * kSide) {
        return;
    }
    let g = tid / 7;
    let d = tid % 7 + 1;
    let a = gridOf(g);
    let b = a + int3(d & 1, (d >> 1) & 1, (d >> 2) & 1);
    edges[tid] = -1;
    if (any(b > kCells)) {
        return;
    }
    let fa = values[g];
    let fb = values[gridIndex(b)];
    if ((fa > 0.0) == (fb > 0.0)) {
        return;
    }
    int v;
    InterlockedAdd(counters[0], 1, v);
    if (v >= push.pointCapacity) {
        return;
    }
    let p = lerp(gridPoint(a), gridPoint(b), fa / (fa - fb));
    let n = outward(p);
    for (int k = 0; k < 3; ++k) {
        push.points[3 * v + k] = p[k];
        push.normals[3 * v + k] = n[k];
    }
    edges[tid] = v;
}

// The point on the edge between two corners, either way round.
int edgeOf(int3 p, int3 q) {
    let swap = p.x + p.y + p.z > q.x + q.y + q.z;
    let a = swap ? q : p;
    let d = (swap ? p : q) - a;
    return edges[gridIndex(a) * 7 + d.x + 2 * d.y + 4 * d.z - 1];
}

// A triangle, turned to face out.
void emit(int i, int j, int k) {
    if (i < 0 || j < 0 || k < 0) {
        return;
    }
    int t;
    InterlockedAdd(counters[1], 1, t);
    if (t >= push.triangleCapacity) {
        return;
    }
    let face = cross(pointAt(j) - pointAt(i), pointAt(k) - pointAt(i));
    let flip = dot(face, normalAt(i) + normalAt(j) + normalAt(k)) < 0.0;
    push.indices[3 * t] = i;
    push.indices[3 * t + 1] = flip ? k : j;
    push.indices[3 * t + 2] = flip ? j : k;
}

// Each tetrahedron's piece of the surface: none, a triangle, or a
// quad as two.
[shader("compute")]
[numthreads(64, 1, 1)]
void triangulate(uint3 id: SV_DispatchThreadID) {
    let tid = int(id.x);
    if (tid >= 6 * kCells * kCells * kCells) {
        return;
    }
    let cell = tid / 6;
    let order = tid % 6;
    let first = order / 2;
    let second = (first + 1 + order % 2) % 3;
    var v = int3[4](int3(cell / (kCells * kCells), (cell / kCells) % kCells, cell % kCells),
                    int3(0), int3(0), int3(0));
    v[1] = v[0];
    v[1][first] += 1;
    v[2] = v[1];
    v[2][second] += 1;
    v[3] = v[0] + int3(1);
    int mask = 0;
    for (int c = 0; c < 4; ++c) {
        mask |= values[gridIndex(v[c])] > 0.0 ? 1 << c : 0;
    }
    if (mask == 0 || mask == 15) {
        return;
    }
    // A lone corner (one in or one out), else two by two.
    let bits = countbits(uint(mask));
    if (bits != 2) {
        let lone = int(firstbitlow(uint(bits == 1 ? mask : 15 - mask)));
        let o1 = (lone + 1) % 4;
        let o2 = (lone + 2) % 4;
        let o3 = (lone + 3) % 4;
        emit(edgeOf(v[lone], v[o1]), edgeOf(v[lone], v[o2]), edgeOf(v[lone], v[o3]));
        return;
    }
    let a = int(firstbitlow(uint(mask)));
    let b = int(firstbithigh(uint(mask)));
    let rest = uint(15 & ~mask);
    let e = int(firstbitlow(rest));
    let f = int(firstbithigh(rest));
    let ae = edgeOf(v[a], v[e]);
    let af = edgeOf(v[a], v[f]);
    let bf = edgeOf(v[b], v[f]);
    let be = edgeOf(v[b], v[e]);
    emit(ae, af, bf);
    emit(ae, bf, be);
}

// The points and triangles in use, within the capacity.
[shader("compute")]
[numthreads(1, 1, 1)]
void finish() {
    push.counts[0] = min(counters[0], push.pointCapacity);
    push.counts[1] = min(counters[1], push.triangleCapacity);
}
)";

/// The shaders' own buffers: the field, edge points and counters.
constexpr std::array<DescriptorBinding, 3> kBindings{{
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
}};

/// What the blobs do now, which every frame's shaders read.
struct BlobState {
    float time = 0.0f;
    int count = 5;
};

/// The compute pipelines and scratch buffers on one renderer's
/// device, and the writer that records them into its frames.
class Polygoniser {
  public:
    Polygoniser(const VulkanContext& context, const BlobState& state)
        : state_(state), layout_(context, kBindings, sizeof(Push)),
          values_(context, kGridPoints * sizeof(float),
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  BufferMemory::Device),
          edges_(context, 7 * kGridPoints * sizeof(std::int32_t),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 BufferMemory::Device),
          counters_(context, 2 * sizeof(std::int32_t),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    BufferMemory::Device) {
        const ShaderCompiler compiler;
        const std::array<ShaderModuleSource, 1> modules{
            {{"polygonise", kPolygoniseSlang}}};
        for (const char* entry :
             {"sampleField", "cutEdges", "triangulate", "finish"}) {
            pipelines_.push_back(std::make_unique<ComputePipeline>(
                context, layout_,
                compiler.CompileSpirv(modules, entry)));
        }
    }

    /// Records the passes into `target`, one after another.
    void Record(VkCommandBuffer commands,
                const MeshWriteTarget& target) const {
        const Push push{
            target.points,
            target.normals,
            target.indices,
            target.counts,
            state_.time,
            state_.count,
            static_cast<std::int32_t>(target.pointCapacity),
            static_cast<std::int32_t>(target.triangleCapacity)};
        const std::array<DescriptorResource, 3> resources{
            DescriptorResource::Of(values_),
            DescriptorResource::Of(edges_),
            DescriptorResource::Of(counters_)};
        const std::array<std::uint32_t, 4> threads{
            kGridPoints, 7 * kGridPoints,
            6 * kCells * kCells * kCells, 1};
        for (std::size_t p = 0; p < pipelines_.size(); ++p) {
            vkCmdBindPipeline(commands,
                              VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipelines_[p]->Handle());
            layout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                         resources);
            layout_.PushConstants(commands, &push, sizeof(push));
            vkCmdDispatch(commands, (threads[p] + 63) / 64, 1, 1);
            VkMemoryBarrier2 memory{
                VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            memory.srcStageMask =
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            memory.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            memory.dstStageMask =
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            memory.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT |
                                   VK_ACCESS_2_SHADER_WRITE_BIT;
            VkDependencyInfo dependency{
                VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.memoryBarrierCount = 1;
            dependency.pMemoryBarriers = &memory;
            vkCmdPipelineBarrier2(commands, &dependency);
        }
    }

    /// The points and triangles the last frame emitted (before the
    /// capacity), for tests.
    std::array<std::int32_t, 2> Emitted() const {
        std::array<std::int32_t, 2> counts{};
        counters_.Read(std::as_writable_bytes(std::span(counts)));
        return counts;
    }

  private:
    struct Push {
        VkDeviceAddress points, normals, indices, counts;
        float time;
        std::int32_t blobs, pointCapacity, triangleCapacity;
    };

    const BlobState& state_;
    PipelineLayout layout_;
    std::vector<std::unique_ptr<ComputePipeline>> pipelines_;
    GpuBuffer values_;
    GpuBuffer edges_;
    GpuBuffer counters_;
};

class GpuTopologyDemo : public app::Demo {
  public:
    GpuTopologyDemo(app::Viewer* viewer,
                    const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 60;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Ground").SetChecker(true);
        DistantLight::Define(scene, "/World/Sun");
        Material::Define(scene, "/World/Looks/Goo")
            .SetAlbedo(glm::dvec3(0.1, 0.4, 0.8));
        // Placeholders at the capacity, which the writer fills.
        const std::vector<float> zeros(3 * kMaxPoints, 0.0f);
        const std::vector<std::int32_t> threes(kMaxTriangles, 3);
        const std::vector<std::int32_t> indices(3 * kMaxTriangles, 0);
        Mesh::Define(scene, kBlobs)
            .SetPoints(Tensor::Copy<float>(zeros, {kMaxPoints, 3}))
            .SetNormals(Tensor::Copy<float>(zeros, {kMaxPoints, 3}))
            .SetFaceVertexCounts(
                Tensor::Copy<std::int32_t>(threes, {kMaxTriangles}))
            .SetFaceVertexIndices(Tensor::Copy<std::int32_t>(
                indices, {3 * kMaxTriangles}))
            .SetMaterialBinding("/World/Looks/Goo");
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(30.0)
            .SetTransform(LookAtTransform(
                {3.2, -3.2, 2.4}, {0.0, 0.0, 0.6}, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({960, 540});

        // Written in the frames of whichever renderers draw the
        // scene.
        Write(GetRenderer().Delegate(), GetRenderer().Context());
        if (viewer != nullptr) {
            Write(viewer->Delegate(), viewer->Context());
            viewer->SetActiveCamera("/World/Camera");
        }
        Widgets().Slider("Blobs", 1.0, 8.0, state_.count,
                         [this](double value) {
                             state_.count = static_cast<int>(value);
                         });
        Widgets().Slider("Speed", 0.0, 3.0, speed_,
                         [this](double value) { speed_ = value; });
    }

    void Step() override {
        state_.time = static_cast<float>(speed_ * time);
    }

    void TestPostStep() override {
        // The viewer's, which draws each step, else the demo's own.
        emitted_.insert(polygonisers_.back()->Emitted());
    }

    /// Stops the writers and waits for the frames that use the
    /// shaders' buffers before they go.
    void Close() override {
        for (const Written& written : written_) {
            written.delegate->SetMeshWriter(kBlobs, nullptr);
            vkDeviceWaitIdle(written.context->Device());
        }
        written_.clear();
        polygonisers_.clear();
    }

    void TestFinal() override {
        const Tensor image = RenderImage();
        const auto [points, triangles] =
            polygonisers_.front()->Emitted();
        std::printf("%d points and %d triangles in use\n", points,
                    triangles);
        Check(points > 0 && points < kMaxPoints && triangles > 0 &&
                  triangles < kMaxTriangles,
              "the surface should fit its capacity");
        Check(emitted_.size() > 1, "the topology should change");
        const auto* texels =
            static_cast<const std::uint8_t*>(image.Data());
        const std::int64_t pixels =
            image.Shape()[0] * image.Shape()[1];
        std::int64_t blue = 0;
        for (std::int64_t i = 0; i < pixels; ++i) {
            const std::uint8_t* t = texels + 4 * i;
            blue += t[2] - std::max(t[0], t[1]) > 30 ? 1 : 0;
        }
        const double covered =
            static_cast<double>(blue) / static_cast<double>(pixels);
        std::printf("the blobs cover %.1f%% of the image\n",
                    100.0 * covered);
        Check(covered > 0.03, "the blobs should be drawn");
    }

  private:
    void Write(RenderDelegate& delegate,
               const VulkanContext& context) {
        polygonisers_.push_back(
            std::make_unique<Polygoniser>(context, state_));
        const Polygoniser* polygoniser = polygonisers_.back().get();
        written_.push_back({&delegate, &context});
        delegate.SetMeshWriter(
            kBlobs, [polygoniser](VkCommandBuffer commands,
                                  const MeshWriteTarget& target) {
                polygoniser->Record(commands, target);
            });
    }

    BlobState state_;
    double speed_ = 1.0;
    std::vector<std::unique_ptr<Polygoniser>> polygonisers_;
    /// The delegates writing the blobs, and their devices.
    struct Written {
        RenderDelegate* delegate;
        const VulkanContext* context;
    };
    std::vector<Written> written_;
    std::set<std::array<std::int32_t, 2>> emitted_;
};

}  // namespace

ERNIE_REGISTER_DEMO(GpuTopologyDemo, "interop", "gpu_topology",
                    "Metaballs polygonised by Vulkan compute shaders "
                    "in each frame, their topology changing within a "
                    "capacity.");
