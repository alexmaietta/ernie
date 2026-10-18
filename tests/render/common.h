// SPDX-License-Identifier: BSD-3-Clause
/// @file common.h
/// @brief Helpers shared by the render tests: a ray-tracing fixture over a device scene, one trace
/// kernel, and test geometry.
#pragma once

#include "ernie/device/kernel.h"
#include "ernie/device/staging.h"
#include "ernie/render/device_scene.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <glm/vec4.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ernie::test {

/// @brief Authors a mesh's points (x, y, z each) and faces.
inline void WriteMesh(Scene& scene, PrimId mesh, const std::vector<float>& points,
                      const std::vector<std::int32_t>& counts,
                      const std::vector<std::int32_t>& indices) {
    const auto size = [](const auto& values) { return static_cast<std::int64_t>(values.size()); };
    scene.Write(mesh, "points", Tensor::Copy(std::span(points), {size(points) / 3, 3}));
    scene.Write(mesh, "faceVertexCounts", Tensor::Copy(std::span(counts), {size(counts)}));
    scene.Write(mesh, "faceVertexIndices", Tensor::Copy(std::span(indices), {size(indices)}));
}

/// @brief Defines a square mesh at height `z`, `half` metres from its centre to each edge, wound
/// to face +z.
inline PrimId DefineSquare(Scene& scene, const Path& path, float half, float z) {
    const PrimId square = scene.Define(path, "Mesh");
    WriteMesh(scene, square, {-half, -half, z, half, -half, z, half, half, z, -half, half, z}, {4},
              {0, 1, 2, 3});
    return square;
}

/// A ray as `kTrace` reads it: origin and direction, each padded to four floats.
struct Ray {
    glm::vec4 origin;
    glm::vec4 direction;
};

/// What `kTrace` writes for a ray that hits nothing, and for one `TraceAny` and `TraceClosest`
/// disagree on.
inline constexpr float kMiss = -1.0f;
inline constexpr float kDisagree = -2.0f;

/// Traces each ray, writing its hit distance and world normal, `kMiss`, or `kDisagree`.
inline const ShaderModule kTrace{"trace", R"(
import ray_tracer;
import ref;
struct Ray { float4 origin; float4 direction; };
struct Parameters { TracerParameters tracer; Ref<Ray> rays; Ref<float4> hits; uint count; };

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uniform Parameters parameters) {
    if (id.x >= parameters.count) {
        return;
    }
    Ray r = parameters.rays[id.x];
    RayDesc ray = { r.origin.xyz, 0, r.direction.xyz, 1e30 };
    RayTracer tracer = RayTracer(parameters.tracer);
    Hit hit = tracer.TraceClosest(ray, 0xff);
    float4 result = float4(-1, 0, 0, 0);
    if (hit.found != tracer.TraceAny(ray, 0xff)) {
        result.x = -2;
    } else if (hit.found) {
        result = float4(hit.t, tracer.WorldNormal(hit));
    }
    parameters.hits[id.x] = result;
}
)"};

/// @brief A validating context whose device ray traces, arenas and a scene, with the scene's
/// device scene made on first use. The test fails on any validation message.
class RayTracingTest : public ::testing::Test {
  protected:
    void SetUp() override {
        context_ = MakeContext();
        if (!context_) {
            GTEST_SKIP() << "No Vulkan device";
        }
        ERNIE_REQUIRE_RAY_QUERIES(*context_);
        arenas_ = std::make_unique<Arenas>(*context_, MemoryUse::Device, "scene", 1 << 20);
    }
    void TearDown() override {
        if (context_) {
            context_->WaitIdle();
            ExpectClean(*context_);
        }
    }

    DeviceScene& Device() {
        if (!device_) {
            device_ = std::make_unique<DeviceScene>(*context_, *arenas_, scene_, compiler_);
        }
        return *device_;
    }

    /// @return Each ray's hit after a frame of the device scene: its distance, then its world
    /// normal.
    std::vector<glm::vec4> Trace(std::span<const Ray> rays);

    /// @return A column's first `rows` rows as `T`s, once `after` is reached.
    template <typename T>
    std::vector<T> Read(const DeviceColumn& column, std::size_t rows, const Ticket& after) {
        Readback readback(*context_, context_->Main(), column.Storage(), column.Range().offset,
                          rows * column.RowBytes(), std::span(&after, 1));
        return readback.Values<T>();
    }

    ShaderCompiler compiler_;
    std::unique_ptr<DeviceContext> context_;
    std::unique_ptr<Arenas> arenas_;
    Scene scene_;
    std::unique_ptr<DeviceScene> device_;

  private:
    struct TraceParameters {
        TracerParameters tracer;
        shaders::Ref<Ray> rays;
        shaders::Ref<glm::vec4> hits;
        std::uint32_t count;
    };
    std::unique_ptr<ComputeKernel> trace_;
    std::unique_ptr<DeviceColumn> rays_;
    std::unique_ptr<DeviceColumn> hits_;
};

inline std::vector<glm::vec4> RayTracingTest::Trace(std::span<const Ray> rays) {
    if (!trace_) {
        std::vector<ShaderModule> modules(RayTracer::Modules().begin(), RayTracer::Modules().end());
        modules.push_back(kTrace);
        trace_ = std::make_unique<ComputeKernel>(*context_, compiler_.Spirv(modules, "main"),
                                                 "trace", std::uint32_t{sizeof(TraceParameters)});
        rays_ = std::make_unique<DeviceColumn>(*arenas_, "rays", DType::Float32,
                                               std::vector<std::int64_t>{8}, 0);
        hits_ = std::make_unique<DeviceColumn>(*arenas_, "hits", DType::Float32,
                                               std::vector<std::int64_t>{4}, 0);
    }
    rays_->Resize(rays.size(), context_->Main());
    hits_->Resize(rays.size(), context_->Main());
    Uploader uploader(*context_, context_->Main());
    uploader.Write(rays_->Storage(), rays_->Range().offset, rays);
    const Ticket uploaded = uploader.Flush();
    const auto count = static_cast<std::uint32_t>(rays.size());
    const Ticket done = Device().Execute(
        [&](graph::Graph& graph) {
            const TraceScene& scene = graph.Board().Get<TraceScene>();
            const graph::Buffer from = graph.Import(*rays_);
            graph::Buffer to = graph.Import(*hits_);
            graph.Add(
                "Trace", graph::Kind::Compute,
                [&](graph::PassBuilder& pass) {
                    scene.Read(pass);
                    pass.ReadAddress(from);
                    to = pass.WriteAddress(to, graph::Use::StorageWrite);
                },
                [this, from, to, count, tracer = scene.parameters](graph::PassContext& pass) {
                    trace_->Dispatch(pass.commands, {(count + 63) / 64, 1, 1},
                                     TraceParameters{tracer, pass.Get(from).Ref<Ray>(),
                                                     pass.Get(to).Ref<glm::vec4>(), count});
                });
        },
        std::span(&uploaded, 1));
    return Read<glm::vec4>(*hits_, rays.size(), done);
}

}  // namespace ernie::test
