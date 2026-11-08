// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <fmt/format.h>

#include "bench.h"
#include "ernie/device/queue.h"
#include "ernie/render/ray_cast/ray_cast.h"
#include "ernie/scene/scene.h"
#include "support/geometry.h"

namespace ernie {
namespace {

using device::Traversal;

constexpr uint32_t kRays = 1 << 20;

device::Tensor Upload(device::Device& device, const std::vector<float>& values) {
    device::Tensor tensor = device.CreateTensor(device::DType::Float32, {kRays, 3}, "rays");
    device.GetQueue().Write(tensor.GetBuffer().Slice(tensor.Offset(), values.size() * 4),
                            values.data());
    return tensor;
}

Mesh::Desc Soup(uint32_t triangles) {
    Mesh::Desc mesh;
    for (const bvh::Triangle& triangle : test::Soup(triangles, 1)) {
        for (const float* corner : {triangle.a, triangle.b, triangle.c}) {
            mesh.faceVertexIndices.push_back(int32_t(mesh.points.size()));
            mesh.points.push_back({corner[0], corner[1], corner[2]});
        }
        mesh.faceVertexCounts.push_back(3);
    }
    return mesh;
}

struct Tracing {
    Ref<Scene> scene;
    RayCaster caster;
    Rays rays;
    RayHits hits;
};

Tracing Prepare(Traversal traversal, uint32_t triangles) {
    device::Device& device = bench::BenchDevice(traversal);
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<Mesh>("/Soup", Soup(triangles));

    std::mt19937 random(1);
    std::uniform_real_distribution<float> unit(0, 1);
    std::normal_distribution<float> normal;
    std::vector<float> origins;
    std::vector<float> directions;
    for (uint32_t i = 0; i < kRays; ++i) {
        const float around[3] = {normal(random), normal(random), normal(random)};
        const float length =
            std::sqrt(around[0] * around[0] + around[1] * around[1] + around[2] * around[2]);
        for (int axis = 0; axis < 3; ++axis) {
            const float origin = 0.5f + 1.5f * around[axis] / length;
            origins.push_back(origin);
            directions.push_back(unit(random) - origin);
        }
    }

    Tracing tracing{scene, RayCaster(scene), {}, {}};
    tracing.rays.origins = Upload(device, origins);
    tracing.rays.directions = Upload(device, directions);
    tracing.caster.Cast(tracing.rays, tracing.hits).Wait();
    return tracing;
}

void TraceRays(benchmark::State& state) {
    const bool any = state.range(0) != 0;
    const auto triangles = uint32_t(state.range(1));
    const auto traversal = state.range(2) != 0 ? Traversal::Hardware : Traversal::Software;
    if (traversal == Traversal::Hardware &&
        bench::BenchDevice().GetTraversal() != Traversal::Hardware) {
        state.SkipWithMessage("the device has no hardware ray tracing");
        return;
    }
    Tracing& tracing = bench::Cached<Tracing>(fmt::format("{}/{}", int(traversal), triangles),
                                              [&] { return Prepare(traversal, triangles); });
    const RayCastDesc desc{.mode = any ? RayCastMode::Any : RayCastMode::Closest,
                           .outputs = RayOutput::Distance};
    for (auto _ : state) {
        tracing.caster.Cast(tracing.rays, tracing.hits, desc).Wait();
    }
    state.SetItemsProcessed(int64_t(state.iterations()) * kRays);
}

BENCHMARK(TraceRays)
    ->Name("Bvh/Trace")
    ->ArgNames({"any", "triangles", "hardware"})
    ->ArgsProduct({{0, 1}, {100'000, 1'000'000}, {0, 1}})
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

}
}
