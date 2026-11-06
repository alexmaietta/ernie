// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <format>
#include <optional>
#include <span>
#include <vector>

#include "bench.h"
#include "ernie/bvh/builder.h"
#include "ernie/bvh/common.h"
#include "ernie/bvh/lbvh.h"
#include "ernie/bvh/reference.h"
#include "support/buffers.h"
#include "support/geometry.h"

namespace ernie::bvh {
namespace {

struct Meshes {
    LbvhBuilder builder;
    Ref<device::Buffer> vertices;
    Ref<device::Buffer> output;
};

void BuildBlas(benchmark::State& state) {
    const auto count = uint32_t(state.range(0));
    Meshes& meshes = bench::Cached<Meshes>(std::format("{}", count), [&] {
        device::Device& device = bench::BenchDevice();
        return Meshes{LbvhBuilder(device, count), test::Upload(device, test::Soup(count, 1)),
                      device.CreateBuffer({.bytes = StorageBytes(count, kBvhTriangles)})};
    });
    const Geometry geometry = device::AccelTriangles{
        .vertices = *meshes.vertices, .vertexCount = count * 3, .count = count};
    const device::BufferRange output = *meshes.output;
    bench::TimeGpu(state, [&](device::CommandEncoder& encoder) {
        meshes.builder.Build(encoder, std::span(&geometry, 1), std::span(&output, 1));
    });
    state.SetItemsProcessed(int64_t(state.iterations()) * count);
}

BENCHMARK(BuildBlas)
    ->Name("Bvh/Lbvh/Blas")
    ->ArgName("triangles")
    ->Arg(10'000)
    ->Arg(100'000)
    ->Arg(1'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

struct Worlds {
    LbvhBuilder builder;
    std::vector<Ref<device::Buffer>> blases;
    Ref<device::Buffer> records;
    Ref<device::Buffer> output;
};

void BuildTlas(benchmark::State& state) {
    const auto count = uint32_t(state.range(0));
    Worlds& worlds = bench::Cached<Worlds>(std::format("{}", count), [&] {
        device::Device& device = bench::BenchDevice();
        Worlds made{LbvhBuilder(device, count),
                    {},
                    nullptr,
                    device.CreateBuffer({.bytes = StorageBytes(count, kBvhInstances)})};
        for (uint32_t b = 0; b < 3; ++b) {
            made.blases.push_back(test::Upload(device, BuildReference(test::Soup(64, b + 1))));
        }
        test::Instances instances = test::Scatter(count, 1);
        for (uint32_t i = 0; i < count; ++i) {
            instances.records[i].blas = made.blases[i % 3]->Address();
        }
        made.records = test::Upload(device, instances.records);
        return made;
    });
    const device::BufferRange output = *worlds.output;
    bench::TimeGpu(state, [&](device::CommandEncoder& encoder) {
        worlds.builder.Build(encoder, *worlds.records, std::span(&count, 1), std::nullopt,
                             std::span(&output, 1));
    });
    state.SetItemsProcessed(int64_t(state.iterations()) * count);
}

BENCHMARK(BuildTlas)
    ->Name("Bvh/Lbvh/Tlas")
    ->ArgName("instances")
    ->Arg(1'000)
    ->Arg(64'000)
    ->Arg(1'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

}
}
