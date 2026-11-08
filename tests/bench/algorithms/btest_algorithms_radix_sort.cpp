// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <format>
#include <utility>
#include <vector>

#include "bench.h"
#include "ernie/algorithms/radix_sort.h"
#include "support/buffers.h"

namespace ernie::algorithms {
namespace {

struct Sorting {
    RadixSort sort;
    Ref<device::Buffer> source;
    Ref<device::Buffer> keys;
    Ref<device::Buffer> values;
};

Sorting Prepare(uint32_t count, uint32_t keyBytes) {
    device::Device& device = bench::BenchDevice();
    Ref<device::Buffer> source = keyBytes == 8
                                     ? test::Upload(device, test::Random<uint64_t>(count, 1))
                                     : test::Upload(device, test::Random<uint32_t>(count, 1));
    return {RadixSort(device, count, keyBytes), source,
            device.CreateBuffer({.bytes = source->Bytes()}),
            device.CreateBuffer({.bytes = uint64_t(count) * 4})};
}

void SortKeys(benchmark::State& state) {
    const auto count = uint32_t(state.range(0));
    const auto keyBytes = uint32_t(state.range(1));
    const auto bits = uint32_t(state.range(2));
    Sorting& sorting = bench::Cached<Sorting>(std::format("{}/{}", count, keyBytes),
                                              [&] { return Prepare(count, keyBytes); });
    bench::TimeGpu(
        state,
        [&](device::CommandEncoder& encoder) {
            sorting.sort.Sort(encoder, *sorting.keys, *sorting.values, count, bits);
        },
        [&](device::CommandEncoder& encoder) { encoder.Copy(*sorting.source, *sorting.keys); });
    state.SetItemsProcessed(int64_t(state.iterations()) * count);
}

BENCHMARK(SortKeys)
    ->Name("Algorithms/RadixSort")
    ->ArgNames({"keys", "bytes", "bits"})
    ->Apply([](benchmark::internal::Benchmark* benchmark) {
        for (const int64_t count : {1 << 16, 1 << 20, 1 << 24}) {
            for (const auto& [bytes, bits] : {std::pair{4, 32}, {4, 30}, {8, 64}, {8, 30}}) {
                benchmark->Args({count, bytes, bits});
            }
        }
    })
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

}
}
