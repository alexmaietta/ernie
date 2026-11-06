// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <format>
#include <vector>

#include "bench.h"
#include "ernie/algorithms/scan.h"
#include "support/buffers.h"

namespace ernie::algorithms {
namespace {

struct Scanning {
    Scan scan;
    Ref<device::Buffer> input;
    Ref<device::Buffer> output;
};

void ExclusiveSum(benchmark::State& state) {
    const auto count = uint32_t(state.range(0));
    Scanning& scanning = bench::Cached<Scanning>(std::format("{}", count), [&] {
        device::Device& device = bench::BenchDevice();
        std::vector<uint32_t> words = test::Random<uint32_t>(count, 1);
        for (uint32_t& word : words) {
            word &= 0xFF;
        }
        return Scanning{Scan(device, count), test::Upload(device, words),
                        device.CreateBuffer({.bytes = uint64_t(count) * 4})};
    });
    bench::TimeGpu(state, [&](device::CommandEncoder& encoder) {
        scanning.scan.ExclusiveSum(encoder, *scanning.input, *scanning.output, count);
    });
    state.SetItemsProcessed(int64_t(state.iterations()) * count);
    state.SetBytesProcessed(int64_t(state.iterations()) * count * 8);
}

BENCHMARK(ExclusiveSum)
    ->Name("Algorithms/Scan")
    ->ArgName("words")
    ->Arg(1 << 16)
    ->Arg(1 << 20)
    ->Arg(1 << 24)
    ->Arg(1 << 26)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

}
}
