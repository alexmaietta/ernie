// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <format>
#include <vector>

#include "bench.h"
#include "ernie/device/queue.h"
#include "ernie/device/tensor.h"

namespace ernie::device {
namespace {

constexpr int64_t kRowWords = 1024;

enum class Layout : int64_t {
    Contiguous,
    EveryOtherRow,
    Transposed,
};

struct Tensors {
    std::vector<uint32_t> host;
    Tensor source;
    Tensor target;
};

Tensors& Prepare(int64_t bytes, int64_t copies) {
    return bench::Cached<Tensors>(std::format("{}/{}", bytes, copies), [&] {
        Device& device = bench::BenchDevice();
        const int64_t words = bytes / 4;
        std::vector<uint32_t> host(size_t(words * copies));
        for (size_t at = 0; at < host.size(); ++at) {
            host[at] = uint32_t(at * 2654435761u);
        }
        const Ref<Buffer> buffer = device.CreateBuffer({.label = "bench tensor"}, host);
        host.resize(size_t(words));
        return Tensors{std::move(host), Tensor(buffer, 0, DType::UInt32, {words * copies}),
                       device.CreateTensor(DType::UInt32, {words}, "bench tensor target")};
    });
}

void Read(benchmark::State& state, Layout layout) {
    const int64_t bytes = state.range(0);
    const int64_t words = bytes / 4;
    const int64_t rows = words / kRowWords;
    const Tensors& tensors = Prepare(bytes, layout == Layout::EveryOtherRow ? 2 : 1);
    const Ref<Buffer> buffer(&tensors.source.GetBuffer());
    const Tensor view =
        layout == Layout::Contiguous ? tensors.source
        : layout == Layout::EveryOtherRow
            ? Tensor(buffer, 0, DType::UInt32, {rows, kRowWords}, {2 * kRowWords, 1})
            : Tensor(buffer, 0, DType::UInt32, {kRowWords, rows}, {1, kRowWords});
    for (auto _ : state) {
        benchmark::DoNotOptimize(view.ToHostBytes());
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * bytes);
}

void ReadTyped(benchmark::State& state) {
    const int64_t bytes = state.range(0);
    const Tensors& tensors = Prepare(bytes, 1);
    for (auto _ : state) {
        benchmark::DoNotOptimize(tensors.source.ToHost<uint32_t>());
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * bytes);
}

void ReadAsync(benchmark::State& state) {
    const int64_t bytes = state.range(0);
    const Tensors& tensors = Prepare(bytes, 1);
    for (auto _ : state) {
        const Download download = tensors.source.ToHostAsync();
        download.Wait();
        benchmark::DoNotOptimize(download.Bytes().data());
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * bytes);
}

void Write(benchmark::State& state) {
    const int64_t bytes = state.range(0);
    Tensors& tensors = Prepare(bytes, 1);
    Queue& queue = bench::BenchDevice().GetQueue();
    for (auto _ : state) {
        queue.Write(tensors.target.GetBuffer().Slice(tensors.target.Offset(), uint64_t(bytes)),
                    tensors.host.data());
        queue.WaitIdle();
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * bytes);
}

void Import(benchmark::State& state) {
    const int64_t bytes = state.range(0);
    Tensors& tensors = Prepare(bytes, 1);
    Device& device = bench::BenchDevice();
    int64_t shape[] = {bytes / 4};
    const DLTensor host{.data = tensors.host.data(),
                        .device = {kDLCPU, 0},
                        .ndim = 1,
                        .dtype = {kDLUInt, 32, 1},
                        .shape = shape};
    for (auto _ : state) {
        const Tensor imported = device.Import(host);
        device.GetQueue().WaitIdle();
        benchmark::DoNotOptimize(imported.Address());
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * bytes);
}

void Copy(benchmark::State& state) {
    const int64_t bytes = state.range(0);
    const Tensors& tensors = Prepare(bytes, 1);
    Device& device = bench::BenchDevice();
    for (auto _ : state) {
        device.Copy(tensors.source, tensors.target).Wait();
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * bytes);
}

const bool kCore =
    bench::Core("^Device/Tensor/Read/layout:(contiguous|every_other_row)/bytes:8388608/") &&
    bench::Core("^Device/Tensor/Write/bytes:8388608/");

const bool kRegistered = [] {
    const std::vector<int64_t> sizes = {4 << 10, 1 << 20, 8 << 20, 64 << 20};
    const auto add = [&](benchmark::internal::Benchmark* benchmark) {
        for (const int64_t size : sizes) {
            benchmark->Arg(size);
        }
        benchmark->ArgName("bytes")->UseRealTime()->Unit(benchmark::kMicrosecond);
    };
    for (const auto& [layout, name] : {std::pair{Layout::Contiguous, "contiguous"},
                                       std::pair{Layout::EveryOtherRow, "every_other_row"},
                                       std::pair{Layout::Transposed, "transposed"}}) {
        add(benchmark::RegisterBenchmark(std::format("Device/Tensor/Read/layout:{}", name), Read,
                                         layout));
    }
    add(benchmark::RegisterBenchmark("Device/Tensor/ReadTyped", ReadTyped));
    add(benchmark::RegisterBenchmark("Device/Tensor/ReadAsync", ReadAsync));
    add(benchmark::RegisterBenchmark("Device/Tensor/Write", Write));
    add(benchmark::RegisterBenchmark("Device/Tensor/Import", Import));
    add(benchmark::RegisterBenchmark("Device/Tensor/Copy", Copy));
    return true;
}();

}
}
