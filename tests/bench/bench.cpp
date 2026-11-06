// SPDX-License-Identifier: BSD-3-Clause
#include "bench.h"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "ernie/device/adapter.h"
#include "ernie/device/queue.h"
#include "ernie/util/error.h"
#include "ernie/util/profiler.h"

namespace ernie::bench {
namespace {

std::string& Wanted() {
    static std::string wanted;
    return wanted;
}

std::map<device::Traversal, Ref<device::Device>>& Devices() {
    static std::map<device::Traversal, Ref<device::Device>> devices;
    return devices;
}

profiler::ZoneId BenchZone() {
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Bench");
    return zone;
}

double GpuSeconds() {
    profiler::BeginFrame();
    const std::vector<profiler::Frame> frames = profiler::Frames();
    if (frames.empty()) {
        return 0;
    }
    const std::vector<profiler::Track> tracks = profiler::Tracks();
    uint64_t ns = 0;
    for (const profiler::Zone& zone : frames.back().zones) {
        if (zone.id == BenchZone() && tracks.at(zone.track).gpu) {
            ns += zone.endNs - zone.startNs;
        }
    }
    return double(ns) * 1e-9;
}

}

void SelectAdapter(std::string name) {
    Wanted() = std::move(name);
}

std::vector<std::string>& Patterns() {
    static std::vector<std::string> patterns;
    return patterns;
}

bool Core(std::string pattern) {
    Patterns().push_back(std::move(pattern));
    return true;
}

const std::vector<std::string>& CorePatterns() {
    return Patterns();
}

device::DeviceDesc BenchDesc(device::Traversal traversal) {
    device::DeviceDesc desc{.traversal = traversal};
    if (Wanted().empty()) {
        return desc;
    }
    for (const device::AdapterInfo& adapter : device::EnumerateAdapters()) {
        if (adapter.unsupported.empty() &&
            (adapter.alias == Wanted() || adapter.name.find(Wanted()) != std::string::npos)) {
            desc.adapter = adapter.alias;
            return desc;
        }
    }
    throw Error("no supported adapter matches " + Wanted());
}

device::Device& BenchDevice(device::Traversal traversal) {
    Ref<device::Device>& device = Devices()[traversal];
    if (!device) {
        device = device::Device::Create(BenchDesc(traversal));
    }
    return *device;
}

std::vector<std::function<void()>>& Releases() {
    static std::vector<std::function<void()>> releases;
    return releases;
}

void Release() {
    for (const std::function<void()>& release : Releases()) {
        release();
    }
    for (auto& [traversal, device] : Devices()) {
        device->GetQueue().WaitIdle();
    }
    Devices().clear();
}

void TimeGpu(benchmark::State& state, const Record& work, const Record& prepare) {
    device::Queue& queue = BenchDevice().GetQueue();
    const auto run = [&] {
        device::CommandEncoder encoder = queue.CreateEncoder();
        if (prepare) {
            prepare(encoder);
        }
        {
            const device::ZoneScope zone(encoder, BenchZone());
            work(encoder);
        }
        queue.Submit(encoder.Finish()).Wait();
        return GpuSeconds();
    };
    run();
    for (auto _ : state) {
        const double seconds = run();
        if (seconds <= 0) {
            state.SkipWithError("the device recorded no GPU time");
            break;
        }
        state.SetIterationTime(seconds);
    }
}

}
