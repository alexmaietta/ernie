// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include "ernie/device/commands.h"
#include "ernie/device/device.h"

namespace ernie::bench {

void SelectAdapter(std::string name);
bool Core(std::string pattern);
const std::vector<std::string>& CorePatterns();
device::DeviceDesc BenchDesc(device::Traversal traversal = device::Traversal::Auto);
device::Device& BenchDevice(device::Traversal traversal = device::Traversal::Auto);
void Release();

std::vector<std::function<void()>>& Releases();

template <class T, class Make> T& Cached(const std::string& key, Make make) {
    static std::string current;
    static std::unique_ptr<T> value;
    static const bool registered = [] {
        Releases().push_back([] { value.reset(); });
        return true;
    }();
    (void)registered;
    if (!value || current != key) {
        value.reset();
        value = std::make_unique<T>(make());
        current = key;
    }
    return *value;
}

using Record = std::function<void(device::CommandEncoder&)>;

void TimeGpu(benchmark::State& state, const Record& work, const Record& prepare = {});

}
