// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/memory_stats.h"

#include <mutex>
#include <unordered_map>
#include <utility>

namespace ernie::memory {

namespace {

struct State {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, Allocation> live;
};

State& Global() {
    static State state;
    return state;
}

}  // namespace

void Track(std::uint64_t handle, Allocation allocation) {
    State& s = Global();
    std::lock_guard lock(s.mutex);
    s.live[handle] = std::move(allocation);
}

void Untrack(std::uint64_t handle) {
    State& s = Global();
    std::lock_guard lock(s.mutex);
    s.live.erase(handle);
}

std::vector<Allocation> Live() {
    State& s = Global();
    std::lock_guard lock(s.mutex);
    std::vector<Allocation> live;
    live.reserve(s.live.size());
    for (const auto& [handle, allocation] : s.live) {
        live.push_back(allocation);
    }
    return live;
}

}  // namespace ernie::memory
