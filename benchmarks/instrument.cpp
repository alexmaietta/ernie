// SPDX-License-Identifier: BSD-3-Clause
#include "instrument.h"

#include "bench.h"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace ernie::bench {

namespace {

thread_local Observations* gActive = nullptr;
thread_local bool gRecords = false;
thread_local std::vector<std::string> gStack;

std::uint64_t NowNs() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

}  // namespace

ActiveObservations::ActiveObservations(Observations& observations,
                                       bool records)
    : previous_(gActive), previousRecords_(gRecords),
      previousStack_(std::move(gStack)) {
    gActive = &observations;
    gRecords = records;
    gStack.clear();
}

ActiveObservations::~ActiveObservations() {
    gActive = previous_;
    gRecords = previousRecords_;
    gStack = std::move(previousStack_);
}

Measure::Measure(std::string name) {
    if (gActive == nullptr) {
        return;
    }
    active_ = true;
    gStack.push_back(std::move(name));
    start_ = NowNs();
}

Measure::~Measure() {
    if (!active_ || gActive == nullptr || gStack.empty()) {
        return;
    }
    const double ms = static_cast<double>(NowNs() - start_) / 1e6;
    std::string path;
    for (const std::string& part : gStack) {
        path += (path.empty() ? "" : "/") + part;
    }
    gStack.pop_back();
    gActive->regions[path].push_back(ms);
}

void Record(const std::string& name, double value,
            const std::string& unit) {
    if (gActive == nullptr || !gRecords) {
        return;
    }
    if (!std::isfinite(value)) {
        throw std::invalid_argument("Recorded metric " + name +
                                    " is not finite");
    }
    auto& [values, kept] = gActive->records[name];
    if (!values.empty() && kept != unit) {
        throw std::invalid_argument("Recorded metric " + name +
                                    " changed units from " + kept +
                                    " to " + unit);
    }
    kept = unit;
    values.push_back(value);
}

}  // namespace ernie::bench
