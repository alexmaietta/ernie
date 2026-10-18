// SPDX-License-Identifier: BSD-3-Clause
/// @file bench.h
/// @brief Registering and timing benchmarks, which report against a target and never fail on time.
#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace ernie::bench {

/// What a benchmark measured: its fastest run and the time it aims for, in milliseconds.
struct Timing {
    double ms = 0.0;
    double targetMs = 0.0;
};

struct Benchmark {
    std::string name;
    std::function<Timing()> run;
};

/// @return Every benchmark registered at static initialization.
inline std::vector<Benchmark>& Registry() {
    static std::vector<Benchmark> benchmarks;
    return benchmarks;
}

/// Registers a benchmark from a namespace-scope `const bool`.
inline bool Register(std::string name, std::function<Timing()> run) {
    Registry().push_back({std::move(name), std::move(run)});
    return true;
}

/// @brief Times `step` `runs` times, after `setup` before each, to see past a noisy machine.
/// @return The fastest run, in milliseconds.
template <typename Setup, typename Step>
double Fastest(const Setup& setup, const Step& step, int runs = 5) {
    double best = 1e30;
    for (int run = 0; run < runs; ++run) {
        setup();
        const auto start = std::chrono::steady_clock::now();
        step();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        best = std::min(best, std::chrono::duration<double, std::milli>(elapsed).count());
    }
    return best;
}

}  // namespace ernie::bench
