// SPDX-License-Identifier: BSD-3-Clause
/// @file main.cpp
/// @brief Runs the registered benchmarks whose names contain the first argument, and prints each
/// fastest time beside its target.
#include "benchmarks/bench.h"

#include <cstdio>
#include <string_view>

int main(int argc, char** argv) {
    const std::string_view filter = argc > 1 ? argv[1] : "";
    std::printf("%-48s %12s %12s\n", "benchmark", "ms", "target ms");
    for (const ernie::bench::Benchmark& benchmark : ernie::bench::Registry()) {
        if (benchmark.name.find(filter) == std::string::npos) {
            continue;
        }
        const ernie::bench::Timing timing = benchmark.run();
        std::printf("%-48s %12.3f %12.3f%s\n", benchmark.name.c_str(), timing.ms, timing.targetMs,
                    timing.ms > timing.targetMs ? "  over" : "");
    }
    return 0;
}
