// SPDX-License-Identifier: BSD-3-Clause
/// @file parallel.h
/// @brief Runs a loop's iterations across CPU threads.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace ernie {

/// @brief Calls body(i) for every i in [0, count), spread over half the machine's threads.
/// Iterations are handed out one at a time, so uneven ones balance. Half the threads leaves the
/// rest of the machine responsive while tables and textures are built.
/// @param count The number of iterations.
/// @param body  Called once per iteration, from any thread, possibly concurrently.
template <typename Body> void ParallelFor(std::size_t count, const Body& body) {
    std::atomic<std::size_t> next{0};
    const auto work = [&] {
        for (std::size_t i; (i = next.fetch_add(1)) < count;) {
            body(i);
        }
    };
    const std::size_t threads = std::clamp<std::size_t>(std::thread::hardware_concurrency() / 2, 1,
                                                        std::max<std::size_t>(count, 1));
    std::vector<std::thread> pool;
    for (std::size_t t = 1; t < threads; ++t) {
        pool.emplace_back(work);
    }
    work();
    for (std::thread& thread : pool) {
        thread.join();
    }
}

}  // namespace ernie
