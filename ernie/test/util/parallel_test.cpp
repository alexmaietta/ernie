// SPDX-License-Identifier: BSD-3-Clause
/// @file parallel_test.cpp
/// @brief Tests of ParallelFor.
#include "ernie/util/parallel.h"

#include <gtest/gtest.h>

#include <atomic>
#include <vector>

namespace ernie {
namespace {

TEST(Parallel, RunsEveryIterationExactlyOnce) {
    std::vector<std::atomic<int>> runs(1000);
    ParallelFor(runs.size(), [&](std::size_t i) { ++runs[i]; });
    for (const std::atomic<int>& count : runs) {
        EXPECT_EQ(count.load(), 1);
    }
}

TEST(Parallel, AcceptsNoIterations) {
    int runs = 0;
    ParallelFor(0, [&](std::size_t) { ++runs; });
    EXPECT_EQ(runs, 0);
}

}  // namespace
}  // namespace ernie
