// SPDX-License-Identifier: BSD-3-Clause
/// @file worker_pool_test.cpp
/// @brief Tests of the background thread pool.
#include "ernie/util/worker_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ernie {
namespace {

TEST(WorkerPool, ReturnsEveryJobsResult) {
    WorkerPool pool(4);
    std::vector<std::future<int>> results;
    for (int i = 0; i < 100; ++i) {
        results.push_back(pool.Submit([i] { return i * i; }));
    }
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(results[static_cast<std::size_t>(i)].get(), i * i);
    }
}

TEST(WorkerPool, RunsJobsInSubmissionOrderOnOneThread) {
    WorkerPool pool(1);
    std::vector<int> order;
    std::vector<std::future<void>> done;
    for (int i = 0; i < 20; ++i) {
        done.push_back(pool.Submit([&order, i] { order.push_back(i); }));
    }
    for (std::future<void>& job : done) {
        job.get();
    }
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(order[static_cast<std::size_t>(i)], i);
    }
}

TEST(WorkerPool, PassesExceptionsToTheFuture) {
    WorkerPool pool(2);
    std::future<int> failing = pool.Submit([]() -> int { throw std::runtime_error("job failed"); });
    EXPECT_THROW(failing.get(), std::runtime_error);
    EXPECT_EQ(pool.Submit([] { return 7; }).get(), 7);
}

TEST(WorkerPool, StartsAtLeastOneThread) {
    WorkerPool pool(0);
    EXPECT_EQ(pool.Submit([] { return 1; }).get(), 1);
}

TEST(WorkerPool, DropsQueuedJobsOnDestruction) {
    std::atomic<int> ran{0};
    std::promise<void> release;
    std::future<void> queued;
    // Released only once the pool is being destroyed (its stop flag is the destructor's first
    // step), so the worker can't reach the queued job first.
    std::thread releaser;
    {
        WorkerPool pool(1);
        pool.Submit([&release] { release.get_future().wait(); });
        queued = pool.Submit([&ran] { ++ran; });
        releaser = std::thread([&release] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            release.set_value();
        });
    }
    releaser.join();
    EXPECT_THROW(queued.get(), std::future_error);
    EXPECT_EQ(ran.load(), 0);
}

}  // namespace
}  // namespace ernie
