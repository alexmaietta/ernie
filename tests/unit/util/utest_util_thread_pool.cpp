// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/thread_pool.h"

#include <atomic>
#include <functional>
#include <future>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace ernie {
namespace {

class Gate {
  public:
    std::function<void()> BlockingJob() {
        return [this] {
            m_started.set_value();
            m_released.wait();
        };
    }

    void WaitUntilStarted() {
        m_started.get_future().wait();
    }

    void Release() {
        m_release.set_value();
    }

  private:
    std::promise<void> m_started;
    std::promise<void> m_release;
    std::shared_future<void> m_released = m_release.get_future().share();
};

TEST(ThreadPoolTest, ZeroThreadsStillMeansOne) {
    ThreadPool pool(0);
    std::promise<std::thread::id> ran;
    pool.Add(nullptr, [&ran] { ran.set_value(std::this_thread::get_id()); });
    EXPECT_NE(ran.get_future().get(), std::this_thread::get_id());
}

TEST(ThreadPoolTest, RunsEveryJobOnItsThreads) {
    ThreadPool pool(4);
    std::atomic<int> ran{0};
    std::vector<std::promise<void>> done(16);
    for (std::promise<void>& promise : done) {
        pool.Add(&promise, [&ran, &promise] {
            ++ran;
            promise.set_value();
        });
    }
    for (std::promise<void>& promise : done) {
        promise.get_future().wait();
    }
    EXPECT_EQ(ran, 16);
}

TEST(ThreadPoolTest, ATakenJobIsTheCallersToRun) {
    Gate gate;
    ThreadPool pool(1);
    pool.Add(nullptr, gate.BlockingJob());
    gate.WaitUntilStarted();

    const int key = 0;
    const int otherKey = 0;
    std::thread::id ranOn;
    pool.Add(&key, [&ranOn] { ranOn = std::this_thread::get_id(); });
    EXPECT_FALSE(pool.Take(&otherKey));
    const std::function<void()> taken = pool.Take(&key);
    ASSERT_TRUE(taken);
    EXPECT_FALSE(pool.Take(&key));
    taken();
    EXPECT_EQ(ranOn, std::this_thread::get_id());
    gate.Release();
}

TEST(ThreadPoolTest, CancelledJobsNeverRun) {
    Gate gate;
    ThreadPool pool(1);
    const int keys[2] = {0, 1};
    pool.Add(&keys[0], gate.BlockingJob());
    gate.WaitUntilStarted();

    std::atomic<int> ran{0};
    std::promise<void> last;
    pool.Add(&keys[1], [&ran] { ++ran; });
    pool.Add(nullptr, [&last] { last.set_value(); });
    EXPECT_FALSE(pool.Cancel(&keys[0]));
    EXPECT_TRUE(pool.Cancel(&keys[1]));
    EXPECT_FALSE(pool.Cancel(&keys[1]));
    gate.Release();
    last.get_future().wait();
    EXPECT_EQ(ran, 0);
}

TEST(ThreadPoolTest, AnUnusedPoolStartsNoThreads) {
    const ThreadPool pool(8);
}

}
}
