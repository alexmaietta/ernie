// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/profile.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace ernie {

/// A few threads running jobs in the order they come, such as
/// textures decoded while frames render. Jobs still queued when it
/// goes are dropped (their futures report `std::future_error`);
/// running ones finish first.
class WorkerPool {
  public:
    /// `threads` threads (at least one), named `name` 1, 2... in the
    /// profilers.
    explicit WorkerPool(unsigned threads,
                        const std::string& name = "Worker") {
        for (unsigned i = 0; i < std::max(threads, 1u); ++i) {
            threads_.emplace_back(
                [this, thread = name + " " + std::to_string(i + 1)] {
                    profiler::SetThreadName(thread);
#if ERNIE_TRACY
                    tracy::SetThreadName(profiler::Intern(thread));
#endif
                    Run();
                });
        }
    }

    ~WorkerPool() {
        {
            const std::lock_guard lock(mutex_);
            stopping_ = true;
            jobs_.clear();
        }
        wake_.notify_all();
        for (std::thread& thread : threads_) {
            thread.join();
        }
    }

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    /// Queues `job`; its result (or exception) comes through the
    /// future.
    template <class Job>
    std::future<std::invoke_result_t<Job>> Submit(Job job) {
        using Result = std::invoke_result_t<Job>;
        auto task = std::make_shared<std::packaged_task<Result()>>(
            std::move(job));
        std::future<Result> result = task->get_future();
        {
            const std::lock_guard lock(mutex_);
            jobs_.emplace_back([task] { (*task)(); });
        }
        wake_.notify_one();
        return result;
    }

  private:
    void Run() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, [this] {
                    return stopping_ || !jobs_.empty();
                });
                if (stopping_) {
                    return;
                }
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            job();
        }
    }

    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::function<void()>> jobs_;
    bool stopping_ = false;
    std::vector<std::thread> threads_;
};

}  // namespace ernie
