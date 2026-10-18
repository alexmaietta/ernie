// SPDX-License-Identifier: BSD-3-Clause
/// @file worker_pool.h
/// @brief A small FIFO thread pool for background jobs.
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

/// @brief A few threads running jobs in submission order.
/// On destruction, running jobs finish and queued ones are dropped
/// (their futures report `std::future_error`).
class WorkerPool {
  public:
    /// @brief Starts the threads.
    /// @param threads The thread count; at least one is started.
    /// @param name    The profiler name prefix ("Worker 1", ...).
    explicit WorkerPool(unsigned threads, const std::string& name = "Worker") {
        for (unsigned i = 0; i < std::max(threads, 1u); ++i) {
            threads_.emplace_back([this, thread = name + " " + std::to_string(i + 1)] {
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

    /// @brief Queues a job.
    /// @param job A callable taking no arguments.
    /// @return A future for its result or exception.
    template <class Job> std::future<std::invoke_result_t<Job>> Submit(Job job) {
        using Result = std::invoke_result_t<Job>;
        auto task = std::make_shared<std::packaged_task<Result()>>(std::move(job));
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
                wake_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
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
