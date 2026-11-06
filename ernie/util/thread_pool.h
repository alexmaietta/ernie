// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ernie {

class ThreadPool {
  public:
    explicit ThreadPool(unsigned threads) : m_size(threads > 0 ? threads : 1) {}

    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void Add(const void* key, std::function<void()> job);

    std::function<void()> Take(const void* key);

    bool Cancel(const void* key);

  private:
    struct Job {
        const void* key = nullptr;
        std::function<void()> run;
    };

    void Run();

    unsigned m_size;
    std::mutex m_mutex;
    std::deque<Job> m_jobs;
    std::vector<std::thread> m_threads;
    bool m_stopping = false;
    std::condition_variable m_ready;
};

}
