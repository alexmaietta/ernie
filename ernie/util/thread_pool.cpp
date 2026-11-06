// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/thread_pool.h"

#include <algorithm>
#include <utility>

namespace ernie {

ThreadPool::~ThreadPool() {
    std::deque<Job> dropped;
    {
        const std::lock_guard lock(m_mutex);
        m_stopping = true;
        dropped.swap(m_jobs);
    }
    m_ready.notify_all();
    for (std::thread& thread : m_threads) {
        thread.join();
    }
}

void ThreadPool::Add(const void* key, std::function<void()> job) {
    {
        const std::lock_guard lock(m_mutex);
        if (m_threads.empty()) {
            for (unsigned i = 0; i < m_size; ++i) {
                m_threads.emplace_back([this] { Run(); });
            }
        }
        m_jobs.push_back({key, std::move(job)});
    }
    m_ready.notify_one();
}

std::function<void()> ThreadPool::Take(const void* key) {
    const std::lock_guard lock(m_mutex);
    const auto found = std::ranges::find(m_jobs, key, &Job::key);
    if (found == m_jobs.end()) {
        return {};
    }

    std::function<void()> taken = std::move(found->run);
    m_jobs.erase(found);
    return taken;
}

bool ThreadPool::Cancel(const void* key) {
    return bool(Take(key));
}

void ThreadPool::Run() {
    for (;;) {
        Job job;
        {
            std::unique_lock lock(m_mutex);
            m_ready.wait(lock, [this] { return m_stopping || !m_jobs.empty(); });
            if (m_stopping) {
                return;
            }
            job = std::move(m_jobs.front());
            m_jobs.pop_front();
        }
        job.run();
    }
}

}
