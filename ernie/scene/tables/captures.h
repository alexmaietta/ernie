// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <map>
#include <mutex>

#include "ernie/scene/tables/products.h"

namespace ernie {

struct SimulationClock {
    double start = 0;
    double end = 0;
    uint64_t steps = 0;
    uint64_t resets = 0;
};

class CaptureSchedule {
  public:
    CaptureSchedule(Scene& scene, const SimulationClock& clock, const ProductTable& products)
        : m_scene(scene), m_clock(clock), m_products(products) {}

    void Schedule();
    bool Due(uint32_t prim) const;

  private:
    struct Entry {
        float frameRate = 0;
        float startTime = 0;
        int64_t next = 0;
        uint64_t step = ~uint64_t(0);
        bool due = true;
    };

    Scene& m_scene;
    const SimulationClock& m_clock;
    const ProductTable& m_products;
    mutable std::mutex m_mutex;
    std::map<uint32_t, Entry> m_entries;
    uint64_t m_resets = 0;
};

}
