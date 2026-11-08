// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/captures.h"

#include <algorithm>
#include <cmath>

#include "ernie/scene/scene.h"
#include "ernie/util/log.h"

namespace ernie {

namespace {

constexpr double kCaptureTolerance = 1e-6;

}

void CaptureSchedule::Schedule() {
    const std::vector<std::shared_ptr<const ResolvedProduct>> products = m_products.Products();
    const std::lock_guard lock(m_mutex);
    if (m_clock.resets != m_resets) {
        m_resets = m_clock.resets;
        m_entries.clear();
    }
    std::erase_if(m_entries, [&](const auto& entry) {
        return std::ranges::none_of(
            products, [&](const auto& product) { return product->prim == entry.first; });
    });
    const double dt = m_clock.end - m_clock.start;
    for (const std::shared_ptr<const ResolvedProduct>& product : products) {
        Entry& entry = m_entries[product->prim];
        const float rate = product->frameRate;
        if (rate <= 0 || dt == 0) {
            entry.due = true;
            continue;
        }
        const float start = product->startTime;
        if (entry.frameRate != rate || entry.startTime != start) {
            const double first = std::ceil((m_clock.start - start) * rate - kCaptureTolerance);
            entry = {rate, start, std::max<int64_t>(0, int64_t(first))};
        }
        if (entry.step == m_clock.steps) {
            continue;
        }
        entry.step = m_clock.steps;
        const double reached = (m_clock.end - start) * rate + kCaptureTolerance;
        entry.due = double(entry.next) <= reached;
        if (!entry.due) {
            continue;
        }
        const auto last = int64_t(std::floor(reached));
        if (last > entry.next) {
            Log(LogLevel::Debug, "{} skipped {} in a step of {} s",
                m_scene.ByIndex(product->prim).Path(),
                Counted(uint64_t(last - entry.next), "capture"), dt);
        }
        entry.next = last + 1;
    }
}

bool CaptureSchedule::Due(uint32_t prim) const {
    const std::lock_guard lock(m_mutex);
    const auto found = m_entries.find(prim);
    return found == m_entries.end() || found->second.due;
}

}
