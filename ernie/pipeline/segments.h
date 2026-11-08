// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "ernie/pipeline/resolve.h"

namespace ernie::pipeline {

inline constexpr uint32_t kBeforeSegments = ~0u;
inline constexpr uint64_t kNotStored = ~uint64_t(0);
inline constexpr uint64_t kArenaAlignment = 256;

struct Segment {
    NodeKind kind = NodeKind::Point;
    std::vector<uint32_t> nodes;
};

struct StoredValue {
    uint32_t value = kNoValue;
    uint32_t first = kBeforeSegments;
    uint32_t last = 0;
};

struct SegmentPlan {
    std::vector<Segment> segments;
    std::vector<StoredValue> stored;
};

struct ArenaLayout {
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
};

SegmentPlan SplitSegments(const ResolvedPipeline& pipeline, const NodeRegistry& registry);

ArenaLayout LayoutArena(const SegmentPlan& plan, std::span<const SignalLayout> layouts,
                        uint64_t elements);

}
