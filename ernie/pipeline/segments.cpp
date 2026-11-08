// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/pipeline/segments.h"

#include <algorithm>

namespace ernie::pipeline {

namespace {

int64_t Start(uint32_t first) {
    return first == kBeforeSegments ? -1 : int64_t(first);
}

}

SegmentPlan SplitSegments(const ResolvedPipeline& pipeline, const NodeRegistry& registry) {
    SegmentPlan plan;
    std::vector<uint32_t> segmentOfNode(pipeline.nodes.size(), kBeforeSegments);
    bool open = false;
    for (uint32_t node = 0; node < pipeline.nodes.size(); ++node) {
        const NodeKind kind = registry[pipeline.nodes[node].def].kind;
        if (kind == NodeKind::Source) {
            continue;
        }
        if (!(open && kind == NodeKind::Point)) {
            plan.segments.push_back({kind, {}});
            open = kind == NodeKind::Point || kind == NodeKind::Stencil;
        }
        plan.segments.back().nodes.push_back(node);
        segmentOfNode[node] = uint32_t(plan.segments.size() - 1);
    }

    const auto end = uint32_t(plan.segments.size());
    for (uint32_t value = 0; value < pipeline.values.size(); ++value) {
        const uint32_t made = segmentOfNode[pipeline.values[value].node];
        bool stored = false;
        uint32_t last = 0;
        for (uint32_t node = 0; node < pipeline.nodes.size(); ++node) {
            if (std::ranges::find(pipeline.nodes[node].inputs, value) ==
                pipeline.nodes[node].inputs.end()) {
                continue;
            }
            const uint32_t used = segmentOfNode[node];
            if (used != made || made == kBeforeSegments) {
                stored = true;
                last = std::max(last, used);
            }
        }
        const bool output = std::ranges::find(pipeline.outputs, value, &ResolvedOutput::value) !=
                            pipeline.outputs.end();
        if (output && made == kBeforeSegments) {
            stored = true;
            last = end;
        }
        if (stored) {
            plan.stored.push_back({value, made, last});
        }
    }
    return plan;
}

ArenaLayout LayoutArena(const SegmentPlan& plan, std::span<const SignalLayout> layouts,
                        uint64_t elements) {
    struct Slot {
        uint64_t offset = 0;
        uint64_t bytes = 0;
        int64_t last = 0;
    };

    ArenaLayout arena;
    arena.offsets.assign(layouts.size(), kNotStored);
    std::vector<const StoredValue*> order;
    for (const StoredValue& stored : plan.stored) {
        order.push_back(&stored);
    }
    std::ranges::stable_sort(order, {},
                             [](const StoredValue* stored) { return Start(stored->first); });
    std::vector<Slot> slots;
    for (const StoredValue* stored : order) {
        const SignalLayout& layout = layouts[stored->value];
        const uint64_t channels =
            layout.element == ElementType::F16 ? (layout.channels + 1) / 2 * 2 : layout.channels;
        const uint64_t raw = elements * channels * kElementBytes[uint32_t(layout.element)];
        const uint64_t bytes = (raw + kArenaAlignment - 1) / kArenaAlignment * kArenaAlignment;
        Slot* best = nullptr;
        for (Slot& slot : slots) {
            if (slot.last < Start(stored->first) && slot.bytes >= bytes &&
                (best == nullptr || slot.bytes < best->bytes)) {
                best = &slot;
            }
        }
        if (best == nullptr) {
            best = &slots.emplace_back(Slot{arena.bytes, bytes, 0});
            arena.bytes += bytes;
        }
        best->last = int64_t(stored->last);
        arena.offsets[stored->value] = best->offset;
    }
    return arena;
}

}
