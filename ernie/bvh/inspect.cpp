// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/bvh/inspect.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <fmt/format.h>

namespace ernie::bvh {

namespace {

constexpr size_t kMaxErrors = 32;

BvhBox Intersect(const BvhBox& a, const BvhBox& b) {
    BvhBox box;
    for (int i = 0; i < 3; ++i) {
        box.lower[i] = std::max(a.lower[i], b.lower[i]);
        box.upper[i] = std::min(a.upper[i], b.upper[i]);
    }
    return box;
}

class Walker {
  public:
    Walker(std::span<const std::byte> bytes, std::span<const BvhBox> boxes, Inspection& found)
        : m_bytes(bytes), m_boxes(boxes), m_found(found) {}

    void Run() {
        if (m_bytes.size() < sizeof(BvhHeader)) {
            Fail(fmt::format("it has {} bytes, fewer than its header's {}", m_bytes.size(),
                             sizeof(BvhHeader)));
            return;
        }
        std::memcpy(&m_header, m_bytes.data(), sizeof(m_header));
        if (m_header.version != kBvhVersion) {
            Fail(fmt::format("its format is version {}; Ernie reads version {}", m_header.version,
                             kBvhVersion));
            return;
        }
        if (m_header.kind != kBvhTriangles && m_header.kind != kBvhInstances &&
            m_header.kind != kBvhBoxes) {
            Fail(fmt::format("its leaf records are of kind {}: not triangles, instances or boxes",
                             m_header.kind));
            return;
        }
        m_words = BvhRecordWords(m_header.kind);
        if (!Fits("nodes", m_header.nodes, m_header.nodeCount, sizeof(BvhNode)) ||
            !Fits("leaf records", m_header.leaves, m_header.leafCount, uint64_t(m_words) * 16)) {
            return;
        }
        const uint64_t nodesEnd = m_header.nodes + uint64_t(m_header.nodeCount) * sizeof(BvhNode);
        const uint64_t leavesEnd = m_header.leaves + uint64_t(m_header.leafCount) * m_words * 16;
        if (m_header.nodes < sizeof(BvhHeader) || m_header.leaves < sizeof(BvhHeader) ||
            (m_header.nodes < leavesEnd && m_header.leaves < nodesEnd)) {
            Fail(fmt::format("its nodes, bytes {} to {}, and leaf records, bytes {} to {}, overlap "
                             "each other or its header",
                             m_header.nodes, nodesEnd, m_header.leaves, leavesEnd));
            return;
        }
        if (m_header.kind == kBvhInstances && !m_boxes.empty() &&
            m_boxes.size() != m_header.primitives) {
            Fail(fmt::format("it has {} instances, but {} boxes were given", m_header.primitives,
                             m_boxes.size()));
            return;
        }

        m_reached.assign(m_header.primitives, 0);
        m_visited.assign(m_header.nodeCount, false);
        if (m_header.primitives > 0) {
            if (m_header.nodeCount == 0) {
                Fail(fmt::format("it has {} primitives but no nodes", m_header.primitives));
                return;
            }
            BvhBox bounds;
            std::memcpy(bounds.lower, m_header.lower, sizeof(bounds.lower));
            std::memcpy(bounds.upper, m_header.upper, sizeof(bounds.upper));
            m_rootArea = BvhArea(bounds);
            double extent = 0;
            for (int i = 0; i < 3; ++i) {
                extent = std::max(extent, double(bounds.upper[i]) - bounds.lower[i]);
                extent = std::max(
                    {extent, std::abs(double(bounds.lower[i])), std::abs(double(bounds.upper[i]))});
            }
            m_slack = float(extent * 1e-5);

            m_found.sahCost = 1;
            m_pending.push_back({0, bounds, 1});
            while (!m_pending.empty()) {
                const Pending next = m_pending.back();
                m_pending.pop_back();
                Visit(next.index, next.within, next.depth);
            }
        }

        const uint32_t least = m_header.kind == kBvhTriangles ? 1 : 0;
        for (uint32_t i = 0; i < m_header.primitives; ++i) {
            if (m_reached[i] < least || m_reached[i] > 1) {
                Fail(fmt::format("primitive {} is in {} leaf records, not one", i, m_reached[i]));
            }
        }
        if (m_found.records != m_header.leafCount) {
            Fail(fmt::format("it has {} leaf records, but its leaves hold {}", m_header.leafCount,
                             m_found.records));
        }
        for (uint32_t i = 0; i < m_header.nodeCount; ++i) {
            if (!m_visited[i]) {
                Fail(fmt::format("node {} is never reached from the root", i));
            }
        }

        if (m_found.nodes > 0) {
            m_found.childrenPerNode = double(m_slots) / m_found.nodes;
        }
        if (m_found.leaves > 0) {
            m_found.recordsPerLeaf = double(m_found.records) / m_found.leaves;
        }
        if (m_dropped > 0) {
            m_found.errors.push_back(fmt::format("and {} more faults", m_dropped));
        }
    }

  private:
    void Fail(std::string message) {
        if (m_found.errors.size() < kMaxErrors) {
            m_found.errors.push_back(std::move(message));
        } else {
            ++m_dropped;
        }
    }

    bool Fits(const char* what, uint64_t offset, uint64_t count, uint64_t bytes) {
        if (offset % 16 != 0 || offset > m_bytes.size() ||
            count > (m_bytes.size() - offset) / bytes) {
            Fail(fmt::format("its {} at byte {}, {} of {} bytes, run past its {} bytes", what,
                             offset, count, bytes, m_bytes.size()));
            return false;
        }
        return true;
    }

    template <class T> T Read(uint64_t offset) const {
        T value;
        std::memcpy(&value, m_bytes.data() + offset, sizeof(T));
        return value;
    }

    void Visit(uint32_t index, const BvhBox& within, uint32_t depth) {
        if (index >= m_header.nodeCount) {
            Fail(fmt::format("a node's child is node {}, past its {} nodes", index,
                             m_header.nodeCount));
            return;
        }
        if (m_visited[index]) {
            Fail(fmt::format("node {} is reached more than once", index));
            return;
        }

        m_visited[index] = true;
        ++m_found.nodes;
        m_found.depth = std::max(m_found.depth, depth);

        BvhNode node = Read<BvhNode>(m_header.nodes + uint64_t(index) * sizeof(BvhNode));
        const uint32_t imask = node.exponents >> 24;
        uint32_t internal = 0;
        for (uint32_t slot = 0; slot < kBvhWidth; ++slot) {
            const uint32_t meta = BvhByte(node.meta, slot);
            const bool isInternal = (imask >> slot & 1) != 0;
            if (meta == 0) {
                if (isInternal) {
                    Fail(fmt::format("node {}'s slot {} is empty but in its imask", index, slot));
                }
                continue;
            }
            ++m_slots;
            const BvhBox box = BvhChildBox(node, slot);
            const double weight = m_rootArea > 0 ? double(BvhArea(box)) / m_rootArea : 1;
            const BvhBox inside = Intersect(within, box);
            if (isInternal) {
                if (meta != BvhInternalMeta(slot)) {
                    Fail(
                        fmt::format("node {}'s slot {} is in its imask, but its meta byte is {:#x}",
                                    index, slot, meta));
                    continue;
                }
                m_found.sahCost += weight;
                m_pending.push_back({node.firstChild + internal++, inside, depth + 1});
                continue;
            }
            if (meta == BvhInternalMeta(slot)) {
                Fail(
                    fmt::format("node {}'s slot {} has an internal child's meta byte, but isn't in "
                                "its imask",
                                index, slot));
                continue;
            }
            const uint32_t count = std::popcount(meta >> 5);
            if ((meta >> 5) != (1u << count) - 1) {
                Fail(fmt::format("node {}'s slot {} has meta byte {:#x}: neither empty, internal "
                                 "nor a leaf's",
                                 index, slot, meta));
                continue;
            }
            ++m_found.leaves;
            m_found.sahCost += weight * count;
            for (uint32_t i = 0; i < count; ++i) {
                VisitRecord(index, node.firstLeaf + m_words * ((meta & 31) + i), inside);
            }
        }
    }

    void VisitRecord(uint32_t node, uint32_t word, const BvhBox& within) {
        if (word % m_words != 0 || word / m_words >= m_header.leafCount) {
            Fail(fmt::format("node {} has a leaf record at word {}, not one of its {} records",
                             node, word, m_header.leafCount));
            return;
        }
        ++m_found.records;
        const uint64_t at = m_header.leaves + uint64_t(word) * 16;

        uint32_t primitive;
        BvhBox box;
        if (m_header.kind == kBvhInstances) {
            primitive = Read<BvhInstance>(at).index;
            if (m_boxes.empty() || primitive >= m_boxes.size()) {
                box = within;
            } else {
                box = m_boxes[primitive];
            }
        } else if (m_header.kind == kBvhBoxes) {
            const auto record = Read<BvhBoxRecord>(at);
            primitive = record.index;
            std::memcpy(box.lower, record.lower, sizeof(box.lower));
            std::memcpy(box.upper, record.upper, sizeof(box.upper));
        } else {
            const auto triangle = Read<BvhTriangle>(at);
            primitive = triangle.index;
            BvhCorners corners;
            for (int i = 0; i < 3; ++i) {
                corners.a[i] = triangle.a[i];
                corners.b[i] = triangle.a[i] + triangle.edgeB[i];
                corners.c[i] = triangle.a[i] + triangle.edgeC[i];
            }
            box = BvhTriangleBox(corners);
        }

        if (primitive >= m_header.primitives) {
            Fail(fmt::format("node {} has primitive {}, past its {} primitives", node, primitive,
                             m_header.primitives));
            return;
        }
        ++m_reached[primitive];
        for (int i = 0; i < 3; ++i) {
            if (box.lower[i] < within.lower[i] - m_slack ||
                box.upper[i] > within.upper[i] + m_slack) {
                Fail(fmt::format("primitive {} of node {} lies outside the boxes above it",
                                 primitive, node));
                return;
            }
        }
    }

    struct Pending {
        uint32_t index;
        BvhBox within;
        uint32_t depth;
    };

    std::span<const std::byte> m_bytes;
    std::span<const BvhBox> m_boxes;
    Inspection& m_found;
    std::vector<Pending> m_pending;
    BvhHeader m_header{};
    uint32_t m_words = 0;
    std::vector<uint32_t> m_reached;
    std::vector<bool> m_visited;
    double m_rootArea = 0;
    float m_slack = 0;
    uint64_t m_slots = 0;
    size_t m_dropped = 0;
};

}

Inspection Inspect(std::span<const std::byte> bytes, std::span<const BvhBox> boxes) {
    Inspection found;
    Walker(bytes, boxes, found).Run();
    return found;
}

}
