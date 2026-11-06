// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/bvh/reference.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "ernie/util/error.h"

namespace ernie::bvh {

namespace {

using Point = std::array<float, 3>;

constexpr int kBins = 16;

void Grow(BvhBox& box, const BvhBox& other) {
    box = BvhUnion(box, other);
}

void Grow(BvhBox& box, const Point& point) {
    Grow(box, BvhBox{{point[0], point[1], point[2]}, {point[0], point[1], point[2]}});
}

Point Center(const BvhBox& box) {
    return {BvhCenter(box, 0), BvhCenter(box, 1), BvhCenter(box, 2)};
}

struct Binary {
    BvhBox box = BvhEmptyBox();
    uint32_t left = 0;
    uint32_t right = 0;
    uint32_t first = 0;
    uint32_t count = 0;
};

template <class Record, class MakeRecord> class ReferenceBuilder {
  public:
    ReferenceBuilder(std::vector<BvhBox> boxes, MakeRecord makeRecord)
        : m_boxes(std::move(boxes)), m_makeRecord(std::move(makeRecord)) {}

    std::vector<std::byte> Build() {
        BvhHeader header{};
        header.version = kBvhVersion;
        header.kind = std::is_same_v<Record, BvhInstance>    ? kBvhInstances
                      : std::is_same_v<Record, BvhBoxRecord> ? kBvhBoxes
                                                             : kBvhTriangles;
        header.primitives = uint32_t(m_boxes.size());
        header.nodes = sizeof(BvhHeader);
        if (!m_boxes.empty()) {
            BuildBinary();
            Encode();
            for (int i = 0; i < 3; ++i) {
                header.lower[i] = m_binary[0].box.lower[i];
                header.upper[i] = m_binary[0].box.upper[i];
            }
        }

        header.nodeCount = uint32_t(m_nodes.size());
        header.leafCount = uint32_t(m_records.size());
        header.leaves = header.nodes + m_nodes.size() * sizeof(BvhNode);

        std::vector<std::byte> bytes(header.leaves + m_records.size() * sizeof(Record));
        std::memcpy(bytes.data(), &header, sizeof(header));
        std::memcpy(bytes.data() + header.nodes, m_nodes.data(), m_nodes.size() * sizeof(BvhNode));
        std::memcpy(bytes.data() + header.leaves, m_records.data(),
                    m_records.size() * sizeof(Record));
        return bytes;
    }

  private:
    void BuildBinary() {
        const auto count = uint32_t(m_boxes.size());
        m_centers.resize(count);
        m_order.resize(count);
        Binary root;
        for (uint32_t i = 0; i < count; ++i) {
            m_centers[i] = Center(m_boxes[i]);
            m_order[i] = i;
            Grow(root.box, m_boxes[i]);
        }
        root.count = count;
        m_binary.push_back(root);

        std::vector<uint32_t> pending = {0};
        while (!pending.empty()) {
            const uint32_t index = pending.back();
            pending.pop_back();
            if (Split(index)) {
                pending.push_back(m_binary[index].left);
                pending.push_back(m_binary[index].right);
            }
        }
    }

    struct SplitChoice {
        int axis = -1;
        int bin = 0;
        BvhBox centers = BvhEmptyBox();
    };

    SplitChoice BestSplit(const Binary& node) const {
        SplitChoice best;
        for (uint32_t i = node.first; i < node.first + node.count; ++i) {
            Grow(best.centers, m_centers[m_order[i]]);
        }

        float bestCost = std::numeric_limits<float>::max();
        for (int axis = 0; axis < 3; ++axis) {
            const float extent = best.centers.upper[axis] - best.centers.lower[axis];
            if (!(extent > 0)) {
                continue;
            }
            BvhBox boxes[kBins];
            std::fill(std::begin(boxes), std::end(boxes), BvhEmptyBox());
            uint32_t counts[kBins] = {};
            for (uint32_t i = node.first; i < node.first + node.count; ++i) {
                const int bin = Bin(m_centers[m_order[i]][axis], best.centers.lower[axis], extent);
                Grow(boxes[bin], m_boxes[m_order[i]]);
                ++counts[bin];
            }

            float leftArea[kBins - 1];
            uint32_t leftCount[kBins - 1];
            BvhBox left = BvhEmptyBox();
            uint32_t below = 0;
            for (int bin = 0; bin < kBins - 1; ++bin) {
                Grow(left, boxes[bin]);
                below += counts[bin];
                leftArea[bin] = BvhArea(left);
                leftCount[bin] = below;
            }

            BvhBox right = BvhEmptyBox();
            uint32_t above = 0;
            for (int bin = kBins - 1; bin > 0; --bin) {
                Grow(right, boxes[bin]);
                above += counts[bin];
                if (leftCount[bin - 1] == 0 || above == 0) {
                    continue;
                }
                const float cost =
                    leftArea[bin - 1] * float(leftCount[bin - 1]) + BvhArea(right) * float(above);
                if (cost < bestCost) {
                    bestCost = cost;
                    best.axis = axis;
                    best.bin = bin;
                }
            }
        }
        return best;
    }

    bool Split(uint32_t index) {
        const Binary node = m_binary[index];
        if (node.count <= kBvhLeafSize) {
            return false;
        }
        const SplitChoice best = BestSplit(node);
        uint32_t middle;
        if (best.axis >= 0) {
            const float lower = best.centers.lower[best.axis];
            const float extent = best.centers.upper[best.axis] - lower;
            const auto split = std::partition(
                m_order.begin() + node.first, m_order.begin() + node.first + node.count,
                [&](uint32_t primitive) {
                    return Bin(m_centers[primitive][best.axis], lower, extent) < best.bin;
                });
            middle = uint32_t(split - m_order.begin());
        } else {
            middle = node.first + node.count / 2;
        }

        Binary left{.first = node.first, .count = middle - node.first};
        Binary right{.first = middle, .count = node.first + node.count - middle};
        for (Binary* child : {&left, &right}) {
            for (uint32_t i = child->first; i < child->first + child->count; ++i) {
                Grow(child->box, m_boxes[m_order[i]]);
            }
        }

        m_binary[index].left = uint32_t(m_binary.size());
        m_binary.push_back(left);
        m_binary[index].right = uint32_t(m_binary.size());
        m_binary.push_back(right);
        m_binary[index].count = 0;
        return true;
    }

    static int Bin(float center, float lower, float extent) {
        const float at = (center - lower) * (float(kBins) / extent);
        return at > 0 ? std::min(kBins - 1, int(std::min(at, float(kBins)))) : 0;
    }

    std::vector<uint32_t> Children(uint32_t index) const {
        const Binary& node = m_binary[index];
        if (node.count > 0) {
            return {index};
        }
        std::vector<uint32_t> children = {node.left, node.right};
        while (children.size() < kBvhWidth) {
            int largest = -1;
            for (size_t i = 0; i < children.size(); ++i) {
                const Binary& child = m_binary[children[i]];
                if (child.count == 0 &&
                    (largest < 0 ||
                     BvhArea(child.box) > BvhArea(m_binary[children[largest]].box))) {
                    largest = int(i);
                }
            }
            if (largest < 0) {
                break;
            }
            const Binary& opened = m_binary[children[largest]];
            children[largest] = opened.left;
            children.push_back(opened.right);
        }
        return children;
    }

    void Encode() {
        m_nodes.emplace_back();
        std::vector<std::pair<uint32_t, uint32_t>> pending = {{0, 0}};
        while (!pending.empty()) {
            const auto [binary, at] = pending.back();
            pending.pop_back();
            const std::vector<uint32_t> children = Children(binary);
            const BvhBox& box = m_binary[binary].box;
            BvhBox boxes[kBvhWidth];
            for (size_t i = 0; i < children.size(); ++i) {
                boxes[i] = m_binary[children[i]].box;
            }
            const BvhSlots slots = BvhAssignSlots(box, boxes, uint32_t(children.size()));
            BvhNode node = BvhQuantize(box, boxes, slots);

            uint32_t imask = 0;
            uint32_t internal = 0;
            uint32_t records = 0;
            for (uint32_t slot = 0; slot < kBvhWidth; ++slot) {
                if (slots.child[slot] == kBvhEmptySlot) {
                    continue;
                }
                const uint32_t index = children[slots.child[slot]];
                const Binary& child = m_binary[index];
                uint32_t meta;
                if (child.count == 0) {
                    if (internal++ == 0) {
                        node.firstChild = uint32_t(m_nodes.size());
                    }
                    imask |= 1u << slot;
                    meta = BvhInternalMeta(slot);
                    pending.emplace_back(index, uint32_t(m_nodes.size()));
                    m_nodes.emplace_back();
                } else {
                    if (records == 0) {
                        node.firstLeaf = uint32_t(m_records.size() * sizeof(Record) / 16);
                    }
                    meta = BvhLeafMeta(child.count, records);
                    records += child.count;
                    for (uint32_t i = child.first; i < child.first + child.count; ++i) {
                        m_records.push_back(m_makeRecord(m_order[i]));
                    }
                }
                node.meta[slot >> 2] = BvhWithByte(node.meta[slot >> 2], slot & 3, meta);
            }

            node.exponents |= imask << 24;
            m_nodes[at] = node;
        }
    }

    std::vector<BvhBox> m_boxes;
    MakeRecord m_makeRecord;
    std::vector<Point> m_centers;
    std::vector<uint32_t> m_order;
    std::vector<Binary> m_binary;
    std::vector<BvhNode> m_nodes;
    std::vector<Record> m_records;
};

template <class Record, class MakeRecord>
std::vector<std::byte> BuildBvh(std::vector<BvhBox> boxes, MakeRecord makeRecord) {
    return ReferenceBuilder<Record, MakeRecord>(std::move(boxes), std::move(makeRecord)).Build();
}

}

std::vector<std::byte> BuildReference(std::span<const Triangle> triangles) {
    std::vector<BvhBox> boxes(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i) {
        boxes[i] = BvhTriangleBox(triangles[i]);
    }
    return BuildBvh<BvhTriangle>(
        std::move(boxes), [&](uint32_t index) { return BvhMakeTriangle(triangles[index], index); });
}

std::vector<std::byte> BuildReference(std::span<const BvhBox> boxes) {
    return BuildBvh<BvhBoxRecord>(std::vector<BvhBox>(boxes.begin(), boxes.end()),
                                  [&](uint32_t index) { return BvhMakeBox(boxes[index], index); });
}

std::vector<std::byte> BuildReference(std::span<const InstanceRecord> instances,
                                      std::span<const BvhBox> bounds) {
    if (bounds.size() != instances.size()) {
        throw Error("BuildReference of {} instances with {} bounds; it takes one an instance",
                    instances.size(), bounds.size());
    }
    std::vector<BvhBox> boxes(instances.size());
    for (size_t i = 0; i < instances.size(); ++i) {
        boxes[i] = BvhInstanceBox(instances[i], bounds[i]);
    }
    return BuildBvh<BvhInstance>(
        std::move(boxes), [&](uint32_t index) { return BvhMakeInstance(instances[index], index); });
}

}
