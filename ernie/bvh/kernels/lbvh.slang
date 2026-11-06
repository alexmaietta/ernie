// SPDX-License-Identifier: BSD-3-Clause

import ernie.algorithms.kernels.group;
import ernie.bvh.kernels.common;

using namespace ernie;

uint FindSegment(LbvhParams p, uint i) {
    return FindOffset(p.offsets, p.segments, i);
}

uint SegmentCapacity(LbvhParams p, uint segment) {
    return p.offsets[segment + 1] - p.offsets[segment];
}

uint ActiveCount(LbvhParams p, uint segment) {
    const uint capacity = SegmentCapacity(p, segment);
    return p.counts == nullptr ? capacity : min(p.counts[segment], capacity);
}

BvhBox PrimitiveBox(LbvhParams p, uint segment, uint i) {
    if (p.kind == kBvhTriangles) {
        return BvhTriangleBox(Corners(p.geometries[segment], i - p.offsets[segment]));
    }
    if (p.kind == kBvhBoxes) {
        return InputBox(p.geometries[segment], i - p.offsets[segment]);
    }
    return BvhInstanceWorldBox(p.instances[i]);
}

[shader("compute")]
[numthreads(kLbvhThreads, 1, 1)]
void Prepare(uniform LbvhParams p, uint3 thread: SV_GroupThreadID) {
    const uint i = LinearThreadIndex();
    const bool inside = i < p.total;
    const uint segment = inside ? FindSegment(p, i) : 0;
    const bool active = inside && i - p.offsets[segment] < ActiveCount(p, segment) &&
                        (p.kind != kBvhInstances || (p.instances[i].custom >> 24) != 0);
    const BvhBox box = active ? PrimitiveBox(p, segment, i) : BvhEmptyBox();
    if (inside) {
        p.order[i] = i;
        for (uint axis = 0; axis < 3; ++axis) {
            p.boxes[i * 6 + axis] = box.lower[axis];
            p.boxes[i * 6 + 3 + axis] = box.upper[axis];
        }
        const uint group = segment * 2 + (active ? 0 : 1);
        p.keys[i * p.keyWords] = group << kLbvhCodeBits;
        if (p.keyWords == 2) {
            p.keys[i * 2 + 1] = group >> (32 - kLbvhCodeBits);
        }
    }
    uint words[kLbvhBoundsWords];
    for (uint axis = 0; axis < 3; ++axis) {
        const float center = BvhCenter(box, axis);
        words[kLbvhMinCenter + axis] = active ? LbvhOrdered(center) : 0xFFFFFFFFu;
        words[kLbvhMinLower + axis] = active ? LbvhOrdered(box.lower[axis]) : 0xFFFFFFFFu;
        words[kLbvhMaxCenter + axis] = active ? LbvhOrdered(-center) : 0xFFFFFFFFu;
        words[kLbvhMaxUpper + axis] = active ? LbvhOrdered(-box.upper[axis]) : 0xFFFFFFFFu;
    }
    const uint first = i - thread.x;
    const uint last = min(first + kLbvhThreads, p.total) - 1;
    if (first < p.total && FindSegment(p, first) == FindSegment(p, last)) {
        for (uint word = 0; word < kLbvhBoundsWords; ++word) {
            const uint least = GroupCombine<MinCombine>(thread.x, kLbvhThreads, words[word]);
            if (thread.x == 0) {
                InterlockedMin(p.bounds[segment * kLbvhBoundsWords + word], least);
            }
        }
    } else if (active) {
        for (uint word = 0; word < kLbvhBoundsWords; ++word) {
            InterlockedMin(p.bounds[segment * kLbvhBoundsWords + word], words[word]);
        }
    }
}

uint Spread(uint bits) {
    bits = (bits * 0x00010001u) & 0xFF0000FFu;
    bits = (bits * 0x00000101u) & 0x0F00F00Fu;
    bits = (bits * 0x00000011u) & 0xC30C30C3u;
    bits = (bits * 0x00000005u) & 0x49249249u;
    return bits;
}

uint Group(LbvhParams p, uint i) {
    const uint low = p.keys[i * p.keyWords] >> kLbvhCodeBits;
    return p.keyWords == 2 ? low | p.keys[i * 2 + 1] << (32 - kLbvhCodeBits) : low;
}

[shader("compute")]
[numthreads(kLbvhThreads, 1, 1)]
void Morton(uniform LbvhParams p) {
    const uint i = LinearThreadIndex();
    if (i >= p.total) {
        return;
    }
    const uint group = Group(p, i);
    if ((group & 1) != 0) {
        return;
    }
    uint* bounds = p.bounds + (group / 2) * kLbvhBoundsWords;
    uint cell[3];
    for (uint axis = 0; axis < 3; ++axis) {
        const float lower = LbvhFloat(bounds[kLbvhMinCenter + axis]);
        const float extent = -LbvhFloat(bounds[kLbvhMaxCenter + axis]) - lower;
        const float center = (p.boxes[i * 6 + axis] + p.boxes[i * 6 + 3 + axis]) * 0.5f;
        const float at = extent > 0 ? (center - lower) / extent : 0;
        cell[axis] = uint(clamp(at * 1024.0f, 0.0f, 1023.0f));
    }
    p.keys[i * p.keyWords] |= Spread(cell[0]) << 2 | Spread(cell[1]) << 1 | Spread(cell[2]);
}

int Common(LbvhParams p, int i, int j) {
    if (j < 0 || j >= int(p.total)) {
        return -1;
    }
    if (p.keyWords == 2) {
        const uint high = p.keys[i * 2 + 1] ^ p.keys[j * 2 + 1];
        if (high != 0) {
            return int(31 - firstbithigh(high));
        }
    }
    const uint low = p.keys[i * p.keyWords] ^ p.keys[j * p.keyWords];
    const int above = p.keyWords == 2 ? 32 : 0;
    if (low != 0) {
        return above + int(31 - firstbithigh(low));
    }
    return above + 32 + int(31 - firstbithigh(uint(i) ^ uint(j)));
}

bool IsRoot(LbvhParams p, uint first, uint last) {
    const uint group = Group(p, first);
    return (group & 1) == 0 && Group(p, last) == group &&
           (first == 0 || Group(p, first - 1) != group) &&
           (last + 1 == p.total || Group(p, last + 1) != group);
}

[shader("compute")]
[numthreads(kLbvhThreads, 1, 1)]
void Hierarchy(uniform LbvhParams p) {
    const int i = int(LinearThreadIndex());
    if (i >= int(p.total)) {
        return;
    }
    if (IsRoot(p, uint(i), uint(i))) {
        p.roots[Group(p, uint(i)) / 2] = kLbvhLeaf | uint(i);
    }
    if (i >= int(p.total) - 1) {
        return;
    }
    const int direction = Common(p, i, i + 1) > Common(p, i, i - 1) ? 1 : -1;
    const int outside = Common(p, i, i - direction);
    int bound = 2;
    while (Common(p, i, i + bound * direction) > outside) {
        bound *= 2;
    }
    int span = 0;
    for (int step = bound / 2; step >= 1; step /= 2) {
        if (Common(p, i, i + (span + step) * direction) > outside) {
            span += step;
        }
    }
    const int end = i + span * direction;
    const int withEnd = Common(p, i, end);
    int past = 0;
    int step = span;
    do {
        step = (step + 1) / 2;
        if (Common(p, i, i + (past + step) * direction) > withEnd) {
            past += step;
        }
    } while (step > 1);
    const int split = i + past * direction + min(direction, 0);
    const uint first = uint(min(i, end));
    const uint last = uint(max(i, end));
    const uint left = first == uint(split) ? kLbvhLeaf | uint(split) : uint(split);
    const uint right = last == uint(split + 1) ? kLbvhLeaf | uint(split + 1) : uint(split + 1);
    uint* node = p.nodes + uint(i) * kLbvhNodeWords;
    node[kLbvhLeft] = left;
    node[kLbvhRight] = right;
    node[kLbvhFirst] = first;
    node[kLbvhLast] = last;
    if (i == 0) {
        node[kLbvhParent] = kLbvhNone;
    }
    for (uint side = 0; side < 2; ++side) {
        const uint child = side == 0 ? left : right;
        if ((child & kLbvhLeaf) != 0) {
            p.parents[child & ~kLbvhLeaf] = uint(i);
        } else {
            p.nodes[child * kLbvhNodeWords + kLbvhParent] = uint(i);
        }
    }
    if (IsRoot(p, first, last)) {
        p.roots[Group(p, first) / 2] = uint(i);
    }
}

BvhBox LeafBox(LbvhParams p, uint position) {
    const uint primitive = p.order[position];
    BvhBox box;
    for (uint axis = 0; axis < 3; ++axis) {
        box.lower[axis] = p.boxes[primitive * 6 + axis];
        box.upper[axis] = p.boxes[primitive * 6 + 3 + axis];
    }
    return box;
}

[shader("compute")]
[numthreads(kLbvhThreads, 1, 1)]
void FitBoxes(uniform LbvhParams p) {
    const uint leaf = LinearThreadIndex();
    if (leaf >= p.total) {
        return;
    }
    BvhBox box = LeafBox(p, leaf);
    uint from = kLbvhLeaf | leaf;
    uint at = p.parents[leaf];
    while (at != kLbvhNone) {
        AllMemoryBarrier();
        uint arrived;
        InterlockedAdd(p.arrivals[at], 1u, arrived);
        if (arrived == 0) {
            return;
        }
        AllMemoryBarrier();
        uint* node = p.nodes + at * kLbvhNodeWords;
        const uint other = node[kLbvhLeft] == from ? node[kLbvhRight] : node[kLbvhLeft];
        BvhBox sibling;
        if ((other & kLbvhLeaf) != 0) {
            sibling = LeafBox(p, other & ~kLbvhLeaf);
        } else {
            uint* words = p.nodes + other * kLbvhNodeWords;
            for (uint axis = 0; axis < 3; ++axis) {
                sibling.lower[axis] = asfloat(Coherent(words + kLbvhLower + axis));
                sibling.upper[axis] = asfloat(Coherent(words + kLbvhUpper + axis));
            }
        }
        box = BvhUnion(box, sibling);
        for (uint axis = 0; axis < 3; ++axis) {
            node[kLbvhLower + axis] = asuint(box.lower[axis]);
            node[kLbvhUpper + axis] = asuint(box.upper[axis]);
        }
        from = at;
        at = node[kLbvhParent];
    }
}

void Enqueue(LbvhParams p, uint ref, uint segment, uint node) {
    uint at;
    InterlockedAdd(p.state[0].next, 1u, at);
    uint* task = p.tasks + ((p.state[0].parity ^ 1) * p.queue + at) * kLbvhTaskWords;
    task[0] = ref;
    task[1] = segment;
    task[2] = node;
}

[shader("compute")]
[numthreads(kLbvhThreads, 1, 1)]
void Seed(uniform LbvhParams p) {
    const uint segment = LinearThreadIndex();
    if (segment >= p.segments || p.roots[segment] == kLbvhNone) {
        return;
    }
    p.counters[segment * 2] = 1;
    Enqueue(p, p.roots[segment], segment, 0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void Advance(uniform LbvhParams p) {
    p.state[0].parity ^= 1;
    p.state[0].level = p.state[0].next;
    p.state[0].next = 0;
    p.state[0].groups[0] = (p.state[0].level + kLbvhThreads - 1) / kLbvhThreads;
    p.state[0].groups[1] = 1;
    p.state[0].groups[2] = 1;
}

uint Count(LbvhParams p, uint ref) {
    if ((ref & kLbvhLeaf) != 0) {
        return 1;
    }
    return p.nodes[ref * kLbvhNodeWords + kLbvhLast] - p.nodes[ref * kLbvhNodeWords + kLbvhFirst] + 1;
}

uint First(LbvhParams p, uint ref) {
    return (ref & kLbvhLeaf) != 0 ? ref & ~kLbvhLeaf : p.nodes[ref * kLbvhNodeWords + kLbvhFirst];
}

BvhBox Box(LbvhParams p, uint ref) {
    if ((ref & kLbvhLeaf) != 0) {
        return LeafBox(p, ref & ~kLbvhLeaf);
    }
    BvhBox box;
    for (uint axis = 0; axis < 3; ++axis) {
        box.lower[axis] = asfloat(p.nodes[ref * kLbvhNodeWords + kLbvhLower + axis]);
        box.upper[axis] = asfloat(p.nodes[ref * kLbvhNodeWords + kLbvhUpper + axis]);
    }
    return box;
}

struct Children {
    uint refs[8];
    uint count;
}

[ForceInline]
Children GatherChildren(LbvhParams p, uint task) {
    Children children;
    if (Count(p, task) <= kBvhLeafSize) {
        children.refs[0] = task;
        children.count = 1;
        return children;
    }
    children.refs[0] = p.nodes[task * kLbvhNodeWords + kLbvhLeft];
    children.refs[1] = p.nodes[task * kLbvhNodeWords + kLbvhRight];
    children.count = 2;
    while (children.count < 8) {
        int largest = -1;
        float area = -1;
        for (uint i = 0; i < children.count; ++i) {
            if (Count(p, children.refs[i]) > kBvhLeafSize) {
                const float a = BvhArea(Box(p, children.refs[i]));
                if (a > area) {
                    area = a;
                    largest = int(i);
                }
            }
        }
        if (largest < 0) {
            break;
        }
        const uint opened = children.refs[largest];
        children.refs[largest] = p.nodes[opened * kLbvhNodeWords + kLbvhLeft];
        children.refs[children.count++] = p.nodes[opened * kLbvhNodeWords + kLbvhRight];
    }
    return children;
}

BvhHeader* SegmentHeader(LbvhParams p, uint segment) {
    return (BvhHeader*)p.headers[segment];
}

BvhNode* SegmentNodes(LbvhParams p, uint segment) {
    return (BvhNode*)((uint*)SegmentHeader(p, segment) + kBvhHeaderBytes / 4);
}

uint4* SegmentRecords(LbvhParams p, uint segment) {
    const uint64_t leaves = BvhLeavesOffset(SegmentCapacity(p, segment));
    return (uint4*)((uint*)SegmentHeader(p, segment) + uint(leaves / 4));
}

[ForceInline]
void LinkChildren(LbvhParams p, inout BvhNode node, uint segment, Children children,
                  BvhSlots slots) {
    uint internal = 0;
    uint records = 0;
    for (uint i = 0; i < children.count; ++i) {
        const uint primitives = Count(p, children.refs[i]);
        if (primitives > kBvhLeafSize) {
            ++internal;
        } else {
            records += primitives;
        }
    }
    uint firstChild = 0;
    uint firstRecord = 0;
    if (internal > 0) {
        InterlockedAdd(p.counters[segment * 2], internal, firstChild);
    }
    if (records > 0) {
        InterlockedAdd(p.counters[segment * 2 + 1], records, firstRecord);
    }
    const uint words = BvhRecordWords(p.kind);
    node.firstChild = firstChild;
    node.firstLeaf = firstRecord * words;

    uint4* output = SegmentRecords(p, segment);
    const uint offset = p.offsets[segment];
    uint imask = 0;
    uint nextChild = firstChild;
    uint before = 0;
    for (uint slot = 0; slot < 8; ++slot) {
        if (slots.child[slot] == kBvhEmptySlot) {
            continue;
        }
        const uint child = children.refs[slots.child[slot]];
        const uint primitives = Count(p, child);
        uint meta;
        if (primitives > kBvhLeafSize) {
            imask |= 1u << slot;
            meta = BvhInternalMeta(slot);
            Enqueue(p, child, segment, nextChild++);
        } else {
            meta = BvhLeafMeta(primitives, before);
            const uint first = First(p, child);
            for (uint t = 0; t < primitives; ++t) {
                const uint primitive = p.order[first + t];
                uint4* at = output + (firstRecord + before + t) * words;
                const uint local = primitive - offset;
                if (p.kind == kBvhTriangles) {
                    *(BvhTriangle*)at = BvhMakeTriangle(Corners(p.geometries[segment], local), local);
                } else if (p.kind == kBvhBoxes) {
                    *(BvhBoxRecord*)at = BvhMakeBox(InputBox(p.geometries[segment], local), local);
                } else {
                    *(BvhInstance*)at = BvhMakeInstance(p.instances[primitive], local);
                }
            }
            before += primitives;
        }
        node.meta[slot >> 2] = BvhWithByte(node.meta[slot >> 2], slot & 3, meta);
    }
    node.exponents |= imask << 24;
}

[shader("compute")]
[numthreads(kLbvhThreads, 1, 1)]
void Collapse(uniform LbvhParams p) {
    const uint t = LinearThreadIndex();
    if (t >= p.state[0].level) {
        return;
    }
    uint* task = p.tasks + (p.state[0].parity * p.queue + t) * kLbvhTaskWords;
    const uint ref = task[0];
    const uint segment = task[1];
    const Children children = GatherChildren(p, ref);
    const BvhBox box = Box(p, ref);
    BvhBox boxes[8];
    for (uint i = 0; i < children.count; ++i) {
        boxes[i] = Box(p, children.refs[i]);
    }
    const BvhSlots slots = BvhAssignSlots(box, boxes, children.count);
    BvhNode node = BvhQuantize(box, boxes, slots);
    LinkChildren(p, node, segment, children, slots);
    SegmentNodes(p, segment)[task[2]] = node;
}

[shader("compute")]
[numthreads(kLbvhThreads, 1, 1)]
void Finish(uniform LbvhParams p) {
    const uint segment = LinearThreadIndex();
    if (segment >= p.segments) {
        return;
    }
    const uint count = ActiveCount(p, segment);
    uint* bounds = p.bounds + segment * kLbvhBoundsWords;
    BvhHeader header;
    for (uint axis = 0; axis < 3; ++axis) {
        header.lower[axis] = count == 0 ? 0 : LbvhFloat(bounds[kLbvhMinLower + axis]);
        header.upper[axis] = count == 0 ? 0 : -LbvhFloat(bounds[kLbvhMaxUpper + axis]);
    }
    header.primitives = count;
    header.version = kBvhVersion;
    header.nodes = kBvhHeaderBytes;
    header.leaves = BvhLeavesOffset(SegmentCapacity(p, segment));
    header.nodeCount = p.counters[segment * 2];
    header.leafCount = p.counters[segment * 2 + 1];
    header.kind = p.kind;
    header.padding = 0;
    *SegmentHeader(p, segment) = header;
}
