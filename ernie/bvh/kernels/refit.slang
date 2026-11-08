// SPDX-License-Identifier: BSD-3-Clause

import ernie.bvh.kernels.common;

using namespace ernie;

BvhHeader* Header(RefitParams p, uint segment) {
    return (BvhHeader*)p.headers[segment];
}

BvhNode* Nodes(BvhHeader* header) {
    return (BvhNode*)((uint*)header + uint(header.nodes) / 4);
}

uint4* Records(BvhHeader* header) {
    return (uint4*)((uint*)header + uint(header.leaves) / 4);
}

groupshared uint s_segment;

uint RefitSegment(RefitParams p, uint i, uint thread) {
    const uint clamped = min(i, p.nodes - 1);
    if (thread == 0) {
        s_segment = FindOffset(p.nodeOffsets, p.segments, clamped);
    }
    GroupMemoryBarrierWithGroupSync();
    const uint segment = s_segment;
    if (clamped < p.nodeOffsets[segment + 1]) {
        return segment;
    }
    return FindOffset(p.nodeOffsets, p.segments, clamped);
}

uint InternalChild(uint firstChild, uint imask, uint slot) {
    return firstChild + countbits(imask & ((1u << slot) - 1));
}

BvhBox RefitLeaf(RefitParams p, uint segment, uint4* at) {
    if (p.kind == kBvhInstances) {
        BvhInstance* record = (BvhInstance*)at;
        const uint index = record.index & kBvhIndexMask;
        const InstanceRecord instance = p.instances[p.instanceOffsets[segment] + index];
        *record = BvhMakeInstance(instance, index);
        return BvhInstanceWorldBox(instance);
    }

    const LbvhGeometry geometry = p.geometries[segment];
    if (p.kind == kBvhBoxes) {
        BvhBoxRecord* record = (BvhBoxRecord*)at;
        const uint index = record.index;
        const BvhBox box = InputBox(geometry, index);
        *record = BvhMakeBox(box, index);
        return box;
    }

    BvhTriangle* record = (BvhTriangle*)at;
    const uint index = record.index;
    const BvhCorners corners = Corners(geometry, index);
    *record = BvhMakeTriangle(corners, index);
    return BvhTriangleBox(corners);
}

[shader("compute")]
[numthreads(kRefitThreads, 1, 1)]
void RefitLeaves(uniform RefitParams p, uint thread: SV_GroupIndex) {
    const uint i = LinearThreadIndex();
    const uint segment = RefitSegment(p, i, thread);
    if (i >= p.nodes) {
        return;
    }
    p.arrivals[i] = 0;
    const uint base = p.nodeOffsets[segment];
    const uint local = i - base;
    BvhHeader* header = Header(p, segment);
    if (local < header.leafCount && header.primitives != 0) {
        const BvhBox box = RefitLeaf(p, segment, Records(header) + local * BvhRecordWords(p.kind));
        float4* stored = (float4*)p.leafBoxes + i * 2;
        stored[0] = float4(box.lower[0], box.lower[1], box.lower[2], 0);
        stored[1] = float4(box.upper[0], box.upper[1], box.upper[2], 0);
    }
    if (local >= header.nodeCount) {
        return;
    }
    if (local == 0) {
        p.parents[i] = kRefitNoParent;
    }
    const BvhNode node = Nodes(header)[local];
    const uint imask = node.exponents >> 24;
    for (uint slot = 0; slot < 8; ++slot) {
        if ((imask >> slot & 1) != 0) {
            p.parents[base + InternalChild(node.firstChild, imask, slot)] = local;
        }
    }
}

BvhBox StoredBox(RefitParams p, uint node) {
    uint* stored = (uint*)p.boxes + node * 6;
    BvhBox box;
    for (uint axis = 0; axis < 3; ++axis) {
        box.lower[axis] = asfloat(Coherent(stored + axis));
        box.upper[axis] = asfloat(Coherent(stored + 3 + axis));
    }
    return box;
}

BvhBox LeafBoxes(RefitParams p, uint first, uint count) {
    BvhBox box = BvhEmptyBox();
    [ForceUnroll]
    for (uint t = 0; t < kBvhLeafSize; ++t) {
        if (t >= count) {
            break;
        }
        float4* stored = (float4*)p.leafBoxes + (first + t) * 2;
        BvhBox leaf;
        for (uint axis = 0; axis < 3; ++axis) {
            leaf.lower[axis] = stored[0][axis];
            leaf.upper[axis] = stored[1][axis];
        }
        box = BvhUnion(box, leaf);
    }
    return box;
}

[shader("compute")]
[numthreads(kRefitThreads, 1, 1)]
void RefitNodes(uniform RefitParams p, uint thread: SV_GroupIndex) {
    const uint i = LinearThreadIndex();
    const uint segment = RefitSegment(p, i, thread);
    if (i >= p.nodes) {
        return;
    }
    const uint base = p.nodeOffsets[segment];
    BvhHeader* header = Header(p, segment);
    BvhNode* nodes = Nodes(header);
    uint at = i - base;
    if (at >= header.nodeCount || header.primitives == 0 || (nodes[at].exponents >> 24) != 0) {
        return;
    }
    const uint words = BvhRecordWords(p.kind);
    while (true) {
        const BvhNode node = nodes[at];
        const uint imask = node.exponents >> 24;
        BvhBox boxes[8];
        uint occupied = 0;
        BvhBox box = BvhEmptyBox();
        [ForceUnroll]
        for (uint slot = 0; slot < 8; ++slot) {
            boxes[slot] = BvhEmptyBox();
            const uint meta = BvhByte(node.meta, slot);
            if (meta == 0) {
                continue;
            }
            occupied |= 1u << slot;
            if ((imask >> slot & 1) != 0) {
                boxes[slot] = StoredBox(p, base + InternalChild(node.firstChild, imask, slot));
            } else {
                boxes[slot] = LeafBoxes(p, base + node.firstLeaf / words + (meta & 0x1F),
                                        countbits(meta >> 5));
            }
            box = BvhUnion(box, boxes[slot]);
        }

        BvhNode refit = BvhQuantizeSlots(box, boxes, occupied);
        refit.exponents |= imask << 24;
        refit.firstChild = node.firstChild;
        refit.firstLeaf = node.firstLeaf;
        refit.meta = node.meta;
        nodes[at] = refit;
        float* stored = p.boxes + (base + at) * 6;
        for (uint axis = 0; axis < 3; ++axis) {
            stored[axis] = box.lower[axis];
            stored[3 + axis] = box.upper[axis];
        }

        const uint parent = p.parents[base + at];
        if (parent == kRefitNoParent) {
            for (uint axis = 0; axis < 3; ++axis) {
                header.lower[axis] = box.lower[axis];
                header.upper[axis] = box.upper[axis];
            }
            return;
        }
        AllMemoryBarrier();
        uint arrived;
        InterlockedAdd(p.arrivals[base + parent], 1u, arrived);
        if (arrived + 1 < countbits(nodes[parent].exponents >> 24)) {
            return;
        }
        AllMemoryBarrier();
        at = parent;
    }
}
