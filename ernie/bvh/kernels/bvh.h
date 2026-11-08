// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

namespace ernie {

static const uint kBvhVersion = 3;

static const uint kBvhWidth = 8;
static const uint kBvhHeaderBytes = 64;
static const uint kBvhNodeBytes = 80;
static const uint kBvhTriangleBytes = 48;
static const uint kBvhInstanceBytes = 64;
static const uint kBvhBoxBytes = 32;

static const uint kBvhLeafSize = 3;

static const uint kBvhTriangles = 0;
static const uint kBvhInstances = 1;
static const uint kBvhBoxes = 2;

static const uint kBvhIndexMask = 0xFFFFFF;
static const uint kInstanceFacingCullDisable = 1u;
static const uint kInstanceFlipFacing = 2u;

struct BvhHeader {
    float lower[3];
    uint primitives;
    float upper[3];
    uint version;
    uint64_t nodes;
    uint64_t leaves;
    uint nodeCount;
    uint leafCount;
    uint kind;
    uint padding;
};

ERNIE_CHECK_SIZE(BvhHeader, kBvhHeaderBytes);

struct BvhNode {
    float origin[3];
    uint exponents;
    uint firstChild;
    uint firstLeaf;
    uint meta[2];
    uint lower[3][2];
    uint upper[3][2];
};

ERNIE_CHECK_SIZE(BvhNode, kBvhNodeBytes);

struct BvhTriangle {
    float edgeC[3];
    uint padding0;
    float edgeB[3];
    uint padding1;
    float a[3];
    uint index;
};

ERNIE_CHECK_SIZE(BvhTriangle, kBvhTriangleBytes);

struct BvhInstance {
    uint64_t blas;
    float toObject[12];
    uint custom;
    uint index;
};

ERNIE_CHECK_SIZE(BvhInstance, kBvhInstanceBytes);

struct BvhBoxRecord {
    float lower[3];
    uint index;
    float upper[3];
    uint padding;
};

ERNIE_CHECK_SIZE(BvhBoxRecord, kBvhBoxBytes);

struct BvhBox {
    float lower[3];
    float upper[3];
};

ERNIE_SHARED uint64_t BvhLeavesOffset(uint capacity) {
    return kBvhHeaderBytes + uint64_t(max(capacity, 1u)) * kBvhNodeBytes;
}

ERNIE_SHARED uint BvhRecordWords(uint kind) {
    return kind == kBvhInstances ? kBvhInstanceBytes / 16
           : kind == kBvhBoxes   ? kBvhBoxBytes / 16
                                 : kBvhTriangleBytes / 16;
}

ERNIE_SHARED uint BvhByte(uint words[2], uint slot) {
    return (words[slot >> 2] >> ((slot & 3) * 8)) & 0xFF;
}

ERNIE_SHARED uint BvhInternalMeta(uint slot) {
    return 1u << 5 | (24 + slot);
}

ERNIE_SHARED uint BvhLeafMeta(uint count, uint before) {
    return ((1u << count) - 1) << 5 | before;
}

ERNIE_SHARED BvhBox BvhChildBox(BvhNode node, uint slot) {
    BvhBox box;
    for (uint axis = 0; axis < 3; ++axis) {
        const float scale = asfloat(((node.exponents >> (axis * 8)) & 0xFF) << 23);
        box.lower[axis] = node.origin[axis] + float(BvhByte(node.lower[axis], slot)) * scale;
        box.upper[axis] = node.origin[axis] + float(BvhByte(node.upper[axis], slot)) * scale;
    }
    return box;
}

static const uint kInstanceCustomMask = 0xFFFFFF;

struct InstanceRecord {
    float transform[12];
    uint custom;
    uint flags;
    uint64_t blas;
};

ERNIE_CHECK_SIZE(InstanceRecord, 64);

struct BvhCorners {
    float a[3];
    float b[3];
    float c[3];
};

ERNIE_CHECK_SIZE(BvhCorners, 36);

static const uint kBvhEmptySlot = 0xFFFFFFFF;
static const float kBvhFar = 3.402823466e+38f;

struct BvhSlots {
    uint child[8];
};

ERNIE_SHARED uint BvhWithByte(uint word, uint index, uint value) {
    const uint shift = index * 8;
    return (word & ~(0xFFu << shift)) | (value << shift);
}

ERNIE_SHARED BvhBox BvhEmptyBox() {
    BvhBox box;
    for (uint axis = 0; axis < 3; ++axis) {
        box.lower[axis] = kBvhFar;
        box.upper[axis] = -kBvhFar;
    }
    return box;
}

ERNIE_SHARED BvhBox BvhUnion(BvhBox a, BvhBox b) {
    BvhBox box;
    for (uint axis = 0; axis < 3; ++axis) {
        box.lower[axis] = min(a.lower[axis], b.lower[axis]);
        box.upper[axis] = max(a.upper[axis], b.upper[axis]);
    }
    return box;
}

ERNIE_SHARED float BvhArea(BvhBox box) {
    const float x = max(box.upper[0] - box.lower[0], 0.0f);
    const float y = max(box.upper[1] - box.lower[1], 0.0f);
    const float z = max(box.upper[2] - box.lower[2], 0.0f);
    return 2 * (x * y + y * z + z * x);
}

ERNIE_SHARED float BvhCenter(BvhBox box, uint axis) {
    return (box.lower[axis] + box.upper[axis]) * 0.5f;
}

ERNIE_SHARED BvhBox BvhTriangleBox(BvhCorners corners) {
    BvhBox box;
    for (uint axis = 0; axis < 3; ++axis) {
        box.lower[axis] = min(corners.a[axis], min(corners.b[axis], corners.c[axis]));
        box.upper[axis] = max(corners.a[axis], max(corners.b[axis], corners.c[axis]));
    }
    return box;
}

ERNIE_SHARED BvhTriangle BvhMakeTriangle(BvhCorners corners, uint index) {
    BvhTriangle triangle;
    for (uint axis = 0; axis < 3; ++axis) {
        triangle.edgeC[axis] = corners.c[axis] - corners.a[axis];
        triangle.edgeB[axis] = corners.b[axis] - corners.a[axis];
        triangle.a[axis] = corners.a[axis];
    }
    triangle.padding0 = 0;
    triangle.padding1 = 0;
    triangle.index = index;
    return triangle;
}

ERNIE_SHARED BvhBoxRecord BvhMakeBox(BvhBox box, uint index) {
    BvhBoxRecord record;
    for (uint axis = 0; axis < 3; ++axis) {
        record.lower[axis] = box.lower[axis];
        record.upper[axis] = box.upper[axis];
    }
    record.index = index;
    record.padding = 0;
    return record;
}

ERNIE_SHARED BvhInstance BvhMakeInstance(InstanceRecord instance, uint index) {
    BvhInstance record;
    record.blas = instance.blas;
    record.custom = instance.custom;
    record.index = index | (instance.flags >> 24 << 24);
    float m[12];
    for (uint i = 0; i < 12; ++i) {
        m[i] = instance.transform[i];
    }
    float inverse[9];
    inverse[0] = m[5] * m[10] - m[6] * m[9];
    inverse[1] = m[2] * m[9] - m[1] * m[10];
    inverse[2] = m[1] * m[6] - m[2] * m[5];
    inverse[3] = m[6] * m[8] - m[4] * m[10];
    inverse[4] = m[0] * m[10] - m[2] * m[8];
    inverse[5] = m[2] * m[4] - m[0] * m[6];
    inverse[6] = m[4] * m[9] - m[5] * m[8];
    inverse[7] = m[1] * m[8] - m[0] * m[9];
    inverse[8] = m[0] * m[5] - m[1] * m[4];
    const float determinant = m[0] * inverse[0] + m[1] * inverse[3] + m[2] * inverse[6];
    const float scale = determinant != 0 ? 1.0f / determinant : 0.0f;
    for (uint row = 0; row < 3; ++row) {
        float moved = 0;
        for (uint column = 0; column < 3; ++column) {
            const float value = inverse[row * 3 + column] * scale;
            record.toObject[row * 4 + column] = value;
            moved += value * m[column * 4 + 3];
        }
        record.toObject[row * 4 + 3] = -moved;
    }
    return record;
}

ERNIE_SHARED BvhBox BvhInstanceBox(InstanceRecord instance, BvhBox bounds) {
    BvhBox box;
    for (uint row = 0; row < 3; ++row) {
        float lower = instance.transform[row * 4 + 3];
        float upper = lower;
        for (uint column = 0; column < 3; ++column) {
            const float m = instance.transform[row * 4 + column];
            const float a = m * bounds.lower[column];
            const float b = m * bounds.upper[column];
            lower += min(a, b);
            upper += max(a, b);
        }
        box.lower[row] = lower;
        box.upper[row] = upper;
    }
    return box;
}

#ifdef __SLANG__
BvhBox BvhInstanceWorldBox(InstanceRecord instance) {
    if (instance.blas == 0) {
        return BvhEmptyBox();
    }
    BvhHeader* blas = (BvhHeader*)instance.blas;
    BvhBox bounds;
    for (uint axis = 0; axis < 3; ++axis) {
        bounds.lower[axis] = blas.lower[axis];
        bounds.upper[axis] = blas.upper[axis];
    }
    return BvhInstanceBox(instance, bounds);
}
#endif

ERNIE_SHARED int BvhGridExponent(float extent) {
    if (!(extent > 0)) {
        return -126;
    }
    return int(clamp(ceil(log2(extent * (1.0f / 255.0f))), -126.0f, 126.0f));
}

ERNIE_SHARED BvhSlots BvhAssignSlots(BvhBox box, BvhBox children[8], uint count) {
    float cost[8][8];
    for (uint i = 0; i < 8; ++i) {
        for (uint slot = 0; slot < 8; ++slot) {
            if (i >= count) {
                cost[slot][i] = kBvhFar;
                continue;
            }
            const float dx = BvhCenter(children[i], 0) - BvhCenter(box, 0);
            const float dy = BvhCenter(children[i], 1) - BvhCenter(box, 1);
            const float dz = BvhCenter(children[i], 2) - BvhCenter(box, 2);
            cost[slot][i] = ((slot & 4) != 0 ? -dx : dx) + ((slot & 2) != 0 ? -dy : dy) +
                            ((slot & 1) != 0 ? -dz : dz);
        }
    }
    BvhSlots slots;
    bool assigned[8];
    for (uint slot = 0; slot < 8; ++slot) {
        slots.child[slot] = kBvhEmptySlot;
        assigned[slot] = false;
    }
    for (uint round = 0; round < 8; ++round) {
        float least = kBvhFar;
        uint bestSlot = kBvhEmptySlot;
        uint bestChild = 0;
        for (uint slot = 0; slot < 8; ++slot) {
            for (uint i = 0; i < count; ++i) {
                if (slots.child[slot] == kBvhEmptySlot && !assigned[i] && cost[slot][i] < least) {
                    least = cost[slot][i];
                    bestSlot = slot;
                    bestChild = i;
                }
            }
        }
        if (bestSlot == kBvhEmptySlot) {
            break;
        }
        slots.child[bestSlot] = bestChild;
        assigned[bestChild] = true;
    }
    for (uint i = 0; i < count; ++i) {
        for (uint slot = 0; slot < 8 && !assigned[i]; ++slot) {
            if (slots.child[slot] == kBvhEmptySlot) {
                slots.child[slot] = i;
                assigned[i] = true;
            }
        }
    }
    return slots;
}

ERNIE_SHARED BvhNode BvhQuantizeSlots(BvhBox box, BvhBox children[8], uint occupied) {
    BvhNode node;
    node.firstChild = 0;
    node.firstLeaf = 0;
    node.exponents = 0;
    node.meta[0] = 0;
    node.meta[1] = 0;
    for (uint axis = 0; axis < 3; ++axis) {
        node.origin[axis] = box.lower[axis];
        const int exponent = BvhGridExponent(box.upper[axis] - box.lower[axis]);
        node.exponents |= uint(exponent + 127) << (axis * 8);
        const float inverse = asfloat(uint(127 - exponent) << 23);
        for (uint word = 0; word < 2; ++word) {
            node.lower[axis][word] = 0;
            node.upper[axis][word] = 0;
        }
        for (uint slot = 0; slot < 8; ++slot) {
            if ((occupied >> slot & 1) == 0) {
                continue;
            }
            const float low = floor((children[slot].lower[axis] - box.lower[axis]) * inverse);
            const float high = ceil((children[slot].upper[axis] - box.lower[axis]) * inverse);
            node.lower[axis][slot >> 2] =
                BvhWithByte(node.lower[axis][slot >> 2], slot & 3, uint(clamp(int(low), 0, 255)));
            node.upper[axis][slot >> 2] =
                BvhWithByte(node.upper[axis][slot >> 2], slot & 3, uint(clamp(int(high), 0, 255)));
        }
    }
    return node;
}

ERNIE_SHARED BvhNode BvhQuantize(BvhBox box, BvhBox children[8], BvhSlots slots) {
    BvhBox ordered[8];
    uint occupied = 0;
    for (uint slot = 0; slot < 8; ++slot) {
        if (slots.child[slot] != kBvhEmptySlot) {
            ordered[slot] = children[slots.child[slot]];
            occupied |= 1u << slot;
        }
    }
    return BvhQuantizeSlots(box, ordered, occupied);
}

}
