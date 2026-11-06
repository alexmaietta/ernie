// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/bvh/kernels/bvh.h"

namespace ernie {

static const uint kLbvhThreads = 64;
static const uint kLbvhLevels = 64;
static const uint kLbvhCodeBits = 30;

static const uint kLbvhLeaf = 0x80000000;
static const uint kLbvhNone = 0xFFFFFFFF;

static const uint kLbvhNodeWords = 12;
static const uint kLbvhLower = 0;
static const uint kLbvhLeft = 3;
static const uint kLbvhUpper = 4;
static const uint kLbvhRight = 7;
static const uint kLbvhParent = 8;
static const uint kLbvhFirst = 9;
static const uint kLbvhLast = 10;

static const uint kLbvhBoundsWords = 12;
static const uint kLbvhMinCenter = 0;
static const uint kLbvhMinLower = 3;
static const uint kLbvhMaxCenter = 6;
static const uint kLbvhMaxUpper = 9;

static const uint kLbvhTaskWords = 3;

struct LbvhGeometry {
    Ptr<float> data;
    Ptr<uint> indices;
    uint stride;
    uint padding;
};

ERNIE_CHECK_SIZE(LbvhGeometry, 24);

struct LbvhState {
    uint parity;
    uint level;
    uint next;
    uint groups[3];
    uint padding[2];
};

ERNIE_CHECK_SIZE(LbvhState, 32);

struct LbvhParams {
    Ptr<LbvhGeometry> geometries;
    Ptr<InstanceRecord> instances;
    Ptr<uint> offsets;
    Ptr<uint> counts;
    Ptr<uint64_t> headers;
    Ptr<float> boxes;
    Ptr<uint> bounds;
    Ptr<uint> keys;
    Ptr<uint> order;
    Ptr<uint> nodes;
    Ptr<uint> parents;
    Ptr<uint> arrivals;
    Ptr<uint> roots;
    Ptr<uint> counters;
    Ptr<uint> tasks;
    Ptr<LbvhState> state;
    uint total;
    uint segments;
    uint kind;
    uint keyWords;
    uint queue;
    uint padding;
};

ERNIE_CHECK_SIZE(LbvhParams, 152);

ERNIE_SHARED uint LbvhOrdered(float value) {
    const uint bits = asuint(value);
    return (bits & 0x80000000) != 0 ? ~bits : bits | 0x80000000;
}

ERNIE_SHARED float LbvhFloat(uint ordered) {
    return asfloat((ordered & 0x80000000) != 0 ? ordered & 0x7FFFFFFF : ~ordered);
}

}
