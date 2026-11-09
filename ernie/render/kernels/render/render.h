// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

#ifndef __SLANG__
#include "ernie/scene/records/instances.h"
#include "ernie/scene/records/lights.h"
#include "ernie/scene/records/render_tables.h"
#include "ernie/scene/records/semantics.h"
#endif

namespace ernie {

struct Capture {
    PlannedCapture plan;
    uint64_t vars[kRenderVarSourceCount];
    uint64_t primIds;
};

ERNIE_CHECK_SIZE(Capture, 192);

struct PointRecord {
    float x;
    float y;
    float z;
    float distance;
    float intensity;
    float radialVelocity;
    uint prim;
    uint padding;
    int64_t instanceId;
    float directionX;
    float directionY;
    float directionZ;
    uint padding1;
};

ERNIE_CHECK_SIZE(PointRecord, 56);

struct PointCloud {
    uint64_t records;
    uint64_t flags;
    uint64_t offsets;
    uint64_t counts;
    uint64_t coordinates;
    uint64_t distance;
    uint64_t intensity;
    uint64_t stableId;
    uint64_t emitter;
    uint64_t firing;
    uint64_t echo;
    uint64_t radialVelocity;
    uint64_t instanceId;
    uint64_t radialVelocityVector;
    uint64_t next;
    uint slotsPerSensor;
    uint width;
    uint returns;
    uint sensors;
    uint world;
    uint padding;
};

ERNIE_CHECK_SIZE(PointCloud, 144);

#ifdef __SLANG__

struct CapturePixel {
    Capture* capture;
    uint2 coord;
    uint index;
};

#endif

struct RenderParams {
    Ptr<RenderTables> tables;
    Ptr<Capture> captures;
    uint64_t firstPixel;
    uint captureCount;
    uint pixelCount;
};

ERNIE_CHECK_SIZE(RenderParams, 32);

struct ScatterParams {
    Ptr<PointCloud> cloud;
    Ptr<RenderTables> tables;
    uint slotCount;
    uint padding;
};

ERNIE_CHECK_SIZE(ScatterParams, 24);

static const uint kRenderGroup = 64;

}
