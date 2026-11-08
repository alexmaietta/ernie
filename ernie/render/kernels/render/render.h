// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

#ifndef __SLANG__
#include "ernie/scene/kernels/instance_layout.h"
#include "ernie/scene/kernels/lights.h"
#include "ernie/scene/kernels/scene.h"
#include "ernie/scene/kernels/semantics.h"
#include "ernie/scene/kernels/tables/render_tables.h"
#endif

namespace ernie {

struct Capture {
    uint64_t vars[kRenderVarSourceCount];
    uint64_t firstPixel;
    uint64_t primIds;
    uint product;
    uint sensor;
    uint layer;
    uint width;
    uint height;
    uint taxonomy;
    float missDistance;
    uint returns;
};

ERNIE_CHECK_SIZE(Capture, 184);

struct PointRecord {
    float x;
    float y;
    float z;
    float distance;
    float intensity;
    float radialVelocity;
    uint prim;
};

ERNIE_CHECK_SIZE(PointRecord, 28);

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
    uint64_t next;
    uint slotsPerSensor;
    uint width;
    uint returns;
    uint sensors;
    uint world;
    uint padding;
};

ERNIE_CHECK_SIZE(PointCloud, 128);

#ifdef __SLANG__

struct CapturePixel {
    Capture* capture;
    uint2 coord;
    uint index;
};

#endif

struct RenderParams {
    Ptr<SceneData> scene;
    Ptr<RenderTables> tables;
    Ptr<Capture> captures;
    uint64_t firstPixel;
    uint captureCount;
    uint pixelCount;
};

ERNIE_CHECK_SIZE(RenderParams, 40);

struct ScatterParams {
    Ptr<PointCloud> cloud;
    Ptr<RenderTables> tables;
    uint slotCount;
    uint padding;
};

ERNIE_CHECK_SIZE(ScatterParams, 24);

static const uint kRenderGroup = 64;

}
