// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"
#include "ernie/overlay/kernels/style.h"

#ifndef __SLANG__
#include "ernie/io/kernels/encoding.h"
#include "ernie/scene/records/instances.h"
#include "ernie/scene/records/render_tables.h"
#endif

namespace ernie {

ERNIE_ENUM(OverlayKind){
    Lines,
    Markers,
    Boxes,
    Spheres,
    Circles,
    Frames,
    Arrows,
    Capsules,
    Trails,
    Labels,
    Texts,
    Numbers,
    Triangles,
    Images,
    Prims,
    Outlines,
};

ERNIE_ENUM(OverlayPiece){
    Segment,
    Marker,
    Glyph,
    Box,
    Triangle,
};

static const uint kOverlaySphereSegments = 24;
static const uint kOverlayCircleSegments = 32;
static const uint kOverlayArrowHeadLines = 6;
static const uint kOverlayCapsuleSegments = 16;
static const uint kOverlayCapsuleArcSegments = 8;

static const uint kOverlayGroup = 64;
static const uint kOverlayTile = 8;
static const uint kOverlayTilePixels = kOverlayTile * kOverlayTile;

static const uint kOverlayDepthMask = 3;
static const uint kOverlayPerspective = 1u << 2;
static const uint kOverlayKindShift = 3;
static const uint kOverlayKindMask = 7;
static const uint kOverlayShapeShift = 6;
static const uint kOverlayShapeMask = 7;
static const uint kOverlayDashShift = 6;
static const uint kOverlayGapShift = 12;
static const uint kOverlayPatternMask = 63;
static const uint kOverlayRadiusShift = 18;
static const float kOverlayRadiusUnits = 8;
static const uint kOverlayMostRadius = (1u << 14) - 1;
static const uint kOverlayHalo = 1u << 6;
static const uint kOverlayEdgeShift = 6;
static const uint kOverlayEdgeMask = 7;
static const uint kOverlayTextured = 1u << 9;
static const uint kOverlayUvShift = 10;
static const uint kOverlayHaloShift = 7;
static const uint kOverlayHaloMask = 255;
static const uint kOverlayMostGroups = 65535;
static const uint kOverlayTileList = 4;
static const uint kOverlayTileWords = 4;

static const uint kOverlayByteColors = 1u << 0;
static const uint kOverlayHasColors = 1u << 1;
static const uint kOverlayHasSizes = 1u << 2;
static const uint kOverlayHasWorlds = 1u << 3;
static const uint kOverlayGrouped = 1u << 4;
static const uint kOverlayHasRotations = 1u << 5;
static const uint kOverlayHasScalars = 1u << 6;
static const uint kOverlayHasExtras = 1u << 7;
static const uint kOverlayPackedText = 1u << 8;
static const uint kOverlayScreen = 1u << 9;
static const uint kOverlayMeters = 1u << 10;
static const uint kOverlayUnited = 1u << 11;
static const uint kOverlayImageLayers = 1u << 12;
static const uint kOverlayScreenSized = 1u << 13;
static const uint kOverlayFacing = 1u << 14;
static const uint kOverlayPickable = 1u << 15;
static const uint kOverlayHasIds = 1u << 16;
static const uint kOverlayScreenPoints = 1u << 17;
static const uint kOverlayNoPick = ~0u;

static const uint kOverlayFrustumLines = 15;
static const uint kOverlayLightLines = 3 * kOverlaySphereSegments;
static const uint kOverlayNoImage = ~0u;
static const uint kOverlayBoxWords = 8;
static const uint kOverlayTextChars = 64;
static const uint kOverlayAffixChars = 16;
static const uint kOverlayMostDecimals = 6;
static const uint kOverlayNewline = 10;
static const uint kOverlayKernFirstCode = 32;
static const uint kOverlayKernCodeCount = 224;
static const float kOverlayLineUnits = 12.0f;
static const float kOverlayFieldEdge = 128.0f;
static const float kOverlayFieldLevels = 16.0f;
static const float kOverlayFieldTexelsPerUnit = 40.0f / 12.0f;
static const uint kOverlayFieldPadding = 8;
static const uint kOverlayFieldShift = 16;

struct OverlayGlyph {
    float x0;
    float y0;
    float x1;
    float y1;
    float advance;
    uint texel;
    uint width;
    uint height;
};

ERNIE_CHECK_SIZE(OverlayGlyph, 32);

static const uint kMostOverlays = 64;

struct OverlayView {
    Ptr<uint> overlay;
    Ptr<float> depth;
    Ptr<uint> mask;
    Ptr<float> maskDepth;
    uint64_t overlays;
    uint camera;
    uint layer;
    uint width;
    uint height;
    uint firstTile;
    uint columnCount;
    float missDistance;
    uint outlined;
    uint product;
    uint padding0;
};

ERNIE_CHECK_SIZE(OverlayView, 80);

struct OverlayOutline {
    Ptr<uint> bits;
    uint bitWords;
    uint color;
    float width;
    uint overlay;
};

ERNIE_CHECK_SIZE(OverlayOutline, 24);

struct OverlayBatch {
    Ptr<float> points;
    Ptr<float> ends;
    Ptr<uint> colors;
    Ptr<float> sizes;
    Ptr<uint> worlds;
    Ptr<uint> itemCounts;
    Ptr<uint> order;
    Ptr<uint> groupStarts;
    Ptr<float> rotations;
    Ptr<float> scalars;
    Ptr<uint> extras;
    uint64_t chars;
    Ptr<uint> prims;
    Ptr<uint> primGroups;
    Ptr<uint> boxes;
    uint64_t image;
    Ptr<uint> ids;
    uint pointStride;
    uint pointWorldStride;
    uint endStride;
    uint endWorldStride;
    uint colorStride;
    uint colorWorldStride;
    uint sizeStride;
    uint sizeWorldStride;
    uint rotationStride;
    uint rotationWorldStride;
    uint scalarStride;
    uint scalarWorldStride;
    uint extraStride;
    uint extraWorldStride;
    uint worldStride;
    uint itemCountStride;
    uint capacity;
    uint worldCount;
    uint firstItem;
    uint kind;
    uint flags;
    uint color;
    float size;
    uint world;
    uint anchor;
    uint overlay;
    uint groups;
    uint pieces;
    uint trailLength;
    float length;
    float fade;
    uint charStride;
    uint maxChars;
    uint prefix[4];
    uint suffix[4];
    uint prefixLength;
    uint suffixLength;
    uint decimals;
    uint background;
    float atX;
    float atY;
    float offsetX;
    float offsetY;
    float alignX;
    float alignY;
    uint depth;
    uint marker;
    uint dash;
    uint gap;
    uint primShape;
    uint primCount;
    uint imageWidth;
    uint imageHeight;
    uint imageRowStride;
    uint imageLayerStride;
    uint halo;
    float justify;
    uint id;
    uint idStride;
    uint padding;
};

ERNIE_CHECK_SIZE(OverlayBatch, 400);

struct OverlayPickQuery {
    uint view;
    float x;
    float y;
    float radius;
};

ERNIE_CHECK_SIZE(OverlayPickQuery, 16);

struct OverlayPickResult {
    uint key;
    uint work;
    uint batch;
    uint row;
    uint piece;
    uint id;
    uint overlay;
    uint world;
    float distance;
    float depth;
    uint hit;
    uint padding;
    float position[3];
    float origin[3];
    float direction[3];
    uint prim;
    uint tail[2];
};

ERNIE_CHECK_SIZE(OverlayPickResult, 96);

struct OverlayGroupParams {
    Ptr<uint> worlds;
    Ptr<uint> itemCounts;
    Ptr<uint> keys;
    Ptr<uint> order;
    Ptr<uint> starts;
    Ptr<uint> needs;
    uint rows;
    uint worldStride;
    uint groups;
    uint need;
};

ERNIE_CHECK_SIZE(OverlayGroupParams, 64);

struct OverlayUniteParams {
    Ptr<RenderTables> tables;
    Ptr<uint> prims;
    Ptr<uint> groups;
    Ptr<uint> boxes;
    uint primCount;
    uint primStride;
    uint groupStride;
    uint boxCount;
};

ERNIE_CHECK_SIZE(OverlayUniteParams, 48);

struct OverlayFill {
    uint64_t pixels;
    float x[3];
    float y[3];
    float depth[3];
    uint width;
    uint height;
    uint rowStride;
    uint opaque;
    uint padding;
};

ERNIE_CHECK_SIZE(OverlayFill, 64);

struct OverlayCameraImage {
    uint64_t pixels;
    uint camera;
    uint width;
    uint height;
    uint padding;
};

ERNIE_CHECK_SIZE(OverlayCameraImage, 24);

struct OverlayRecord {
    float ax;
    float ay;
    float bx;
    float by;
    float depthA;
    float depthB;
    uint color;
    uint flags;
};

ERNIE_CHECK_SIZE(OverlayRecord, 32);

struct OverlayParams {
    Ptr<RenderTables> tables;
    Ptr<OverlayView> views;
    Ptr<OverlayBatch> batches;
    Ptr<OverlayRecord> records;
    Ptr<uint> tileCounts;
    Ptr<uint> offsets;
    Ptr<uint> keys;
    Ptr<uint> values;
    Ptr<uint> pairCount;
    Ptr<uint> occupied;
    uint64_t font;
    Ptr<OverlayGlyph> glyphs;
    uint64_t kerning;
    Ptr<OverlayFill> fills;
    Ptr<uint> fillCount;
    Ptr<OverlayCameraImage> cameraImages;
    Ptr<OverlayOutline> outlines;
    uint cameraImageCount;
    uint fillCapacity;
    uint outlineCount;
    uint pickCount;
    Ptr<OverlayPickQuery> picks;
    Ptr<OverlayPickResult> pickResults;
    uint viewCount;
    uint batchCount;
    uint viewItemCount;
    uint workCount;
    uint tileCount;
    uint pairCapacity;
    float textHeight;
    uint padding;
};

ERNIE_CHECK_SIZE(OverlayParams, 200);

struct BlendParams {
    Ptr<uint> color;
    Ptr<uint> overlay;
    Ptr<uint> blended;
    uint count;
    uint padding;
};

ERNIE_CHECK_SIZE(BlendParams, 32);

ERNIE_SHARED uint PackOverlayColor(float r, float g, float b, float a) {
    return EncodeUnorm8(EncodeSrgb(r)) | EncodeUnorm8(EncodeSrgb(g)) << 8 |
           EncodeUnorm8(EncodeSrgb(b)) << 16 | EncodeUnorm8(a) << 24;
}

}
