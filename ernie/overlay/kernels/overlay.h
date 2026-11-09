// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"
#include "ernie/overlay/kernels/fills.h"
#include "ernie/overlay/kernels/mask.h"
#include "ernie/overlay/kernels/picks.h"
#include "ernie/overlay/kernels/style.h"
#include "ernie/overlay/kernels/text.h"

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
static const uint kOverlayEndVectors = 1u << 18;

static const uint kOverlayBoxWords = 8;

static const uint kMostOverlays = 64;

struct OverlayView {
    Ptr<uint> overlay;
    Ptr<float> depth;
    Ptr<uint> mask;
    Ptr<float> maskDepth;
    Ptr<uint> maskTiles;
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

ERNIE_CHECK_SIZE(OverlayView, 88);

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
    float vectorScale;
};

ERNIE_CHECK_SIZE(OverlayBatch, 400);

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

ERNIE_SHARED uint PackOverlayColor(float r, float g, float b, float a) {
    return EncodeUnorm8(EncodeSrgb(r)) | EncodeUnorm8(EncodeSrgb(g)) << 8 |
           EncodeUnorm8(EncodeSrgb(b)) << 16 | EncodeUnorm8(a) << 24;
}

}
