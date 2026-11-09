// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

namespace ernie {

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

}
