// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

#ifndef __SLANG__
#include "ernie/io/kernels/encoding.h"
#endif

namespace ernie {

#ifndef __SLANG__
using std::abs;
using std::isfinite;
using std::log;
#endif

static const uint kColorizeThreads = 256;
static const uint kNoRange = 0xFFFFFFFFu;
static const uint kFixedMinimum = 1;
static const uint kFixedMaximum = 2;
static const uint kLogarithmic = 4;
static const uint kReversed = 8;
static const uint kCentered = 16;
static const uint kWriteBounds = 32;
static const uint kColormapGrayscale = 0;
static const uint kColormapTurbo = 1;
static const uint kColormapViridis = 2;
static const float kSmallestLogValue = 1e-5f;

struct ColorizeParams {
    Ptr<uint> input;
    Ptr<uint> output;
    Ptr<uint> range;
    Ptr<float> bounds;
    uint count;
    uint channels;
    uint channel;
    uint colormap;
    uint flags;
    float minimum;
    float maximum;
    float scale;
};

ERNIE_CHECK_SIZE(ColorizeParams, 64);

struct ScalarBounds {
    float low;
    float high;
};

ERNIE_SHARED uint PackRgba8(uint r, uint g, uint b, uint a) {
    return r | (g << 8) | (b << 16) | (a << 24);
}

ERNIE_SHARED uint Unorm8(float value) {
    return uint(floor(clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f));
}

ERNIE_SHARED float Polynomial(float t, float c0, float c1, float c2, float c3, float c4, float c5,
                              float c6) {
    return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), 0.0f, 1.0f);
}

ERNIE_SHARED uint ColormapRgba(uint colormap, float position) {
    const float t = clamp(position, 0.0f, 1.0f);
    if (colormap == kColormapTurbo) {
        return PackRgba8(Unorm8(Polynomial(t, 0.13572138f, 4.61539260f, -42.66032258f,
                                           132.13108234f, -152.94239396f, 59.28637943f, 0.0f)),
                         Unorm8(Polynomial(t, 0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f,
                                           4.27729857f, 2.82956604f, 0.0f)),
                         Unorm8(Polynomial(t, 0.10667330f, 12.64194608f, -60.58204836f,
                                           110.36276771f, -89.90310912f, 27.34824973f, 0.0f)),
                         255);
    }
    if (colormap == kColormapViridis) {
        return PackRgba8(Unorm8(Polynomial(t, 0.27772733f, 0.10509304f, -0.33086183f, -4.63423050f,
                                           6.22826994f, 4.77638500f, -5.43545586f)),
                         Unorm8(Polynomial(t, 0.00540734f, 1.40461353f, 0.21484756f, -5.79910097f,
                                           14.17993337f, -13.74514538f, 4.64585261f)),
                         Unorm8(Polynomial(t, 0.33409981f, 1.38459016f, 0.09509516f, -19.33244096f,
                                           56.69055260f, -65.35303263f, 26.31243525f)),
                         255);
    }
    const uint shade = Unorm8(t);
    return PackRgba8(shade, shade, shade, 255);
}

ERNIE_SHARED bool ScalarShown(float value, uint flags) {
    return isfinite(value) && ((flags & kLogarithmic) == 0 || value > 0.0f);
}

ERNIE_SHARED uint OrderedBits(float value) {
    const uint bits = asuint(value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

ERNIE_SHARED float OrderedFloat(uint bits) {
    return asfloat((bits & 0x80000000u) != 0 ? bits & 0x7FFFFFFFu : ~bits);
}

ERNIE_SHARED ScalarBounds ResolveBounds(float low, float high, float minimum, float maximum,
                                        uint flags) {
    ScalarBounds bounds;
    bounds.low = (flags & kFixedMinimum) != 0 ? minimum : low;
    bounds.high = (flags & kFixedMaximum) != 0 ? maximum : high;
    if ((flags & kCentered) != 0 && isfinite(bounds.low) && isfinite(bounds.high)) {
        const float extent = max(abs(bounds.low), abs(bounds.high));
        bounds.low = -extent;
        bounds.high = extent;
    }
    if ((flags & kLogarithmic) != 0) {
        bounds.low = max(bounds.low, kSmallestLogValue);
    }
    bounds.high = max(bounds.high, bounds.low);
    return bounds;
}

ERNIE_SHARED uint ScalarRgba(float value, ScalarBounds bounds, uint flags, uint colormap) {
    if (!ScalarShown(value, flags)) {
        return PackRgba8(0, 0, 0, 255);
    }
    const bool logarithmic = (flags & kLogarithmic) != 0;
    const float low = logarithmic ? log(bounds.low) : bounds.low;
    const float span = (logarithmic ? log(bounds.high) : bounds.high) - low;
    const float clamped = clamp(value, bounds.low, bounds.high);
    float t = (flags & kCentered) != 0 ? 0.5f : 0.0f;
    if (span > 0.0f) {
        t = ((logarithmic ? log(clamped) : clamped) - low) / span;
    }
    return ColormapRgba(colormap, (flags & kReversed) != 0 ? 1.0f - t : t);
}

ERNIE_SHARED uint NormalRgba(float x, float y, float z) {
    if (x == 0 && y == 0 && z == 0) {
        return 0;
    }
    return PackRgba8(Unorm8(x * 0.5f + 0.5f), Unorm8(y * 0.5f + 0.5f), Unorm8(z * 0.5f + 0.5f),
                     255);
}

ERNIE_SHARED uint HdrRgba(float r, float g, float b, float a, float scale) {
    return PackRgba8(Unorm8(EncodeSrgb(r * scale)), Unorm8(EncodeSrgb(g * scale)),
                     Unorm8(EncodeSrgb(b * scale)), Unorm8(a));
}

ERNIE_SHARED uint MixIdWord(uint key, uint word) {
    return (key ^ word) * 16777619u;
}

ERNIE_SHARED uint IdRgba(uint key) {
    const uint scaled = ((key * 2654435769u) >> 16) * 6u;
    const uint sector = scaled >> 16;
    const uint f = (scaled >> 8) & 255u;
    const uint high = 242u;
    const uint low = 73u;
    const uint rise = low + (high - low) * f / 255u;
    const uint fall = high - (high - low) * f / 255u;
    if (sector == 0) {
        return PackRgba8(high, rise, low, 255);
    }
    if (sector == 1) {
        return PackRgba8(fall, high, low, 255);
    }
    if (sector == 2) {
        return PackRgba8(low, high, rise, 255);
    }
    if (sector == 3) {
        return PackRgba8(low, fall, high, 255);
    }
    if (sector == 4) {
        return PackRgba8(rise, low, high, 255);
    }
    return PackRgba8(high, low, fall, 255);
}

}
