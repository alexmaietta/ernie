// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

namespace ernie {

ERNIE_SHARED float HalfToFloat(uint bits) {
    const uint sign = (bits & 0x8000) << 16;
    const uint exponent = (bits >> 10) & 0x1F;
    const uint mantissa = bits & 0x3FF;
    if (exponent == 0) {
        return asfloat(sign | asuint(float(mantissa) / 16777216.0f));
    }
    if (exponent == 31) {
        return asfloat(sign | 0x7F800000 | (mantissa << 13));
    }
    return asfloat(sign | ((exponent + 112) << 23) | (mantissa << 13));
}

ERNIE_SHARED uint FloatToHalf(float value) {
    const uint bits = asuint(value);
    const uint sign = (bits >> 16) & 0x8000;
    const uint magnitude = bits & 0x7FFFFFFF;
    if (magnitude >= 0x7F800000) {
        return sign | 0x7C00 | (magnitude > 0x7F800000 ? 0x200u : 0u);
    }
    if (magnitude >= 0x477FF000) {
        return sign | 0x7C00;
    }
    if (magnitude < 0x38800000) {
        const uint exponent = magnitude >> 23;
        const uint shift = 126 - exponent;
        if (shift > 24) {
            return sign;
        }
        const uint mantissa = (magnitude & 0x7FFFFF) | 0x800000;
        uint half = mantissa >> shift;
        const uint rest = mantissa & ((1u << shift) - 1);
        const uint midway = 1u << (shift - 1);
        if (rest > midway || (rest == midway && (half & 1) != 0)) {
            half += 1;
        }
        return sign | half;
    }
    const uint rounded = magnitude + 0xFFF + ((magnitude >> 13) & 1);
    return sign | ((rounded - 0x38000000) >> 13);
}

ERNIE_SHARED float DecodeSrgb(float encoded) {
    return encoded <= 0.04045f ? encoded / 12.92f : pow((encoded + 0.055f) / 1.055f, 2.4f);
}

ERNIE_SHARED float EncodeSrgb(float linear) {
    const float c = clamp(linear, 0.0f, 1.0f);
    return c <= 0.0031308f ? 12.92f * c : 1.055f * pow(c, 1 / 2.4f) - 0.055f;
}

ERNIE_SHARED uint EncodeUnorm8(float value) {
    return uint(floor(clamp(value, 0.0f, 1.0f) * 255 + 0.5f));
}

}
