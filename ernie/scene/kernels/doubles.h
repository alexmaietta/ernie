// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

#ifdef __SLANG__

namespace ernie {

float2 SplitDouble(uint low, uint high) {
    const uint exponent = (high >> 20) & 0x7FFu;
    const float sign = (high >> 31) != 0 ? -1.0f : 1.0f;
    if (exponent == 0) {
        return float2(0.0f);
    }
    if (exponent == 0x7FFu) {
        return float2(sign * asfloat(0x7F800000u), 0.0f);
    }
    const float e = float(int(exponent) - 1023);
    const uint top = ((high & 0xFFFFFu) << 3) | (low >> 29);
    const uint rest = low & 0x1FFFFFFFu;
    const float h = sign * ldexp(1.0f + float(top) * exp2(-23.0f), e);
    const float l = sign * ldexp(float(rest), e - 52.0f);
    return float2(h, l);
}

float Narrow(uint low, uint high) {
    const float2 split = SplitDouble(low, high);
    return split.x + split.y;
}

uint2 DoubleBits(float value) {
    const uint bits = asuint(value);
    const uint sign = bits & 0x80000000u;
    uint exponent = (bits >> 23) & 0xFFu;
    uint mantissa = bits & 0x7FFFFFu;
    if (exponent == 0 && mantissa == 0) {
        return uint2(0, sign);
    }
    if (exponent == 0xFFu) {
        return uint2(mantissa << 29, sign | 0x7FF00000u | (mantissa >> 3));
    }
    int unbiased = int(exponent) - 127;
    if (exponent == 0) {
        unbiased = -126;
        while ((mantissa & 0x800000u) == 0) {
            mantissa <<= 1;
            unbiased -= 1;
        }
        mantissa &= 0x7FFFFFu;
    }
    const uint wide = uint(unbiased + 1023);
    return uint2(mantissa << 29, sign | (wide << 20) | (mantissa >> 3));
}

}

#endif
