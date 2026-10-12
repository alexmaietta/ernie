// SPDX-License-Identifier: BSD-3-Clause
// What precedes PNanoVDB.h in the volume module (`VolumePass`): its HLSL flavour, and a half to
// float decode in 32-bit integer arithmetic in place of `f16tof32`, which Slang lowers to 16-bit
// floats (SPIR-V's Float16 and Int16 capabilities, which Ernie's devices need not enable).
//
// Reference: IEEE 754-2019, binary16.

#define PNANOVDB_HLSL

float ernieHalfToFloat(uint bits) {
    let sign = (bits >> 15) & 1u;
    let exponent = (bits >> 10) & 31u;
    let mantissa = bits & 1023u;
    float value;
    if (exponent == 0u) {
        value = float(mantissa) * exp2(-24.0);
    } else if (exponent == 31u) {
        value = asfloat(mantissa == 0u ? 0x7f800000u : 0x7fc00000u);
    } else {
        value = asfloat(((exponent + 112u) << 23) | (mantissa << 13));
    }
    return sign != 0u ? -value : value;
}

#define f16tof32 ernieHalfToFloat
