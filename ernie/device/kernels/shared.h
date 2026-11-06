// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#ifdef __SLANG__

#define ERNIE_SHARED
#define ERNIE_CHECK_SIZE(type, bytes)
#define ERNIE_ENUM(name) enum class name : uint

#else

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#define ERNIE_SHARED inline
#define ERNIE_CHECK_SIZE(type, bytes)                                                              \
    static_assert(sizeof(type) == (bytes), #type " keeps its shared layout")
#define ERNIE_ENUM(name) enum class name : uint32_t

namespace ernie {

using uint = uint32_t;

constexpr uint64_t RoundUp(uint64_t value, uint64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

using std::ceil;
using std::clamp;
using std::floor;
using std::log2;
using std::max;
using std::min;
using std::pow;

template <class T> struct Ptr {
    uint64_t address = 0;
};

inline float asfloat(uint bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline uint asuint(float value) {
    uint bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

}

#endif
