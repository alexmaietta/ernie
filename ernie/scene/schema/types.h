// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/columns/kernels/paged.h"

#ifndef __SLANG__
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <variant>
#include <vector>
#endif

namespace ernie {

#ifdef __SLANG__

struct Float2 {
    float x;
    float y;

    __init() {
        x = 0;
        y = 0;
    }

    __init(float2 v) {
        x = v.x;
        y = v.y;
    }

    float2 Vector() {
        return float2(x, y);
    }
};

struct Uint2 {
    uint x;
    uint y;
};

struct Float3 {
    float x;
    float y;
    float z;

    __init() {
        x = 0;
        y = 0;
        z = 0;
    }

    __init(float3 v) {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    float3 Vector() {
        return float3(x, y, z);
    }
};

struct Float4 {
    float x;
    float y;
    float z;
    float w;

    __init() {
        x = 0;
        y = 0;
        z = 0;
        w = 0;
    }

    __init(float4 v) {
        x = v.x;
        y = v.y;
        z = v.z;
        w = v.w;
    }

    float4 Vector() {
        return float4(x, y, z, w);
    }
};

struct Double3 {
    uint x[2];
    uint y[2];
    uint z[2];
};

struct Quatd {
    uint x[2];
    uint y[2];
    uint z[2];
    uint w[2];
};

#else

struct Float2 {
    float x = 0, y = 0;

    bool operator==(const Float2& other) const {
        return x == other.x && y == other.y;
    }

    bool operator!=(const Float2& other) const {
        return !(*this == other);
    }
};

struct Uint2 {
    uint32_t x = 0, y = 0;

    bool operator==(const Uint2& other) const {
        return x == other.x && y == other.y;
    }

    bool operator!=(const Uint2& other) const {
        return !(*this == other);
    }
};

struct Float3 {
    float x = 0, y = 0, z = 0;

    bool operator==(const Float3& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator!=(const Float3& other) const {
        return !(*this == other);
    }
};

struct Float4 {
    float x = 0, y = 0, z = 0, w = 0;

    bool operator==(const Float4& other) const {
        return x == other.x && y == other.y && z == other.z && w == other.w;
    }

    bool operator!=(const Float4& other) const {
        return !(*this == other);
    }
};

struct Double3 {
    double x = 0, y = 0, z = 0;

    bool operator==(const Double3& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator!=(const Double3& other) const {
        return !(*this == other);
    }
};

struct Quatd {
    double x = 0, y = 0, z = 0, w = 1;

    bool operator==(const Quatd& other) const {
        return x == other.x && y == other.y && z == other.z && w == other.w;
    }

    bool operator!=(const Quatd& other) const {
        return !(*this == other);
    }
};

using MaterialValue = std::variant<bool, int32_t, float, Float2, Float3, Float4, std::string>;

struct MaterialInput {
    std::string name;
    MaterialValue value;
};

using InstancedTokens = std::map<std::string, std::vector<std::string>, std::less<>>;

#endif

ERNIE_CHECK_SIZE(Float2, 8);
ERNIE_CHECK_SIZE(Uint2, 8);
ERNIE_CHECK_SIZE(Float3, 12);
ERNIE_CHECK_SIZE(Float4, 16);
ERNIE_CHECK_SIZE(Double3, 24);
ERNIE_CHECK_SIZE(Quatd, 32);

struct ArrayRef {
    uint64_t address;
    uint count;
    uint capacity;
};

ERNIE_CHECK_SIZE(ArrayRef, 16);

struct WorldTransform {
    Float3 x;
    Float3 y;
    Float3 z;
    Float3 high;
    Float3 low;

#ifdef __SLANG__
    float3 Translation() {
        return high.Vector() + low.Vector();
    }

    float3 TranslationFrom(WorldTransform origin) {
        return (high.Vector() - origin.high.Vector()) + (low.Vector() - origin.low.Vector());
    }

    float3 Apply(float3 v) {
        return x.Vector() * v.x + y.Vector() * v.y + z.Vector() * v.z;
    }

    float3 ApplyToNormal(float3 n) {
        return cross(y.Vector(), z.Vector()) * n.x + cross(z.Vector(), x.Vector()) * n.y +
               cross(x.Vector(), y.Vector()) * n.z;
    }
#endif
};

ERNIE_CHECK_SIZE(WorldTransform, 60);

}
