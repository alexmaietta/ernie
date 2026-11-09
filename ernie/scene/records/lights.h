// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/io/kernels/ies.h"
#include "ernie/scene/records/instances.h"
#include "ernie/scene/records/textures.h"
#include "ernie/stage/schema/ids.h"

namespace ernie {

ERNIE_ENUM(LightType) {
    Distant,
    Sphere,
    Rect,
    Disk,
    Cylinder,
    Dome,
    Portal,
    Unsupported,
};

static const uint kLightTreatAsPoint = 1u << 0;
static const uint kLightTreatAsLine = 1u << 1;
static const uint kLightNormalize = 1u << 2;

static const uint kLightShaped = 1u << 3;
static const uint kLightVisible = 1u << 4;
static const uint kLightShadowless = 1u << 5;

struct LightRecord {
    Float3 position;
    uint prim;
    Float3 axisX;
    uint type;
    Float3 axisY;
    uint flags;
    Float3 axisZ;
    float power;
    Float3 radiance;
    float radius;
    Float3 focusTint;
    float focus;
    float width;
    float height;
    float length;
    float cosAngle;
    float diffuse;
    float specular;
    float coneAngle;
    float coneSoftness;
    uint forwardAxis;
    float spread;
    uint lightLink;
    uint shadowLink;
    Float3 shadowColor;
    float shadowDistance;
    float shadowFalloff;
    float shadowFalloffGamma;
    uint textures;
    uint textureFormat;
    Ptr<float> map;
    uint portals;
    uint ies;
    float iesAngleScale;
    float iesScale;
    uint instancer;
    uint copyOffset;
    uint64_t portalPrims;
};

ERNIE_CHECK_SIZE(LightRecord, 216);

struct LightLayout {
    Ptr<LightRecord> records;
    Ptr<uint> firstLights;
    Ptr<uint> prims;
    Ptr<float> cdf;
    Ptr<float> maps;
    Ptr<uint> domePrims;
    Ptr<uint> domeSlots;
    Ptr<uint> offsetBases;
    Ptr<Float3> offsets;
    Ptr<uint> copies;
    uint worldCount;
    uint domeCount;
};

ERNIE_CHECK_SIZE(LightLayout, 88);

static const uint kDomeMapWidth = 256;
static const uint kDomeMapHeight = 128;
static const uint kDomeMapCells = kDomeMapWidth * kDomeMapHeight;
static const uint kDomeMapFloats = kDomeMapCells + kDomeMapHeight + 1;

static const uint kSpectralRows = 95;
static const float kSpectralFirstNm = 360;
static const float kSpectralStepNm = 5;
static const uint kTemperatureRows = 256;
static const float kMaxMired = 1000;
static const uint kBlackbodyRow = kSpectralRows;
static const uint kDaylightRow = kBlackbodyRow + kTemperatureRows;
static const uint kWhiteRow = kDaylightRow + kTemperatureRows;

#ifndef __SLANG__

struct SpectralTables {
    Float3 cmf[kSpectralRows];
    Float3 blackbody[kTemperatureRows];
    Float3 daylight[kTemperatureRows];
    Float3 white;
    uint padding;
};

ERNIE_CHECK_SIZE(SpectralTables, 7300);

#endif

struct CollectionRecord {
    ArrayRef includes;
    ArrayRef excludes;
    uint includeRoot;
    uint explicitOnly;
    uint padding0;
    uint padding1;
};

ERNIE_CHECK_SIZE(CollectionRecord, 48);

struct LightShapeRecord {
    float radius;
    float width;
    float height;
    float length;
    uint treatAsPoint;
    uint padding0;
    uint padding1;
    uint padding2;
};

ERNIE_CHECK_SIZE(LightShapeRecord, 32);

static const uint kLoopedLights = 8;

#ifdef __SLANG__

bool Lists(ArrayRef targets, uint prim) {
    Ptr<uint> items = Ptr<uint>(targets.address);
    for (uint i = 0; i < targets.count; ++i) {
        if (items[i] == prim) {
            return true;
        }
    }
    return false;
}

float3 ShapingWeight(LightRecord light, float cosine) {
    if ((light.flags & kLightShaped) == 0) {
        return float3(1);
    }
    const float angle = acos(clamp(cosine, -1.0f, 1.0f));
    float cone = angle <= light.coneAngle ? 1 : 0;
    const float inner = light.coneAngle * (1 - light.coneSoftness);
    if (light.coneSoftness > 0 && angle > inner && angle <= light.coneAngle) {
        cone = smoothstep(light.coneAngle, inner, angle);
    }
    const float focused = light.focus > 0 ? pow(max(cosine, 0.0f), light.focus) : 1;
    const float3 tint = light.focusTint.Vector();
    return cone * (tint + (1 - tint) * focused);
}

float3 ForwardAxis(LightRecord light) {
    switch (ShapingAxis(light.forwardAxis)) {
    case ShapingAxis::PlusZ:
        return light.axisZ.Vector();
    case ShapingAxis::MinusY:
        return -light.axisY.Vector();
    case ShapingAxis::PlusY:
        return light.axisY.Vector();
    case ShapingAxis::MinusX:
        return -light.axisX.Vector();
    case ShapingAxis::PlusX:
        return light.axisX.Vector();
    default:
        return -light.axisZ.Vector();
    }
}

float3 EmissionAcross(LightRecord light) {
    const ShapingAxis axis = ShapingAxis(light.forwardAxis);
    return axis == ShapingAxis::MinusX || axis == ShapingAxis::PlusX ? light.axisY.Vector()
                                                                     : light.axisX.Vector();
}

float2 IesSt(LightRecord light, float3 emitted) {
    const float3 forward = ForwardAxis(light);
    const float3 across = EmissionAcross(light);
    const float polar =
        acos(clamp(dot(emitted, forward), -1.0f, 1.0f)) / max(1 + light.iesAngleScale, 1e-3f);
    if (polar > kPi) {
        return float2(-1);
    }
    float around = atan2(dot(emitted, cross(forward, across)), dot(emitted, across)) / (2 * kPi);
    around -= floor(around);
    return float2((around * (kIesColumns - 1) + 0.5f) / kIesColumns,
                  1 - (polar / kPi * (kIesRows - 1) + 0.5f) / kIesRows);
}

float2 CrossFace(uint column, uint row, float a, float b) {
    const float edge = 1 - 1e-4f;
    const float2 at = float2(clamp(a, -edge, edge), clamp(b, -edge, edge));
    return float2((column + 0.5f * (at.x + 1)) / 3, 1 - (row + 0.5f * (1 - at.y)) / 4);
}

float2 DomeTextureSt(DomeTextureFormat format, float3 n) {
    switch (format) {
    case DomeTextureFormat::MirroredBall: {
        const float3 h = n + float3(0, 0, 1);
        const float size = length(h);
        const float2 xy = size > 1e-6f ? h.xy / size : float2(1, 0);
        return 0.5f + 0.5f * xy;
    }
    case DomeTextureFormat::Angular: {
        const float side = length(n.xy);
        const float radius = side > 1e-6f ? acos(clamp(n.z, -1.0f, 1.0f)) / (kPi * side) : 0;
        return 0.5f + 0.5f * radius * n.xy;
    }
    case DomeTextureFormat::CubeMapVerticalCross: {
        const float3 a = abs(n);
        if (a.x >= a.y && a.x >= a.z) {
            return n.x > 0 ? CrossFace(2, 1, n.z / a.x, n.y / a.x)
                           : CrossFace(0, 1, -n.z / a.x, n.y / a.x);
        }
        if (a.y >= a.z) {
            return n.y > 0 ? CrossFace(1, 0, n.x / a.y, n.z / a.y)
                           : CrossFace(1, 2, n.x / a.y, -n.z / a.y);
        }
        return n.z < 0 ? CrossFace(1, 1, n.x / a.z, n.y / a.z)
                       : CrossFace(1, 3, n.x / a.z, -n.y / a.z);
    }
    default: {
        const float u = (atan2(n.z, n.x) + 0.5f * kPi) / (2 * kPi);
        return float2(u - floor(u), 1 - acos(clamp(n.y, -1.0f, 1.0f)) / kPi);
    }
    }
}

float3 DomeMapDirection(float2 uv) {
    const float phi = 2 * kPi * uv.x;
    const float theta = kPi * uv.y;
    return float3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
}

#endif

}
