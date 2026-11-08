// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

#include "ernie/scene/tables/kernels/tangents.h"

using namespace ernie;

float3 CornerPoint(TangentParams p, uint corner) {
    return p.points[uint(p.indices[corner])].Vector();
}

float3 CornerNormal(TangentParams p, uint corner) {
    if (p.cornerNormals != nullptr) {
        return p.cornerNormals[corner].Vector();
    }
    return p.smoothNormals[uint(p.indices[corner])].Vector();
}

[shader("compute")]
[numthreads(kTangentsGroup, 1, 1)]
void GroupTangents(uniform TangentParams p) {
    const uint group = LinearThreadIndex();
    if (group >= p.groupCount) {
        return;
    }
    float3 tangent = float3(0);
    float3 bitangent = float3(0);
    uint first = kNoRow;
    float3 facet = float3(0, 0, 1);
    for (uint at = p.groupStarts[group]; at < p.groupStarts[group + 1]; ++at) {
        const uint entry = p.groupCorners[at];
        const uint triangle = entry / 3;
        const uint k = entry % 3;
        const uint a = p.triangles[triangle * 3 + k];
        const uint b = p.triangles[triangle * 3 + (k + 1) % 3];
        const uint c = p.triangles[triangle * 3 + (k + 2) % 3];
        const float3 e1 = CornerPoint(p, b) - CornerPoint(p, a);
        const float3 e2 = CornerPoint(p, c) - CornerPoint(p, a);
        if (first == kNoRow) {
            first = a;
            facet = cross(e1, e2);
        }
        const float2 d1 = p.cornerSt[b].Vector() - p.cornerSt[a].Vector();
        const float2 d2 = p.cornerSt[c].Vector() - p.cornerSt[a].Vector();
        const float det = d1.x * d2.y - d2.x * d1.y;
        const float lengths = length(e1) * length(e2);
        if (abs(det) < 1e-20f || lengths <= 0) {
            continue;
        }
        const float angle = acos(clamp(dot(e1, e2) / lengths, -1.0f, 1.0f));
        const float3 t = (e1 * d2.y - e2 * d1.y) / det;
        const float3 s = (e2 * d1.x - e1 * d2.x) / det;
        const float tl = length(t);
        const float sl = length(s);
        if (tl > 0) {
            tangent += angle * t / tl;
        }
        if (sl > 0) {
            bitangent += angle * s / sl;
        }
    }
    float4 result = float4(1, 0, 0, 1);
    if (first != kNoRow) {
        const float3 n = normalize(p.faceted != 0 ? facet : CornerNormal(p, first));
        const float3 projected = tangent - n * dot(n, tangent);
        const float size = length(projected);
        if (size > 0) {
            const float3 t = projected / size;
            result = float4(t, dot(cross(n, t), bitangent) < 0 ? -1.0f : 1.0f);
        }
    }
    p.groupTangents[group] = Float4(result);
}

[shader("compute")]
[numthreads(kTangentsGroup, 1, 1)]
void CornerTangents(uniform TangentParams p) {
    const uint corner = LinearThreadIndex();
    if (corner >= p.cornerCount) {
        return;
    }
    const uint group = p.cornerGroups[corner];
    p.tangents[corner] = group == kNoRow ? Float4(float4(1, 0, 0, 1)) : p.groupTangents[group];
}
