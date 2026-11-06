// SPDX-License-Identifier: BSD-3-Clause

struct Basis : IDifferentiable {
    float3 tangent;
    float3 bitangent;
    float3 normal;

    [Differentiable]
    __init(float3 normal) {
        const float s = normal.z >= 0 ? 1.0f : -1.0f;
        const float a = -1 / (s + normal.z);
        const float b = normal.x * normal.y * a;
        tangent = float3(1 + s * normal.x * normal.x * a, s * b, -s * normal.x);
        bitangent = float3(b, s + normal.y * normal.y * a, -normal.y);
        this.normal = normal;
    }

    [Differentiable]
    float3 ToLocal(float3 v) {
        return float3(dot(v, tangent), dot(v, bitangent), dot(v, normal));
    }

    [Differentiable]
    float3 ToWorld(float3 v) {
        return v.x * tangent + v.y * bitangent + v.z * normal;
    }
}
