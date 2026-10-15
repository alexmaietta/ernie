// SPDX-License-Identifier: BSD-3-Clause
// Instance transforms composed on the GPU: each instance's transform ops, read as rows of the raw
// arrays a compute API wrote (translations, scales, quaternions, Euler angles, 4x4 matrices),
// multiplied in order, then put under the instance it is posed beneath, written to the instance
// pools as the position, orientation and scale the mesh and cull passes compose
// (`placeVertex`, `placeAt`). One thread per instance of one instancer at one level of the
// hierarchy; levels run in order, so a parent's pose is written before its children read it.
//
// Poses are kept as translate, rotate, scale (T R S): composing two is exact unless the outer
// scales non-uniformly something the inner rotates, which T R S cannot hold (shear).
//
// Bindings: 0 the raw arrays, 1 the ops (kind, offset in floats from the instancer's raw arrays,
// width, flags; one uint4 each), 2 the instance indices of each dispatch, 3 per instance its
// parent (slot in the pools or ~0; then the transform from the parent's frame to its own
// parent's, as translate, quaternion, scale), 4 to 6 the pools written.
//
// References:
// - Pixar, UsdGeomXformable, https://openusd.org/release/api/class_usd_geom_xformable.html
// - K. Shoemake, "Animating Rotation with Quaternion Curves", SIGGRAPH 1985 (quaternions).
// - K. Shoemake, "Euler Angle Conversion", Graphics Gems IV, 1994.

import math;

static const uint kNone = 0xFFFFFFFFu;

// Op kinds, as `InstanceOpKind`.
static const uint kTranslate = 0;
static const uint kScale = 1;
static const uint kOrient = 2;
static const uint kRotateX = 3;
static const uint kRotateY = 4;
static const uint kRotateZ = 5;
static const uint kRotateXYZ = 6;  // ... to kRotateZYX = 11, in `RotationOrder`'s order
static const uint kTransform = 12;
// Op flags.
static const uint kWFirst = 1;

struct ComposeConstants {
    uint count;          // instances in this dispatch
    uint list;           // first entry of binding 2
    uint raw;            // the instancer's first raw float
    uint ops;            // its first op
    uint opCount;
    uint out;            // its first slot in the pools
    uint parents;        // its first parent record
    uint rows;           // its instances (rows of each raw array)
}

[[vk::binding(0, 0)]] StructuredBuffer<float> raw;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> ops;
[[vk::binding(2, 0)]] StructuredBuffer<uint> lists;
[[vk::binding(3, 0)]] StructuredBuffer<float4> parents;  // three per instance
[[vk::binding(4, 0)]] RWStructuredBuffer<float> positions;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> orientations;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> scales;
[[vk::push_constant]] ConstantBuffer<ComposeConstants> constants;

struct Trs {
    float3 t;
    float4 q;  // x, y, z, w
    float3 s;
}

Trs identity() {
    Trs pose;
    pose.t = float3(0.0);
    pose.q = float4(0.0, 0.0, 0.0, 1.0);
    pose.s = float3(1.0);
    return pose;
}

float4 qmul(float4 a, float4 b) {
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

// a after b: points go through b, then a.
Trs compose(Trs a, Trs b) {
    Trs pose;
    pose.t = a.t + rotate(a.q, a.s * b.t);
    pose.q = normalize(qmul(a.q, b.q));
    pose.s = a.s * b.s;
    return pose;
}

float4 axisAngle(uint axis, float degrees) {
    let half = radians(degrees) * 0.5;
    float4 q = float4(0.0, 0.0, 0.0, cos(half));
    q[axis] = sin(half);
    return q;
}

// The axes of Euler order `order` (0 XYZ ... 5 ZYX) in the order they apply.
uint3 eulerAxes(uint order) {
    switch (order) {
    case 0: return uint3(0, 1, 2);
    case 1: return uint3(0, 2, 1);
    case 2: return uint3(1, 0, 2);
    case 3: return uint3(1, 2, 0);
    case 4: return uint3(2, 0, 1);
    default: return uint3(2, 1, 0);
    }
}

// A rotation matrix's quaternion (Shepperd's method), given its columns.
float4 quatOfColumns(float3 c0, float3 c1, float3 c2) {
    let trace = c0.x + c1.y + c2.z;
    float4 q;
    if (trace > 0.0) {
        let s = sqrt(trace + 1.0) * 2.0;
        q = float4((c1.z - c2.y) / s, (c2.x - c0.z) / s, (c0.y - c1.x) / s, 0.25 * s);
    } else if (c0.x > c1.y && c0.x > c2.z) {
        let s = sqrt(1.0 + c0.x - c1.y - c2.z) * 2.0;
        q = float4(0.25 * s, (c1.x + c0.y) / s, (c2.x + c0.z) / s, (c1.z - c2.y) / s);
    } else if (c1.y > c2.z) {
        let s = sqrt(1.0 + c1.y - c0.x - c2.z) * 2.0;
        q = float4((c1.x + c0.y) / s, 0.25 * s, (c2.y + c1.z) / s, (c2.x - c0.z) / s);
    } else {
        let s = sqrt(1.0 + c2.z - c0.x - c1.y) * 2.0;
        q = float4((c2.x + c0.z) / s, (c2.y + c1.z) / s, 0.25 * s, (c0.y - c1.x) / s);
    }
    return normalize(q);
}

// One op of instance `row` as a pose.
Trs opPose(uint4 op, uint row) {
    let kind = op.x;
    let at = constants.raw + op.y + row * op.z;
    Trs pose = identity();
    if (kind == kTranslate) {
        pose.t = float3(raw[at], raw[at + 1], raw[at + 2]);
    } else if (kind == kScale) {
        pose.s = op.z == 1 ? float3(raw[at]) : float3(raw[at], raw[at + 1], raw[at + 2]);
    } else if (kind == kOrient) {
        let a = float4(raw[at], raw[at + 1], raw[at + 2], raw[at + 3]);
        pose.q = normalize((op.w & kWFirst) != 0 ? float4(a.y, a.z, a.w, a.x) : a);
    } else if (kind <= kRotateZ) {
        pose.q = axisAngle(kind - kRotateX, raw[at]);
    } else if (kind < kTransform) {
        // Fixed axes, the order's first axis first: the last applied is outermost.
        let degrees = float3(raw[at], raw[at + 1], raw[at + 2]);
        let axes = eulerAxes(kind - kRotateXYZ);
        pose.q = qmul(axisAngle(axes.z, degrees[axes.z]),
                      qmul(axisAngle(axes.y, degrees[axes.y]), axisAngle(axes.x, degrees[axes.x])));
    } else {
        // Row-major 4 x 4, column vectors: translation in the last column.
        float3 c0 = float3(raw[at], raw[at + 4], raw[at + 8]);
        float3 c1 = float3(raw[at + 1], raw[at + 5], raw[at + 9]);
        float3 c2 = float3(raw[at + 2], raw[at + 6], raw[at + 10]);
        pose.t = float3(raw[at + 3], raw[at + 7], raw[at + 11]);
        pose.s = float3(length(c0), length(c1), length(c2));
        if (dot(c0, cross(c1, c2)) < 0.0) {
            pose.s.x = -pose.s.x;
        }
        pose.q = quatOfColumns(c0 / pose.s.x, c1 / pose.s.y, c2 / pose.s.z);
    }
    return pose;
}

Trs readPose(uint slot) {
    Trs pose;
    pose.t = float3(positions[3 * slot], positions[3 * slot + 1], positions[3 * slot + 2]);
    pose.q = float4(orientations[4 * slot], orientations[4 * slot + 1],
                    orientations[4 * slot + 2], orientations[4 * slot + 3]);
    pose.s = float3(scales[3 * slot], scales[3 * slot + 1], scales[3 * slot + 2]);
    return pose;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void composeMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= constants.count) {
        return;
    }
    let row = lists[constants.list + id.x];
    // The ops in order, the first outermost.
    Trs pose = identity();
    for (uint i = 0; i < constants.opCount; ++i) {
        pose = compose(pose, opPose(ops[constants.ops + i], row));
    }
    // Under its parent: the parent's pose, then the CPU transforms between.
    let record = 3 * (constants.parents + row);
    let link = parents[record];
    if (asuint(link.x) != kNone) {
        Trs between;
        between.t = link.yzw;
        between.q = parents[record + 1];
        between.s = parents[record + 2].xyz;
        pose = compose(compose(readPose(asuint(link.x)), between), pose);
    }
    let slot = constants.out + row;
    positions[3 * slot] = pose.t.x;
    positions[3 * slot + 1] = pose.t.y;
    positions[3 * slot + 2] = pose.t.z;
    orientations[4 * slot] = pose.q.x;
    orientations[4 * slot + 1] = pose.q.y;
    orientations[4 * slot + 2] = pose.q.z;
    orientations[4 * slot + 3] = pose.q.w;
    scales[3 * slot] = pose.s.x;
    scales[3 * slot + 1] = pose.s.y;
    scales[3 * slot + 2] = pose.s.z;
}
