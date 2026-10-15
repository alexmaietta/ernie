// SPDX-License-Identifier: BSD-3-Clause
// Skinned meshes and point caches posed on the GPU, a thread a point, as Unreal's GPU skin cache
// does: the rest points blended between the two samples either side of the time (one for a mesh
// without a point cache), then blend shapes
// add their weighted offsets to the rest points, then the joints' transforms blend, linearly (with
// the bind and the skeleton-to-mesh transforms folded into each joint's) or as dual quaternions
// (scale blended linearly first, signs taken against the point's heaviest joint). A subdivision
// surface's control points go to a scratch buffer, which `refineMain` turns into its refined
// points by their stencils; other meshes' go straight into the points pool. Normals follow from
// the triangles (mesh_normals.slang).
//
// References:
// - Pixar, UsdSkel, https://openusd.org/release/api/_usd_skel__intro.html (blend shapes before
//   skinning, geomBindTransform, skinning transforms).
// - N. Magnenat-Thalmann, R. Laperrière, D. Thalmann, "Joint-dependent local deformations for hand
//   animation and object grasping", Graphics Interface '88, 26-33, 1988.
// - L. Kavan, S. Collins, J. Žára, C. O'Sullivan, "Geometric Skinning with Approximate Dual
//   Quaternion Blending", ACM TOG 27(4), 105, 2008. https://doi.org/10.1145/1409625.1409627
// - Epic Games, Unreal Engine's GPU skin cache (GpuSkinCacheComputeShader.usf).

import math;

struct SkinConstants {
    // x: points, y: 0 linear or 1 dual quaternions, z: 1 when one set of influences serves all
    // points, w: 1 with blend shapes.
    uint4 info;
    // The rest points' samples (a point cache) blended: x the lower sample, y the upper, z how far
    // between them (a float's bits); one sample is 0, 0, 0.
    uint4 samples;
    float* rest;
    // Each point's influences from starts[point] to starts[point + 1].
    uint* influenceStarts;
    uint* influenceJoints;
    float* influenceWeights;
    // Each point's blend shape offsets from starts[point] to starts[point + 1], each with the
    // sub-shape it belongs to.
    uint* shapeStarts;
    float* shapeOffsets;
    uint* shapeIndices;
    float* shapeWeights;
    // Rows of 3x4 transforms: geomBindTransform and skeleton-to-mesh (dual quaternions only), then
    // each joint's: its 3 rows (linear), or its real and dual quaternions (x, y, z, w) and the 3
    // rows of its scale (dual quaternions).
    float4* pose;
    float* output;
}

struct RefineConstants {
    // x: refined points.
    uint4 info;
    float* control;
    uint* starts;
    uint* indices;
    float* weights;
    float* output;
}

[[vk::push_constant]] ConstantBuffer<SkinConstants> skin;

float3 transformRows(float4 r0, float4 r1, float4 r2, float3 p) {
    let h = float4(p, 1.0);
    return float3(dot(r0, h), dot(r1, h), dot(r2, h));
}

float4 multiply(float4 a, float4 b) {
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void skinMain(uint3 id : SV_DispatchThreadID) {
    let point = id.x;
    if (point >= skin.info.x) {
        return;
    }
    let lower = 3 * (skin.samples.x * skin.info.x + point);
    let upper = 3 * (skin.samples.y * skin.info.x + point);
    let alpha = asfloat(skin.samples.z);
    var p = lerp(float3(skin.rest[lower], skin.rest[lower + 1], skin.rest[lower + 2]),
                 float3(skin.rest[upper], skin.rest[upper + 1], skin.rest[upper + 2]), alpha);
    if (skin.info.w != 0) {
        for (uint k = skin.shapeStarts[point]; k < skin.shapeStarts[point + 1]; ++k) {
            let w = skin.shapeWeights[skin.shapeIndices[k]];
            p += w * float3(skin.shapeOffsets[3 * k], skin.shapeOffsets[3 * k + 1],
                            skin.shapeOffsets[3 * k + 2]);
        }
    }
    let first = skin.info.z != 0 ? skin.influenceStarts[0] : skin.influenceStarts[point];
    let last = skin.info.z != 0 ? skin.influenceStarts[1] : skin.influenceStarts[point + 1];
    var out = p;
    if (first < last) {
        if (skin.info.y == 0) {
            // The blended matrix, applied once.
            var r0 = float4(0.0);
            var r1 = float4(0.0);
            var r2 = float4(0.0);
            for (uint k = first; k < last; ++k) {
                let w = skin.influenceWeights[k];
                let at = 6 + 3 * skin.influenceJoints[k];
                r0 += w * skin.pose[at];
                r1 += w * skin.pose[at + 1];
                r2 += w * skin.pose[at + 2];
            }
            out = transformRows(r0, r1, r2, p);
        } else {
            let bound = transformRows(skin.pose[0], skin.pose[1], skin.pose[2], p);
            uint pivot = first;
            for (uint k = first + 1; k < last; ++k) {
                if (skin.influenceWeights[k] > skin.influenceWeights[pivot]) {
                    pivot = k;
                }
            }
            let pivotReal = skin.pose[6 + 5 * skin.influenceJoints[pivot]];
            var scaled = float3(0.0);
            var real = float4(0.0);
            var dual = float4(0.0);
            for (uint k = first; k < last; ++k) {
                var w = skin.influenceWeights[k];
                let at = 6 + 5 * skin.influenceJoints[k];
                scaled += w * transformRows(skin.pose[at + 2], skin.pose[at + 3],
                                            skin.pose[at + 4], bound);
                if (dot(skin.pose[at], pivotReal) < 0.0) {
                    w = -w;
                }
                real += w * skin.pose[at];
                dual += w * skin.pose[at + 1];
            }
            let length = sqrt(dot(real, real));
            if (length > 0.0) {
                real /= length;
                dual /= length;
            }
            let conjugate = float4(-real.xyz, real.w);
            let translation = 2.0 * multiply(dual, conjugate).xyz;
            out = transformRows(skin.pose[3], skin.pose[4], skin.pose[5],
                                rotate(real, scaled) + translation);
        }
    }
    skin.output[3 * point] = out.x;
    skin.output[3 * point + 1] = out.y;
    skin.output[3 * point + 2] = out.z;
}

[[vk::push_constant]] ConstantBuffer<RefineConstants> refine;

[shader("compute")]
[numthreads(64, 1, 1)]
void refineMain(uint3 id : SV_DispatchThreadID) {
    let point = id.x;
    if (point >= refine.info.x) {
        return;
    }
    var sum = float3(0.0);
    for (uint k = refine.starts[point]; k < refine.starts[point + 1]; ++k) {
        let c = 3 * refine.indices[k];
        sum += refine.weights[k] *
               float3(refine.control[c], refine.control[c + 1], refine.control[c + 2]);
    }
    refine.output[3 * point] = sum.x;
    refine.output[3 * point + 1] = sum.y;
    refine.output[3 * point + 2] = sum.z;
}

struct InstanceBlendConstants {
    // x: instances; y, z, w: the samples of positions, orientations and scales (0 for an array
    // not sampled, left as it is).
    uint4 info;
    // x: the lower sample, y: the upper, z: how far between them (a float's bits).
    uint4 blend;
    float* positions;
    float* orientations;
    float* scales;
    // The instancer's arrays in the instance pools' current half.
    float* outPositions;
    float* outOrientations;
    float* outScales;
}

[[vk::push_constant]] ConstantBuffer<InstanceBlendConstants> instances;

// A sampled instancer's arrays at the time code, a thread an instance: positions and scales
// blended linearly, orientations normalized after blending the shorter way (nlerp).
[shader("compute")]
[numthreads(64, 1, 1)]
void blendInstancesMain(uint3 id : SV_DispatchThreadID) {
    let i = id.x;
    let n = instances.info.x;
    if (i >= n) {
        return;
    }
    let alpha = asfloat(instances.blend.z);
    if (instances.info.y != 0) {
        let lo = 3 * (min(instances.blend.x, instances.info.y - 1) * n + i);
        let hi = 3 * (min(instances.blend.y, instances.info.y - 1) * n + i);
        let p = lerp(float3(instances.positions[lo], instances.positions[lo + 1],
                            instances.positions[lo + 2]),
                     float3(instances.positions[hi], instances.positions[hi + 1],
                            instances.positions[hi + 2]),
                     alpha);
        instances.outPositions[3 * i] = p.x;
        instances.outPositions[3 * i + 1] = p.y;
        instances.outPositions[3 * i + 2] = p.z;
    }
    if (instances.info.z != 0) {
        let lo = 4 * (min(instances.blend.x, instances.info.z - 1) * n + i);
        let hi = 4 * (min(instances.blend.y, instances.info.z - 1) * n + i);
        let a = float4(instances.orientations[lo], instances.orientations[lo + 1],
                       instances.orientations[lo + 2], instances.orientations[lo + 3]);
        var b = float4(instances.orientations[hi], instances.orientations[hi + 1],
                       instances.orientations[hi + 2], instances.orientations[hi + 3]);
        if (dot(a, b) < 0.0) {
            b = -b;
        }
        let q = lerp(a, b, alpha);
        let length = sqrt(dot(q, q));
        let unit = length > 0.0 ? q / length : float4(0.0, 0.0, 0.0, 1.0);
        instances.outOrientations[4 * i] = unit.x;
        instances.outOrientations[4 * i + 1] = unit.y;
        instances.outOrientations[4 * i + 2] = unit.z;
        instances.outOrientations[4 * i + 3] = unit.w;
    }
    if (instances.info.w != 0) {
        let lo = 3 * (min(instances.blend.x, instances.info.w - 1) * n + i);
        let hi = 3 * (min(instances.blend.y, instances.info.w - 1) * n + i);
        let s = lerp(float3(instances.scales[lo], instances.scales[lo + 1],
                            instances.scales[lo + 2]),
                     float3(instances.scales[hi], instances.scales[hi + 1], instances.scales[hi + 2]),
                     alpha);
        instances.outScales[3 * i] = s.x;
        instances.outScales[3 * i + 1] = s.y;
        instances.outScales[3 * i + 2] = s.z;
    }
}
