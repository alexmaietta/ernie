// SPDX-License-Identifier: BSD-3-Clause
// Camera poses straight from the GPU: each view's camera fields from a pose another API wrote
// (`ernie:poses` as a CUDA array, copied device to device into `poses`), patched into a buffer
// of per-view blocks (uniforms, cull views) the CPU wrote everything else of. One thread per
// view; a field at an offset of `kNone` is left alone.
//
// Matrices here are built row by row, so `m[r][c]` is row r, column c; buffers hold glm's
// column-major order.
//
// A camera rendered on cube faces (cube_faces.slang) has a block per view and face for its raster
// passes: block b is pose b / faces turned onto face b % faces, projected by the faces' own
// projection.
//
// Bindings: 0 the camera (parent-to-world, projection and its inverse, then the cube faces'
// projection and its inverse, as glm columns), 1 the poses (row-major 4x4, one per view; or with
// flag bit 1 seven numbers per view, a position and a unit quaternion x, y, z, w), 2 the per-view
// blocks written.
//
// References:
// - G. Gribb and K. Hartmann, "Fast Extraction of Viewing Frustum Planes from the
//   World-View-Projection Matrix", 2001.
//   https://www.gamedevs.org/uploads/fast-extraction-viewing-frustum-planes-from-world-view-projection-matrix.pdf
// - MESA GLU, `__gluInvertMatrixd` (cofactor inverse of a 4x4 matrix).
//   https://gitlab.freedesktop.org/mesa/glu/-/blob/master/src/libutil/project.c

static const uint kNone = 0xFFFFFFFFu;

struct ViewConstants {
    // Views, block stride and first block's byte offset, flags (bit 0: forward.w is the eye's
    // depth along it; bit 1: the poses are positions and quaternions).
    uint4 layout;
    // Byte offsets in a block: view-projection, its inverse, eye (xyz), forward (xyz).
    uint4 fields;
    // Byte offsets in a block: six normalised frustum planes, camera-to-world pose; then the cube
    // faces per pose (0 or 1 for blocks of whole views), unused.
    uint4 more;
}

import cube_faces;
import math;

[[vk::binding(0, 0)]] StructuredBuffer<float4> camera;
[[vk::binding(1, 0)]] StructuredBuffer<float> poses;
[[vk::binding(2, 0)]] RWByteAddressBuffer blocks;
[[vk::push_constant]] ConstantBuffer<ViewConstants> constants;

/// The matrix of four glm columns of `camera` from `first`.
float4x4 cameraMatrix(uint first) {
    return transpose(float4x4(camera[first], camera[first + 1], camera[first + 2],
                              camera[first + 3]));
}

/// Stores `matrix` at `address` in glm's column-major order.
void storeMatrix(uint address, float4x4 matrix) {
    let columns = transpose(matrix);
    for (uint c = 0; c < 4; ++c) {
        blocks.Store4(address + 16 * c, asuint(columns[c]));
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void viewsMain(uint3 id : SV_DispatchThreadID) {
    let layout = constants.layout;
    let block = id.x;
    if (block >= layout.x) {
        return;
    }
    let faces = max(constants.more.z, 1u);
    let view = block / faces;
    float4x4 local;
    if ((layout.w & 2u) != 0u) {
        // A position and a unit quaternion (x, y, z, w): the rotation's matrix, then the move.
        let at = view * 7;
        let q = normalize(float4(poses[at + 3], poses[at + 4], poses[at + 5], poses[at + 6]));
        let x = q.x;
        let y = q.y;
        let z = q.z;
        let w = q.w;
        local[0] = float4(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                          poses[at]);
        local[1] = float4(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                          poses[at + 1]);
        local[2] = float4(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
                          poses[at + 2]);
        local[3] = float4(0, 0, 0, 1);
    } else {
        for (uint r = 0; r < 4; ++r) {
            let at = view * 16 + r * 4;
            local[r] = float4(poses[at], poses[at + 1], poses[at + 2], poses[at + 3]);
        }
    }
    // Camera to world is the parent's world transform after the pose, as `ViewPosesOf`; a face's
    // turns it onto the face (`FrameParams::Faces`), whose axes are the rows of `cubeFaceToFace`.
    var pose = mul(cameraMatrix(0), local);
    var projection = 4u;
    if (faces > 1) {
        let axes = transpose(cubeFaceToFace(int(block % faces)));
        float4x4 turn;
        turn[0] = float4(axes[0], 0.0);
        turn[1] = float4(axes[1], 0.0);
        turn[2] = float4(axes[2], 0.0);
        turn[3] = float4(0.0, 0.0, 0.0, 1.0);
        pose = mul(pose, turn);
        projection = 12u;
    }
    let viewProjection = mul(cameraMatrix(projection), invert(pose));
    let eye = float3(pose[0][3], pose[1][3], pose[2][3]);
    let forward = -normalize(float3(pose[0][2], pose[1][2], pose[2][2]));

    let address = layout.z + block * layout.y;
    let fields = constants.fields;
    if (fields.x != kNone) {
        storeMatrix(address + fields.x, viewProjection);
    }
    if (fields.y != kNone) {
        // inverse(projection * view) = pose * inverse(projection).
        storeMatrix(address + fields.y, mul(pose, cameraMatrix(projection + 4)));
    }
    if (fields.z != kNone) {
        blocks.Store3(address + fields.z, asuint(eye));
    }
    if (fields.w != kNone) {
        if ((layout.w & 1u) != 0u) {
            blocks.Store4(address + fields.w, asuint(float4(forward, dot(eye, forward))));
        } else {
            blocks.Store3(address + fields.w, asuint(forward));
        }
    }
    let more = constants.more;
    if (more.x != kNone) {
        // Gribb and Hartmann's planes for [0, 1] depth (`Frustum::FromMatrix`); an infinite far
        // plane's is (0, 0, 0, near), which everything is inside.
        let m = viewProjection;
        float4 planes[6] = {m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1], m[2], m[3] - m[2]};
        for (uint p = 0; p < 6; ++p) {
            let size = length(planes[p].xyz);
            let plane = size > 1e-30 ? planes[p] / size : float4(0.0, 0.0, 0.0, 1.0);
            blocks.Store4(address + more.x + 16 * p, asuint(plane));
        }
    }
    if (more.y != kNone) {
        storeMatrix(address + more.y, pose);
    }
}
