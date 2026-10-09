// Camera poses straight from the GPU: each view's camera fields from a pose another API wrote
// (`ernie:poses` as a CUDA array, copied device to device into `poses`), patched into a buffer
// of per-view blocks (uniforms, cull views) the CPU wrote everything else of. One thread per
// view; a field at an offset of `kNone` is left alone.
//
// Matrices here are built row by row, so `m[r][c]` is row r, column c; buffers hold glm's
// column-major order.
//
// Bindings: 0 the camera (parent-to-world, projection and its inverse, as glm columns), 1 the
// poses (row-major 4x4, one per view; or with flag bit 1 seven numbers per view, a position and a
// unit quaternion x, y, z, w), 2 the per-view blocks written.
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
    // Byte offsets in a block: six normalised frustum planes, camera-to-world pose.
    uint4 more;
}

[[vk::binding(0, 0)]] StructuredBuffer<float4> camera;
[[vk::binding(1, 0)]] StructuredBuffer<float> poses;
[[vk::binding(2, 0)]] RWByteAddressBuffer blocks;
[[vk::push_constant]] ConstantBuffer<ViewConstants> constants;

/// The matrix of four glm columns of `camera` from `first`.
float4x4 cameraMatrix(uint first) {
    return transpose(float4x4(camera[first], camera[first + 1], camera[first + 2],
                              camera[first + 3]));
}

float4x4 invert(float4x4 matrix) {
    float m[16];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            m[r * 4 + c] = matrix[r][c];
        }
    }
    float inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
             m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
             m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
             m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
              m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
             m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
             m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
             m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
              m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
             m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
             m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
              m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
              m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    let scale = det != 0.0 ? 1.0 / det : 0.0;
    float4x4 result;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            result[r][c] = inv[r * 4 + c] * scale;
        }
    }
    return result;
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
    let view = id.x;
    if (view >= layout.x) {
        return;
    }
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
    // Camera to world is the parent's world transform after the pose, as `ViewPosesOf`.
    let pose = mul(cameraMatrix(0), local);
    let viewProjection = mul(cameraMatrix(4), invert(pose));
    let eye = float3(pose[0][3], pose[1][3], pose[2][3]);
    let forward = -normalize(float3(pose[0][2], pose[1][2], pose[2][2]));

    let block = layout.z + view * layout.y;
    let fields = constants.fields;
    if (fields.x != kNone) {
        storeMatrix(block + fields.x, viewProjection);
    }
    if (fields.y != kNone) {
        // inverse(projection * view) = pose * inverse(projection).
        storeMatrix(block + fields.y, mul(pose, cameraMatrix(8)));
    }
    if (fields.z != kNone) {
        blocks.Store3(block + fields.z, asuint(eye));
    }
    if (fields.w != kNone) {
        if ((layout.w & 1u) != 0u) {
            blocks.Store4(block + fields.w, asuint(float4(forward, dot(eye, forward))));
        } else {
            blocks.Store3(block + fields.w, asuint(forward));
        }
    }
    let more = constants.more;
    if (more.x != kNone) {
        // Gribb and Hartmann's planes, with [0, 1] depth's near plane z_clip >= 0
        // (`Frustum::FromMatrix`).
        let m = viewProjection;
        float4 planes[6] = {m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1], m[2], m[3] - m[2]};
        for (uint p = 0; p < 6; ++p) {
            let plane = planes[p] / max(length(planes[p].xyz), 1e-30);
            blocks.Store4(block + more.x + 16 * p, asuint(plane));
        }
    }
    if (more.y != kNone) {
        storeMatrix(block + more.y, pose);
    }
}
