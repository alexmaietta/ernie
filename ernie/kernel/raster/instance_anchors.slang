// Transforms posed under prims posed on the GPU (a camera or a point cloud on a robot's link):
// each one relative to the frame's origin, from its frame's transform (the CPU's), its anchor's
// pose in the instance pools (as the compose pass or a compute API left it), and the transforms
// from the anchor down to it (the CPU's): pose = frame * anchor * below. One thread per record;
// a record without an anchor takes frame * below.
//
// Written by `constants.y`: 0, a view's camera to world pose, row-major into the views' GPU poses,
// which the views pass turns into each view's camera fields (views.slang); 1, a point cloud's
// model matrix into its draw (points.slang's PointsDraw: `model` and `previous`, column-major,
// and the largest scale as `info.w`).
//
// Bindings: 0 per record its fields (the frame's rows, then below's rows, then the anchor's slot
// in the pools or ~0 and the index written: nine float4), 1 to 3 the instance pools, 4 what is
// written.

static const uint kNone = 0xFFFFFFFFu;
static const uint kViewPoses = 0;
static const uint kPointsDraws = 1;
static const uint kPointsDrawFloats = 44;

[[vk::binding(0, 0)]] StructuredBuffer<float4> records;
[[vk::binding(1, 0)]] StructuredBuffer<float> positions;
[[vk::binding(2, 0)]] StructuredBuffer<float> orientations;
[[vk::binding(3, 0)]] StructuredBuffer<float> scales;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> written;
[[vk::push_constant]] ConstantBuffer<uint4> constants;  // records, what is written

float4x4 rows(uint first) {
    return float4x4(records[first], records[first + 1], records[first + 2], records[first + 3]);
}

// The matrix of translate, rotate (x, y, z, w), scale at instance pool slot `slot`.
float4x4 anchorMatrix(uint slot) {
    let t = float3(positions[3 * slot], positions[3 * slot + 1], positions[3 * slot + 2]);
    let q = normalize(float4(orientations[4 * slot], orientations[4 * slot + 1],
                             orientations[4 * slot + 2], orientations[4 * slot + 3]));
    let s = float3(scales[3 * slot], scales[3 * slot + 1], scales[3 * slot + 2]);
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;
    return float4x4(
        (1 - 2 * (y * y + z * z)) * s.x, (2 * (x * y - w * z)) * s.y, (2 * (x * z + w * y)) * s.z, t.x,
        (2 * (x * y + w * z)) * s.x, (1 - 2 * (x * x + z * z)) * s.y, (2 * (y * z - w * x)) * s.z, t.y,
        (2 * (x * z - w * y)) * s.x, (2 * (y * z + w * x)) * s.y, (1 - 2 * (x * x + y * y)) * s.z, t.z,
        0, 0, 0, 1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void anchorsMain(uint3 id : SV_DispatchThreadID) {
    let record = id.x;
    if (record >= constants.x) {
        return;
    }
    let first = 9 * record;
    let slot = asuint(records[first + 8].x);
    let target = asuint(records[first + 8].y);
    float4x4 pose = rows(first);
    if (slot != kNone) {
        pose = mul(pose, anchorMatrix(slot));
    }
    pose = mul(pose, rows(first + 4));
    if (constants.y == kViewPoses) {
        for (uint r = 0; r < 4; ++r) {
            for (uint c = 0; c < 4; ++c) {
                written[16 * target + 4 * r + c] = pose[r][c];
            }
        }
        return;
    }
    // A points draw: the model matrix now and at the last frame (which the GPU does not keep:
    // the cloud stands still for motion vectors), then its largest scale for the spheres' radii.
    let base = kPointsDrawFloats * target;
    for (uint c = 0; c < 4; ++c) {
        for (uint r = 0; r < 4; ++r) {
            written[base + 4 * c + r] = pose[r][c];
            written[base + 16 + 4 * c + r] = pose[r][c];
        }
    }
    let scale = max(length(float3(pose[0][0], pose[1][0], pose[2][0])),
                    max(length(float3(pose[0][1], pose[1][1], pose[2][1])),
                        length(float3(pose[0][2], pose[1][2], pose[2][2]))));
    written[base + 39] = scale;
}
