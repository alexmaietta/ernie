// SPDX-License-Identifier: BSD-3-Clause
// The nearest and farthest view depth of any surface in one layer of the depth, for sample
// distribution shadow maps: the sun's cascades are fitted to the depths the view sees rather than
// to its whole clipping range. Each wave reduces its pixels and one lane folds them into the
// result with atomics; positive floats order as their bits do, so the result stays uint.
//
// Bindings: 0 the views' depth ([0, 1]), 1 the result: [0] the nearest depth's bits,
// cleared to the largest float, and [1] the farthest's, cleared to 0. Pixels without a surface
// (depth 1) count for neither.
//
// Reference: A. Lauritzen, M. Salvi and A. Lefohn, "Sample Distribution Shadow Maps", Proc. ACM
// I3D 2011, 97-102. https://doi.org/10.1145/1944745.1944761

struct DepthRangeConstants {
    // xy: image size, z: the layer, w: 1 when the projection is orthographic (depth linear).
    uint4 size;
    // x: near plane, y: far plane.
    float4 clip;
}

[[vk::binding(0, 0)]] Texture2DArray<float> depth;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> range;
[[vk::push_constant]] ConstantBuffer<DepthRangeConstants> constants;

[shader("compute")]
[numthreads(16, 16, 1)]
void depthRangeMain(uint3 id : SV_DispatchThreadID) {
    var nearest = 3.402823e38;
    var farthest = 0.0;
    if (all(id.xy < constants.size.xy)) {
        let d = depth.Load(int4(int2(id.xy), int(constants.size.z), 0));
        if (d < 1.0) {
            let near = constants.clip.x;
            let far = constants.clip.y;
            let viewDepth = constants.size.w != 0 ? near + d * (far - near)
                                                  : near * far / (far - d * (far - near));
            nearest = viewDepth;
            farthest = viewDepth;
        }
    }
    nearest = WaveActiveMin(nearest);
    farthest = WaveActiveMax(farthest);
    if (WaveIsFirstLane()) {
        InterlockedMin(range[0], asuint(nearest));
        InterlockedMax(range[1], asuint(farthest));
    }
}
