// SPDX-License-Identifier: BSD-3-Clause
// The nearest and farthest view depth of any surface in one layer of the depth, for the sun's
// cascades: a later frame fits how far they reach to the depths the view sees rather than to its
// whole clipping range (`RasterRenderer::FitCascadeDistance`). Each wave reduces its pixels and one
// lane folds them into the result with atomics; positive floats order as their bits do, so the
// result stays uint.
//
// A camera rendered on cube faces reduces its faces' layers, one per dispatch z, its depths
// along each face's own axis as shading compares them.
//
// Bindings: 0 the views' depth ([0, 1], reverse-Z), 1 the result: [0] the nearest depth's bits,
// cleared to the largest float, and [1] the farthest's, cleared to 0. Pixels without a surface
// (the cleared depth 0) count for neither.
//
// Reference: A. Lauritzen, M. Salvi and A. Lefohn, "Sample Distribution Shadow Maps", Proc. ACM
// I3D 2011, 97-102. https://doi.org/10.1145/1944745.1944761

struct DepthRangeConstants {
    // xy: image size, z: the first layer, w: 1 when the projection is orthographic (depth linear).
    uint4 size;
    // x: near plane, y: far plane, which may be infinite.
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
        let d = depth.Load(int4(int2(id.xy), int(constants.size.z + id.z), 0));
        if (d > 0.0) {
            // Reverse-Z (`ReversedPerspective`, `ReversedOrthographic`): near f / (near + d
            // (f - near)) in perspective, near / d with an infinite far plane, and linear from far
            // (an infinite one 10^12 past the near plane) orthographic.
            let near = constants.clip.x;
            let far = constants.clip.y;
            let viewDepth =
                constants.size.w != 0
                    ? lerp(isinf(far) ? near + 1e12 : far, near, d)
                    : (isinf(far) ? near / d : near * far / (near + d * (far - near)));
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
