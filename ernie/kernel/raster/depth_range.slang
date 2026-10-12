// SPDX-License-Identifier: BSD-3-Clause
// The nearest and farthest view depth of any surface in one layer of the depth, for sample
// distribution shadow maps: the sun's cascades are fitted to the depths the view sees rather than
// to its whole clipping range. Each wave reduces its pixels and one lane folds them into the
// result with atomics; positive floats order as their bits do, so the result stays uint.
//
// A camera rendered on cube faces reduces its faces' layers, one per dispatch z, its depths
// along each face's own axis as shading compares them.
//
// Bindings: 0 the views' depth ([0, 1], reverse-Z), 1 the result: [0] the nearest depth's bits,
// cleared to the largest float, and [1] the farthest's, cleared to 0. Pixels without a surface
// (the cleared depth 0) count for neither.
//
// `cascadeFitMain` then fits a distant light's cascades over the next frame's view to the depths
// the last one saw, on the GPU, as `FitCascades` (shadow_cascades.cpp) does on the host: the
// practical splits between them, a sphere about each slice, snapped to whole texels, and an
// orthographic map round it, written where the shadow passes, the meshes' culling and shading
// read them. No frame waits for the host to read the depths.
//
// References:
// - A. Lauritzen, M. Salvi and A. Lefohn, "Sample Distribution Shadow Maps", Proc. ACM I3D 2011,
//   97-102. https://doi.org/10.1145/1944745.1944761
// - F. Zhang, H. Sun, L. Xu and L. K. Lun, "Parallel-Split Shadow Maps for Large-Scale Virtual
//   Environments", Proc. ACM VRCIA 2006, 311-318 (the practical splits).
// - M. Valient, "Stable Rendering of Cascaded Shadow Maps", in ShaderX6, 2008 (spheres, snapping).

// One distant light's cascades to fit (`RasterRenderer::CascadeJob`), matrices stored by column.
struct CascadeJob {
    // World to the light's view (glm::lookAtRH from the light's direction).
    float4x4 lightView;
    // The camera's view to world; for an orthographic camera, its inverse projection.
    float4x4 cameraToWorld;
    float4x4 toView;
    // x, y: tangents of the half fields, z: near plane, w: a cube camera's face extent, else 0.
    float4 camera;
    // x: the shadow distance the casters reach, y: lambda, z: texels across, w: cascades.
    float4 fit;
    // x, y: light depths spanning the scene, padded; z: 1 to clamp casters; w: 1 orthographic.
    float4 scene;
    // x: 1 to fit to the depths seen, y: how far the eye moved since, z: how far it may move per
    // unit of the nearest depth seen.
    float4 seen;
    // x, y: the nearest and farthest view depth of the volumes' bounds, which write no depth for
    // the views to see; y 0 without volumes.
    float4 volumes;
    // x: its first view, y: views with Shadows blocks, z: flags (1 to write the frame's splits, 2
    // to write into `subframes` rather than the frame's views), w: 1 to write minimal shading's
    // maps.
    uint4 out;
}

static const uint kWriteSplits = 1;
static const uint kSubframe = 2;

struct DepthRangeConstants {
    // xy: image size, z: the first layer, w: 1 when the projection is orthographic (depth linear).
    uint4 size;
    // x: near plane, y: far plane, which may be infinite.
    float4 clip;
    // Fitting cascades: the jobs, the depths the last frame saw (`depthRangeMain`'s result),
    // every light view's matrix and their Shadows blocks (the first matrix of each,
    // `layout.y` float4s apart), the frame's Shadows block (frame.slang), and the matrices of
    // lights turning across the shutter at each sub-frame, which its shadow passes copy in.
    CascadeJob* jobs;
    uint* seen;
    float4x4* matrices;
    float4* blocks;
    float4* frame;
    float4x4* subframes;
    // x: jobs, y: Shadows block stride (float4s).
    uint4 layout;
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

// glm::orthoRH_ZO.
float4x4 orthographic(float left, float right, float bottom, float top, float near, float far) {
    return float4x4(float4(2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)),
                    float4(0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)),
                    float4(0.0, 0.0, -1.0 / (far - near), -near / (far - near)),
                    float4(0.0, 0.0, 0.0, 1.0));
}

// A corner of the camera's frustum between view depths `near` and `far` (corner bits: x, y, far).
float3 sliceCorner(CascadeJob job, uint i, float near, float far) {
    let depth = (i & 4) != 0 ? far : near;
    let sx = (i & 1) != 0 ? 1.0 : -1.0;
    let sy = (i & 2) != 0 ? 1.0 : -1.0;
    var inView = float4(sx * job.camera.x * depth, sy * job.camera.y * depth, -depth, 1.0);
    if (job.scene.w != 0.0) {
        let edge = mul(job.toView, float4(sx, sy, 0.0, 1.0));
        inView = float4(edge.x, edge.y, -depth, 1.0);
    }
    return mul(job.cameraToWorld, inView).xyz;
}

// `depth` rounded to quarter stops, `up` or down.
float quarterStop(float depth, bool up) {
    let stops = 4.0 * log2(max(depth, 1e-6));
    return exp2((up ? ceil(stops) : floor(stops)) / 4.0);
}

// Split `k` of `count` between `start` and `end`: logarithmic blended with uniform by `lambda`.
float practicalSplit(float start, float end, float lambda, uint count, uint k) {
    if (k == 0) {
        return start;
    }
    if (k == count) {
        return end;
    }
    let f = float(k) / float(count);
    return lambda * start * pow(end / start, f) + (1.0 - lambda) * (start + (end - start) * f);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cascadeFitMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= constants.layout.x) {
        return;
    }
    let job = constants.jobs[id.x];
    let near = job.camera.z;
    var distance = job.fit.x;
    var splitNear = near;
    // Sample distribution: within the depths seen, with margins, while the eye has moved little.
    var nearest = asfloat(constants.seen[0]);
    var farthest = asfloat(constants.seen[1]);
    if (job.volumes.y > 0.0) {
        nearest = min(nearest, job.volumes.x);
        farthest = max(farthest, job.volumes.y);
    }
    if (job.seen.x != 0.0 && farthest > 0.0 && nearest <= farthest &&
        job.seen.y <= job.seen.z * nearest) {
        distance = max(min(distance, 1.1 * farthest), near * 1.01);
        splitNear = clamp(0.9 * nearest, near, 0.95 * distance);
    }
    let count = uint(job.fit.w);
    let lambda = job.fit.y;
    let texels = job.fit.z;
    // The span in quarter stops (powers of 2^(1/4)), the start down and the distance up, as
    // `FitCascades` takes it: a span that drifts as the view turns or what it sees changes keeps
    // the same cascades, whose snapped texels then hold still, until it crosses a step.
    let start = max(quarterStop(splitNear, false), near);
    let end = max(quarterStop(distance, true), start * 1.001);
    for (uint c = 0; c < count; ++c) {
        let sliceNear = practicalSplit(start, end, lambda, count, c);
        let sliceFar = practicalSplit(start, end, lambda, count, c + 1);
        // A sphere about the slice: steady as the camera turns.
        var center = float3(0.0);
        var radius = 0.0;
        if (job.camera.w > 0.0) {
            center = mul(job.cameraToWorld, float4(0.0, 0.0, 0.0, 1.0)).xyz;
            radius = sliceFar * sqrt(1.0 + 2.0 * job.camera.w * job.camera.w);
        } else {
            for (uint i = 0; i < 8; ++i) {
                center += sliceCorner(job, i, sliceNear, sliceFar) / 8.0;
            }
            for (uint i = 0; i < 8; ++i) {
                radius = max(radius, length(sliceCorner(job, i, sliceNear, sliceFar) - center));
            }
        }
        radius = ceil(radius * 16.0) / 16.0 * texels / (texels - 2.0);
        let texel = 2.0 * radius / texels;
        // Snapped to whole texels, so edges do not shimmer as the camera moves.
        let inLight = mul(job.lightView, float4(center, 1.0)).xyz;
        let x = floor(inLight.x / texel) * texel;
        let y = floor(inLight.y / texel) * texel;
        var zNear = job.scene.x;
        var zFar = job.scene.y;
        if (job.scene.z != 0.0) {
            let padding = 0.01 * radius + 1e-3;
            zNear = max(zNear, -inLight.z - radius - padding);
            zFar = min(zFar, -inLight.z + radius + padding);
        }
        let matrix =
            mul(orthographic(x - radius, x + radius, y - radius, y + radius, zNear, zFar),
                job.lightView);
        let view = job.out.x + c;
        if ((job.out.z & kSubframe) != 0) {
            constants.subframes[view] = matrix;
            continue;
        }
        constants.matrices[view] = matrix;
        if (view < job.out.y) {
            for (uint column = 0; column < 4; ++column) {
                constants.blocks[view * constants.layout.y + column] = float4(
                    matrix[0][column], matrix[1][column], matrix[2][column], matrix[3][column]);
            }
        }
        // frame.slang's Shadows: lightViewProj[4] (16 float4s), splits, texelWorld.
        if (c < 4) {
            if ((job.out.z & kWriteSplits) != 0) {
                constants.frame[16][c] = sliceFar;
            }
            if (job.out.w != 0) {
                for (uint column = 0; column < 4; ++column) {
                    constants.frame[c * 4 + column] = float4(matrix[0][column], matrix[1][column],
                                                            matrix[2][column], matrix[3][column]);
                }
                constants.frame[17][c] = texel;
            }
        }
    }
}
