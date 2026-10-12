// SPDX-License-Identifier: BSD-3-Clause
// Lights posed on the GPU (a lamp on a robot's link): from each one's pose, which the anchors pass
// composed under its instance (instance_anchors.slang), its entry of the shading pass's light
// table and its shadow views, fitted as `FitLightShadowViews` fits them on the CPU; and the cull
// views of those shadow views, for the meshes' GPU culling.
//
// `lightsMain`, one thread per light. Bindings: 0 per light two float4 records (light index,
// first shadow view, shadow views, kind: 0 distant, 1 spot, 2 cube, 3 a laser's spot; then the
// spot's field of view in radians and the light's radius, or the laser's widest waist radius,
// shortest Rayleigh range and waist distance), 1 the poses (row-major 4x4 per light, relative to the
// frame's origin), 2 the light table (LightGpu, `constants.x` float4s each), 3 the shadow views'
// matrices (glm columns), 4 the shadow views' Shadows blocks (frame.slang; the first matrix of
// each, `constants.z` float4s apart), 8 the poses at the last frame. The scene's bounds, relative
// to the frame's origin, come in `low` and `high`. With motion blur it runs once per sub-frame,
// each light's pose blended `motion.x` of the way from the last frame's to this one's (each
// axis and the position linearly, close for the small turns of a frame), into the sub-frame's
// copy of the table (`layout.w` lights on) and over the views' matrices.
//
// `cullViewsMain`, one thread per cull view (`constants.layout.x` of them). Bindings: 5 the
// shadow views' matrices, 6 per cull view its matrix and slot, 7 the cull views (cull.slang, 12
// float4 each): its frustum planes and matrix columns are written, the rest left as the CPU
// wrote it.
//
// Matrices follow glm: lookAtRH, perspectiveRH_ZO and orthoRH_ZO, stored by column.
//
// References:
// - G. Gribb and K. Hartmann, "Fast Extraction of Viewing Frustum Planes from the
//   World-View-Projection Matrix", 2001.
//   https://www.gamedevs.org/uploads/fast-extraction-viewing-frustum-planes-from-world-view-projection-matrix.pdf

struct LightConstants {
    // Lights, LightGpu stride and Shadows block stride (float4s), and the first light of the
    // table's copy to write (a sub-frame's).
    uint4 layout;
    float4 low;
    float4 high;
    // x: the share of the way from the last frame's poses to this one's; y: 1 to blend them.
    float4 motion;
}

[[vk::binding(0, 0)]] StructuredBuffer<float4> records;
[[vk::binding(1, 0)]] StructuredBuffer<float> poses;
[[vk::binding(8, 0)]] StructuredBuffer<float> previousPoses;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> lights;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> matrices;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> blocks;
[[vk::push_constant]] ConstantBuffer<LightConstants> constants;

static const uint kDistant = 0;
static const uint kSpot = 1;
static const uint kLaserSpot = 3;

// LightGpu's fields, in float4s.
static const uint kPosition = 0;
static const uint kAxis = 1;
static const uint kExtentX = 4;
static const uint kExtentY = 5;

// Column `column` of light `light`'s pose, blended from the last frame's by `motion` (its axes
// then normalised where they are used).
float3 poseColumn(uint light, uint column) {
    let first = 16 * light;
    let now = float3(poses[first + column], poses[first + 4 + column], poses[first + 8 + column]);
    if (constants.motion.y < 0.5) {
        return now;
    }
    let was = float3(previousPoses[first + column], previousPoses[first + 4 + column],
                     previousPoses[first + 8 + column]);
    return lerp(was, now, constants.motion.x);
}

// glm's UpFor: an up vector not along `d`.
float3 upFor(float3 d) {
    return abs(d.z) > 0.99 ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0);
}

float4x4 lookAt(float3 eye, float3 centre, float3 up) {
    let f = normalize(centre - eye);
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    return float4x4(s.x, s.y, s.z, -dot(s, eye),
                    u.x, u.y, u.z, -dot(u, eye),
                    -f.x, -f.y, -f.z, dot(f, eye),
                    0.0, 0.0, 0.0, 1.0);
}

float4x4 perspective(float fov, float near, float far) {
    let t = tan(0.5 * fov);
    return float4x4(1.0 / t, 0.0, 0.0, 0.0,
                    0.0, 1.0 / t, 0.0, 0.0,
                    0.0, 0.0, far / (near - far), -(far * near) / (far - near),
                    0.0, 0.0, -1.0, 0.0);
}

float4x4 ortho(float r, float near, float far) {
    return float4x4(1.0 / r, 0.0, 0.0, 0.0,
                    0.0, 1.0 / r, 0.0, 0.0,
                    0.0, 0.0, -1.0 / (far - near), -near / (far - near),
                    0.0, 0.0, 0.0, 1.0);
}

void writeView(uint view, float4x4 m) {
    for (uint c = 0; c < 4; ++c) {
        let column = float4(m[0][c], m[1][c], m[2][c], m[3][c]);
        matrices[4 * view + c] = column;
        blocks[constants.layout.z * view + c] = column;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void lightsMain(uint3 id : SV_DispatchThreadID) {
    let light = id.x;
    if (light >= constants.layout.x) {
        return;
    }
    let record = asuint(records[2 * light]);
    let shape = records[2 * light + 1];
    let index = record.x;
    let firstView = record.y;
    let views = record.z;
    let kind = record.w;
    let entry = constants.layout.y * (index + constants.layout.w);
    let low = constants.low.xyz;
    let high = constants.high.xyz;
    let centre = 0.5 * (low + high);

    if (kind == kDistant) {
        // +z points towards a distant light.
        let towards = normalize(poseColumn(light, 2));
        lights[entry + kPosition] = float4(towards, lights[entry + kPosition].w);
        lights[entry + kAxis] = float4(towards, lights[entry + kAxis].w);
        if (views > 0) {
            let r = max(0.5 * length(high - low), 1e-3);
            writeView(firstView,
                      mul(ortho(r, r, 3.0 * r), lookAt(centre + 2.0 * r * towards, centre, upFor(towards))));
        }
        return;
    }

    // Others emit along -z; a rect's or disk's width runs along +x, kept square to the axis, its
    // height along the axis cross that. Their half sizes stay as the CPU estimated them.
    let position = poseColumn(light, 3);
    let axis = normalize(-poseColumn(light, 2));
    let x = poseColumn(light, 0);
    let tangent = normalize(x - dot(x, axis) * axis);
    let halfX = length(lights[entry + kExtentX].xyz);
    let halfY = length(lights[entry + kExtentY].xyz);
    lights[entry + kPosition] = float4(position, lights[entry + kPosition].w);
    lights[entry + kAxis] = float4(axis, lights[entry + kAxis].w);
    lights[entry + kExtentX] = float4(tangent * halfX, lights[entry + kExtentX].w);
    lights[entry + kExtentY] = float4(cross(tangent, axis) * halfY, lights[entry + kExtentY].w);
    if (views == 0) {
        return;
    }

    // From the light (its radius) to the scene's farthest corner.
    float far = 1e-3;
    for (uint corner = 0; corner < 8; ++corner) {
        let at = float3((corner & 1) != 0 ? high.x : low.x, (corner & 2) != 0 ? high.y : low.y,
                        (corner & 4) != 0 ? high.z : low.z);
        far = max(far, 1.01 * length(at - position));
    }
    let near = max(shape.y, 1e-4 * far);
    if (kind == kSpot) {
        writeView(firstView, mul(perspective(shape.x, near, far),
                                 lookAt(position, position + axis, upFor(axis))));
        return;
    }
    if (kind == kLaserSpot) {
        // Orthographic along the beam, as wide as the beam's widest (at the aperture or the
        // scene's far side) out to 2.5 1/e^2 radii.
        var widest = 0.0;
        for (uint end = 0; end < 2; ++end) {
            let z = (end == 0 ? 0.0 : far) - shape.z;
            let t = z / max(shape.y, 1e-12);
            widest = max(widest, shape.x * sqrt(1.0 + t * t));
        }
        writeView(firstView, mul(ortho(max(2.5 * widest, 1e-6), 0.0, far),
                                 lookAt(position, position + axis, upFor(axis))));
        return;
    }
    // A cube's faces: +x, -x, +y, -y, +z, -z.
    let faces = perspective(1.5707963267948966, near, far);
    for (uint face = 0; face < 6; ++face) {
        float3 d = float3(0.0);
        d[face / 2] = (face % 2) == 0 ? 1.0 : -1.0;
        writeView(firstView + face, mul(faces, lookAt(position, position + d, upFor(d))));
    }
}

[[vk::binding(5, 0)]] StructuredBuffer<float4> fitted;
[[vk::binding(6, 0)]] StructuredBuffer<uint2> cullSlots;
[[vk::binding(7, 0)]] RWStructuredBuffer<float4> cullViews;

static const uint kCullViewStride = 12;

[shader("compute")]
[numthreads(64, 1, 1)]
void cullViewsMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= constants.layout.x) {
        return;
    }
    let slot = cullSlots[id.x];
    let first = kCullViewStride * slot.y;
    float4 c[4];
    for (uint k = 0; k < 4; ++k) {
        c[k] = fitted[4 * slot.x + k];
        cullViews[first + 6 + k] = c[k];
    }
    // The matrix's rows, from its columns.
    float4 m[4];
    for (uint r = 0; r < 4; ++r) {
        m[r] = float4(c[0][r], c[1][r], c[2][r], c[3][r]);
    }
    float4 planes[6] = {m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1], m[2], m[3] - m[2]};
    for (uint p = 0; p < 6; ++p) {
        cullViews[first + p] = planes[p] / max(length(planes[p].xyz), 1e-30);
    }
}
