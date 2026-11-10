// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

#include "ernie/stage/columns/kernels/doubles.h"
#include "ernie/scene/tables/kernels/transforms.h"

using namespace ernie;

float2 TwoSum(float a, float b, uint zero) {
    const float s = asfloat(asuint(a + b) ^ zero);
    const float v = s - a;
    return float2(s, (a - (s - v)) + (b - v));
}

struct Local {
    float3 x;
    float3 y;
    float3 z;
    float3 high;
    float3 low;
    bool reset;

    float3 Apply(float3 v) {
        return x * v.x + y * v.y + z * v.z;
    }
}

Local LocalTransform(XformableData xform, uint row) {
    const Double3 t = xform.translate[row];
    const float2 tx = SplitDouble(t.x[0], t.x[1]);
    const float2 ty = SplitDouble(t.y[0], t.y[1]);
    const float2 tz = SplitDouble(t.z[0], t.z[1]);

    const Quatd q = xform.orient[row];
    float4 r = float4(Narrow(q.x[0], q.x[1]), Narrow(q.y[0], q.y[1]), Narrow(q.z[0], q.z[1]),
                      Narrow(q.w[0], q.w[1]));
    const float norm = length(r);
    r = norm > 0.0f ? r / norm : float4(0.0f, 0.0f, 0.0f, 1.0f);
    const float3 s = xform.scale[row].Vector();

    Local local;
    local.x = float3(1.0f - 2.0f * (r.y * r.y + r.z * r.z), 2.0f * (r.x * r.y + r.z * r.w),
                     2.0f * (r.x * r.z - r.y * r.w)) * s.x;
    local.y = float3(2.0f * (r.x * r.y - r.z * r.w), 1.0f - 2.0f * (r.x * r.x + r.z * r.z),
                     2.0f * (r.y * r.z + r.x * r.w)) * s.y;
    local.z = float3(2.0f * (r.x * r.z + r.y * r.w), 2.0f * (r.y * r.z - r.x * r.w),
                     1.0f - 2.0f * (r.x * r.x + r.y * r.y)) * s.z;
    local.high = float3(tx.x, ty.x, tz.x);
    local.low = float3(tx.y, ty.y, tz.y);
    local.reset = xform.resetXformStack[row] != 0;
    return local;
}

Local Apply(Local up, Local below, uint zero) {
    Local world;
    world.x = up.Apply(below.x);
    world.y = up.Apply(below.y);
    world.z = up.Apply(below.z);
    const float3 a = up.Apply(below.high);
    const float3 b = up.Apply(below.low);
    [ForceUnroll]
    for (uint i = 0; i < 3; ++i) {
        const float2 sum = TwoSum(up.high[i], a[i], zero);
        const float2 total = TwoSum(sum.x, sum.y + up.low[i] + b[i], zero);
        world.high[i] = total.x;
        world.low[i] = total.y;
    }
    world.reset = up.reset;
    return world;
}

WorldTransform ComposeWorld(HierarchyTree tree, XformableData xform, uint row, uint zero) {
    Local world = LocalTransform(xform, tree.xform[row]);
    uint at = tree.parent[row];
    for (uint depth = 0; !world.reset && at != kNoRow && depth < kMaxHierarchyDepth; ++depth) {
        const uint up = tree.xform[at];
        if (up != kNoRow) {
            world = Apply(LocalTransform(xform, up), world, zero);
        }
        at = tree.parent[at];
    }
    WorldTransform result;
    result.x = Float3(world.x);
    result.y = Float3(world.y);
    result.z = Float3(world.z);
    result.high = Float3(world.high);
    result.low = Float3(world.low);
    return result;
}

Local IdentityLocal() {
    Local local;
    local.x = float3(1, 0, 0);
    local.y = float3(0, 1, 0);
    local.z = float3(0, 0, 1);
    local.high = float3(0);
    local.low = float3(0);
    local.reset = false;
    return local;
}

Local StoredLocal(WorldTransform world) {
    Local local;
    local.x = world.x.Vector();
    local.y = world.y.Vector();
    local.z = world.z.Vector();
    local.high = world.high.Vector();
    local.low = world.low.Vector();
    local.reset = false;
    return local;
}

WorldTransform StoreLocal(Local local) {
    WorldTransform world;
    world.x = Float3(local.x);
    world.y = Float3(local.y);
    world.z = Float3(local.z);
    world.high = Float3(local.high);
    world.low = Float3(local.low);
    return world;
}

float4 QuatFromAxes(float3 x, float3 y, float3 z) {
    const float trace = x.x + y.y + z.z;
    float4 q;
    if (trace > 0) {
        const float s = sqrt(trace + 1) * 2;
        q = float4((y.z - z.y) / s, (z.x - x.z) / s, (x.y - y.x) / s, s / 4);
    } else if (x.x > y.y && x.x > z.z) {
        const float s = sqrt(1 + x.x - y.y - z.z) * 2;
        q = float4(s / 4, (y.x + x.y) / s, (z.x + x.z) / s, (y.z - z.y) / s);
    } else if (y.y > z.z) {
        const float s = sqrt(1 + y.y - x.x - z.z) * 2;
        q = float4((y.x + x.y) / s, s / 4, (z.y + y.z) / s, (z.x - x.z) / s);
    } else {
        const float s = sqrt(1 + z.z - x.x - y.y) * 2;
        q = float4((z.x + x.z) / s, (z.y + y.z) / s, s / 4, (x.y - y.x) / s);
    }
    return normalize(q);
}

float3 QuatRotate(float4 q, float3 v) {
    const float3 t = 2 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

float4 QuatMultiply(float4 a, float4 b) {
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

float4 QuatConjugate(float4 q) {
    return float4(-q.xyz, q.w);
}

float4 Nlerp(float4 a, float4 b, float t) {
    const float4 toward = dot(a, b) < 0 ? -b : b;
    return normalize(lerp(a, toward, t));
}

struct Frame {
    float3 high;
    float3 low;
    float4 orient;
    float3 scale;
}

Frame FrameFrom(Local local) {
    Frame frame;
    frame.high = local.high;
    frame.low = local.low;
    frame.scale = float3(length(local.x), length(local.y), length(local.z));
    const float3 x = frame.scale.x > 0 ? local.x / frame.scale.x : float3(1, 0, 0);
    const float3 y = frame.scale.y > 0 ? local.y / frame.scale.y : float3(0, 1, 0);
    const float3 z = frame.scale.z > 0 ? local.z / frame.scale.z : float3(0, 0, 1);
    frame.orient = QuatFromAxes(x, y, z);
    return frame;
}

Local LocalFrom(Frame frame) {
    Local local;
    local.x = QuatRotate(frame.orient, float3(1, 0, 0)) * frame.scale.x;
    local.y = QuatRotate(frame.orient, float3(0, 1, 0)) * frame.scale.y;
    local.z = QuatRotate(frame.orient, float3(0, 0, 1)) * frame.scale.z;
    local.high = frame.high;
    local.low = frame.low;
    local.reset = false;
    return local;
}

float3 Separation(float3 high, float3 low, float3 fromHigh, float3 fromLow) {
    return (high - fromHigh) + (low - fromLow);
}

void Shift(inout float3 high, inout float3 low, float3 delta, uint zero) {
    [ForceUnroll]
    for (uint i = 0; i < 3; ++i) {
        const float2 sum = TwoSum(high[i], delta[i] + low[i], zero);
        high[i] = sum.x;
        low[i] = sum.y;
    }
}

float4 LookRotation(float3 toward, float3 up, float4 fallback) {
    const float distance = length(toward);
    if (distance < 1e-6f) {
        return fallback;
    }
    const float3 z = -toward / distance;
    float3 x = cross(up, z);
    if (length(x) < 1e-6f) {
        x = cross(abs(z.y) < 0.999f ? float3(0, 1, 0) : float3(1, 0, 0), z);
    }
    x = normalize(x);
    return QuatFromAxes(x, cross(z, x), z);
}

uint FirstXformable(AuthoredData* scene, ArrayRef targets) {
    if (targets.count == 0) {
        return kNoPrim;
    }
    const uint prim = Ptr<uint>(targets.address)[0];
    if (prim >= scene->primCount || (scene->prims.flags[prim] & kPrimAlive) == 0) {
        return kNoPrim;
    }
    return SchemaIsA(scene->prims.schema[prim], SchemaId::Xformable) ? prim : kNoPrim;
}

float Share(float lag, float dt) {
    if (lag <= 0) {
        return 1;
    }
    return dt > 0 ? 1 - exp(-dt / lag) : 0;
}

Frame FollowFrame(FollowParams p, uint api, uint target, Frame parent) {
    AuthoredData* scene = p.scene;
    FollowAPIData follow = scene->schemas.followAPI;
    const Frame source =
        FrameFrom(StoredLocal(p.transforms[PrimRow(scene, target, kSlotXformable)]));
    const FollowRotate rotate = FollowRotate(follow.followRotate[api]);
    float4 placing = source.orient;
    if (rotate == FollowRotate::Yaw) {
        const float3 x = QuatRotate(source.orient, float3(1, 0, 0));
        const float half = atan2(x.y, x.x) / 2;
        placing = float4(0, 0, sin(half), cos(half));
    } else if (rotate == FollowRotate::None) {
        placing = parent.orient;
    }
    const float3 scale = follow.followInheritScale[api] != 0 ? source.scale : parent.scale;
    Frame frame;
    const bool moves = FollowTranslate(follow.followTranslate[api]) == FollowTranslate::Inherit;
    frame.high = moves ? source.high : parent.high;
    frame.low = moves ? source.low : parent.low;
    Shift(frame.high, frame.low,
          QuatRotate(placing, follow.followOffsetTranslate[api].Vector() * scale), p.zero);
    float4 orient = placing;
    if (rotate == FollowRotate::LookAt) {
        orient = source.orient;
        const uint look = FirstXformable(scene, follow.followLookAt[api]);
        if (look != kNoPrim) {
            const Local seen = StoredLocal(p.transforms[PrimRow(scene, look, kSlotXformable)]);
            const float3 toward = Separation(seen.high, seen.low, frame.high, frame.low) +
                                  seen.Apply(follow.followLookAtOffsetTranslate[api].Vector());
            float3 up = float3(0, 0, 1);
            if (FollowUp(follow.followUp[api]) == FollowUp::Target) {
                up = QuatRotate(source.orient, float3(0, 0, 1));
            }
            orient = LookRotation(toward, up, orient);
        }
    }
    const Quatd offset = follow.followOffsetOrient[api];
    float4 turn = float4(Narrow(offset.x[0], offset.x[1]), Narrow(offset.y[0], offset.y[1]),
                         Narrow(offset.z[0], offset.z[1]), Narrow(offset.w[0], offset.w[1]));
    turn = length(turn) > 0 ? normalize(turn) : float4(0, 0, 0, 1);
    frame.orient = normalize(QuatMultiply(orient, turn));
    frame.scale = scale * follow.followOffsetScale[api].Vector();

    FollowState state = p.states[api];
    if (p.snap != 0 || state.settled == 0 || state.target != target) {
        state.high = Float3(frame.high);
        state.low = Float3(frame.low);
        state.orient = Float4(frame.orient);
        state.target = target;
        state.settled = 1;
    } else {
        const float3 lags = follow.followPositionLag[api].Vector();
        const float3 behind = QuatRotate(
            QuatConjugate(frame.orient),
            Separation(frame.high, frame.low, state.high.Vector(), state.low.Vector()));
        const float3 keep = float3(1 - Share(lags.x, p.dt), 1 - Share(lags.y, p.dt),
                                   1 - Share(lags.z, p.dt));
        float3 trailing = QuatRotate(frame.orient, behind * keep);
        const float most = follow.followLagMaxDistance[api];
        const float trail = length(trailing);
        if (most > 0 && trail > most) {
            trailing *= most / trail;
        }
        float3 high = frame.high;
        float3 low = frame.low;
        Shift(high, low, -trailing, p.zero);
        state.high = Float3(high);
        state.low = Float3(low);
        state.orient = Float4(Nlerp(state.orient.Vector(), frame.orient,
                                    Share(follow.followRotationLag[api], p.dt)));
    }
    p.states[api] = state;
    frame.high = state.high.Vector();
    frame.low = state.low.Vector();
    frame.orient = state.orient.Vector();

    const float weight = saturate(follow.followWeight[api]);
    if (weight < 1) {
        const float3 along = Separation(frame.high, frame.low, parent.high, parent.low) * weight;
        frame.high = parent.high;
        frame.low = parent.low;
        Shift(frame.high, frame.low, along, p.zero);
        frame.orient = Nlerp(parent.orient, frame.orient, weight);
        frame.scale = lerp(parent.scale, frame.scale, weight);
    }
    return frame;
}

[shader("compute")]
[numthreads(kHierarchyGroup, 1, 1)]
void FollowFrames(uniform FollowParams p) {
    const uint index = LinearThreadIndex();
    if (index >= p.count) {
        return;
    }
    const uint row = p.rows[p.first + index];
    AuthoredData* scene = p.scene;
    XformableData xform = scene->schemas.xformable;
    const uint own = p.tree.xform[row];
    const Local local = LocalTransform(xform, own);
    Local parent = IdentityLocal();
    if (!local.reset) {
        uint at = p.tree.parent[row];
        for (uint depth = 0;
             at != kNoRow && p.tree.xform[at] == kNoRow && depth < kMaxHierarchyDepth; ++depth) {
            at = p.tree.parent[at];
        }
        if (at != kNoRow && p.tree.xform[at] != kNoRow) {
            parent = StoredLocal(p.transforms[p.tree.xform[at]]);
        }
    }
    const uint prim = scene->schemas.imageable.prim[row];
    const uint api = ApiRow(scene, prim, kApiFollowAPI);
    const uint target = FirstXformable(scene, scene->schemas.followAPI.followTarget[api]);
    Local frame = parent;
    if (target != kNoPrim) {
        frame = LocalFrom(FollowFrame(p, api, target, FrameFrom(parent)));
    }
    p.transforms[own] = StoreLocal(Apply(frame, local, p.zero));
}

[shader("compute")]
[numthreads(kHierarchyGroup, 1, 1)]
void FollowOwned(uniform FollowParams p) {
    const uint index = LinearThreadIndex();
    if (index >= p.count) {
        return;
    }
    const uint row = p.rows[p.first + index];
    const uint owner = p.owners[p.first + index];
    XformableData xform = p.scene->schemas.xformable;
    Local world = LocalTransform(xform, p.tree.xform[row]);
    uint at = p.tree.parent[row];
    for (uint depth = 0; !world.reset && at != kNoRow && at != owner && depth < kMaxHierarchyDepth;
         ++depth) {
        const uint up = p.tree.xform[at];
        if (up != kNoRow) {
            world = Apply(LocalTransform(xform, up), world, p.zero);
        }
        at = p.tree.parent[at];
    }
    if (!world.reset && at == owner) {
        world = Apply(StoredLocal(p.transforms[p.tree.xform[owner]]), world, p.zero);
    }
    p.transforms[p.tree.xform[row]] = StoreLocal(world);
}

bool Changed(uint64_t stamp, uint64_t since, uint64_t until) {
    return stamp > since && stamp <= until;
}

[shader("compute")]
[numthreads(kHierarchyGroup, 1, 1)]
void ResolveRows(uniform ResolveParams p) {
    const uint3 thread = ThreadIndex();
    const uint segment = p.segments[thread.y];
    if (thread.x >= p.tree.segmentCount[segment]) {
        return;
    }
    const uint row = p.tree.segmentRows[p.tree.segmentStart[segment] + thread.x];
    AuthoredData* scene = p.scene;
    ImageableData image = scene->schemas.imageable;
    XformableData xform = scene->schemas.xformable;
    if (image.prim[row] == kNoPrim) {
        return;
    }

    bool transform = p.all != 0;
    bool inherited = p.all != 0 || p.inherit != 0;
    bool reset = false;
    uint at = row;
    for (uint depth = 0; at != kNoRow && depth < kMaxHierarchyDepth; ++depth) {
        const uint binding = BindingRow(scene, image.prim[at]);
        inherited = inherited || Changed(p.visibilityStamps[at], p.since, p.until) ||
                    Changed(p.purposeStamps[at], p.since, p.until) ||
                    Changed(p.materialBindingStamps[binding], p.since, p.until);
        const uint x = p.tree.xform[at];
        if (!reset && x != kNoRow) {
            transform = transform || Changed(p.translateStamps[x], p.since, p.until) ||
                        Changed(p.orientStamps[x], p.since, p.until) ||
                        Changed(p.scaleStamps[x], p.since, p.until) ||
                        Changed(p.resetXformStackStamps[x], p.since, p.until);
            reset = xform.resetXformStack[x] != 0;
        }
        at = p.tree.parent[at];
    }

    const uint own = p.tree.xform[row];
    if (transform && own != kNoRow) {
        p.transforms[own] = ComposeWorld(p.tree, xform, row, p.zero);
    }
    if (!inherited) {
        return;
    }

    const uint invisible = uint(Visibility::Invisible);
    uint visibility = uint(Visibility::Inherited);
    uint purpose = uint(Purpose::Default);
    uint material = kNoPrim;
    uint motion = kNoPrim;
    bool purposeFound = false;
    bool materialFound = false;
    at = row;
    for (uint depth = 0; at != kNoRow && depth < kMaxHierarchyDepth; ++depth) {
        if (image.visibility[at] == invisible) {
            visibility = invisible;
        }
        if (!purposeFound && image.purpose[at] != uint(Purpose::Default)) {
            purpose = image.purpose[at];
            purposeFound = true;
        }
        if (motion == kNoPrim && HasApi(scene, image.prim[at], kApiMotionAPI)) {
            motion = image.prim[at];
        }
        if (!materialFound) {
            bool binds;
            const uint bound = BoundMaterial(scene, image.prim[at], binds);
            if (binds) {
                material = bound;
                materialFound = true;
            }
        }
        at = p.tree.parent[at];
    }
    ResolvedImageable resolved;
    resolved.visibility = visibility;
    resolved.purpose = purpose;
    resolved.material = material;
    resolved.motion = motion;
    p.imageables[row] = resolved;
}

[shader("compute")]
[numthreads(kHierarchyGroup, 1, 1)]
void WriteMotions(uniform MotionParams p) {
    const uint row = LinearThreadIndex();
    if (row >= p.rows) {
        return;
    }
    MotionAPIData* motion = &p.scene->schemas.motionAPI;
    MotionRecord record;
    record.linear = motion.motionVelocity[row];
    record.angular = Float3(radians(motion.motionAngularVelocity[row].Vector()));
    record.padding0 = 0;
    record.padding1 = 0;
    p.motions[row] = record;
}
