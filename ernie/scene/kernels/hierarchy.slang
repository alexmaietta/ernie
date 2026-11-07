// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

#include "ernie/scene/kernels/doubles.h"
#include "ernie/scene/kernels/hierarchy.h"

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

bool Changed(uint64_t stamp, uint64_t since, uint64_t until) {
    return stamp > since && stamp <= until;
}

uint BindingRow(SceneData* scene, uint prim) {
    return ApiRow(scene, prim, kApiMaterialBindingAPI);
}

uint BoundMaterial(SceneData* scene, uint prim, out bool binds) {
    const ArrayRef binding = scene->schemas.materialBindingAPI.materialBinding[BindingRow(scene, prim)];
    binds = binding.count > 0;
    if (!binds) {
        return kNoPrim;
    }
    const uint target = Ptr<uint>(binding.address)[0];
    const bool bound = target < scene->primCount && (scene->prims.flags[target] & kPrimAlive) != 0 &&
                       SchemaIsA(scene->prims.schema[target], SchemaId::Material);
    return bound ? target : kNoPrim;
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
    SceneData* scene = p.scene;
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
        inherited = inherited || Changed(image.visibilityStamp[at], p.since, p.until) ||
                    Changed(image.purposeStamp[at], p.since, p.until) ||
                    Changed(scene->schemas.materialBindingAPI.materialBindingStamp[binding],
                            p.since, p.until);
        const uint x = p.tree.xform[at];
        if (!reset && x != kNoRow) {
            transform = transform || Changed(xform.translateStamp[x], p.since, p.until) ||
                        Changed(xform.orientStamp[x], p.since, p.until) ||
                        Changed(xform.scaleStamp[x], p.since, p.until) ||
                        Changed(xform.resetXformStackStamp[x], p.since, p.until);
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
    resolved.padding = 0;
    p.imageables[row] = resolved;
}

[shader("compute")]
[numthreads(kHierarchyGroup, 1, 1)]
void GatherRows(uniform GatherParams p) {
    const uint index = LinearThreadIndex();
    if (index >= p.count) {
        return;
    }
    const uint row = p.rows[index];
    for (uint word = 0; word < p.words; ++word) {
        p.out[index * p.words + word] = row == kNoRow ? 0 : p.source[row * p.words + word];
    }
}
