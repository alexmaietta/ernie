// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.overlay.kernels.common;

using namespace ernie;

[shader("compute")]
[numthreads(kOverlayGroup, 1, 1)]
void ClearOverlayBoxes(uniform OverlayUniteParams u) {
    const uint word = LinearThreadIndex();
    if (word >= u.boxCount * kOverlayBoxWords) {
        return;
    }
    const uint at = word % kOverlayBoxWords;
    u.boxes[word] = at < 3 || at == 6 ? ~0u : 0u;
}

[shader("compute")]
[numthreads(kOverlayGroup, 1, 1)]
void UniteOverlayBoxes(uniform OverlayUniteParams u) {
    const uint entry = LinearThreadIndex();
    if (entry >= u.primCount) {
        return;
    }
    RenderTables* tables = u.tables;
    const uint prim = u.prims[entry * u.primStride];
    const uint group = u.groups[entry * u.groupStride];
    if (group >= u.boxCount || !PrimBounded(tables, prim)) {
        return;
    }
    const float3 low = tables->primBounds[prim].lower.Vector();
    const float3 high = tables->primBounds[prim].upper.Vector();
    if (low.x == high.x && low.y == high.y && low.z == high.z) {
        return;
    }
    const uint at = group * kOverlayBoxWords;
    for (uint axis = 0; axis < 3; ++axis) {
        InterlockedMin(u.boxes[at + axis], OrderedFloat(low[axis]));
        InterlockedMax(u.boxes[at + 3 + axis], OrderedFloat(high[axis]));
    }
    InterlockedMin(u.boxes[at + 6], tables->prims.world[prim]);
}

[shader("compute")]
[numthreads(kOverlayTile, kOverlayTile, 1)]
void OutlineOverlays(uniform OverlayParams p) {
    const uint tile = LinearGroupIndex();
    if (tile >= p.tileCount) {
        return;
    }
    const OverlayView view = p.views[FindView(p, tile)];
    if (view.outlined == 0) {
        return;
    }
    const uint thread = LocalThreadIndex();
    const int2 pixel =
        int2(view.TileCorner(tile) + uint2(thread % kOverlayTile, thread / kOverlayTile));
    const int2 size = int2(view.width, view.height);
    if (any(pixel >= size)) {
        return;
    }
    const uint index = view.PixelIndex(uint2(pixel));
    const uint centreIndex = MaskIndex(view, uint2(pixel));
    const uint centreMask = view.mask[centreIndex];
    float4 under = float4(0);
    for (uint at = 0; at < min(p.outlineCount, 32u); ++at) {
        const OverlayOutline outline = p.outlines[at];
        if (((view.overlays >> outline.overlay) & 1) == 0) {
            continue;
        }
        const uint bit = 1u << at;
        const bool centre = (centreMask & bit) != 0;
        const int reach = int(ceil(outline.width));
        float nearest = 1e9f;
        bool hidden = false;
        for (int dy = -reach; dy <= reach; ++dy) {
            for (int dx = -reach; dx <= reach; ++dx) {
                const float distance = length(float2(dx, dy));
                if ((dx == 0 && dy == 0) || distance >= nearest ||
                    distance > outline.width + 0.5f) {
                    continue;
                }
                const uint2 other = uint2(clamp(pixel + int2(dx, dy), int2(0), size - 1));
                const uint otherIndex = MaskIndex(view, other);
                if (((view.mask[otherIndex] & bit) != 0) == centre) {
                    continue;
                }
                const uint2 inside = centre ? uint2(pixel) : other;
                const float sceneDepth = view.depth[view.PixelIndex(inside)];
                nearest = distance;
                hidden = sceneDepth != view.missDistance &&
                         view.maskDepth[centre ? centreIndex : otherIndex] >
                             sceneDepth * (1 + kDepthSlack);
            }
        }
        const float coverage = saturate(outline.width + 0.5f - nearest);
        if (coverage > 0) {
            const float4 color = DecodeOverlayColor(outline.color);
            AddUnder(under, color.xyz, color.w * coverage * (hidden ? kXRayOpacity : 1.0f));
        }
    }
    if (!(under.w > 0)) {
        return;
    }
    const float4 over = DecodeOverlayColor(view.overlay[index]);
    const float alpha = over.w + under.w * (1 - over.w);
    const float3 premultiplied = over.xyz * over.w + under.xyz * (1 - over.w);
    view.overlay[index] = EncodeOverlayColor(float4(premultiplied / alpha, alpha));
}
