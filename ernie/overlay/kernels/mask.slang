// SPDX-License-Identifier: BSD-3-Clause

import ernie.overlay.kernels.common;
import ernie.render.kernels.query.hit;

using namespace ernie;

uint Covering(Ptr<OverlayOutline> outlines, uint count, uint64_t overlays, uint prim) {
    uint covering = 0;
    for (uint at = 0; at < min(count, 32u); ++at) {
        const OverlayOutline outline = outlines[at];
        if (((overlays >> outline.overlay) & 1) != 0 && Outlined(outline, prim)) {
            covering |= 1u << at;
        }
    }
    return covering;
}

struct OutlineFilter : IHitFilter {
    static const uint kRayFlags = 0;

    Ptr<OverlayOutline> outlines;
    uint64_t overlays;
    Ptr<SlotHit> hits;
    uint count;
    uint first;

    static bool AcceptsAll() {
        return false;
    }

    bool Opaque() {
        return false;
    }

    bool Accept(Ray ray, Hit candidate) {
        return Covering(outlines, count, overlays, hits[first + candidate.instance].prim) != 0;
    }
}

[shader("compute")]
[numthreads(kOverlayTile, kOverlayTile, 1)]
void MaskOutlines(uniform OverlayParams p) {
    const uint tile = LinearGroupIndex();
    if (tile >= p.tileCount) {
        return;
    }
    const OverlayView view = p.views[FindView(p, tile)];
    if (view.outlined == 0) {
        return;
    }
    const uint thread = LocalThreadIndex();
    const uint2 pixel = view.TileCorner(tile) + uint2(thread % kOverlayTile, thread / kOverlayTile);
    if (pixel.x >= view.width || pixel.y >= view.height) {
        return;
    }
    RenderTables* tables = p.tables;
    const CameraModel camera = CameraModel.Load(tables, view.camera);
    const float2 film = float2((float(pixel.x) + 0.5f) / float(view.width) * 2 - 1,
                               1 - (float(pixel.y) + 0.5f) / float(view.height) * 2);
    const CameraRay generated = camera.GenerateRay(CameraSample(film, float2(0.5f), 0.5f));
    const HitQuery query = HitQuery(tables, tables->prims.world[view.camera]);
    OutlineFilter filter;
    filter.outlines = p.outlines;
    filter.overlays = view.overlays;
    filter.count = p.outlineCount;
    filter.hits = query.layout->hits;
    filter.first = query.layout->firstInstances[query.tlas];
    uint covering = 0;
    float depth = 0;
    if (generated.weight > 0) {
        const Ray ray = generated.ray;
        const Hit hit = Tracer.Closest(query.layout->handles[query.tlas], ray, kGeometryMask,
                                       filter, query.Intersector());
        if (hit.IsHit()) {
            covering = Covering(p.outlines, p.outlineCount, view.overlays,
                                query.layout->hits[filter.first + hit.instance].prim);
            depth = camera.ImagePlaneDistance(ray.origin + ray.direction * hit.t);
        }
    }
    const uint at = MaskIndex(view, pixel);
    view.mask[at] = covering;
    view.maskDepth[at] = depth;
}
