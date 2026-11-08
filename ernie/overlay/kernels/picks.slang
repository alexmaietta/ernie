// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.overlay.kernels.common;

using namespace ernie;

struct PickMeasure {
    float distance;
    float depth;
    float2 nearest;
    float along;
}

bool MeasurePick(OverlayParams p, OverlayRecord record, float2 point, out PickMeasure measure) {
    measure = {};
    const float2 a = float2(record.ax, record.ay);
    const float2 b = float2(record.bx, record.by);
    const uint kind = (record.flags >> kOverlayKindShift) & kOverlayKindMask;
    if (kind == uint(OverlayPiece::Segment)) {
        const float2 along = b - a;
        const float lengthSquared = dot(along, along);
        const float t = lengthSquared > 0 ? saturate(dot(point - a, along) / lengthSquared) : 0;
        measure.nearest = a + along * t;
        measure.along = t;
        measure.distance = max(length(point - measure.nearest) - RecordRadius(record), 0.0f);
        measure.depth = (record.flags & kOverlayPerspective) != 0
                            ? 1 / lerp(1 / record.depthA, 1 / record.depthB, t)
                            : lerp(record.depthA, record.depthB, t);
        return true;
    }
    if (kind == uint(OverlayPiece::Marker)) {
        measure.nearest = a;
        measure.distance = max(length(point - a) - RecordRadius(record), 0.0f);
        measure.depth = record.depthA;
        return true;
    }
    if (kind == uint(OverlayPiece::Triangle)) {
        const OverlayFill fill = p.fills[asuint(record.depthA)];
        const float2 corners[3] = {float2(fill.x[0], fill.y[0]), float2(fill.x[1], fill.y[1]),
                                   float2(fill.x[2], fill.y[2])};
        const float area = Cross2(corners[1] - corners[0], corners[2] - corners[0]);
        const float side = area > 0 ? 1 : -1;
        float outside = 0;
        for (uint edge = 0; edge < 3; ++edge) {
            const float2 from = corners[edge];
            const float2 along = corners[(edge + 1) % 3] - from;
            const float inside = side * Cross2(along, point - from) / max(length(along), 1e-6f);
            outside = max(outside, -inside);
        }
        measure.distance = outside;
        const float first = Cross2(corners[1] - point, corners[2] - point) / area;
        const float second = Cross2(corners[2] - point, corners[0] - point) / area;
        const float3 shares = saturate(float3(first, second, 1 - first - second));
        const float total = max(shares.x + shares.y + shares.z, 1e-6f);
        measure.nearest = (corners[0] * shares.x + corners[1] * shares.y + corners[2] * shares.z) /
                          total;
        const float3 depths = float3(fill.depth[0], fill.depth[1], fill.depth[2]);
        measure.depth = (record.flags & kOverlayPerspective) != 0
                            ? total / dot(shares, 1 / depths)
                            : dot(shares, depths) / total;
        return true;
    }
    const float2 low = min(a, b);
    const float2 high = max(a, b);
    measure.nearest = clamp(point, low, high);
    measure.distance = length(point - measure.nearest);
    measure.depth = record.depthA;
    return true;
}

uint PickKey(OverlayParams p, uint pick, uint local, out PickMeasure measure) {
    measure = {};
    const OverlayPickQuery query = p.picks[pick];
    const uint work = query.view * p.viewItemCount + local;
    if (p.tileCounts[work] == 0) {
        return ~0u;
    }
    const OverlayBatch batch = p.batches[FindBatch(p, local)];
    if ((batch.flags & kOverlayPickable) == 0) {
        return ~0u;
    }
    const OverlayRecord record = p.records[work];
    const float2 point = float2(query.x, query.y);
    if (!MeasurePick(p, record, point, measure) || !(measure.distance <= query.radius)) {
        return ~0u;
    }
    const OverlayView view = p.views[query.view];
    const uint mode = record.flags & kOverlayDepthMask;
    if (mode == uint(OverlayDepth::Test)) {
        const uint2 pixel = uint2(clamp(measure.nearest, float2(0), float2(view.width - 1, view.height - 1)));
        const float sceneDepth = view.depth[(view.layer * view.height + pixel.y) * view.width + pixel.x];
        if (sceneDepth != view.missDistance && measure.depth > sceneDepth * (1 + kDepthSlack)) {
            return ~0u;
        }
    }
    const uint distance = uint(min(measure.distance * 8, 4095.0f));
    return distance << 20 | (OrderedFloat(measure.depth) >> 12);
}

[shader("compute")]
[numthreads(kOverlayGroup, 1, 1)]
void ClearOverlayPicks(uniform OverlayParams p) {
    const uint pick = LinearThreadIndex();
    if (pick >= p.pickCount) {
        return;
    }
    OverlayPickResult result = {};
    result.key = ~0u;
    result.work = ~0u;
    p.pickResults[pick] = result;
}

[shader("compute")]
[numthreads(kOverlayGroup, 1, 1)]
void MeasureOverlayPicks(uniform OverlayParams p) {
    const uint thread = LinearThreadIndex();
    if (p.viewItemCount == 0 || thread >= p.pickCount * p.viewItemCount) {
        return;
    }
    PickMeasure measure;
    const uint key = PickKey(p, thread / p.viewItemCount, thread % p.viewItemCount, measure);
    if (key != ~0u) {
        InterlockedMin(p.pickResults[thread / p.viewItemCount].key, key);
    }
}

[shader("compute")]
[numthreads(kOverlayGroup, 1, 1)]
void ChooseOverlayPicks(uniform OverlayParams p) {
    const uint thread = LinearThreadIndex();
    if (p.viewItemCount == 0 || thread >= p.pickCount * p.viewItemCount) {
        return;
    }
    const uint pick = thread / p.viewItemCount;
    const uint local = thread % p.viewItemCount;
    PickMeasure measure;
    const uint key = PickKey(p, pick, local, measure);
    if (key != ~0u && key == p.pickResults[pick].key) {
        InterlockedMin(p.pickResults[pick].work, p.picks[pick].view * p.viewItemCount + local);
    }
}

void PickRay(OverlayParams p, OverlayView view, float2 pixel, out float3 origin,
             out float3 direction) {
    RenderTables* tables = p.tables;
    const CameraModel camera = CameraModel.Load(p.tables, view.camera);
    const float2 film = float2(pixel.x / float(view.width) * 2 - 1, 1 - pixel.y / float(view.height) * 2);
    const CameraRay ray = camera.GenerateRay(CameraSample(film, float2(0.5f), 0.5f));
    origin = ray.ray.origin;
    direction = normalize(ray.ray.direction);
}

[shader("compute")]
[numthreads(kOverlayGroup, 1, 1)]
void ResolveOverlayPicks(uniform OverlayParams p) {
    const uint pick = LinearThreadIndex();
    if (pick >= p.pickCount) {
        return;
    }
    OverlayPickResult result = p.pickResults[pick];
    const OverlayPickQuery query = p.picks[pick];
    const OverlayView view = p.views[query.view];
    RenderTables* tables = p.tables;
    result.world = tables->prims.world[view.camera];
    float3 origin;
    float3 direction;
    PickRay(p, view, float2(query.x, query.y), origin, direction);
    const float3 eye = origin - RenderShift(p.tables, result.world, result.world);
    result.origin = {eye.x, eye.y, eye.z};
    result.direction = {direction.x, direction.y, direction.z};
    if (result.work == ~0u || p.viewItemCount == 0) {
        p.pickResults[pick] = result;
        return;
    }
    const uint local = result.work % p.viewItemCount;
    const uint batchIndex = FindBatch(p, local);
    const OverlayBatch batch = p.batches[batchIndex];
    const uint item = (local - batch.firstItem) / batch.pieces;
    uint slice;
    uint row;
    ResolveRow(batch, result.world, item, slice, row);
    PickMeasure measure;
    PickKey(p, pick, local, measure);
    float3 through;
    float3 along;
    PickRay(p, view, measure.nearest, through, along);
    const CameraModel camera = CameraModel.Load(p.tables, view.camera);
    const float start = camera.ImagePlaneDistance(through);
    const float rate = camera.ImagePlaneDistance(through + along) - start;
    const float3 position = through + along * (rate != 0 ? (measure.depth - start) / rate : 0);
    result.hit = 1;
    result.batch = batchIndex;
    result.row = row;
    result.piece = (local - batch.firstItem) % batch.pieces;
    result.id = (batch.flags & kOverlayHasIds) != 0 ? batch.ids[row * batch.idStride] : batch.id;
    result.overlay = batch.overlay;
    result.prim = batch.kind == uint(OverlayKind::Prims) && (batch.flags & kOverlayUnited) == 0
                      ? batch.prims[row * batch.pointStride]
                      : kNoPrim;
    result.distance = measure.distance;
    result.depth = measure.depth;
    const uint world = tables->prims.world[view.camera];
    const float3 found = position - RenderShift(p.tables, world, world);
    result.position = {found.x, found.y, found.z};
    p.pickResults[pick] = result;
}
