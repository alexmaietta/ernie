// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.overlay.kernels.fills;
__exported import ernie.overlay.kernels.text;

using namespace ernie;

bool Project(OverlayParams p, uint work, out OverlayRecord record) {
    record = {};
    const uint viewIndex = work / p.viewItemCount;
    const uint local = work % p.viewItemCount;
    const OverlayBatch batch = p.batches[FindBatch(p, local)];
    const uint item = (local - batch.firstItem) / batch.pieces;
    const uint piece = (local - batch.firstItem) % batch.pieces;
    const OverlayView view = p.views[viewIndex];
    RenderTables* tables = p.tables;
    if (((view.overlays >> batch.overlay) & 1) == 0) {
        return false;
    }
    const uint viewWorld = tables->prims.world[view.camera];
    uint slice;
    uint row;
    if (!ResolveRow(batch, viewWorld, item, slice, row)) {
        return false;
    }

    const uint colorAt = slice * batch.colorWorldStride + row * batch.colorStride;
    float4 color = DecodeOverlayColor(batch.color);
    if ((batch.flags & kOverlayHasColors) != 0) {
        color = (batch.flags & kOverlayByteColors) != 0
                    ? DecodeOverlayColor(batch.colors[colorAt])
                    : float4(asfloat(batch.colors[colorAt]), asfloat(batch.colors[colorAt + 1]),
                             asfloat(batch.colors[colorAt + 2]), asfloat(batch.colors[colorAt + 3]));
    }
    const float loadedSize = batch.sizes[slice * batch.sizeWorldStride + row * batch.sizeStride];
    const float size = (batch.flags & kOverlayHasSizes) != 0 ? loadedSize : batch.size;
    if (!(size > 0) || !(color.w > 0)) {
        return false;
    }
    const ShapeRow shape = {batch, slice, row};
    const OverlayCamera camera = OverlayCamera(tables, view.camera);
    if (batch.kind >= uint(OverlayKind::Labels) && batch.kind <= uint(OverlayKind::Numbers)) {
        return ProjectText(p, shape, view, camera, viewWorld, item, piece, color, size, record);
    }
    if (batch.kind == uint(OverlayKind::Triangles) || batch.kind == uint(OverlayKind::Images) ||
        (batch.kind == uint(OverlayKind::Prims) &&
         batch.primShape == uint(OverlayPrimShape::CameraImages))) {
        return ProjectFill(p, shape, view, camera, viewWorld, piece, color, record);
    }

    float3 a;
    float3 b;
    if (batch.kind == uint(OverlayKind::Prims)) {
        if (!GeneratePrim(p, shape, piece, view, viewWorld, a, b, color)) {
            return false;
        }
    } else {
        if (!Generate(shape, piece, a, b, color)) {
            return false;
        }
        const Sizing sizing = Sizing(p, batch, view, camera, viewWorld, shape.Point());
        a = sizing.Place(p, batch, viewWorld, a);
        b = sizing.Place(p, batch, viewWorld, b);
    }
    if (any(isnan(a)) || any(isnan(b)) || any(isinf(a)) || any(isinf(b))) {
        return false;
    }
    float depthA = camera.model.ImagePlaneDistance(a);
    float depthB = camera.model.ImagePlaneDistance(b);
    if (depthA < camera.near && depthB < camera.near) {
        return false;
    }
    if (depthA < camera.near) {
        a = lerp(a, b, (camera.near - depthA) / (depthB - depthA));
        depthA = camera.near;
    } else if (depthB < camera.near) {
        b = lerp(b, a, (camera.near - depthB) / (depthA - depthB));
        depthB = camera.near;
    }

    const float2 pixelA = Pixel(camera.model.Project(a).film, view);
    const float2 pixelB = Pixel(camera.model.Project(b).film, view);
    float pixelRadius = 0.5f * size;
    if ((batch.flags & kOverlayMeters) != 0) {
        const float3 middle = 0.5f * (a + b);
        const float3 right = normalize(PrimWorld(tables, view.camera).x.Vector());
        const float2 centre = Pixel(camera.model.Project(middle).film, view);
        const float2 side = Pixel(camera.model.Project(middle + right * (0.5f * size)).film, view);
        pixelRadius = length(side - centre);
    }
    if (!(pixelRadius > 0)) {
        return false;
    }
    const uint radius =
        uint(clamp(pixelRadius * kOverlayRadiusUnits + 0.5f, 1.0f, float(kOverlayMostRadius)));
    const bool marker = batch.kind == uint(OverlayKind::Markers);
    const uint pattern = marker ? (batch.marker & kOverlayShapeMask) << kOverlayShapeShift
                                : (min(batch.dash, kOverlayPatternMask) << kOverlayDashShift) |
                                      (min(batch.gap, kOverlayPatternMask) << kOverlayGapShift);
    record.ax = pixelA.x;
    record.ay = pixelA.y;
    record.bx = pixelB.x;
    record.by = pixelB.y;
    record.depthA = depthA;
    record.depthB = depthB;
    record.color = EncodeOverlayColor(color);
    record.flags = batch.depth | pattern |
                   (uint(marker ? OverlayPiece::Marker : OverlayPiece::Segment) << kOverlayKindShift) |
                   (camera.perspective ? kOverlayPerspective : 0) | (radius << kOverlayRadiusShift);
    return true;
}

uint VisitTiles(OverlayParams p, OverlayRecord record, uint recordIndex, uint offset, bool write) {
    const OverlayView view = p.views[recordIndex / p.viewItemCount];
    const uint kind = (record.flags >> kOverlayKindShift) & kOverlayKindMask;
    const bool flat = kind == uint(OverlayPiece::Glyph) || kind == uint(OverlayPiece::Box) ||
                      kind == uint(OverlayPiece::Triangle);
    const float reach = (flat ? 0 : RecordRadius(record)) + kFeather;
    const float tile = float(kOverlayTile);
    const float lowX = min(record.ax, record.bx) - reach;
    const float highX = max(record.ax, record.bx) + reach;
    const float lowY = min(record.ay, record.by) - reach;
    const float highY = max(record.ay, record.by) + reach;
    if (highX < 0 || highY < 0 || lowX >= float(view.width) || lowY >= float(view.height)) {
        return 0;
    }
    const int rows = int((view.height + kOverlayTile - 1) / kOverlayTile);
    const int columns = int(view.columnCount);
    const int firstRow = max(0, int(floor(lowY / tile)));
    const int lastRow = min(rows - 1, int(floor(highY / tile)));
    const float dy = record.by - record.ay;
    const float dx = record.bx - record.ax;
    uint count = 0;
    for (int tileRow = firstRow; tileRow <= lastRow; ++tileRow) {
        const float top = float(tileRow) * tile - reach;
        const float bottom = float(tileRow + 1) * tile + reach;
        float start = 0;
        float end = 1;
        if (!flat && abs(dy) > 1e-6f) {
            const float t0 = (top - record.ay) / dy;
            const float t1 = (bottom - record.ay) / dy;
            start = max(0.0f, min(t0, t1));
            end = min(1.0f, max(t0, t1));
            if (start > end) {
                continue;
            }
        } else if (!flat && (record.ay < top || record.ay > bottom)) {
            continue;
        }
        const float x0 = flat ? lowX + reach : record.ax + dx * start;
        const float x1 = flat ? highX - reach : record.ax + dx * end;
        const int firstColumn = max(0, int(floor((min(x0, x1) - reach) / tile)));
        const int lastColumn = min(columns - 1, int(floor((max(x0, x1) + reach) / tile)));
        if (firstColumn > lastColumn) {
            continue;
        }
        if (write) {
            for (int column = firstColumn; column <= lastColumn; ++column) {
                const uint at = offset + count + uint(column - firstColumn);
                if (at < p.pairCapacity) {
                    p.keys[at] = view.firstTile + uint(tileRow * columns + column);
                    p.values[at] = recordIndex;
                }
            }
        }
        count += uint(lastColumn - firstColumn + 1);
    }
    return count;
}

[shader("compute")]
[numthreads(kOverlayGroup, 1, 1)]
void ProjectOverlays(uniform OverlayParams p) {
    const uint work = LinearThreadIndex();
    if (work >= p.workCount) {
        return;
    }
    OverlayRecord record;
    const bool shown = Project(p, work, record);
    p.records[work] = record;
    p.tileCounts[work] = shown ? VisitTiles(p, record, work, 0, false) : 0;
}

[shader("compute")]
[numthreads(kOverlayGroup, 1, 1)]
void ScatterOverlays(uniform OverlayParams p) {
    const uint work = LinearThreadIndex();
    if (work >= p.workCount) {
        return;
    }
    const uint count = p.tileCounts[work];
    const uint offset = p.offsets[work];
    if (work + 1 == p.workCount) {
        p.pairCount[0] = offset + count;
    }
    if (count > 0) {
        VisitTiles(p, p.records[work], work, offset, true);
    }
}

[shader("compute")]
[numthreads(kOverlayGroup, 1, 1)]
void FindOverlayTiles(uniform OverlayParams p) {
    const uint pair = LinearThreadIndex();
    const uint pairs = min(p.pairCount[0], p.pairCapacity);
    if (pair >= pairs) {
        return;
    }
    const uint key = p.keys[pair];
    if (pair > 0 && p.keys[pair - 1] == key) {
        return;
    }
    uint matched = pair;
    uint step = 1;
    while (pair + step < pairs && p.keys[pair + step] == key) {
        matched = pair + step;
        step *= 2;
    }
    uint low = matched + 1;
    uint high = min(pair + step, pairs);
    while (low < high) {
        const uint middle = (low + high) / 2;
        if (p.keys[middle] == key) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    uint slot;
    InterlockedAdd(p.occupied[0], 1u, slot);
    const uint at = kOverlayTileList + kOverlayTileWords * slot;
    p.occupied[at] = key;
    p.occupied[at + 1] = FindView(p, key);
    p.occupied[at + 2] = pair;
    p.occupied[at + 3] = low;
}
