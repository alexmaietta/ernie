// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.overlay.kernels.guides;

using namespace ernie;

bool ProjectFill(OverlayParams p, ShapeRow shape, OverlayView view, OverlayCamera camera,
                 uint viewWorld, uint piece, float4 color, out OverlayRecord record) {
    record = {};
    RenderTables* tables = p.tables;
    const OverlayBatch batch = shape.batch;
    float3 corners[3];
    uint edges = 7;
    uint uvs = 0;
    OverlayFill fill = {};
    const Sizing sizing = Sizing(p, batch, view, camera, viewWorld, shape.Point());
    if (batch.kind == uint(OverlayKind::Triangles)) {
        corners[0] = sizing.Place(p, batch, viewWorld, shape.Point());
        corners[1] = sizing.Place(p, batch, viewWorld, shape.End());
        corners[2] = sizing.Place(p, batch, viewWorld, shape.Third());
        edges = shape.Extra(7) & kOverlayEdgeMask;
    } else {
        float3 origin;
        float3 right;
        float3 down;
        if (batch.kind == uint(OverlayKind::Images)) {
            const float3 local = shape.Point();
            origin = sizing.Place(p, batch, viewWorld, local);
            right = sizing.Place(p, batch, viewWorld, local + shape.End());
            down = sizing.Place(p, batch, viewWorld, local + shape.Third());
        } else {
            const uint prim = shape.Prim();
            if (!PrimAlive(tables, prim) || prim == view.camera ||
                !SchemaIsA(tables->prims.schema[prim], SchemaId::Camera)) {
                return false;
            }
            const uint world = tables->prims.world[prim];
            if (!WorldShown(tables->instances, viewWorld, world)) {
                return false;
            }
            const Lens lens = Lens(tables, prim);
            const float distance = batch.length > 0 ? batch.length : 1;
            const WorldTransform w = PrimWorld(tables, prim);
            const float3 offset = ViewOrigin(tables, w, world, viewWorld);
            origin = w.Apply(lens.Corner(float2(-1, 1), distance)) + offset;
            right = w.Apply(lens.Corner(float2(1, 1), distance)) + offset;
            down = w.Apply(lens.Corner(float2(-1, -1), distance)) + offset;
            fill.opaque = 1;
            if (batch.image == 0) {
                bool found = false;
                for (uint at = 0; at < p.cameraImageCount && !found; ++at) {
                    const OverlayCameraImage image = p.cameraImages[at];
                    if (image.camera == prim) {
                        fill.pixels = image.pixels;
                        fill.width = image.width;
                        fill.height = image.height;
                        fill.rowStride = image.width;
                        found = true;
                    }
                }
                if (!found) {
                    return false;
                }
            }
        }
        if (fill.pixels == 0) {
            const uint layer = (batch.flags & kOverlayImageLayers) != 0 ? shape.row : 0;
            fill.pixels = batch.image + uint64_t(layer) * batch.imageLayerStride * 4;
            fill.width = batch.imageWidth;
            fill.height = batch.imageHeight;
            fill.rowStride = batch.imageRowStride;
        }
        const float3 far = right + down - origin;
        corners[0] = origin;
        corners[1] = piece == 0 ? right : far;
        corners[2] = piece == 0 ? far : down;
        edges = piece == 0 ? 3u : 6u;
        uvs = piece == 0 ? 0x34u : 0x2Cu;
    }

    float2 pixels[3];
    for (uint corner = 0; corner < 3; ++corner) {
        if (any(isnan(corners[corner])) || any(isinf(corners[corner]))) {
            return false;
        }
        const float depth = camera.model.ImagePlaneDistance(corners[corner]);
        if (!(depth >= camera.near)) {
            return false;
        }
        pixels[corner] = Pixel(camera.model.Project(corners[corner]).film, view);
        fill.x[corner] = pixels[corner].x;
        fill.y[corner] = pixels[corner].y;
        fill.depth[corner] = depth;
    }
    if (!(abs(Cross2(pixels[1] - pixels[0], pixels[2] - pixels[0])) > 1e-6f)) {
        return false;
    }
    uint slot;
    InterlockedAdd(p.fillCount[0], 1u, slot);
    if (slot >= p.fillCapacity) {
        return false;
    }
    p.fills[slot] = fill;
    record.ax = min(pixels[0].x, min(pixels[1].x, pixels[2].x));
    record.ay = min(pixels[0].y, min(pixels[1].y, pixels[2].y));
    record.bx = max(pixels[0].x, max(pixels[1].x, pixels[2].x));
    record.by = max(pixels[0].y, max(pixels[1].y, pixels[2].y));
    record.depthA = asfloat(slot);
    record.color = EncodeOverlayColor(color);
    record.flags = batch.depth | (camera.perspective ? kOverlayPerspective : 0) |
                   (uint(OverlayPiece::Triangle) << kOverlayKindShift) |
                   (edges << kOverlayEdgeShift) | (fill.pixels != 0 ? kOverlayTextured : 0) |
                   (uvs << kOverlayUvShift);
    return true;
}
