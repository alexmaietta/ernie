// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.overlay.kernels.shapes;

using namespace ernie;

struct Lens {
    float2 aperture;
    float2 offset;
    float focal;
    float near;
    bool orthographic;

    __init(RenderTables* tables, uint prim) {
        const CameraRecord record = tables->cameras[PrimRow(tables, prim, kSlotCamera)];
        aperture = record.aperture.Vector();
        offset = record.offset.Vector();
        focal = max(record.focalLength, 1e-6f);
        near = record.clipping.x;
        orthographic = Projection(record.projection) == Projection::Orthographic;
    }

    float3 Corner(float2 film, float depth) {
        const float2 plane = film * 0.5f * aperture + offset;
        const float2 across = orthographic ? plane * kOrthographicApertureUnits
                                           : plane / focal * depth;
        return float3(across, -depth);
    }
}

float2 FilmCorner(uint corner) {
    return float2(corner == 1 || corner == 2 ? 1 : -1, corner >= 2 ? 1 : -1);
}

bool FrustumPiece(RenderTables* tables, uint prim, uint piece, float far, out float3 a,
                  out float3 b) {
    const Lens lens = Lens(tables, prim);
    const uint corner = piece % 4;
    if (piece < 4) {
        a = lens.orthographic ? lens.Corner(FilmCorner(corner), lens.near) : float3(0);
        b = lens.Corner(FilmCorner(corner), far);
    } else if (piece < 12) {
        const float depth = piece < 8 ? far : lens.near;
        a = lens.Corner(FilmCorner(corner), depth);
        b = lens.Corner(FilmCorner((corner + 1) % 4), depth);
    } else {
        const float2 points[3] = {float2(-0.5f, 1.1f), float2(0, 1.5f), float2(0.5f, 1.1f)};
        a = lens.Corner(points[piece - 12], far);
        b = lens.Corner(points[(piece - 11) % 3], far);
    }
    return true;
}

bool LightPiece(RenderTables* tables, uint prim, uint piece, float length, out float3 a,
                out float3 b) {
    a = float3(0);
    b = float3(0);
    const LightShapeRecord shape = HasApi(tables, prim, kApiLightAPI)
                                       ? tables->lightShapes[ApiRow(tables, prim, kApiLightAPI)]
                                       : LightShapeRecord();
    switch (PrimSchema(tables, prim)) {
    case SchemaId::SphereLight: {
        const float radius =
            shape.treatAsPoint != 0 ? 0.05f * length : max(shape.radius, 0.02f * length);
        const uint circle = piece / kOverlaySphereSegments;
        const float3 u = circle == 0 ? float3(0, 1, 0) : circle == 1 ? float3(0, 0, 1) : float3(1, 0, 0);
        const float3 v = circle == 0 ? float3(0, 0, 1) : circle == 1 ? float3(1, 0, 0) : float3(0, 1, 0);
        RingPiece(float3(0), u, v, radius, piece % kOverlaySphereSegments, kOverlaySphereSegments,
                  a, b);
        return true;
    }
    case SchemaId::RectLight: {
        const float2 half = 0.5f * float2(shape.width, shape.height);
        if (piece < 4) {
            a = float3(FilmCorner(piece) * half, 0);
            b = float3(FilmCorner((piece + 1) % 4) * half, 0);
            return true;
        }
        return piece < 9 &&
               ArrowPiece(float3(0), float3(0, 0, -0.5f * length), 0, piece - 4, 4, a, b);
    }
    case SchemaId::DiskLight: {
        const float radius = shape.radius;
        if (piece < kOverlayCircleSegments) {
            RingPiece(float3(0), float3(1, 0, 0), float3(0, 1, 0), radius, piece,
                      kOverlayCircleSegments, a, b);
            return true;
        }
        return piece < kOverlayCircleSegments + 5 &&
               ArrowPiece(float3(0), float3(0, 0, -0.5f * length), 0,
                          piece - kOverlayCircleSegments, 4, a, b);
    }
    case SchemaId::CylinderLight: {
        const float half = 0.5f * shape.length;
        const float radius = shape.radius;
        if (piece < 2 * kOverlayCapsuleSegments) {
            const float x = piece < kOverlayCapsuleSegments ? -half : half;
            RingPiece(float3(x, 0, 0), float3(0, 1, 0), float3(0, 0, 1), radius,
                      piece % kOverlayCapsuleSegments, kOverlayCapsuleSegments, a, b);
            return true;
        }
        if (piece >= 2 * kOverlayCapsuleSegments + 4) {
            return false;
        }
        const float2 turn = Turn(piece - 2 * kOverlayCapsuleSegments, 4) * radius;
        a = float3(-half, turn);
        b = float3(half, turn);
        return true;
    }
    case SchemaId::DistantLight: {
        if (piece >= 9) {
            return false;
        }
        const float2 offset = Turn(piece / 3, 3) * (0.15f * length);
        const float3 start = float3(offset, 0);
        return ArrowPiece(start, start + float3(0, 0, -length), 0.25f * length, piece % 3, 2, a,
                          b);
    }
    default:
        return false;
    }
}

bool GeneratePrim(OverlayParams p, ShapeRow shape, uint piece, OverlayView view, uint viewWorld,
                  out float3 a, out float3 b, inout float4 color) {
    a = float3(0);
    b = float3(0);
    RenderTables* tables = p.tables;
    const OverlayBatch batch = shape.batch;
    if (batch.primShape == uint(OverlayPrimShape::Bounds)) {
        float3 low;
        float3 high;
        uint world;
        if ((batch.flags & kOverlayUnited) != 0) {
            const uint at = shape.row * kOverlayBoxWords;
            low = float3(FloatFromOrdered(batch.boxes[at]), FloatFromOrdered(batch.boxes[at + 1]),
                         FloatFromOrdered(batch.boxes[at + 2]));
            high = float3(FloatFromOrdered(batch.boxes[at + 3]),
                          FloatFromOrdered(batch.boxes[at + 4]),
                          FloatFromOrdered(batch.boxes[at + 5]));
            world = batch.boxes[at + 6];
        } else {
            const uint prim = shape.Prim();
            if (!PrimBounded(tables, prim)) {
                return false;
            }
            low = p.tables->primBounds[prim].lower.Vector();
            high = p.tables->primBounds[prim].upper.Vector();
            world = tables->prims.world[prim];
            if (low.x == high.x && low.y == high.y && low.z == high.z) {
                return false;
            }
        }
        if (!all(low <= high) || !WorldShown(p.tables->instances, viewWorld, world)) {
            return false;
        }
        float3 lowCorner;
        float3 highCorner;
        BoxEdge(piece, lowCorner, highCorner);
        const float3 center = 0.5f * (low + high);
        const float3 half = 0.5f * (high - low);
        a = center + lowCorner * half;
        b = center + highCorner * half;
        if (world == kGlobalWorld) {
            a = RenderSpace(p, batch, viewWorld, a);
            b = RenderSpace(p, batch, viewWorld, b);
        } else {
            float3 shift = RenderShift(p.tables, world, viewWorld);
            if (world != viewWorld) {
                shift += ShownOffset(p.tables->instances, viewWorld, world);
            }
            a += shift;
            b += shift;
        }
        return true;
    }
    const uint prim = shape.Prim();
    if (!PrimAlive(tables, prim)) {
        return false;
    }
    const uint world = tables->prims.world[prim];
    if (!WorldShown(p.tables->instances, viewWorld, world)) {
        return false;
    }
    const WorldTransform w = PrimWorld(p.tables, prim);
    const float3 origin = RenderTranslation(p.tables, w, viewWorld) +
                          ShownOffset(p.tables->instances, viewWorld, world);
    float3 from;
    float3 to;
    switch (batch.primShape) {
    case uint(OverlayPrimShape::Frames): {
        float3 axis = float3(0);
        axis[piece] = 1;
        const float3 direction = w.Apply(axis);
        const float span = length(direction);
        if (!(span > 0)) {
            return false;
        }
        a = origin;
        b = origin + direction / span * batch.length;
        color = float4(AxisTint(piece), color.w);
        return true;
    }
    case uint(OverlayPrimShape::Frustums):
        if (prim == view.camera || !SchemaIsA(tables->prims.schema[prim], SchemaId::Camera) ||
            !FrustumPiece(tables, prim, piece, batch.length, from, to)) {
            return false;
        }
        break;
    case uint(OverlayPrimShape::Lights):
        if (!LightPiece(tables, prim, piece, batch.length, from, to)) {
            return false;
        }
        break;
    default:
        return false;
    }
    a = w.Apply(from) + origin;
    b = w.Apply(to) + origin;
    return true;
}
