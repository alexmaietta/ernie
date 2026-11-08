// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
import ernie.io.kernels.encoding;
import ernie.render.kernels.sensors.camera.camera;

#include "ernie/overlay/kernels/overlay.h"

using namespace ernie;

static const float kDepthSlack = 2e-3f;
static const float kXRayOpacity = 0.3f;
static const float kFeather = 1.0f;

float4 DecodeOverlayColor(uint packed) {
    return float4(DecodeSrgb(float(packed & 0xFF) / 255), DecodeSrgb(float((packed >> 8) & 0xFF) / 255),
                  DecodeSrgb(float((packed >> 16) & 0xFF) / 255), float(packed >> 24) / 255);
}

uint EncodeOverlayColor(float4 color) {
    return PackOverlayColor(color.x, color.y, color.z, color.w);
}

uint FindBatch(OverlayParams p, uint item) {
    uint low = 0;
    uint high = p.batchCount;
    while (high - low > 1) {
        const uint middle = (low + high) / 2;
        if (p.batches[middle].firstItem <= item) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return low;
}

uint FindView(OverlayParams p, uint tile) {
    uint low = 0;
    uint high = p.viewCount;
    while (high - low > 1) {
        const uint middle = (low + high) / 2;
        if (p.views[middle].firstTile <= tile) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return low;
}

float3 RenderSpace(OverlayParams p, OverlayBatch batch, uint world, float3 point) {
    RenderTables* tables = p.tables;
    InstanceLayout* layout = p.tables->instances;
    if (batch.anchor != kNoPrim) {
        const WorldTransform transform = PrimWorld(p.tables, batch.anchor);
        return transform.Apply(point) + RenderTranslation(p.tables, transform, world) +
               ShownOffset(layout, world, tables->prims.world[batch.anchor]);
    }
    return RenderPoint(p.tables, point, world);
}

float RecordRadius(OverlayRecord record) {
    return float(record.flags >> kOverlayRadiusShift) / kOverlayRadiusUnits;
}

float3 LoadPoint(Ptr<float> values, uint at) {
    return float3(values[at], values[at + 1], values[at + 2]);
}

static const float kTau = 6.28318530718f;

struct ShapeRow {
    OverlayBatch batch;
    uint slice;
    uint row;

    float3 Point() {
        return LoadPoint(batch.points, slice * batch.pointWorldStride + row * batch.pointStride);
    }

    float3 End() {
        return LoadPoint(batch.ends, slice * batch.endWorldStride + row * batch.endStride);
    }

    float3 Sample(uint index) {
        return LoadPoint(batch.points, row * batch.pointStride + index * batch.endStride);
    }

    float4 Rotation() {
        const uint at = slice * batch.rotationWorldStride + row * batch.rotationStride;
        const float4 loaded = float4(batch.rotations[at], batch.rotations[at + 1],
                                     batch.rotations[at + 2], batch.rotations[at + 3]);
        return (batch.flags & kOverlayHasRotations) != 0 ? loaded : float4(0, 0, 0, 1);
    }

    float Scalar() {
        const float loaded =
            batch.scalars[slice * batch.scalarWorldStride + row * batch.scalarStride];
        return (batch.flags & kOverlayHasScalars) != 0 ? loaded : batch.length;
    }

    float3 Third() {
        return LoadPoint(batch.rotations,
                         slice * batch.rotationWorldStride + row * batch.rotationStride);
    }

    uint Prim() {
        return batch.prims[row * batch.pointStride];
    }

    uint Extra(uint fallback) {
        const uint loaded = batch.extras[slice * batch.extraWorldStride + row * batch.extraStride];
        return (batch.flags & kOverlayHasExtras) != 0 ? loaded : fallback;
    }
}

float3 Rotate(float4 q, float3 v) {
    const float3 t = 2 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

void Perpendiculars(float3 axis, out float3 u, out float3 v) {
    const float3 n = normalize(axis);
    const float3 other = abs(n.z) < 0.9f ? float3(0, 0, 1) : float3(1, 0, 0);
    u = normalize(cross(other, n));
    v = cross(n, u);
}

float2 Turn(uint step, uint steps) {
    const float angle = kTau * float(step) / float(steps);
    return float2(cos(angle), sin(angle));
}

void BoxEdge(uint piece, out float3 low, out float3 high) {
    const uint axis = piece / 4;
    const float first = (piece & 1) != 0 ? 1 : -1;
    const float second = (piece & 2) != 0 ? 1 : -1;
    low = axis == 0   ? float3(-1, first, second)
          : axis == 1 ? float3(first, -1, second)
                      : float3(first, second, -1);
    high = low;
    high[axis] = 1;
}

void RingPiece(float3 center, float3 u, float3 v, float radius, uint step, uint steps,
               out float3 a, out float3 b) {
    const float2 from = Turn(step, steps) * radius;
    const float2 to = Turn(step + 1, steps) * radius;
    a = center + u * from.x + v * from.y;
    b = center + u * to.x + v * to.y;
}

bool ArrowPiece(float3 from, float3 to, float head, uint piece, uint headLines, out float3 a,
                out float3 b) {
    a = from;
    b = to;
    const float3 along = to - from;
    const float span = length(along);
    if (!(span > 0)) {
        return false;
    }
    if (piece == 0) {
        return true;
    }
    const float reach = min(head > 0 ? head : 0.2f * span, span);
    float3 u;
    float3 v;
    Perpendiculars(along, u, v);
    const float2 turn = Turn(piece - 1, headLines) * (0.35f * reach);
    a = to;
    b = to - along / span * reach + u * turn.x + v * turn.y;
    return true;
}

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

bool PrimBounded(RenderTables* tables, uint prim) {
    return PrimAlive(tables, prim) && SchemaIsA(tables->prims.schema[prim], SchemaId::Gprim);
}


float3 AxisTint(uint axis) {
    return axis == 0 ? float3(0.8f, 0.05f, 0.03f)
           : axis == 1 ? float3(0.1f, 0.6f, 0.05f)
                       : float3(0.03f, 0.15f, 0.9f);
}

bool Generate(ShapeRow shape, uint piece, out float3 a, out float3 b, inout float4 color) {
    const OverlayBatch batch = shape.batch;
    a = shape.Point();
    b = a;
    switch (batch.kind) {
    case uint(OverlayKind::Lines):
        b = shape.End();
        return true;
    case uint(OverlayKind::Boxes): {
        const float3 half = shape.End();
        float3 low;
        float3 high;
        BoxEdge(piece, low, high);
        const float4 q = shape.Rotation();
        b = a + Rotate(q, high * half);
        a = a + Rotate(q, low * half);
        return true;
    }
    case uint(OverlayKind::Spheres): {
        const float radius = shape.Scalar();
        const uint circle = piece / kOverlaySphereSegments;
        const uint step = piece % kOverlaySphereSegments;
        const float2 from = Turn(step, kOverlaySphereSegments) * radius;
        const float2 to = Turn(step + 1, kOverlaySphereSegments) * radius;
        const float3 u = circle == 0 ? float3(0, 1, 0) : circle == 1 ? float3(0, 0, 1) : float3(1, 0, 0);
        const float3 v = circle == 0 ? float3(0, 0, 1) : circle == 1 ? float3(1, 0, 0) : float3(0, 1, 0);
        b = a + u * to.x + v * to.y;
        a = a + u * from.x + v * from.y;
        return true;
    }
    case uint(OverlayKind::Circles): {
        float3 u;
        float3 v;
        Perpendiculars(shape.End(), u, v);
        const float radius = shape.Scalar();
        const float2 from = Turn(piece, kOverlayCircleSegments) * radius;
        const float2 to = Turn(piece + 1, kOverlayCircleSegments) * radius;
        b = a + u * to.x + v * to.y;
        a = a + u * from.x + v * from.y;
        return true;
    }
    case uint(OverlayKind::Frames): {
        float3 axis = float3(0);
        axis[piece] = 1;
        b = a + Rotate(shape.Rotation(), axis) * shape.Scalar();
        color = float4(AxisTint(piece), color.w);
        return true;
    }
    case uint(OverlayKind::Arrows): {
        const float3 from = a;
        return ArrowPiece(from, shape.End(), shape.Scalar(), piece, kOverlayArrowHeadLines, a, b);
    }
    case uint(OverlayKind::Capsules): {
        const float radius = shape.Scalar();
        const float half = asfloat(shape.Extra(0));
        float3 from;
        float3 to;
        const uint rings = 2 * kOverlayCapsuleSegments;
        if (piece < rings) {
            const uint ring = piece / kOverlayCapsuleSegments;
            const uint step = piece % kOverlayCapsuleSegments;
            const float z = ring == 0 ? half : -half;
            from = float3(Turn(step, kOverlayCapsuleSegments) * radius, z);
            to = float3(Turn(step + 1, kOverlayCapsuleSegments) * radius, z);
        } else if (piece < rings + 4) {
            const float2 turn = Turn(piece - rings, 4) * radius;
            from = float3(turn, -half);
            to = float3(turn, half);
        } else {
            const uint arc = (piece - rings - 4) / kOverlayCapsuleArcSegments;
            const uint step = (piece - rings - 4) % kOverlayCapsuleArcSegments;
            const float2 start = Turn(step, 2 * kOverlayCapsuleArcSegments) * radius;
            const float2 end = Turn(step + 1, 2 * kOverlayCapsuleArcSegments) * radius;
            const float side = arc < 2 ? 1 : -1;
            const bool across = (arc & 1) != 0;
            from = across ? float3(0, start.x, side * (half + start.y))
                          : float3(start.x, 0, side * (half + start.y));
            to = across ? float3(0, end.x, side * (half + end.y))
                        : float3(end.x, 0, side * (half + end.y));
        }
        const float4 q = shape.Rotation();
        b = a + Rotate(q, to);
        a = a + Rotate(q, from);
        return true;
    }
    case uint(OverlayKind::Trails): {
        const uint samples = batch.trailLength;
        const uint head = shape.Extra(samples - 1) % samples;
        a = shape.Sample((head + 1 + piece) % samples);
        b = shape.Sample((head + 2 + piece) % samples);
        color.w *= lerp(batch.fade, 1.0f, float(piece + 1) / float(samples - 1));
        return true;
    }
    default:
        return true;
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

float2 Pixel(float2 film, OverlayView view) {
    return float2((film.x + 1) * 0.5f * float(view.width), (1 - film.y) * 0.5f * float(view.height));
}

uint ReadByte(uint64_t address, uint index) {
    const uint64_t at = address + index;
    const Ptr<uint> word = Ptr<uint>(at & ~uint64_t(3));
    return (word[0] >> ((uint(at) & 3u) * 8u)) & 0xFFu;
}

uint AffixCode(uint words[4], uint index) {
    return (words[index / 4] >> ((index % 4) * 8)) & 0xFFu;
}

static const uint kMinus = 45;
static const uint kPoint = 46;
static const uint kZero = 48;

uint FormatNumber(OverlayBatch batch, float value, out uint codes[kOverlayTextChars]) {
    for (uint i = 0; i < kOverlayTextChars; ++i) {
        codes[i] = 0;
    }
    uint count = 0;
    for (uint i = 0; i < batch.prefixLength; ++i) {
        codes[count++] = AffixCode(batch.prefix, i);
    }
    uint64_t scale = 1;
    for (uint i = 0; i < batch.decimals; ++i) {
        scale *= 10;
    }
    const float scaled = abs(value) * float(scale) + 0.5f;
    if (isnan(value) || isinf(value) || scaled >= 1.8e19f) {
        if (value < 0) {
            codes[count++] = kMinus;
        }
        const bool undefined = isnan(value);
        codes[count++] = undefined ? 110u : 105u;
        codes[count++] = undefined ? 97u : 110u;
        codes[count++] = undefined ? 110u : 102u;
    } else {
        const uint64_t whole = uint64_t(scaled);
        uint64_t integer = whole / max(scale, uint64_t(1));
        const uint64_t fraction = whole % max(scale, uint64_t(1));
        if (value < 0 && whole > 0) {
            codes[count++] = kMinus;
        }
        uint digits[20];
        uint written = 0;
        do {
            digits[written++] = uint(integer % 10);
            integer /= 10;
        } while (integer > 0 && written < 20);
        while (written > 0) {
            codes[count++] = kZero + digits[--written];
        }
        if (batch.decimals > 0) {
            codes[count++] = kPoint;
            uint64_t place = scale / 10;
            for (uint i = 0; i < batch.decimals; ++i) {
                codes[count++] = kZero + uint((fraction / max(place, uint64_t(1))) % 10);
                place /= 10;
            }
        }
    }
    for (uint i = 0; i < batch.suffixLength; ++i) {
        codes[count++] = AffixCode(batch.suffix, i);
    }
    return count;
}

uint GatherCodes(ShapeRow shape, out uint codes[kOverlayTextChars]) {
    const OverlayBatch batch = shape.batch;
    if (batch.kind == uint(OverlayKind::Numbers)) {
        return FormatNumber(batch, shape.Scalar(), codes);
    }
    for (uint i = 0; i < kOverlayTextChars; ++i) {
        codes[i] = 0;
    }
    uint start = shape.row * batch.charStride;
    uint length = min(batch.maxChars, kOverlayTextChars);
    if ((batch.flags & kOverlayPackedText) != 0) {
        const uint packed = shape.Extra(0);
        start = packed & 0xFFFFFFu;
        length = min(packed >> 24, kOverlayTextChars);
    }
    uint count = 0;
    for (uint i = 0; i < length; ++i) {
        const uint code = ReadByte(batch.chars, start + i);
        if (code == 0) {
            break;
        }
        codes[count++] = code;
    }
    return count;
}

float PenAdvance(OverlayParams p, uint code, uint next) {
    float units = p.glyphs[code].advance;
    if (code >= kOverlayKernFirstCode && next >= kOverlayKernFirstCode) {
        const uint entry = (code - kOverlayKernFirstCode) * kOverlayKernCodeCount +
                           (next - kOverlayKernFirstCode);
        units += float(int(ReadByte(p.kerning, entry) << 24) >> 24) / 16;
    }
    return units;
}

float2 PenPosition(OverlayParams p, uint codes[kOverlayTextChars], uint count, uint index,
                   out float2 size, out float lineWidth) {
    float2 pen = float2(0);
    float2 found = float2(0);
    float widest = 0;
    bool inLine = false;
    lineWidth = 0;
    for (uint i = 0; i < count; ++i) {
        if (i == index) {
            found = pen;
            inLine = true;
        }
        if (codes[i] == kOverlayNewline) {
            widest = max(widest, pen.x);
            if (inLine) {
                lineWidth = pen.x;
                inLine = false;
            }
            pen = float2(0, pen.y + kOverlayLineUnits);
        } else {
            pen.x += PenAdvance(p, codes[i], i + 1 < count ? codes[i + 1] : 0);
        }
    }
    if (inLine) {
        lineWidth = pen.x;
    }
    size = float2(max(widest, pen.x), pen.y + p.textHeight);
    return found;
}

bool ProjectText(OverlayParams p, ShapeRow shape, OverlayView view, uint viewWorld, uint item,
                 uint piece, float4 color, float size, out OverlayRecord record) {
    record = {};
    const OverlayBatch batch = shape.batch;
    uint codes[kOverlayTextChars];
    const uint count = GatherCodes(shape, codes);
    const float4 haloColor = DecodeOverlayColor(batch.halo);
    const bool haloed = haloColor.w > 0;
    const uint perPass = max((batch.pieces - 1) / (haloed ? 2u : 1u), 1u);
    const bool haloPiece = haloed && piece >= 1 && piece <= perPass;
    const uint index = (piece - 1) % perPass;
    if (piece > 0 && (index >= count || codes[index] == kOverlayNewline)) {
        return false;
    }
    float2 block;
    float lineWidth;
    float2 pen = PenPosition(p, codes, count, piece > 0 ? index : count, block, lineWidth);
    const float scale = size / kOverlayLineUnits;
    const float pad = 0.25f * size;
    const float2 blockPixels = block * scale;
    float2 corner;
    float depth = 0;
    uint mode = batch.depth;
    float2 align = float2(batch.alignX, batch.alignY);
    if (batch.kind == uint(OverlayKind::Texts) || (batch.flags & kOverlayScreen) != 0) {
        float2 at = float2(batch.atX, batch.atY);
        float2 offset = float2(batch.offsetX, batch.offsetY + float(item) * 1.25f * size);
        if (batch.kind == uint(OverlayKind::Texts)) {
            at = shape.Point().xy;
            offset = shape.End().xy;
            align = shape.Rotation().xy;
        }
        const float2 anchor = at * float2(view.width, view.height) + offset;
        corner = anchor - align * (blockPixels + 2 * pad) + pad;
        mode = uint(OverlayDepth::Always);
    } else {
        RenderTables* tables = p.tables;
        const CameraModel camera = CameraModel.Load(p.tables, view.camera);
        const uint cameraRow = PrimRow(tables, view.camera, kSlotCamera);
        const float near = tables->cameras[cameraRow].clipping.x;
        const float3 point = RenderSpace(p, batch, viewWorld, shape.Point());
        if (any(isnan(point)) || any(isinf(point))) {
            return false;
        }
        depth = camera.ImagePlaneDistance(point);
        if (depth < near) {
            return false;
        }
        const float2 anchor = Pixel(camera.Project(point).film, view);
        corner = anchor - align * blockPixels + (1 - 2 * align) * 0.6f * size;
    }
    pen.x += batch.justify * (block.x - lineWidth);
    record.depthA = depth;
    record.depthB = depth;
    if (piece == 0) {
        const float4 background = DecodeOverlayColor(batch.background);
        if (!(background.w > 0) || count == 0) {
            return false;
        }
        record.ax = corner.x - pad;
        record.ay = corner.y - pad;
        record.bx = corner.x + blockPixels.x + pad;
        record.by = corner.y + blockPixels.y + pad;
        record.color = EncodeOverlayColor(background);
        record.flags = mode | (uint(OverlayPiece::Box) << kOverlayKindShift);
        return true;
    }
    const OverlayGlyph glyph = p.glyphs[codes[index]];
    if (glyph.width == 0) {
        return false;
    }
    record.ax = corner.x + (pen.x + glyph.x0) * scale;
    record.ay = corner.y + (pen.y + glyph.y0) * scale;
    record.bx = corner.x + (pen.x + glyph.x1) * scale;
    record.by = corner.y + (pen.y + glyph.y1) * scale;
    record.depthB = asfloat(glyph.texel);
    record.color = EncodeOverlayColor(haloPiece ? float4(haloColor.xyz, haloColor.w * color.w) : color);
    record.flags = mode | (uint(OverlayPiece::Glyph) << kOverlayKindShift) |
                   (glyph.width << kOverlayFieldShift) | (glyph.height << 24);
    if (haloPiece) {
        const uint quarters = uint(clamp(batch.length * 4 + 0.5f, 0.0f, float(kOverlayHaloMask)));
        record.flags |= kOverlayHalo | (quarters << kOverlayHaloShift);
    }
    return true;
}

struct Sizing {
    bool active;
    bool facing;
    float3 localCentre;
    float3 centre;
    float3 right;
    float3 up;
    float3 back;
    float scale;

    __init(OverlayParams p, OverlayBatch batch, OverlayView view, uint viewWorld,
           float3 localCentre) {
        this.localCentre = localCentre;
        active = (batch.flags & (kOverlayScreenSized | kOverlayFacing)) != 0;
        facing = (batch.flags & kOverlayFacing) != 0;
        centre = RenderSpace(p, batch, viewWorld, localCentre);
        scale = 1;
        right = float3(1, 0, 0);
        up = float3(0, 1, 0);
        back = float3(0, 0, 1);
        if (!active) {
            return;
        }
        RenderTables* tables = p.tables;
        const WorldTransform frame = PrimWorld(p.tables, view.camera);
        right = normalize(frame.x.Vector());
        up = normalize(frame.y.Vector());
        back = normalize(frame.z.Vector());
        if ((batch.flags & kOverlayScreenSized) == 0) {
            return;
        }
        const CameraModel camera = CameraModel.Load(p.tables, view.camera);
        const uint cameraRow = PrimRow(tables, view.camera, kSlotCamera);
        const float near = max(tables->cameras[cameraRow].clipping.x, 1e-4f);
        const float depth = camera.ImagePlaneDistance(centre);
        const float probe = 1e-2f * max(depth, near);
        const float2 from = Pixel(camera.Project(centre).film, view);
        const float2 to = Pixel(camera.Project(centre + right * probe).film, view);
        const float pixels = length(to - from) / probe;
        scale = depth > 0 && pixels > 0 ? 1 / pixels : 0;
    }

    float3 Place(OverlayParams p, OverlayBatch batch, uint viewWorld, float3 local) {
        if (!active) {
            return RenderSpace(p, batch, viewWorld, local);
        }
        const float3 offset = local - localCentre;
        const float3 moved = facing ? right * offset.x + up * offset.y + back * offset.z
                                    : RenderSpace(p, batch, viewWorld, local) - centre;
        return centre + moved * scale;
    }
}

bool Filled(OverlayBatch batch) {
    return batch.kind == uint(OverlayKind::Triangles) || batch.kind == uint(OverlayKind::Images) ||
           (batch.kind == uint(OverlayKind::Prims) &&
            batch.primShape == uint(OverlayPrimShape::CameraImages));
}

float Cross2(float2 a, float2 b) {
    return a.x * b.y - a.y * b.x;
}

bool ProjectFill(OverlayParams p, ShapeRow shape, OverlayView view, uint viewWorld, uint piece,
                 float4 color, out OverlayRecord record) {
    record = {};
    RenderTables* tables = p.tables;
    const OverlayBatch batch = shape.batch;
    float3 corners[3];
    uint edges = 7;
    uint uvs = 0;
    OverlayFill fill = {};
    const Sizing sizing = Sizing(p, batch, view, viewWorld, shape.Point());
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
            if (!WorldShown(p.tables->instances, viewWorld, world)) {
                return false;
            }
            const Lens lens = Lens(tables, prim);
            const float distance = batch.length > 0 ? batch.length : 1;
            const WorldTransform w = PrimWorld(p.tables, prim);
            const float3 offset = RenderTranslation(p.tables, w, viewWorld) +
                                  ShownOffset(p.tables->instances, viewWorld, world);
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

    const CameraModel camera = CameraModel.Load(p.tables, view.camera);
    const uint cameraRow = PrimRow(tables, view.camera, kSlotCamera);
    const float near = tables->cameras[cameraRow].clipping.x;
    const bool perspective =
        Projection(tables->cameras[cameraRow].projection) != Projection::Orthographic;
    float2 pixels[3];
    for (uint corner = 0; corner < 3; ++corner) {
        if (any(isnan(corners[corner])) || any(isinf(corners[corner]))) {
            return false;
        }
        const float depth = camera.ImagePlaneDistance(corners[corner]);
        if (!(depth >= near)) {
            return false;
        }
        pixels[corner] = Pixel(camera.Project(corners[corner]).film, view);
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
    record.flags = batch.depth | (perspective ? kOverlayPerspective : 0) |
                   (uint(OverlayPiece::Triangle) << kOverlayKindShift) |
                   (edges << kOverlayEdgeShift) | (fill.pixels != 0 ? kOverlayTextured : 0) |
                   (uvs << kOverlayUvShift);
    return true;
}

bool ResolveRow(OverlayBatch batch, uint viewWorld, uint item, out uint slice, out uint row) {
    slice = 0;
    row = item;
    if ((batch.flags & kOverlayGrouped) != 0) {
        const uint globalStart = batch.groupStarts[0];
        const uint globalLength = batch.groupStarts[1] - globalStart;
        uint at = globalStart + item;
        if (item >= globalLength) {
            const bool placed = viewWorld < batch.groups;
            const uint bucket = placed ? viewWorld + 1 : batch.groups + 1;
            const uint start = batch.groupStarts[bucket];
            const uint length = placed ? batch.groupStarts[bucket + 1] - start : 0;
            if (item - globalLength >= length) {
                return false;
            }
            at = start + item - globalLength;
        }
        row = batch.order[at];
        return true;
    }
    if (batch.worldCount > 0) {
        if (viewWorld == kGlobalWorld || viewWorld >= batch.worldCount) {
            return false;
        }
        slice = viewWorld;
    }
    if (item >= min(batch.itemCounts[slice * batch.itemCountStride], batch.capacity / batch.pieces)) {
        return false;
    }
    const uint loadedWorld = batch.worlds[item * batch.worldStride];
    const uint itemWorld = (batch.flags & kOverlayHasWorlds) != 0 ? loadedWorld : batch.world;
    return batch.worldCount > 0 || itemWorld == kGlobalWorld || itemWorld == viewWorld;
}

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
    if (batch.kind >= uint(OverlayKind::Labels) && batch.kind <= uint(OverlayKind::Numbers)) {
        return ProjectText(p, shape, view, viewWorld, item, piece, color, size, record);
    }
    if (Filled(batch)) {
        return ProjectFill(p, shape, view, viewWorld, piece, color, record);
    }

    const CameraModel camera = CameraModel.Load(p.tables, view.camera);
    const uint cameraRow = PrimRow(tables, view.camera, kSlotCamera);
    const float near = tables->cameras[cameraRow].clipping.x;
    const bool perspective =
        Projection(tables->cameras[cameraRow].projection) != Projection::Orthographic;

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
        const Sizing sizing = Sizing(p, batch, view, viewWorld, shape.Point());
        a = sizing.Place(p, batch, viewWorld, a);
        b = sizing.Place(p, batch, viewWorld, b);
    }
    if (any(isnan(a)) || any(isnan(b)) || any(isinf(a)) || any(isinf(b))) {
        return false;
    }
    float depthA = camera.ImagePlaneDistance(a);
    float depthB = camera.ImagePlaneDistance(b);
    if (depthA < near && depthB < near) {
        return false;
    }
    if (depthA < near) {
        a = lerp(a, b, (near - depthA) / (depthB - depthA));
        depthA = near;
    } else if (depthB < near) {
        b = lerp(b, a, (near - depthB) / (depthA - depthB));
        depthB = near;
    }

    const float2 pixelA = Pixel(camera.Project(a).film, view);
    const float2 pixelB = Pixel(camera.Project(b).film, view);
    float pixelRadius = 0.5f * size;
    if ((batch.flags & kOverlayMeters) != 0) {
        const float3 middle = 0.5f * (a + b);
        const float3 right = normalize(PrimWorld(p.tables, view.camera).x.Vector());
        const float2 centre = Pixel(camera.Project(middle).film, view);
        const float2 side = Pixel(camera.Project(middle + right * (0.5f * size)).film, view);
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
                   (perspective ? kOverlayPerspective : 0) | (radius << kOverlayRadiusShift);
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
void KeyOverlayRows(uniform OverlayGroupParams g) {
    const uint row = LinearThreadIndex();
    if (row >= g.rows) {
        return;
    }
    const uint world = g.worlds[row * g.worldStride];
    uint bucket = world == kGlobalWorld ? 0 : (world < g.groups ? world + 1 : g.groups + 1);
    if (row >= min(g.itemCounts[0], g.rows)) {
        bucket = g.groups + 1;
    }
    g.keys[row] = bucket;
    g.order[row] = row;
}

[shader("compute")]
[numthreads(kOverlayGroup, 1, 1)]
void StartOverlayGroups(uniform OverlayGroupParams g) {
    const uint bucket = LinearThreadIndex();
    if (bucket > g.groups + 2) {
        return;
    }
    uint low = 0;
    uint high = g.rows;
    while (low < high) {
        const uint middle = (low + high) / 2;
        if (g.keys[middle] < bucket) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    g.starts[bucket] = low;
}

[shader("compute")]
[numthreads(kOverlayGroup, 1, 1)]
void MeasureOverlayGroups(uniform OverlayGroupParams g) {
    const uint world = LinearThreadIndex();
    if (world >= max(g.groups, 1u)) {
        return;
    }
    const uint global = g.starts[1] - g.starts[0];
    const uint own = g.groups > 0 ? g.starts[world + 2] - g.starts[world + 1] : 0;
    InterlockedMax(g.needs[g.need], global + own);
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

float MarkerDistance(uint shape, float2 offset, float radius) {
    const float2 folded = abs(offset);
    const float stroke = max(1.5f, 0.25f * radius);
    switch (shape) {
    case uint(MarkerShape::Ring):
        return 0.5f * stroke - abs(length(offset) - (radius - 0.5f * stroke));
    case uint(MarkerShape::Square):
        return radius - max(folded.x, folded.y);
    case uint(MarkerShape::Diamond):
        return (radius - (folded.x + folded.y)) * 0.7071f;
    case uint(MarkerShape::Cross):
        return min(0.5f * stroke - abs(folded.x - folded.y) * 0.7071f,
                   radius * 0.7071f - max(folded.x, folded.y));
    case uint(MarkerShape::Plus):
        return min(0.5f * stroke - min(folded.x, folded.y), radius - max(folded.x, folded.y));
    default:
        return radius - length(offset);
    }
}

float GlyphCoverage(OverlayRecord record, float2 point, uint64_t font) {
    const float spread = (record.flags & kOverlayHalo) != 0
                             ? float((record.flags >> kOverlayHaloShift) & kOverlayHaloMask) / 4
                             : 0;
    const float2 a = float2(record.ax, record.ay);
    const float2 b = float2(record.bx, record.by);
    if (any(point < a - 1) || any(point > b + 1)) {
        return 0;
    }
    const uint width = (record.flags >> kOverlayFieldShift) & 0xFFu;
    const uint height = record.flags >> 24;
    const uint first = asuint(record.depthB);
    const float2 texel = (point - a) / (b - a) * float2(width, height) - 0.5f;
    const int2 low = int2(floor(texel));
    const float2 blend = texel - float2(low);
    float corners[4];
    for (uint corner = 0; corner < 4; ++corner) {
        const int x = clamp(low.x + int(corner & 1), 0, int(width) - 1);
        const int y = clamp(low.y + int(corner >> 1), 0, int(height) - 1);
        corners[corner] = float(ReadByte(font, first + uint(y) * width + uint(x)));
    }
    const float value = lerp(lerp(corners[0], corners[1], blend.x),
                             lerp(corners[2], corners[3], blend.x), blend.y);
    const float distance = (value - kOverlayFieldEdge) / kOverlayFieldLevels;
    return saturate(distance * (b.x - a.x) / float(width) + spread + 0.5f);
}

float4 SampleImage(OverlayFill fill, float2 uv) {
    const float2 texel = saturate(uv) * float2(fill.width, fill.height) - 0.5f;
    const int2 low = int2(floor(texel));
    const float2 blend = texel - float2(low);
    const Ptr<uint> pixels = Ptr<uint>(fill.pixels);
    float4 corners[4];
    for (uint corner = 0; corner < 4; ++corner) {
        const int x = clamp(low.x + int(corner & 1), 0, int(fill.width) - 1);
        const int y = clamp(low.y + int(corner >> 1), 0, int(fill.height) - 1);
        corners[corner] = DecodeOverlayColor(pixels[uint(y) * fill.rowStride + uint(x)]);
    }
    float4 sampled =
        lerp(lerp(corners[0], corners[1], blend.x), lerp(corners[2], corners[3], blend.x), blend.y);
    if (fill.opaque != 0) {
        sampled.w = 1;
    }
    return sampled;
}

float TriangleCoverage(OverlayRecord record, OverlayFill fill, float2 point, inout float4 color,
                       out float depth) {
    depth = 0;
    const float2 corners[3] = {float2(fill.x[0], fill.y[0]), float2(fill.x[1], fill.y[1]),
                               float2(fill.x[2], fill.y[2])};
    const float area = Cross2(corners[1] - corners[0], corners[2] - corners[0]);
    const float side = area > 0 ? 1 : -1;
    const uint edges = (record.flags >> kOverlayEdgeShift) & kOverlayEdgeMask;
    float coverage = 1;
    for (uint edge = 0; edge < 3; ++edge) {
        const float2 from = corners[edge];
        const float2 along = corners[(edge + 1) % 3] - from;
        const float inside = side * Cross2(along, point - from) / max(length(along), 1e-6f);
        if (((edges >> edge) & 1) != 0) {
            coverage = min(coverage, saturate(inside + 0.5f));
        } else if (inside < 0) {
            coverage = 0;
        }
    }
    if (coverage <= 0) {
        return 0;
    }
    const float first = Cross2(corners[1] - point, corners[2] - point) / area;
    const float second = Cross2(corners[2] - point, corners[0] - point) / area;
    const float3 shares = float3(first, second, 1 - first - second);
    const bool perspective = (record.flags & kOverlayPerspective) != 0;
    const float3 depths = float3(fill.depth[0], fill.depth[1], fill.depth[2]);
    const float3 weights = perspective ? shares / depths : shares;
    const float total = weights.x + weights.y + weights.z;
    depth = perspective ? 1 / total : dot(shares, depths);
    if ((record.flags & kOverlayTextured) != 0) {
        const uint uvs = record.flags >> kOverlayUvShift;
        float2 uv = float2(0);
        for (uint corner = 0; corner < 3; ++corner) {
            const float2 cornerUv = float2((uvs >> (2 * corner)) & 1, (uvs >> (2 * corner + 1)) & 1);
            uv += weights[corner] * cornerUv;
        }
        color *= SampleImage(fill, uv / total);
    }
    return coverage;
}

float4 Shade(OverlayRecord record, float4 color, float2 point, float sceneDepth,
             float missDistance, uint64_t font, Ptr<OverlayFill> fills) {
    const float2 a = float2(record.ax, record.ay);
    const float2 b = float2(record.bx, record.by);
    const uint kind = (record.flags >> kOverlayKindShift) & kOverlayKindMask;
    float t = 0;
    float coverage;
    float fillDepth = 0;
    if (kind == uint(OverlayPiece::Triangle)) {
        coverage = TriangleCoverage(record, fills[asuint(record.depthA)], point, color, fillDepth);
    } else if (kind == uint(OverlayPiece::Glyph)) {
        coverage = GlyphCoverage(record, point, font);
    } else if (kind == uint(OverlayPiece::Box)) {
        const float2 inside = min(point - a, b - point);
        coverage = saturate(min(inside.x, inside.y) + 0.5f);
    } else if (kind == uint(OverlayPiece::Segment)) {
        const float2 along = b - a;
        const float lengthSquared = dot(along, along);
        t = lengthSquared > 0 ? saturate(dot(point - a, along) / lengthSquared) : 0;
        coverage = saturate(RecordRadius(record) + 0.5f - length(point - (a + along * t)));
        const uint dash = (record.flags >> kOverlayDashShift) & kOverlayPatternMask;
        if (dash > 0 && coverage > 0) {
            const float gap = float(max((record.flags >> kOverlayGapShift) & kOverlayPatternMask, 1u));
            const float period = float(dash) + gap;
            const float phase = fmod(t * sqrt(lengthSquared), period);
            const float inside = phase < float(dash) ? min(phase, float(dash) - phase)
                                                     : -min(phase - float(dash), period - phase);
            coverage *= saturate(inside + 0.5f);
        }
    } else {
        const uint shape = (record.flags >> kOverlayShapeShift) & kOverlayShapeMask;
        coverage = saturate(MarkerDistance(shape, point - a, RecordRadius(record)) + 0.5f);
    }
    if (coverage <= 0) {
        return float4(0);
    }
    const bool flat = kind == uint(OverlayPiece::Glyph) || kind == uint(OverlayPiece::Box);
    const float depth = kind == uint(OverlayPiece::Triangle) ? fillDepth
                        : flat ? record.depthA
                        : (record.flags & kOverlayPerspective) != 0
                            ? 1 / lerp(1 / record.depthA, 1 / record.depthB, t)
                            : lerp(record.depthA, record.depthB, t);
    const uint mode = record.flags & kOverlayDepthMask;
    if (mode != uint(OverlayDepth::Always) && sceneDepth != missDistance &&
        depth > sceneDepth * (1 + kDepthSlack)) {
        if (mode != uint(OverlayDepth::XRay)) {
            return float4(0);
        }
        coverage *= kXRayOpacity;
    }
    return float4(color.xyz, color.w * coverage);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void FoldOverlayTiles(uniform OverlayParams p) {
    const uint tiles = p.occupied[0];
    const uint across = min(tiles, kOverlayMostGroups);
    p.occupied[1] = across;
    p.occupied[2] = across == 0 ? 0 : (tiles + across - 1) / across;
    p.occupied[3] = 1;
}

static const float kOpaque = 254.5f / 255;

groupshared OverlayRecord s_records[kOverlayTilePixels];
groupshared float4 s_colors[kOverlayTilePixels];
groupshared uint s_open;

struct CompositePixel {
    uint index;
    float2 point;
    float sceneDepth;
    bool inside;
    float4 under;
    bool done;

    __init(OverlayView view, uint2 pixel) {
        inside = pixel.x < view.width && pixel.y < view.height;
        const uint2 clamped = min(pixel, uint2(view.width, view.height) - 1);
        index = (view.layer * view.height + clamped.y) * view.width + clamped.x;
        point = float2(pixel) + 0.5f;
        sceneDepth = view.depth[index];
        under = float4(0);
        done = !inside;
    }

    [mutating]
    void Add(OverlayRecord record, float4 color, float missDistance, uint64_t font,
             Ptr<OverlayFill> fills) {
        if (done) {
            return;
        }
        const float4 shaded = Shade(record, color, point, sceneDepth, missDistance, font, fills);
        const float weight = (1 - under.w) * shaded.w;
        under = float4(under.xyz + shaded.xyz * weight, under.w + weight);
        done = under.w >= kOpaque;
    }

    void Write(OverlayView view) {
        if (inside && under.w > 0) {
            view.overlay[index] = EncodeOverlayColor(float4(under.xyz / under.w, under.w));
        }
    }
}

[shader("compute")]
[numthreads(kOverlayTile, kOverlayTile, 1)]
void CompositeOverlays(uniform OverlayParams p) {
    const uint listed = LinearGroupIndex();
    if (listed >= p.occupied[0]) {
        return;
    }
    const uint at = kOverlayTileList + kOverlayTileWords * listed;
    const uint tile = p.occupied[at];
    const uint start = p.occupied[at + 2];
    const uint end = p.occupied[at + 3];
    const uint thread = LocalThreadIndex();
    const OverlayView view = p.views[p.occupied[at + 1]];
    const uint local = tile - view.firstTile;
    const uint2 corner =
        uint2((local % view.columnCount) * kOverlayTile, (local / view.columnCount) * kOverlayTile);
    CompositePixel pixel =
        CompositePixel(view, corner + uint2(thread % kOverlayTile, thread / kOverlayTile));

    uint chunkEnd = end;
    while (chunkEnd > start) {
        const uint chunkStart = chunkEnd - min(kOverlayTilePixels, chunkEnd - start);
        const uint loaded = chunkEnd - chunkStart;
        if (thread < loaded) {
            const OverlayRecord record = p.records[p.values[chunkStart + thread]];
            s_records[thread] = record;
            s_colors[thread] = DecodeOverlayColor(record.color);
        }
        if (thread == 0) {
            s_open = 0;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint i = loaded; i > 0 && !pixel.done; --i) {
            pixel.Add(s_records[i - 1], s_colors[i - 1], view.missDistance, p.font, p.fills);
        }
        if (chunkStart == start) {
            break;
        }
        if (!pixel.done) {
            InterlockedAdd(s_open, 1u);
        }
        GroupMemoryBarrierWithGroupSync();
        const bool open = s_open > 0;
        GroupMemoryBarrierWithGroupSync();
        if (!open) {
            break;
        }
        chunkEnd = chunkStart;
    }
    pixel.Write(view);
}

[shader("compute")]
[numthreads(kOverlayGroup, 1, 1)]
void BlendOverlay(uniform BlendParams p) {
    const uint pixel = LinearThreadIndex();
    if (pixel >= p.count) {
        return;
    }
    const uint covering = p.overlay[pixel];
    const uint color = p.color[pixel];
    if ((covering >> 24) == 0) {
        p.blended[pixel] = color;
        return;
    }
    const float4 over = DecodeOverlayColor(covering);
    const float4 under = DecodeOverlayColor(color);
    p.blended[pixel] = EncodeOverlayColor(float4(over.xyz * over.w + under.xyz * (1 - over.w),
                                                 over.w + under.w * (1 - over.w)));
}

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
    const float3 low = u.tables->primBounds[prim].lower.Vector();
    const float3 high = u.tables->primBounds[prim].upper.Vector();
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

bool Outlined(OverlayOutline outline, uint id) {
    const uint word = id / 32;
    return id != kNoPrim && word < outline.bitWords && ((outline.bits[word] >> (id % 32)) & 1) != 0;
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
    const uint local = tile - view.firstTile;
    const int2 pixel = int2((local % view.columnCount) * kOverlayTile + thread % kOverlayTile,
                            (local / view.columnCount) * kOverlayTile + thread / kOverlayTile);
    const int2 size = int2(view.width, view.height);
    if (any(pixel >= size)) {
        return;
    }
    const uint base = view.layer * view.height * view.width;
    const uint index = base + uint(pixel.y) * view.width + uint(pixel.x);
    float4 under = float4(0);
    for (uint at = 0; at < p.outlineCount; ++at) {
        const OverlayOutline outline = p.outlines[at];
        if (((view.overlays >> outline.overlay) & 1) == 0) {
            continue;
        }
        const bool centre = Outlined(outline, view.primIds[index]);
        const int reach = int(ceil(outline.width));
        float nearest = 1e9f;
        for (int dy = -reach; dy <= reach; ++dy) {
            for (int dx = -reach; dx <= reach; ++dx) {
                const float distance = length(float2(dx, dy));
                if ((dx == 0 && dy == 0) || distance >= nearest ||
                    distance > outline.width + 0.5f) {
                    continue;
                }
                const int2 other = clamp(pixel + int2(dx, dy), int2(0), size - 1);
                if (Outlined(outline, view.primIds[base + uint(other.y) * view.width +
                                                   uint(other.x)]) != centre) {
                    nearest = distance;
                }
            }
        }
        const float coverage = saturate(outline.width + 0.5f - nearest);
        if (coverage > 0) {
            const float4 color = DecodeOverlayColor(outline.color);
            const float weight = (1 - under.w) * color.w * coverage;
            under = float4(under.xyz + color.xyz * weight, under.w + weight);
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
