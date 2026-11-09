// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.overlay.kernels.common;

#include "ernie/overlay/kernels/shapes.h"

using namespace ernie;

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

void SphereRing(uint circle, out float3 u, out float3 v) {
    u = circle == 0 ? float3(0, 1, 0) : circle == 1 ? float3(0, 0, 1) : float3(1, 0, 0);
    v = circle == 0 ? float3(0, 0, 1) : circle == 1 ? float3(1, 0, 0) : float3(0, 1, 0);
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
        float3 u;
        float3 v;
        SphereRing(piece / kOverlaySphereSegments, u, v);
        RingPiece(a, u, v, shape.Scalar(), piece % kOverlaySphereSegments, kOverlaySphereSegments,
                  a, b);
        return true;
    }
    case uint(OverlayKind::Circles): {
        float3 u;
        float3 v;
        Perpendiculars(shape.End(), u, v);
        RingPiece(a, u, v, shape.Scalar(), piece, kOverlayCircleSegments, a, b);
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

float PixelsPerMeter(OverlayCamera camera, OverlayView view, float3 centre, float3 right) {
    const float depth = camera.model.ImagePlaneDistance(centre);
    const float probe = 1e-2f * max(depth, max(camera.near, 1e-4f));
    const float2 from = Pixel(camera.model.Project(centre).film, view);
    const float2 to = Pixel(camera.model.Project(centre + right * probe).film, view);
    return depth > 0 ? length(to - from) / probe : 0;
}

struct Sizing {
    bool screen;
    bool active;
    bool facing;
    float3 localCentre;
    float3 centre;
    float3 right;
    float3 up;
    float3 back;
    float scale;

    __init(OverlayParams p, OverlayBatch batch, OverlayView view, OverlayCamera camera,
           uint viewWorld, float3 localCentre) {
        this.localCentre = localCentre;
        screen = (batch.flags & kOverlayScreenPoints) != 0;
        active = !screen && (batch.flags & (kOverlayScreenSized | kOverlayFacing)) != 0;
        facing = (batch.flags & kOverlayFacing) != 0;
        centre = screen ? localCentre : RenderSpace(p, batch, viewWorld, localCentre);
        scale = 1;
        right = float3(1, 0, 0);
        up = float3(0, 1, 0);
        back = float3(0, 0, 1);
        if (!active) {
            return;
        }
        const WorldTransform frame = PrimWorld(p.tables, view.camera);
        right = normalize(frame.x.Vector());
        up = normalize(frame.y.Vector());
        back = normalize(frame.z.Vector());
        if ((batch.flags & kOverlayScreenSized) == 0) {
            return;
        }
        const float pixels = PixelsPerMeter(camera, view, centre, right);
        scale = pixels > 0 ? 1 / pixels : 0;
    }

    float3 Place(OverlayParams p, OverlayBatch batch, uint viewWorld, float3 local) {
        if (screen) {
            return local;
        }
        if (!active) {
            return RenderSpace(p, batch, viewWorld, local);
        }
        const float3 offset = local - localCentre;
        const float3 moved = facing ? right * offset.x + up * offset.y + back * offset.z
                                    : RenderSpace(p, batch, viewWorld, local) - centre;
        return centre + moved * scale;
    }
}
