// SPDX-License-Identifier: BSD-3-Clause
// Overlay pass: debug drawing over a view's display image, after the sensor filters. Segments
// become quads `width` pixels wide facing the image, sprites (markers and label text) rectangles of
// fixed pixel size around their anchor's projection, and triangles stay as they are; each is
// depth-tested by hand against the view's depth, so items can be hidden, faded or always shown.
//
// Bindings: 0 the view's Uniforms (frame.slang; only its view-projection and pushbroom strip are
// read, through module `projection`), 1 the views' depth, 2 segments, 3 triangles, 4 sprites.
//
// References:
// - M. Rougier, "Shader-Based Antialiased, Dashed, Stroked Polylines", JCGT 2(2), 2013: lines as
//   screen-space quads with coverage from the distance across them.
// - I. Quilez, "2D distance functions", https://iquilezles.org/articles/distfunctions2d/: the
//   marker shapes.

import projection;

// frame.slang's Uniforms, as far as this pass reads.
struct ViewUniforms {
    float4x4 viewProj;
    float4 sun;
    float4 forward;
    float4 eye;
    uint4 flags;
    float4x4 sunViewProj;
    float4 sunShadow;
    float4 motion;
    float4 strip;
}

// Clip coordinates of world point `world` in the view.
float4 project(float3 world) {
    return cameraProject(u.viewProj, u.strip, world);
}

// Every item: x packed RGBA8 display colour (r in the low byte), y depth mode (kDepth*) in its
// low byte and marker shape (kShape*) in the next, z the world whose views show it (-1 all).
struct Segment {
    float4 a;  // xyz relative to the frame's origin, w width in pixels.
    float4 b;
    uint4 info;
}

struct Triangle {
    float4 p0;
    float4 p1;
    float4 p2;
    uint4 info;
}

struct Sprite {
    float4 anchor;  // xyz relative to the frame's origin.
    float4 rect;    // x0, y0, x1, y1 pixels from the anchor's projection, y down.
    uint4 info;
}

struct OverlayConstants {
    // x: the view's world, y: its layer of the depth, z: 1 when the attachment has an sRGB format,
    // w: 1 when the projection is orthographic.
    int4 view;
    // Image width and height in pixels, near and far planes.
    float4 viewport;
}

static const uint kDepthTest = 0;
static const uint kDepthXRay = 1;
static const uint kDepthAlways = 2;

static const uint kShapeCircle = 0;
static const uint kShapeRing = 1;
static const uint kShapeSquare = 2;
static const uint kShapeDiamond = 3;
static const uint kShapeCross = 4;
static const uint kShapePlus = 5;

// Segments, triangles and sprites draw from their own vertex shaders through one fragment shader;
// this says which made a fragment.
static const uint kKindSegment = 0;
static const uint kKindTriangle = 1;
static const uint kKindSprite = 2;

[[vk::binding(0, 0)]] ConstantBuffer<ViewUniforms> u;
[[vk::binding(1, 0)]] Texture2DArray<float> sceneDepth;
[[vk::binding(2, 0)]] StructuredBuffer<Segment> segments;
[[vk::binding(3, 0)]] StructuredBuffer<Triangle> triangles;
[[vk::binding(4, 0)]] StructuredBuffer<Sprite> sprites;
[[vk::push_constant]] ConstantBuffer<OverlayConstants> constants;

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
    // Segments: x pixels across the line from its centre, y half its width. Sprites: xy the
    // position in the rectangle in [-1, 1], zw its half size in pixels.
    noperspective float4 shape : SHAPE;
    // x kind (kKind*), y depth mode, z marker shape.
    nointerpolation uint4 info : INFO;
}

float4 unpackColor(uint packed) {
    return float4(float(packed & 0xff), float((packed >> 8) & 0xff), float((packed >> 16) & 0xff),
                  float(packed >> 24)) / 255.0;
}

// Whether the item shows in this view's world.
bool inWorld(uint4 info) {
    let world = int(info.z);
    return world < 0 || constants.view.x < 0 || world == constants.view.x;
}

// A vertex no primitive covers, for items culled in the vertex stage.
VertexOutput culled() {
    VertexOutput out;
    out.position = float4(2.0, 2.0, 2.0, 1.0);
    out.color = float4(0.0);
    out.shape = float4(0.0);
    out.info = uint4(0);
    return out;
}

// Pixels per unit of normalized device coordinates, along x and y (y pointing down the image).
float2 pixelScale() {
    return 0.5 * constants.viewport.xy * float2(1.0, -1.0);
}

[shader("vertex")]
VertexOutput segmentVertex(uint vertexIndex : SV_VertexID) {
    let segment = segments[vertexIndex / 6];
    if (!inWorld(segment.info)) {
        return culled();
    }
    // Two triangles: corners (end, side) of the quad.
    static const uint2 kCorners[6] = { uint2(0, 0), uint2(1, 0), uint2(1, 1),
                                       uint2(0, 0), uint2(1, 1), uint2(0, 1) };
    let corner = kCorners[vertexIndex % 6];
    var p0 = project(segment.a.xyz);
    var p1 = project(segment.b.xyz);
    // Clipped at the near plane (w is the view depth), so segments passing behind the eye keep
    // their visible part; closer to the eye an end's w, which the clip computes from far-off
    // ends, may round to or past 0.
    // An orthographic view's w is 1 throughout.
    let near = constants.view.w != 0 ? -1.0 : constants.viewport.z;
    if (p0.w < near && p1.w < near) {
        return culled();
    }
    if (p0.w < near) {
        p0 = lerp(p0, p1, (near - p0.w) / (p1.w - p0.w));
    } else if (p1.w < near) {
        p1 = lerp(p1, p0, (near - p1.w) / (p0.w - p1.w));
    }
    let scale = pixelScale();
    let s0 = p0.xy / p0.w * scale;
    let s1 = p1.xy / p1.w * scale;
    let delta = s1 - s0;
    let span = sqrt(dot(delta, delta));
    let along = span > 1e-6 ? delta / span : float2(1.0, 0.0);
    let across = float2(-along.y, along.x);
    // Half the width plus a pixel for the coverage ramp, and as much past each end for a square
    // cap, so polylines join without gaps.
    let halfWidth = 0.5 * max(segment.a.w, 1.0);
    let extent = halfWidth + 1.0;
    let side = corner.y == 0 ? -1.0 : 1.0;
    let end = corner.x == 0 ? -1.0 : 1.0;
    let p = corner.x == 0 ? p0 : p1;
    let offset = (across * side * extent + along * end * extent) / scale;
    VertexOutput out;
    out.position = float4(p.xy + offset * p.w, p.z, p.w);
    out.color = unpackColor(segment.info.x);
    out.shape = float4(side * extent, halfWidth, 0.0, 0.0);
    out.info = uint4(kKindSegment, segment.info.y & 0xff, 0, 0);
    return out;
}

[shader("vertex")]
VertexOutput triangleVertex(uint vertexIndex : SV_VertexID) {
    let triangle = triangles[vertexIndex / 3];
    if (!inWorld(triangle.info)) {
        return culled();
    }
    let corner = vertexIndex % 3;
    let p = corner == 0 ? triangle.p0 : (corner == 1 ? triangle.p1 : triangle.p2);
    VertexOutput out;
    out.position = project(p.xyz);
    out.color = unpackColor(triangle.info.x);
    out.shape = float4(0.0);
    out.info = uint4(kKindTriangle, triangle.info.y & 0xff, 0, 0);
    return out;
}

[shader("vertex")]
VertexOutput spriteVertex(uint vertexIndex : SV_VertexID) {
    let sprite = sprites[vertexIndex / 6];
    let anchor = project(sprite.anchor.xyz);
    if (!inWorld(sprite.info) || anchor.w <= 0.0) {
        return culled();
    }
    static const float2 kCorners[6] = { float2(0, 0), float2(1, 0), float2(1, 1),
                                        float2(0, 0), float2(1, 1), float2(0, 1) };
    let t = kCorners[vertexIndex % 6];
    let pixel = lerp(sprite.rect.xy, sprite.rect.zw, t);
    let offset = pixel / pixelScale();
    VertexOutput out;
    out.position = float4(anchor.xy + offset * anchor.w, anchor.z, anchor.w);
    out.color = unpackColor(sprite.info.x);
    out.shape = float4(t * 2.0 - 1.0, 0.5 * abs(sprite.rect.zw - sprite.rect.xy));
    out.info = uint4(kKindSprite, sprite.info.y & 0xff, (sprite.info.y >> 8) & 0xff, 0);
    return out;
}

// View depth (distance along the view axis) of a [0, 1] depth: from a perspective projection, or
// linear in it from an orthographic one.
float linearDepth(float depth) {
    let near = constants.viewport.z;
    let far = constants.viewport.w;
    if (constants.view.w != 0) {
        return near + depth * (far - near);
    }
    return near * far / (far - depth * (far - near));
}

// Coverage of a marker of `shape` at `p` (in [-1, 1] over the sprite), `halfSize` pixels across.
float markerCoverage(uint shape, float2 p, float2 halfSize) {
    let pixels = min(halfSize.x, halfSize.y);
    switch (shape) {
    case kShapeCircle:
        return saturate((1.0 - length(p)) * pixels + 0.5);
    case kShapeRing: {
        let thickness = max(1.5 / pixels, 0.18);
        return saturate((thickness - abs(length(p) - (1.0 - thickness))) * pixels + 0.5);
    }
    case kShapeDiamond:
        return saturate((1.0 - abs(p.x) - abs(p.y)) * pixels * 0.7071 + 0.5);
    case kShapeCross: {
        let d = min(abs(p.x - p.y), abs(p.x + p.y)) * 0.7071;
        return saturate((max(1.0 / pixels, 0.12) - d) * pixels + 0.5);
    }
    case kShapePlus: {
        let d = min(abs(p.x), abs(p.y));
        return saturate((max(1.0 / pixels, 0.12) - d) * pixels + 0.5);
    }
    default:
        return 1.0;
    }
}

float decodeSrgb(float encoded) {
    return encoded <= 0.04045 ? encoded / 12.92 : pow((encoded + 0.055) / 1.055, 2.4);
}

[shader("fragment")]
float4 overlayFragment(VertexOutput in) : SV_Target {
    var color = in.color;
    if (in.info.x == kKindSegment) {
        color.a *= saturate(in.shape.y + 0.5 - abs(in.shape.x));
    } else if (in.info.x == kKindSprite) {
        color.a *= markerCoverage(in.info.z, in.shape.xy, in.shape.zw);
    }
    let mode = in.info.y;
    if (mode != kDepthAlways) {
        let pixel = int2(in.position.xy);
        let surface = linearDepth(sceneDepth.Load(int4(pixel, constants.view.y, 0)));
        let depth = linearDepth(in.position.z);
        // A little slack, so lines drawn on surfaces stay in front of them.
        if (depth > surface * 1.003 + 1e-4) {
            if (mode == kDepthTest) {
                discard;
            }
            color.a *= 0.3;
        }
    }
    if (color.a <= 0.0) {
        discard;
    }
    if (constants.view.z == 1) {
        color.rgb = float3(decodeSrgb(color.r), decodeSrgb(color.g), decodeSrgb(color.b));
    }
    return color;
}
