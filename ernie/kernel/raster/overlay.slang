// SPDX-License-Identifier: BSD-3-Clause
// Overlay pass: debug drawing over a view's display image, after the sensor filters. Segments
// become quads `width` pixels wide facing the image, sprites (markers, text's characters and its
// backing boxes) rectangles of fixed pixel size around their anchor's projection, and triangles
// stay as they are; each is depth-tested by hand against the view's depth, so items can be hidden,
// faded or always shown. Items posed under a prim are in its space, moved by its transform this
// frame from the anchor table; screen sprites sit at a point of the image, over everything.
//
// Bindings: 0 the view's Uniforms (frame.slang; only its view-projection and pushbroom strip are
// read, through module `projection`), 1 the views' depth, 2 segments, 3 triangles, 4 sprites, 5
// batches' arrays, 6 batches, 7 batches' items in world order (overlay_sort.slang), 8 the anchor
// table (four rows of a transform per anchor, relative to the frame's origin), 9 the font, 10 the
// anchor table's rows' worlds (-1 for none).
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
    float4 forward;
    float4 eye;
    uint4 flags;
    float4 minimalLights[4];
    float4 motion;
    float4 strip;
}

// Clip coordinates of world point `world` in the view.
float4 project(float3 world) {
    return cameraProject(u.viewProj, u.strip, world);
}

// Every item: x packed RGBA8 display colour (r in the low byte), y depth mode (kDepth*) in its
// low byte, marker shape (kShape*) in the next, a sprite's character (or 0) in the third and 1 for
// a screen sprite in the fourth, z the world whose views show it (-1 all), w 1 + its anchor in the
// anchor table (0 for none).
// Every item's `extra`: x a segment's colour at `b` (packed as `info.x`), y the item's id
// (`OverlayStyle::id`, for picks), z kExtra* flags.
static const uint kExtraEndColor = 1;  // x holds the segment's colour at its end
static const uint kExtraHalo = 2;      // a character's halo: x its colour, w its width in pixels

struct Segment {
    float4 a;  // xyz relative to the frame's origin, w width in pixels.
    float4 b;  // w the dashes' and gaps' length in pixels, 0 for a solid line.
    uint4 info;
    uint4 extra;
}

struct Triangle {
    float4 p0;
    float4 p1;
    float4 p2;
    uint4 info;
    uint4 extra;
}

struct Sprite {
    float4 anchor;  // xyz relative to the frame's origin; a screen sprite's xy fractions of the
                    // image from its top left.
    float4 rect;    // x0, y0, x1, y1 pixels from the anchor's projection, y down; metres with
                    // kInfoMetric.
    uint4 info;
    uint4 extra;
}

// `info.y` above its depth mode, marker shape and character: 1 << 24 a screen sprite, 1 << 25 a
// sprite sized in metres (`OverlayStyle::metric`).
static const uint kInfoMetric = 1u << 25;

struct OverlayConstants {
    // x: the view's world, y: its layer of the depth, z: 1 when the attachment has an sRGB format,
    // w: 1 when the projection is orthographic.
    int4 view;
    // Image width and height in pixels, near and far planes.
    float4 viewport;
    // x: the batch the batch shaders draw (`batches`), or the first item the segment, triangle and
    // sprite shaders draw; y: 1 when the draw takes one slot's items in world order
    // (`sortedItems`): z the slot, w the batch's slots.
    uint4 batch;
    // x: the first row in the anchor table of the drawn layer's anchors, which its items' count
    // from (0 for batches, whose items' are rows of the table); yz: for a pick, the picked pixel,
    // which the viewport moves to the 1 x 1 target's; w: what a pick reports drew the item (1 +
    // the layer, or kPickBatch | the batch).
    uint4 layer;
}

static const uint kPickBatch = 0x80000000u;

// Many items of one kind read from arrays in `batchData` (`OverlayBatch`), expanded here into
// segments, triangles or sprites: lines and arrow shafts, box edges, sphere circles and frame axes
// as segments; arrow heads as triangles; markers, and numbers' characters, as sprites.
struct Batch {
    // x kind (kBatch*), y items, z segments (or triangles, or a number's sprites) per item, w
    // kBatch* flags.
    uint4 shape;
    // First floats in `batchData` of: x points, y ends, half extents, radii or values, z
    // orientations, w colours.
    uint4 arrays;
    // x sizes, y worlds, z its first item in world order in `sortedItems` (overlay_sort.slang), w
    // 1 + its anchor in the anchor table (the first of one per item with kBatchItemAnchors), or 0.
    uint4 extras;
    // The style's: x packed colour, y depth mode and marker shape as `info.y`, z world, w packed
    // background colour.
    uint4 style;
    // xyz the arrays' origin relative to the frame's (or in their anchors' space); w the style's
    // width, or marker or text size (pixels).
    float4 origin;
    // x arrows' head length (0 for a fifth of each), y frames' axis length; z lines' dashes in
    // pixels (0 solid) of kinds but numbers, and of numbers zw where their text
    // boxes meet their places (fractions of the box from its top left).
    float4 params;
    // Numbers: x their text's first character in `batchData`, y and z its prefix's and suffix's
    // lengths, w the decimals.
    uint4 text;
    // Screen numbers without points: xy the first's place (fractions of the image), zw its offset
    // in pixels.
    float4 place;
    // x capsules' half heights' or trails' heads' first float, y 1 + pose indices' (0 for none;
    // read by overlay_sort.slang), z 1 + its count's (0 for none), w points per trail.
    uint4 more;
}

static const uint kBatchLines = 0;
static const uint kBatchArrows = 1;
static const uint kBatchMarkers = 2;
static const uint kBatchBoxes = 3;
static const uint kBatchSpheres = 4;
static const uint kBatchFrames = 5;
static const uint kBatchNumbers = 6;
static const uint kBatchTrails = 7;
static const uint kBatchCapsules = 8;
static const uint kBatchTexts = 9;
static const uint kBatchImage = 10;

static const uint kBatchHasColors = 1;
static const uint kBatchHasAlpha = 2;  // colours with opacity, 4 floats each
static const uint kBatchHasSizes = 4;
static const uint kBatchHasWorlds = 8;
static const uint kBatchHasOrientations = 16;
static const uint kBatchItemAnchors = 32;
static const uint kBatchScreen = 64;
static const uint kBatchHasPoints = 128;
static const uint kBatchHasHeads = 256;
static const uint kBatchFade = 512;
static const uint kBatchDirections = 1024;  // ends are vectors from the points, times params.y
static const uint kBatchPointColors = 2048;  // a trail's colours one per point
static const uint kBatchEndColors = 4096;    // lines' and arrows' colours one per end

// The font (`OverlayTextFont`), on a grid of which a line is 12 units, y down from the top of its
// capitals: per Latin-1 code kGlyphWords words, its field's box (x0, y0, x1, y1 floats, units
// from the pen), its advance (a float), its field's first texel, and its texels across and down
// (16 bits each); then kFontWords of the whole font's (x the text's height in a line, from the top
// of its capitals to the bottom of its descenders); then its kerning between codes 32 to 255, a
// signed byte of sixteenths of a unit per pair, four to a word; then its distance fields, a byte
// per texel, four to a word: 128 on an outline, kFieldLevels more per texel inside.
static const uint kGlyphWords = 8;
static const uint kFontWords = 4;
static const uint kKernFirst = 32;
static const uint kKernCodes = 224;
static const uint kFontBase = 256 * kGlyphWords;
static const uint kKernBase = kFontBase + kFontWords;
static const uint kFieldBase = kKernBase + kKernCodes * kKernCodes / 4;
static const float kFieldLevels = 16.0;
static const float kLineHeight = 12.0;

// Segments of each of a sphere's circles, and triangles of an arrow head's cone (its side, then its
// base), as `Overlay::Sphere` and `Overlay::Arrow` draw them.
static const uint kCircleSegments = 48;
static const uint kConeSides = 8;

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
static const uint kKindImage = 3;  // a batch's image, read in the fragment shader

[[vk::binding(0, 0)]] ConstantBuffer<ViewUniforms> u;
[[vk::binding(1, 0)]] Texture2DArray<float> sceneDepth;
[[vk::binding(2, 0)]] StructuredBuffer<Segment> segments;
[[vk::binding(3, 0)]] StructuredBuffer<Triangle> triangles;
[[vk::binding(4, 0)]] StructuredBuffer<Sprite> sprites;
[[vk::binding(5, 0)]] StructuredBuffer<float> batchData;
[[vk::binding(6, 0)]] StructuredBuffer<Batch> batches;
[[vk::binding(7, 0)]] StructuredBuffer<uint> sortedItems;
[[vk::binding(8, 0)]] StructuredBuffer<float4> anchors;
[[vk::binding(9, 0)]] StructuredBuffer<uint> glyphs;
[[vk::binding(10, 0)]] StructuredBuffer<int> anchorWorlds;
[[vk::push_constant]] ConstantBuffer<OverlayConstants> constants;

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
    // Segments: x pixels across the line from its centre, y half its width, z pixels along it
    // from its start, w its dashes' length (0 solid). Sprites: xy the
    // position in the rectangle in [-1, 1], zw its half size in pixels; a character's z its
    // halo's width in pixels instead.
    noperspective float4 shape : SHAPE;
    // x kind (kKind*), y depth mode, z marker shape and, above it, the character drawn (or 0), w
    // what a pick reports of the item (`extra.y`): a host item's id, a batch item's index. The
    // colour pass reads all of `info`, so the pick needs no output of its own that it would not
    // read.
    nointerpolation uint4 info : INFO;
    // Segments: x their length in pixels, for round ends. Characters: their halo's colour.
    noperspective float4 span : SPAN;
}

// Point `p` of an item with `info`, in its anchor's space if it has one, relative to the frame's
// origin.
// 1 + the row in the anchor table of an item's anchor: its layer's base on (`constants.layer`).
uint anchorRow(uint4 info) {
    return info.w + constants.layer.x;
}

float3 anchored(uint4 info, float3 p) {
    if (info.w == 0) {
        return p;
    }
    let first = 4 * (anchorRow(info) - 1);
    let m = float4x4(anchors[first], anchors[first + 1], anchors[first + 2], anchors[first + 3]);
    return mul(m, float4(p, 1.0)).xyz;
}

// Whether the item is a screen sprite.
bool onScreen(uint4 info) {
    return ((info.y >> 24) & 1) != 0;
}

float4 unpackColor(uint packed) {
    return float4(float(packed & 0xff), float((packed >> 8) & 0xff), float((packed >> 16) & 0xff),
                  float(packed >> 24)) / 255.0;
}

// Whether the item shows in this view's world.
bool inWorld(uint4 info) {
    var world = int(info.z);
    // Without a world of its own, its anchor's.
    if (world < 0 && info.w != 0) {
        world = anchorWorlds[anchorRow(info) - 1];
    }
    return world < 0 || constants.view.x < 0 || world == constants.view.x;
}

// A vertex no primitive covers, for items culled in the vertex stage.
VertexOutput culled() {
    VertexOutput out;
    out.position = float4(2.0, 2.0, 2.0, 1.0);
    out.color = float4(0.0);
    out.shape = float4(0.0);
    out.info = uint4(0);
    out.span = float4(0.0);
    return out;
}

// Pixels a metre spans across the view at `p` (relative to the frame's origin), for items sized
// in metres.
float pixelsPerMetre(float3 p) {
    let forward = u.forward.xyz;
    let side = normalize(cross(forward, abs(forward.z) < 0.9 ? float3(0, 0, 1) : float3(1, 0, 0)));
    let a = project(p);
    let b = project(p + side);
    if (a.w <= 0.0 || b.w <= 0.0) {
        return 0.0;
    }
    return length((b.xy / b.w - a.xy / a.w) * pixelScale());
}

// Pixels per unit of normalized device coordinates, along x and y (y pointing down the image).
float2 pixelScale() {
    return 0.5 * constants.viewport.xy * float2(1.0, -1.0);
}

[shader("vertex")]
VertexOutput segmentVertex(uint vertexIndex : SV_VertexID) {
    return segmentCorner(segments[constants.batch.x + vertexIndex / 6], vertexIndex);
}

// Corner `vertexIndex % 6` of a segment's quad.
VertexOutput segmentCorner(Segment segment, uint vertexIndex) {
    if (!inWorld(segment.info)) {
        return culled();
    }
    // Two triangles: corners (end, side) of the quad.
    static const uint2 kCorners[6] = { uint2(0, 0), uint2(1, 0), uint2(1, 1),
                                       uint2(0, 0), uint2(1, 1), uint2(0, 1) };
    let corner = kCorners[vertexIndex % 6];
    var p0 = project(anchored(segment.info, segment.a.xyz));
    var p1 = project(anchored(segment.info, segment.b.xyz));
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
    // Its colour, blending to its end's if it has one.
    let ended = corner.x != 0 && (segment.extra.z & kExtraEndColor) != 0;
    out.color = unpackColor(ended ? segment.extra.x : segment.info.x);
    out.shape = float4(side * extent, halfWidth, corner.x == 0 ? -extent : span + extent,
                       segment.b.w);
    out.info = uint4(kKindSegment, segment.info.y & 0xff, 0, segment.extra.y);
    out.span = float4(span, 0.0, 0.0, 0.0);
    return out;
}

[shader("vertex")]
VertexOutput triangleVertex(uint vertexIndex : SV_VertexID) {
    return triangleCorner(triangles[constants.batch.x + vertexIndex / 3], vertexIndex);
}

// Corner `vertexIndex % 3` of a triangle.
VertexOutput triangleCorner(Triangle triangle, uint vertexIndex) {
    if (!inWorld(triangle.info)) {
        return culled();
    }
    let corner = vertexIndex % 3;
    let p = corner == 0 ? triangle.p0 : (corner == 1 ? triangle.p1 : triangle.p2);
    VertexOutput out;
    out.position = project(anchored(triangle.info, p.xyz));
    out.color = unpackColor(triangle.info.x);
    out.shape = float4(0.0);
    out.info = uint4(kKindTriangle, triangle.info.y & 0xff, 0, triangle.extra.y);
    out.span = float4(0.0);
    return out;
}

[shader("vertex")]
VertexOutput spriteVertex(uint vertexIndex : SV_VertexID) {
    return spriteCorner(sprites[constants.batch.x + vertexIndex / 6], vertexIndex);
}

// Corner `vertexIndex % 6` of a sprite's rectangle.
VertexOutput spriteCorner(Sprite sprite, uint vertexIndex) {
    // A screen sprite's point of the image, y down, in clip coordinates (y up).
    let world = anchored(sprite.info, sprite.anchor.xyz);
    let anchor = onScreen(sprite.info)
                     ? float4(sprite.anchor.x * 2.0 - 1.0, 1.0 - sprite.anchor.y * 2.0, 0.5, 1.0)
                     : project(world);
    if (!inWorld(sprite.info) || anchor.w <= 0.0) {
        return culled();
    }
    // A rectangle in metres, as many pixels as it spans where it is.
    var rect = sprite.rect;
    if ((sprite.info.y & kInfoMetric) != 0 && !onScreen(sprite.info)) {
        rect *= pixelsPerMetre(world);
    }
    static const float2 kCorners[6] = { float2(0, 0), float2(1, 0), float2(1, 1),
                                        float2(0, 0), float2(1, 1), float2(0, 1) };
    let t = kCorners[vertexIndex % 6];
    let pixel = lerp(rect.xy, rect.zw, t);
    let offset = pixel / pixelScale();
    VertexOutput out;
    out.position = float4(anchor.xy + offset * anchor.w, anchor.z, anchor.w);
    out.color = unpackColor(sprite.info.x);
    out.shape = float4(t * 2.0 - 1.0, 0.5 * abs(rect.zw - rect.xy));
    out.info = uint4(kKindSprite, sprite.info.y & 0xff,
                     ((sprite.info.y >> 8) & 0xff) | (((sprite.info.y >> 16) & 0xff) << 8),
                     sprite.extra.y);
    out.span = float4(0.0);
    if ((sprite.extra.z & kExtraHalo) != 0) {
        out.shape.z = asfloat(sprite.extra.w);
        out.span = unpackColor(sprite.extra.x);
    }
    return out;
}

// Batches: each item read from the batch's arrays and expanded into its segments, triangles or
// sprites, as the overlay's own methods reduce one item on the host.

float3 batchVector(uint first, uint i) {
    let at = first + 3 * i;
    return float3(batchData[at], batchData[at + 1], batchData[at + 2]);
}

// Item `i`'s point of the array at `first`, relative to the frame's origin.
float3 batchPoint(Batch b, uint first, uint i) {
    return batchVector(first, i) + b.origin.xyz;
}

float4 batchOrientation(Batch b, uint i) {
    if ((b.shape.w & kBatchHasOrientations) == 0) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    let at = b.arrays.z + 4 * i;
    return float4(batchData[at], batchData[at + 1], batchData[at + 2], batchData[at + 3]);
}

// `v` turned by unit quaternion `q` (x, y, z, w).
float3 rotate(float4 q, float3 v) {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

uint packColor(float4 color) {
    let bytes = uint4(round(saturate(color) * 255.0));
    return bytes.x | (bytes.y << 8) | (bytes.z << 16) | (bytes.w << 24);
}

// Item `i`'s `info`: its colour or the style's, the style's depth mode and marker shape, its
// world or the style's, and its anchor.
// Row `row` of a batch's colours, packed, its opacity the style's without its own.
uint batchColorAt(Batch b, uint row) {
    let alpha = (b.shape.w & kBatchHasAlpha) != 0;
    let at = b.arrays.w + (alpha ? 4 : 3) * row;
    return packColor(float4(batchData[at], batchData[at + 1], batchData[at + 2],
                            alpha ? batchData[at + 3] : unpackColor(b.style.x).a));
}

uint4 batchInfo(Batch b, uint i) {
    var color = b.style.x;
    if ((b.shape.w & kBatchHasColors) != 0) {
        let alpha = (b.shape.w & kBatchHasAlpha) != 0;
        let at = b.arrays.w + (alpha ? 4 : 3) * i;
        color = packColor(float4(batchData[at], batchData[at + 1], batchData[at + 2],
                                 alpha ? batchData[at + 3] : unpackColor(b.style.x).a));
    }
    let world = (b.shape.w & kBatchHasWorlds) != 0 ? asuint(batchData[b.extras.y + i]) : b.style.z;
    var anchor = b.extras.w;
    if (anchor != 0 && (b.shape.w & kBatchItemAnchors) != 0) {
        anchor += i;
    }
    return uint4(color, b.style.y, world, anchor);
}

// The item at position `at` of the draw's items: for a draw of one slot's (overlay_sort.slang),
// its items in world order from the slot's first, the slot's cursor less its count.
uint batchItem(Batch b, uint at) {
    if (constants.batch.y == 0) {
        return at;
    }
    let slot = constants.batch.z;
    let slots = constants.batch.w;
    let counts = b.extras.z - 2 * slots;
    let first = sortedItems[counts + slots + slot] - sortedItems[counts + slot];
    return sortedItems[b.extras.z + first + at];
}

// Whether item `i` is in use: below the batch's count, if it has one.
bool batchInUse(Batch b, uint i) {
    return b.more.z == 0 || int(i) < asint(batchData[b.more.z - 1]);
}

// Item `i`'s line width or marker size in pixels.
float batchSize(Batch b, uint i) {
    return (b.shape.w & kBatchHasSizes) != 0 ? batchData[b.extras.x + i] : b.origin.w;
}

// Two unit vectors perpendicular to unit `axis` and to each other.
void perpendiculars(float3 axis, out float3 u, out float3 v) {
    let other = abs(axis.x) < 0.9 ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
    u = normalize(cross(axis, other));
    v = cross(axis, u);
}

// Item `i`'s end, of a line or an arrow from `start`: its point, or with kBatchDirections its
// vector from `start`, scaled.
float3 batchEnd(Batch b, uint i, float3 start) {
    if ((b.shape.w & kBatchDirections) != 0) {
        return start + batchVector(b.arrays.y, i) * b.params.y;
    }
    return batchPoint(b, b.arrays.y, i);
}

// Segment `k` of capsule `i` in its own space, along z: a ring around each end of its cylinder
// (24 segments each), its four sides, then half circles over its caps in the xz and yz planes
// (12 segments each), as `kCapsuleSegments` in overlay_pass.cpp counts them.
void capsuleSegment(float radius, float half, uint k, out float3 a, out float3 b) {
    static const float kPi = 3.14159265;
    if (k < 48) {
        let z = k < 24 ? -half : half;
        let j = k % 24;
        let a0 = 2.0 * kPi * float(j) / 24.0;
        let a1 = 2.0 * kPi * float(j + 1) / 24.0;
        a = float3(radius * cos(a0), radius * sin(a0), z);
        b = float3(radius * cos(a1), radius * sin(a1), z);
        return;
    }
    if (k < 52) {
        let angle = 0.5 * kPi * float(k - 48);
        let side = float2(radius * cos(angle), radius * sin(angle));
        a = float3(side, -half);
        b = float3(side, half);
        return;
    }
    let arc = (k - 52) / 12;
    let j = (k - 52) % 12;
    let across = arc < 2 ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
    let up = (arc & 1) == 0 ? 1.0 : -1.0;
    let t0 = kPi * float(j) / 12.0;
    let t1 = kPi * float(j + 1) / 12.0;
    a = radius * (cos(t0) * across + float3(0.0, 0.0, up * sin(t0))) + float3(0.0, 0.0, up * half);
    b = radius * (cos(t1) * across + float3(0.0, 0.0, up * sin(t1))) + float3(0.0, 0.0, up * half);
}

// Arrow `i`'s head from `end` back to its base, a fifth of the arrow unless the batch sets its
// length, as `Overlay::Arrow`.
float3 arrowHead(Batch b, float3 start, float3 end) {
    let shaft = end - start;
    let length = sqrt(dot(shaft, shaft));
    let head = min(b.params.x > 0.0 ? b.params.x : 0.2 * length, length);
    return length > 0.0 ? shaft / length * head : float3(0.0);
}

[shader("vertex")]
VertexOutput batchSegmentVertex(uint vertexIndex : SV_VertexID) {
    let b = batches[constants.batch.x];
    let segment = vertexIndex / 6;
    let i = batchItem(b, segment / b.shape.z);
    if (!batchInUse(b, i)) {
        return culled();
    }
    let k = segment % b.shape.z;  // the item's segment
    let p = batchPoint(b, b.arrays.x, i);
    Segment s;
    s.info = batchInfo(b, i);
    s.extra = uint4(0, i, 0, 0);
    s.a = float4(p, batchSize(b, i));
    s.b = float4(p, b.params.z);
    // Lines' and arrows' colours at both ends, rows 2i and 2i + 1.
    if ((b.shape.w & kBatchEndColors) != 0) {
        s.info.x = batchColorAt(b, 2 * i);
        s.extra.x = batchColorAt(b, 2 * i + 1);
        s.extra.z = kExtraEndColor;
    }
    switch (b.shape.x) {
    case kBatchLines:
        s.b.xyz = batchEnd(b, i, p);
        break;
    case kBatchArrows: {
        let end = batchEnd(b, i, p);
        s.b.xyz = end - arrowHead(b, p, end);
        break;
    }
    case kBatchTrails: {
        // Segment k from the newest point back, round the ring; one with a NaN end (a point not
        // yet written) is left out.
        let length = b.more.w;
        let head = (b.shape.w & kBatchHasHeads) != 0
                       ? uint(max(asint(batchData[b.more.x + i]), 0)) % length
                       : length - 1;
        let newer = (head + length - k) % length;
        let older = (head + 2 * length - k - 1) % length;
        let a = batchVector(b.arrays.x, i * length + newer);
        let c = batchVector(b.arrays.x, i * length + older);
        if (any(isnan(a)) || any(isnan(c))) {
            return culled();
        }
        s.a.xyz = a + b.origin.xyz;
        s.b.xyz = c + b.origin.xyz;
        // Its points' colours, blended between them, or the item's at both.
        s.extra.x = s.info.x;
        s.extra.z = kExtraEndColor;
        if ((b.shape.w & kBatchPointColors) != 0) {
            s.info.x = batchColorAt(b, i * length + newer);
            s.extra.x = batchColorAt(b, i * length + older);
        }
        if ((b.shape.w & kBatchFade) != 0) {
            let span = float(max(length - 1, 1));
            var newest = unpackColor(s.info.x);
            newest.a *= 1.0 - float(k) / span;
            s.info.x = packColor(newest);
            var oldest = unpackColor(s.extra.x);
            oldest.a *= 1.0 - float(k + 1) / span;
            s.extra.x = packColor(oldest);
        }
        break;
    }
    case kBatchCapsules: {
        float3 a;
        float3 c;
        capsuleSegment(batchData[b.arrays.y + i], batchData[b.more.x + i], k, a, c);
        let q = batchOrientation(b, i);
        s.a.xyz = p + rotate(q, a);
        s.b.xyz = p + rotate(q, c);
        break;
    }
    case kBatchBoxes: {
        // Four edges along each axis, at the corners of the other two.
        let half = batchVector(b.arrays.y, i);
        let axis = k / 4;
        let other = uint2((axis + 1) % 3, (axis + 2) % 3);
        var low = float3(0.0);
        low[other.x] = (k & 1) != 0 ? 1.0 : -1.0;
        low[other.y] = (k & 2) != 0 ? 1.0 : -1.0;
        var high = low;
        low[axis] = -1.0;
        high[axis] = 1.0;
        let q = batchOrientation(b, i);
        s.a.xyz = p + rotate(q, low * half);
        s.b.xyz = p + rotate(q, high * half);
        break;
    }
    case kBatchSpheres: {
        // A great circle around each axis.
        let radius = batchData[b.arrays.y + i];
        let axis = k / kCircleSegments;
        let j = k % kCircleSegments;
        let u = axis == 0 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
        let v = axis == 2 ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0);
        let a0 = 2.0 * 3.14159265 * float(j) / float(kCircleSegments);
        let a1 = 2.0 * 3.14159265 * float(j + 1) / float(kCircleSegments);
        s.a.xyz = p + radius * (cos(a0) * u + sin(a0) * v);
        s.b.xyz = p + radius * (cos(a1) * u + sin(a1) * v);
        break;
    }
    case kBatchFrames: {
        // Axis k in `Overlay::Axes`' red, green or blue, with the item's opacity.
        static const float3 kAxisColors[3] = { float3(0.9, 0.2, 0.2), float3(0.3, 0.85, 0.3),
                                               float3(0.25, 0.45, 1.0) };
        var unit = float3(0.0);
        unit[k] = 1.0;
        s.b.xyz = p + rotate(batchOrientation(b, i), unit) * b.params.y;
        s.info.x = packColor(float4(kAxisColors[k], unpackColor(s.info.x).a));
        break;
    }
    default:
        break;
    }
    return segmentCorner(s, vertexIndex);
}

// Arrow heads: cones of kConeSides sides and their bases.
[shader("vertex")]
VertexOutput batchTriangleVertex(uint vertexIndex : SV_VertexID) {
    let b = batches[constants.batch.x];
    let triangle = vertexIndex / 3;
    let i = batchItem(b, triangle / (2 * kConeSides));
    if (!batchInUse(b, i)) {
        return culled();
    }
    let k = triangle % (2 * kConeSides);
    let start = batchPoint(b, b.arrays.x, i);
    let end = batchEnd(b, i, start);
    let head = arrowHead(b, start, end);
    let length = sqrt(dot(head, head));
    if (!(length > 0.0)) {
        return culled();
    }
    let base = end - head;
    float3 u;
    float3 v;
    perpendiculars(head / length, u, v);
    let radius = 0.35 * length;
    let side = k % kConeSides;
    let a0 = 2.0 * 3.14159265 * float(side) / float(kConeSides);
    let a1 = 2.0 * 3.14159265 * float(side + 1) / float(kConeSides);
    let p0 = base + radius * (cos(a0) * u + sin(a0) * v);
    let p1 = base + radius * (cos(a1) * u + sin(a1) * v);
    Triangle t;
    t.info = batchInfo(b, i);
    t.extra = uint4(0, i, 0, 0);
    // An arrow's head in its end's colour.
    if ((b.shape.w & kBatchEndColors) != 0) {
        t.info.x = batchColorAt(b, 2 * i + 1);
    }
    if (k < kConeSides) {
        t.p0 = float4(end, 0.0);
        t.p1 = float4(p0, 0.0);
        t.p2 = float4(p1, 0.0);
    } else {
        t.p0 = float4(base, 0.0);
        t.p1 = float4(p1, 0.0);
        t.p2 = float4(p0, 0.0);
    }
    return triangleCorner(t, vertexIndex);
}

// Markers: squares of their size in pixels around their points.
[shader("vertex")]
VertexOutput batchSpriteVertex(uint vertexIndex : SV_VertexID) {
    let b = batches[constants.batch.x];
    let i = batchItem(b, vertexIndex / 6);
    if (!batchInUse(b, i)) {
        return culled();
    }
    let half = 0.5 * batchSize(b, i);
    Sprite s;
    s.extra = uint4(0, i, 0, 0);
    s.anchor = float4(batchPoint(b, b.arrays.x, i), 0.0);
    s.rect = float4(-half, -half, half, half);
    s.info = batchInfo(b, i);
    return spriteCorner(s, vertexIndex);
}

// The font's words of character `c` (a Latin-1 code; the font's '?' stands in for what it lacks).
uint glyphBase(uint c) {
    return kGlyphWords * min(c, 255u);
}

// How far character `c` moves the pen, in units.
float glyphAdvance(uint c) {
    return asfloat(glyphs[glyphBase(c) + 4]);
}

// The pen's move between characters `a` and `b` besides `a`'s advance, in units.
float glyphKerning(uint a, uint b) {
    if (a < kKernFirst || b < kKernFirst || a > 255 || b > 255) {
        return 0.0;
    }
    let pair = (a - kKernFirst) * kKernCodes + (b - kKernFirst);
    let byte = (glyphs[kKernBase + pair / 4] >> (8 * (pair % 4))) & 0xff;
    return float(int(byte << 24) >> 24) / 16.0;
}

// The text's height in a line, in units: from the top of its capitals to the bottom of its
// descenders.
float textHeight() {
    return asfloat(glyphs[kFontBase]);
}

// The box character `c`'s field covers: x0, y0, x1, y1 in units from the pen; empty for one with
// no outline, such as a space.
float4 glyphBox(uint c) {
    let base = glyphBase(c);
    return float4(asfloat(glyphs[base]), asfloat(glyphs[base + 1]), asfloat(glyphs[base + 2]),
                  asfloat(glyphs[base + 3]));
}

// Numbers' longest number but its decimals: a sign, ten digits and the point.
static const uint kNumberChars = 12;

// A value as a number's text prints it (`OverlayNumberFormat`): its prefix, the value with `text.w`
// decimals (or "nan", or "inf" past 4e9 units of its last decimal), its suffix. Or a texts' item's
// row of character codes, up to its first 0.
struct NumberText {
    uint length;        // characters in all
    uint first;         // the number's first character
    uint count;         // the number's characters
    uint scaled;        // the value's magnitude in units of its last decimal
    uint digits;        // the number's digits, its integer part's and its decimals
    uint whole;         // its integer part's digits
    bool negative;
    uint special;       // 0 a number, 1 nan, 2 inf
    uint row;           // a texts' item's first character code in `batchData`
}

// Item `i`'s text.
NumberText numberText(Batch b, uint i) {
    NumberText t;
    t.row = 0;
    if (b.shape.x == kBatchTexts) {
        // Codes `text.y` to a row, ending at the first 0.
        t.row = b.text.x + i * b.text.y;
        t.length = 0;
        while (t.length < b.text.y && asint(batchData[t.row + t.length]) != 0) {
            ++t.length;
        }
        t.first = t.length;
        t.count = 0;
        t.negative = false;
        t.special = 0;
        return t;
    }
    let value = batchData[b.arrays.y + i];
    let decimals = b.text.w;
    var unit = 1.0;
    for (uint d = 0; d < decimals; ++d) {
        unit *= 10.0;
    }
    let magnitude = abs(value) * unit;
    t.special = isnan(value) ? 1 : (!(magnitude < 4.0e9) ? 2 : 0);
    t.scaled = t.special == 0 ? uint(round(magnitude)) : 0;
    t.negative = value < 0.0 && (t.special == 2 || t.scaled != 0);
    var integer = t.scaled;
    for (uint d = 0; d < decimals; ++d) {
        integer /= 10;
    }
    t.whole = 1;
    for (var rest = integer / 10; rest > 0; rest /= 10) {
        ++t.whole;
    }
    t.digits = t.whole + decimals;
    t.count = (t.negative ? 1 : 0) +
              (t.special != 0 ? 3 : t.digits + (decimals > 0 ? 1 : 0));
    t.first = b.text.y;
    t.length = b.text.y + t.count + b.text.z;
    return t;
}

// Character `k` of a number's text.
uint numberChar(Batch b, NumberText t, uint k) {
    if (b.shape.x == kBatchTexts) {
        return uint(asint(batchData[t.row + k]));
    }
    if (k < t.first) {
        return uint(batchData[b.text.x + k]);
    }
    if (k >= t.first + t.count) {
        return uint(batchData[b.text.x + b.text.y + k - t.first - t.count]);
    }
    var j = k - t.first;
    if (t.negative) {
        if (j == 0) {
            return 45;  // '-'
        }
        --j;
    }
    if (t.special != 0) {
        static const uint kNan[3] = { 110, 97, 110 };
        static const uint kInf[3] = { 105, 110, 102 };
        return t.special == 1 ? kNan[j] : kInf[j];
    }
    if (j == t.whole) {
        return 46;  // '.'
    }
    let q = j < t.whole ? j : j - 1;  // the digit, most significant first
    var power = 1u;
    for (uint p = q + 1; p < t.digits; ++p) {
        power *= 10;
    }
    return 48 + (t.scaled / power) % 10;
}

// Numbers: per item a backing box, then a sprite per character of its text, formatted from its
// value here, beside its point as `Overlay::Label`'s or placed on the screen as `Overlay::Text`.
[shader("vertex")]
VertexOutput batchNumberVertex(uint vertexIndex : SV_VertexID) {
    let b = batches[constants.batch.x];
    let sprite = vertexIndex / 6;
    let i = batchItem(b, sprite / b.shape.z);
    if (!batchInUse(b, i)) {
        return culled();
    }
    let k = sprite % b.shape.z;
    let t = numberText(b, i);
    let size = batchSize(b, i);
    let scale = size / kLineHeight;
    var width = 0.0;
    var pen = 0.0;
    var next = t.length > 0 ? numberChar(b, t, 0) : 0;
    for (uint c = 0; c < t.length; ++c) {
        let code = next;
        next = c + 1 < t.length ? numberChar(b, t, c + 1) : 0;
        var advance = glyphAdvance(code);
        if (c + 1 < t.length) {
            advance += glyphKerning(code, next);
        }
        if (c + 1 < k) {
            pen += advance;
        }
        width += advance;
    }
    let box = float2(width, textHeight()) * scale;
    let backing = b.style.w;
    // A halo (`params.x` pixels) in the backing colour around each character in place of the box.
    let halo = unpackColor(backing).a > 0.0 ? b.params.x : 0.0;
    let pad = unpackColor(backing).a > 0.0 && halo <= 0.0 ? 0.25 * size : 0.0;
    let screen = (b.shape.w & kBatchScreen) != 0;
    let points = (b.shape.w & kBatchHasPoints) != 0;
    Sprite s;
    s.info = batchInfo(b, i);
    s.extra = uint4(0, i, 0, 0);
    float2 corner;
    if (screen) {
        let at = points ? batchPoint(b, b.arrays.x, i).xy : b.place.xy;
        var offset = points ? float2(0.0) : b.place.zw;
        offset.y += points ? 0.0 : float(i) * (kLineHeight * scale + 2.0 * pad);
        corner = offset - b.params.zw * (box + 2.0 * pad) + pad;
        s.anchor = float4(at, 0.0, 0.0);
        s.info.y = kDepthAlways | (1u << 24);
        s.info.w = 0;
    } else {
        corner = float2(0.5 * size, -box.y - 3.0 * scale - 0.25 * size);
        s.anchor = float4(batchPoint(b, b.arrays.x, i), 0.0);
        s.info.y &= 0xffffu | kInfoMetric;
    }
    if (k == 0) {
        if (pad == 0.0) {
            return culled();
        }
        s.rect = float4(corner - pad, corner + box + pad);
        s.info.x = backing;
        s.info.y = (s.info.y & ~0xff00u) | (kShapeSquare << 8);
        return spriteCorner(s, vertexIndex);
    }
    if (k > t.length) {
        return culled();
    }
    let c = numberChar(b, t, k - 1);
    let glyph = glyphBox(c);
    if (!(glyph.z > glyph.x)) {
        return culled();
    }
    s.rect = float4(corner + (float2(pen, 0.0) + glyph.xy) * scale,
                    corner + (float2(pen, 0.0) + glyph.zw) * scale);
    s.info.y |= c << 16;
    if (halo > 0.0) {
        s.extra = uint4(backing, i, kExtraHalo, asuint(halo));
    }
    return spriteCorner(s, vertexIndex);
}

// An image (`OverlayBatch::image`) on its quad: in the world between its four corners, the rows of
// its points (top left, top right, bottom right, bottom left), or on the screen at its place
// (`place` xy, offset zw), `params.xy` pixels (its own size when 0), its point `params.zw` there.
// Two triangles; the fragment shader reads the image at `shape.xy`.
[shader("vertex")]
VertexOutput batchImageVertex(uint vertexIndex : SV_VertexID) {
    let b = batches[constants.batch.x];
    if (!batchInUse(b, 0)) {
        return culled();
    }
    static const uint kQuad[6] = { 0, 1, 2, 0, 2, 3 };
    static const float2 kUv[4] = { float2(0, 0), float2(1, 0), float2(1, 1), float2(0, 1) };
    let corner = kQuad[vertexIndex % 6];
    let uv = kUv[corner];
    let info = batchInfo(b, 0);
    if (!inWorld(info)) {
        return culled();
    }
    VertexOutput out;
    if ((b.shape.w & kBatchScreen) != 0) {
        let size = b.params.x > 0.0 && b.params.y > 0.0 ? b.params.xy
                                                        : float2(float(b.text.y), float(b.text.z));
        let pixel = b.place.zw - b.params.zw * size + uv * size;
        let at = float2(b.place.x * 2.0 - 1.0, 1.0 - b.place.y * 2.0);
        out.position = float4(at + pixel / pixelScale(), 0.5, 1.0);
        out.info = uint4(kKindImage, kDepthAlways, 0, 0);
    } else {
        out.position = project(anchored(info, batchVector(b.arrays.x, corner) + b.origin.xyz));
        out.info = uint4(kKindImage, info.y & 0xff, 0, 0);
    }
    out.color = unpackColor(info.x);
    // uv / w and 1 / w, which vary linearly across the image (`shape` is noperspective): the
    // fragment shader divides them out for the perspective-correct uv.
    out.shape = float4(uv / out.position.w, 1.0 / out.position.w, 0.0);
    out.span = float4(0.0);
    return out;
}

// Byte `index` of the bytes from word `first` of `batchData`, little-endian.
uint batchByte(uint first, uint index) {
    return (asuint(batchData[first + index / 4]) >> (8 * (index % 4))) & 0xff;
}

// Texel (x, y) of a batch's image (`text`: x its first word, y width, z height, w channels), as
// display colour and opacity in [0, 1].
float4 imageTexel(Batch b, int2 texel) {
    let at = uint2(clamp(texel, int2(0), int2(int(b.text.y) - 1, int(b.text.z) - 1)));
    let first = (at.y * b.text.y + at.x) * b.text.w;
    return float4(float(batchByte(b.text.x, first)), float(batchByte(b.text.x, first + 1)),
                  float(batchByte(b.text.x, first + 2)),
                  b.text.w == 4 ? float(batchByte(b.text.x, first + 3)) : 255.0) / 255.0;
}

// A batch's image at `uv` ([0, 1] from its top left), filtered between its four nearest texels.
float4 sampleImage(Batch b, float2 uv) {
    let p = uv * float2(float(b.text.y), float(b.text.z)) - 0.5;
    let low = int2(floor(p));
    let f = p - floor(p);
    let top = lerp(imageTexel(b, low), imageTexel(b, low + int2(1, 0)), f.x);
    let bottom = lerp(imageTexel(b, low + int2(0, 1)), imageTexel(b, low + int2(1, 1)), f.x);
    return lerp(top, bottom, f.y);
}

// Texel (x, y) of character `c`'s distance field, in [0, 1]; 0 (far outside) past its edges.
float fieldTexel(uint first, uint2 size, int2 texel) {
    if (any(texel < int2(0)) || any(texel >= int2(size))) {
        return 0.0;
    }
    let at = first + uint(texel.y) * size.x + uint(texel.x);
    return float((glyphs[kFieldBase + at / 4] >> (8 * (at % 4))) & 0xff) / 255.0;
}

// Character `c`'s distance field at `q` (texels from its top left), between its four nearest
// texels, in texels from its outline: positive inside.
float fieldDistance(uint first, uint2 size, float2 q) {
    let p = q - 0.5;
    let low = int2(floor(p));
    let f = p - floor(p);
    let top = lerp(fieldTexel(first, size, low), fieldTexel(first, size, low + int2(1, 0)), f.x);
    let bottom = lerp(fieldTexel(first, size, low + int2(0, 1)),
                      fieldTexel(first, size, low + int2(1, 1)), f.x);
    return (lerp(top, bottom, f.y) * 255.0 - 128.0) / kFieldLevels;
}

// Coverage of character `c` at `p` (in [-1, 1] over the sprite, which spans its field's box), and
// of its halo `halo` pixels wide: the field's distance in pixels at four taps across the
// fragment's footprint, each covered within half a pixel of the outline.
float2 glyphCoverage(uint c, float2 p, float halo) {
    let base = glyphBase(c);
    let first = glyphs[base + 5];
    let size = uint2(glyphs[base + 6] & 0xffff, glyphs[base + 6] >> 16);
    let q = (p * 0.5 + 0.5) * float2(size);
    let footprint = fwidth(q);
    // Pixels per texel: the field's distances are in texels.
    let pixels = 1.0 / max(0.5 * (footprint.x + footprint.y), 1e-6);
    var covered = float2(0.0);
    for (uint tap = 0; tap < 4; ++tap) {
        let at = q + (float2(float(tap & 1), float(tap >> 1)) - 0.5) * 0.5 * footprint;
        let d = fieldDistance(first, size, at) * pixels;
        covered += saturate(float2(d, d + halo) + 0.5);
    }
    return 0.25 * covered;
}

// View depth (distance along the view axis) of a [0, 1] depth: from a perspective projection, or
// linear in it from an orthographic one.
float linearDepth(float depth) {
    // Reverse-Z (`ReversedPerspective`, `ReversedOrthographic`): 1 at the near plane, 0 at the far,
    // which may be infinite (an orthographic one then 10^12 past the near plane).
    let near = constants.viewport.z;
    let far = constants.viewport.w;
    if (constants.view.w != 0) {
        return lerp(isinf(far) ? near + 1e12 : far, near, depth);
    }
    return isinf(far) ? near / depth : near * far / (near + depth * (far - near));
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
    var color = shade(in);
    if (color.a <= 0.0) {
        discard;
    }
    if (constants.view.z == 1) {
        color.rgb = float3(decodeSrgb(color.r), decodeSrgb(color.g), decodeSrgb(color.b));
    }
    return color;
}

// What a pick finds at its pixel: the item drawn last there with a third or more of its opacity,
// as x what drew it (`constants.layer.w`), y its id (a host item's `OverlayStyle::id`, a batch
// item's index), z 1; all 0 where nothing is.
[shader("fragment")]
uint4 overlayPickFragment(VertexOutput in) : SV_Target {
    if (shade(in).a < 0.3 * in.color.a) {
        discard;
    }
    return uint4(constants.layer.w, in.info.w, 1, 0);
}

// An item's colour and opacity at a fragment: its coverage, and how its depth mode meets the
// view's surfaces.
float4 shade(VertexOutput in) {
    var color = in.color;
    if (in.info.x == kKindSegment) {
        // Round ends: the distance to the segment from its start to its end, so joined segments
        // meet without notches.
        let past = max(0.0, max(-in.shape.z, in.shape.z - in.span.x));
        color.a *= saturate(in.shape.y + 0.5 - length(float2(past, in.shape.x)));
        // Dashes from the segment's start, each as long as the gap after it.
        if (in.shape.w > 0.0) {
            let phase = in.shape.z - floor(in.shape.z / (2.0 * in.shape.w)) * 2.0 * in.shape.w;
            color.a *= saturate(min(phase, in.shape.w - phase) + 0.5);
        }
    } else if (in.info.x == kKindSprite) {
        let character = in.info.z >> 8;
        if (character != 0) {
            // The character over its halo, if it has one (`span` its colour).
            let covered = glyphCoverage(character, in.shape.xy, in.shape.z);
            let fill = color.a * covered.x;
            let halo = in.span.a * (in.span.a > 0.0 ? covered.y : 0.0) * (1.0 - fill);
            let alpha = fill + halo;
            color = float4(alpha > 0.0 ? (color.rgb * fill + in.span.rgb * halo) / alpha
                                       : color.rgb,
                           alpha);
        } else {
            color.a *= markerCoverage(in.info.z & 0xff, in.shape.xy, in.shape.zw);
        }
    } else if (in.info.x == kKindImage) {
        color *= sampleImage(batches[constants.batch.x], in.shape.xy / in.shape.z);
    }
    let mode = in.info.y;
    if (mode != kDepthAlways) {
        // The view's pixel: a pick's, moved to the target's first.
        let pixel = int2(in.position.xy) + int2(constants.layer.yz);
        let surface = linearDepth(sceneDepth.Load(int4(pixel, constants.view.y, 0)));
        let depth = linearDepth(in.position.z);
        // A little slack, so lines drawn on surfaces stay in front of them.
        if (depth > surface * 1.003 + 1e-4) {
            if (mode == kDepthTest) {
                return float4(0.0);
            }
            color.a *= 0.3;
        }
    }
    return color;
}
