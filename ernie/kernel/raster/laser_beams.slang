// SPDX-License-Identifier: BSD-3-Clause
// Lasers' beams crossed by a pixel's ray, in closed form, for the media that scatter them
// (volume.slang's dust and smoke, shading.slang's air): a beam millimetres thick then lights the
// medium once per crossing, not by the chance of a march's step landing in it.
//
// A beam is in pieces, a ray's crossings walked in turn (`nextCrossing`): a spot's beam; a grid's
// zero order on the axis and its beamlets; a line's zero order and its sheet; a scan's zero order
// and the sheets its strokes sweep, a stroke's light spread over its sheet's angle as the beam
// sweeps it; a field of dots' zero order and the dots whose cells the ray's track on the pattern's
// plane crosses. Each beamlet is a Gaussian beam from the aperture, of power p and 1/e^2 radius
// w across the common normal of the ray and its axis there, whose light along the ray, closest d
// to the axis at angle a, is p sqrt(2 / pi) exp(-2 d^2 / w^2) / (w sin a), averaged over the
// pixel's width across that normal. A sheet carries, at fan angle a and distance r from the aperture, the fan's power per
// radian over r per unit length, Gaussian across its thickness, so along a ray at cosine c to its
// normal it gives that over |c|, whatever its thickness.
//
// Reference: A. E. Siegman, Lasers, University Science Books, 1986, ch. 17 (the Gaussian beam).
module laser_beams;

import light_pattern;

static const float kPi = 3.14159265358979;

// The most beamlets of a grid integrated exactly (the renderer's `patternExtent[2]`).
public static const uint kMaxBeamlets = 64;

// A laser's beam, as its `LightGpu` holds it.
public struct Beam {
    public float3 position;  // the aperture
    public float3 axis;      // unit
    public float3 x;         // unit, the pattern's x
    public float4 beam;      // 1/e^2 waist radii (xy) and Rayleigh ranges (zw), m
    public float waistAt;    // the waist's distance along the axis
    public float kind;       // 1 a spot, 2 a grid, 3 a line, 4 a scan, 5 dots
    public float4 params;    // the pattern's first parameters
    public float4 more;      // and its next
}

// A scan's strokes (light_pattern.cpp's LaserScan, as many).
static const uint kScanStrokes = 64;

// Where a ray crosses a piece of a beam: `t` along it (0 for nowhere), the light it gathers there
// per unit of the laser's spectrum (its irradiance integrated along the ray, W m^-1 per W), and
// towards the laser (xyz) and how far (w).
public struct BeamCrossing {
    public float t;
    public float value;
    public float4 towards;
}

BeamCrossing noCrossing() {
    BeamCrossing none;
    none.t = 0.0;
    none.value = 0.0;
    none.towards = float4(0.0);
    return none;
}

// A ray's walk over a beam's pieces (`beamWalk`, `nextCrossing`).
public struct BeamWalk {
    uint next;    // the next piece, the zero order first
    uint pieces;  // a spot's, a grid's, a line's or a scan's
    // A field of dots: the cell of its grid the ray's track is in, the way it steps along each
    // axis, the track's parameter at the next cell boundary along each and between boundaries, as
    // Amanatides and Woo's walk; the cells left.
    int2 cell;
    int2 step;
    float2 boundary;
    float2 spacing;
    uint left;
}

// The most cells of a field of dots a ray's track walks.
static const uint kMaxDotCells = 512;

// The 32-bit integer hash light_pattern.cpp's LaserDots places its dots by, Wellons' lowbias32.
//
// Reference: C. Wellons, "Prospecting for Hash Functions", 2018,
// https://nullprogram.com/blog/2018/07/31/
uint dotHash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Where in its cell, from 0 to 1 along each axis, the dot of cell `cell` of a tile lies (as
// LaserDots's `dotPlace`).
float2 dotPlace(uint2 cell, uint seed) {
    let h = dotHash(cell.x + dotHash(cell.y + dotHash(seed)));
    return float2(h & 0xffffu, h >> 16) / 65536.0;
}

// A field of dots' cells along each axis and their size on the tangent plane (LaserDots: p0 x p1
// a tile, p2 x p3 tiles over |t.x| < p5, |t.y| < p6).
float2 dotCells(Beam beam) {
    return beam.params.xy * beam.params.zw;
}

// The walk of the ray from `origin` along unit `direction`, as far as `far`, over `beam`'s pieces:
// for a field of dots, the ray's track on the tangent plane (a straight line, as the plane's
// central projection keeps lines), clipped to the field, and its first cell.
//
// Reference: J. Amanatides and A. Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing",
// Eurographics 1987 (the walk over the cells).
public BeamWalk beamWalk(Beam beam, float3 origin, float3 direction, float far) {
    BeamWalk walk;
    walk.next = 0;
    walk.pieces = beam.kind < 1.5   ? 1
                  : beam.kind < 2.5 ? 1 + uint(beam.params.x) * uint(beam.params.y)
                  : beam.kind < 3.5 ? 2
                  : beam.kind < 4.5 ? 1 + kScanStrokes
                                    : 1;
    walk.cell = int2(0);
    walk.step = int2(0);
    walk.boundary = float2(0.0);
    walk.spacing = float2(0.0);
    walk.left = 0;
    if (beam.kind < 4.5) {
        return walk;
    }
    // Where the ray lies in front of the aperture, a millimetre at least.
    let y = cross(beam.x, beam.axis);
    let o = origin - beam.position;
    let a0 = dot(o, beam.axis);
    let ad = dot(direction, beam.axis);
    var start = 0.0;
    var end = far;
    if (abs(ad) < 1e-9) {
        if (a0 < 1e-3) {
            return walk;
        }
    } else if (ad > 0.0) {
        start = max(start, (1e-3 - a0) / ad);
    } else {
        end = min(end, (1e-3 - a0) / ad);
    }
    if (!(end > start)) {
        return walk;
    }
    // Its track's ends on the tangent plane: past a far end, where the ray heads (its vanishing
    // point).
    let at = o + direction * start;
    let t0 = float2(dot(at, beam.x), dot(at, y)) / dot(at, beam.axis);
    float2 t1;
    if (end > 1e20 && ad > 0.0) {
        t1 = float2(dot(direction, beam.x), dot(direction, y)) / ad;
    } else {
        let to = o + direction * min(end, 1e20);
        t1 = float2(dot(to, beam.x), dot(to, y)) / max(dot(to, beam.axis), 1e-3);
    }
    // In grid units, clipped to the field (Liang and Barsky's clip).
    let half = beam.more.yz;
    let cells = dotCells(beam);
    let g0 = (t0 + half) / (2.0 * half) * cells;
    let g1 = (t1 + half) / (2.0 * half) * cells;
    let delta = g1 - g0;
    var lo = 0.0;
    var hi = 1.0;
    for (uint axis = 0; axis < 2; ++axis) {
        if (abs(delta[axis]) < 1e-12) {
            if (g0[axis] < 0.0 || g0[axis] > cells[axis]) {
                return walk;
            }
            continue;
        }
        let ta = (0.0 - g0[axis]) / delta[axis];
        let tb = (cells[axis] - g0[axis]) / delta[axis];
        lo = max(lo, min(ta, tb));
        hi = min(hi, max(ta, tb));
    }
    if (!(hi > lo)) {
        return walk;
    }
    let enter = g0 + delta * lo;
    let exit = g0 + delta * hi;
    walk.cell = clamp(int2(floor(enter)), int2(0), int2(cells) - 1);
    let last = clamp(int2(floor(exit)), int2(0), int2(cells) - 1);
    walk.step = int2(sign(delta));
    let span = exit - enter;
    for (uint axis = 0; axis < 2; ++axis) {
        if (walk.step[axis] == 0 || abs(span[axis]) < 1e-12) {
            walk.boundary[axis] = 1e30;
            walk.spacing[axis] = 1e30;
            walk.step[axis] = 0;
            continue;
        }
        let edge = float(walk.cell[axis] + (walk.step[axis] > 0 ? 1 : 0));
        walk.boundary[axis] = (edge - enter[axis]) / span[axis];
        walk.spacing[axis] = abs(1.0 / span[axis]);
    }
    walk.left = min(uint(abs(last.x - walk.cell.x) + abs(last.y - walk.cell.y)) + 1, kMaxDotCells);
    return walk;
}

// The walk's next crossing in `crossing` (its value 0 where the ray passes the piece), false when
// it has none left: the zero order first (a spot's whole beam, of power `zeroOrder` plus
// `efficiency`), then a line's sheet, a scan's strokes' sheets, a grid's beamlets
// (light_pattern.cpp's LaserGrid: p0 x p1, p2 and p3 radians apart) or the dots of the cells the
// walk crosses, sharing `efficiency`; the pixel `footprintX` and `footprintY` across per unit
// distance.
public bool nextCrossing(inout BeamWalk walk, Beam beam, float zeroOrder, float efficiency,
                         float3 origin, float3 direction, float3 footprintX, float3 footprintY,
                         out BeamCrossing crossing) {
    if (walk.next == 0) {
        walk.next = 1;
        let share = beam.kind < 1.5 ? zeroOrder + efficiency : zeroOrder;
        crossing = beamletCrossing(beam, float2(0.0), share, origin, direction, footprintX,
                                   footprintY);
        return true;
    }
    if (beam.kind > 4.5) {
        if (walk.left == 0) {
            crossing = noCrossing();
            return false;
        }
        --walk.left;
        // The cell's dot (as LaserDots places it), then on to the next cell.
        let cells = dotCells(beam);
        let half = beam.more.yz;
        let count = beam.params.xy;
        let at = float2(walk.cell);
        let local = uint2(at - count * floor(at / count));
        let offset = (dotPlace(local, uint(beam.more.w)) - 0.5) * beam.more.x;
        let centre = (at + 0.5 + offset) * (2.0 * half / cells) - half;
        crossing = beamletCrossing(beam, centre, efficiency / (cells.x * cells.y), origin,
                                   direction, footprintX, footprintY);
        let axis = walk.boundary.x < walk.boundary.y ? 0 : 1;
        walk.cell[axis] += walk.step[axis];
        walk.boundary[axis] += walk.spacing[axis];
        return true;
    }
    if (walk.next >= walk.pieces) {
        crossing = noCrossing();
        return false;
    }
    let i = walk.next;
    ++walk.next;
    if (beam.kind > 3.5) {
        crossing = strokeCrossing(beam, i - 1, efficiency / float(kScanStrokes), origin, direction);
    } else if (beam.kind > 2.5) {
        crossing = sheetCrossing(beam, efficiency, origin, direction, footprintX, footprintY);
    } else {
        let cells = uint2(beam.params.xy);
        let cell = i - 1;
        let centre = tan((float2(cell / cells.y, cell % cells.y) - 0.5 * (float2(cells) - 1.0)) *
                         beam.params.zw);
        crossing = beamletCrossing(beam, centre, efficiency / float(max(cells.x * cells.y, 1u)),
                                   origin, direction, footprintX, footprintY);
    }
    return true;
}

// Beamlet `centre` (on the tangent plane) of `beam`, carrying `share` of its power.
BeamCrossing beamletCrossing(Beam beam, float2 centre, float share, float3 origin,
                             float3 direction, float3 footprintX, float3 footprintY) {
    let y = cross(beam.x, beam.axis);
    let u = normalize(beam.x * centre.x + y * centre.y + beam.axis);
    // The closest points of the ray and the beamlet's axis, from their common normal's cross
    // products, which keep their precision for a ray nearly along the beam (a telescope beside a
    // guide star's laser sees it a third of a milliradian off), where 1 - (d.u)^2 loses it.
    let c = cross(direction, u);
    let denom = dot(c, c);
    if (denom < 1e-14 || !(share > 0.0)) {
        return noCrossing();
    }
    let w0 = beam.position - origin;
    let s = dot(cross(w0, u), c) / denom;
    let z = dot(cross(w0, direction), c) / denom;
    if (s <= 0.0 || z <= 0.0) {
        return noCrossing();
    }
    let normal = c * rsqrt(denom);
    let offset = dot(w0, normal);
    // The beam's radii there, from its waist along the axis.
    let along = z * dot(u, beam.axis) - beam.waistAt;
    let zr = max(beam.beam.zw, float2(1e-12));
    let w = beam.beam.xy * sqrt(1.0 + (along / zr) * (along / zr));
    let radius = length(float2(w.x * dot(normal, beam.x), w.y * dot(normal, y)));
    let width = abs(dot(footprintX, normal)) * s + abs(dot(footprintY, normal)) * s;
    if (abs(offset) > 3.0 * radius + width) {
        return noCrossing();
    }
    BeamCrossing crossing;
    crossing.t = s;
    crossing.value = share * beamAverage(offset, radius, width) / sqrt(denom);
    crossing.towards = float4(-u, z);
    return crossing;
}

// The sheet of line laser `beam`, carrying `share` of its power (light_pattern.cpp's LaserLine:
// fanned along x, half fan angle p0, flat or Gaussian for p1 = 1, the beam's Gaussian across it).
// Every ray crosses the sheet's plane once.
BeamCrossing sheetCrossing(Beam beam, float share, float3 origin, float3 direction,
                           float3 footprintX, float3 footprintY) {
    let y = cross(beam.x, beam.axis);
    let cosine = dot(direction, y);
    if (abs(cosine) < 1e-4 || !(share > 0.0)) {
        return noCrossing();
    }
    let s = dot(beam.position - origin, y) / cosine;
    if (s <= 0.0) {
        return noCrossing();
    }
    let offset = origin + direction * s - beam.position;
    let along = dot(offset, beam.axis);
    if (along <= 0.0) {
        return noCrossing();
    }
    let r = length(offset);
    let a = atan(dot(offset, beam.x) / along);
    // The fan's density per radian, its edges softened by the beam's own width and the pixel's
    // angle from the aperture (as LaserLine's).
    let z = along - beam.waistAt;
    let zr = max(beam.beam.z, 1e-12);
    let w = beam.beam.x * sqrt(1.0 + (z / zr) * (z / zr));
    let pixel = 0.5 * (length(footprintX) + length(footprintY)) * s / r;
    let sigma2 = 0.25 * (w / along) * (w / along) + pixel * pixel / 12.0;
    let halfFan = beam.params.x;
    var density = 0.0;
    if (beam.params.y > 0.5) {
        let spread = sqrt(0.25 * halfFan * halfFan + sigma2);
        density = exp(-0.5 * a * a / (spread * spread)) / (sqrt(2.0 * kPi) * spread);
    } else {
        let spread = sqrt(max(sigma2, 1e-20));
        density = (normalBelow((a + halfFan) / spread) - normalBelow((a - halfFan) / spread)) /
                  (2.0 * halfFan);
    }
    // Looking along the sheet, its thickness bounds the path: at most a hundredth's cosine.
    BeamCrossing crossing;
    crossing.t = s;
    crossing.value = share * density / (r * max(abs(cosine), 0.01));
    crossing.towards = float4(-offset / r, r);
    return crossing;
}

// Where scan `beam` is at `s` of its period, on the tangent plane (light_pattern.cpp's
// `scanPoint`).
float2 scanPoint(Beam beam, float s) {
    let turn = 2.0 * kPi * s;
    return float2(beam.params.x * sin(turn * beam.params.z + beam.more.x),
                  beam.params.y * sin(turn * beam.params.w));
}

// The sheet scan `beam`'s stroke `stroke` sweeps from the aperture, carrying `share` of its power
// evenly over the stroke's angle (the beam's own thickness, which cancels, aside): a sheet as a
// line's, between the stroke's ends.
BeamCrossing strokeCrossing(Beam beam, uint stroke, float share, float3 origin,
                            float3 direction) {
    let y = cross(beam.x, beam.axis);
    let a = scanPoint(beam, float(stroke) / float(kScanStrokes));
    let b = scanPoint(beam, float(stroke + 1) / float(kScanStrokes));
    let u0 = normalize(beam.x * a.x + y * a.y + beam.axis);
    let u1 = normalize(beam.x * b.x + y * b.y + beam.axis);
    let normal = cross(u0, u1);
    let sine = length(normal);
    if (sine < 1e-7 || !(share > 0.0)) {
        return noCrossing();
    }
    let unit = normal / sine;
    let cosine = dot(direction, unit);
    if (abs(cosine) < 1e-6) {
        return noCrossing();
    }
    let s = dot(beam.position - origin, unit) / cosine;
    if (s <= 0.0) {
        return noCrossing();
    }
    let offset = origin + direction * s - beam.position;
    // Within the wedge between the stroke's ends.
    if (dot(offset, u0 + u1) <= 0.0 || dot(cross(u0, offset), normal) < 0.0 ||
        dot(cross(offset, u1), normal) < 0.0) {
        return noCrossing();
    }
    let r = length(offset);
    let angle = atan2(sine, dot(u0, u1));
    // Looking along the sheet, its thickness bounds the path: at most a hundredth's cosine.
    BeamCrossing crossing;
    crossing.t = s;
    crossing.value = share / (angle * r * max(abs(cosine), 0.01));
    crossing.towards = float4(-offset / r, r);
    return crossing;
}
