// SPDX-License-Identifier: BSD-3-Clause
// Lasers' beams crossed by a pixel's ray, in closed form, for the media that scatter them
// (volume.slang's dust and smoke, shading.slang's air): a beam millimetres thick then lights the
// medium once per crossing, not by the chance of a march's step landing in it.
//
// A beam is in pieces (`beamPiece`): a spot's beam; a grid's zero order on the axis and its
// beamlets; a line's zero order and its sheet; a scan's zero order and the sheets its strokes
// sweep, a stroke's light spread over its sheet's angle as the beam sweeps it. Each beamlet is a Gaussian beam from the aperture,
// of power p and 1/e^2 radius w across the common normal of the ray and its axis there, whose
// light along the ray, closest d to the axis at angle a, is
// p sqrt(2 / pi) exp(-2 d^2 / w^2) / (w sin a), averaged over the pixel's width across that
// normal. A sheet carries, at fan angle a and distance r from the aperture, the fan's power per
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
    public float kind;       // 1 a spot, 2 a grid, 3 a line, 4 a scan
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

// How many pieces `beam` has.
public uint beamPieces(Beam beam) {
    if (beam.kind < 1.5) {
        return 1;
    }
    if (beam.kind > 3.5) {
        return 1 + kScanStrokes;
    }
    if (beam.kind > 2.5) {
        return 2;
    }
    return 1 + uint(beam.params.x) * uint(beam.params.y);
}

// Where the ray from `origin` along unit `direction` crosses piece `i` of `beam`, the pixel
// `footprintX` and `footprintY` across per unit distance: the zero order first (a spot's whole
// beam, of power `zeroOrder` plus `efficiency`), then a line's sheet or a grid's beamlets
// (light_pattern.cpp's LaserGrid: p0 x p1, p2 and p3 radians apart), sharing `efficiency`.
public BeamCrossing beamPiece(Beam beam, uint i, float zeroOrder, float efficiency, float3 origin,
                              float3 direction, float3 footprintX, float3 footprintY) {
    if (i == 0) {
        let share = beam.kind < 1.5 ? zeroOrder + efficiency : zeroOrder;
        return beamletCrossing(beam, float2(0.0), share, origin, direction, footprintX,
                               footprintY);
    }
    if (beam.kind > 3.5) {
        return strokeCrossing(beam, i - 1, efficiency / float(kScanStrokes), origin, direction);
    }
    if (beam.kind > 2.5) {
        return sheetCrossing(beam, efficiency, origin, direction, footprintX, footprintY);
    }
    let cells = uint2(beam.params.xy);
    let cell = i - 1;
    let centre = tan((float2(cell / cells.y, cell % cells.y) - 0.5 * (float2(cells) - 1.0)) *
                     beam.params.zw);
    return beamletCrossing(beam, centre, efficiency / float(max(cells.x * cells.y, 1u)), origin,
                           direction, footprintX, footprintY);
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
