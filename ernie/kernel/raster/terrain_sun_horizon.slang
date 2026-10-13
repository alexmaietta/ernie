// SPDX-License-Identifier: BSD-3-Clause
// Terrain horizons towards the sun, for the sun's shadows over the whole terrain whatever the
// distance (shading.slang's `sunHorizonVisibility`), where the cascades' texels grow coarse.
//
// `heightMaxMain` builds a chain of the heights' maxima: level 0 the heights with no-data and NaN
// cells as kNoHeight, each level after the highest of the 2 x 2 cells under a texel (and of the
// odd cell left at an edge). `sunHorizonMain` then marches from each texel of the sun horizon map,
// a grid coarser than the heights by `scale` cells, towards the sun's azimuth, reading the maxima
// at steps a few per cent of the distance apart from the mip as wide as a step, so that no ridge
// between steps is missed. It keeps the point that sets the horizon's elevation that way, its
// height (before vertical exaggeration) and how far along the azimuth it lies (in cells, from the
// grid's origin), so that shading finds the horizon from each receiver's own place and height
// rather than the texel's centre; a height of kNoHeight with no ground there or nothing above the
// nadir. Only the azimuth counts: the map holds while the sun rises and sets along it.
//
// References:
// - N. L. Max, "Horizon mapping: shadows for bump-mapped surfaces", The Visual Computer 4(2),
//   1988. https://doi.org/10.1007/BF01905559
// - V. Timonen and J. Westerholm, "Scalable Height Field Self-Shadowing", Computer Graphics Forum
//   29(2):723-731, 2010 (horizons towards the light, for shadows over large height fields).
//   https://doi.org/10.1111/j.1467-8659.2009.01642.x

// `heightMaxMain`: the finer level (the heights, for level 0); `sunHorizonMain`: unused.
[[vk::binding(0, 0)]] Texture2D<float> finer;
// `heightMaxMain`: the level written.
[[vk::image_format("r32f")]] [[vk::binding(1, 0)]] RWTexture2D<float> target;
// `sunHorizonMain`: every level of the maxima, and the sun horizon map's layer to write (x the
// height, y the distance along).
[[vk::binding(2, 0)]] Texture2D<float> maxima;
[[vk::image_format("rg32f")]] [[vk::binding(3, 0)]] RWTexture2D<float2> horizon;

struct SunHorizonConstants {
    // The finer level's (or the heights') width and height, and the target's.
    uint4 size;
    // `sunHorizonMain`: cells per horizon texel and the cell size in world units;
    // `heightMaxMain`: the no-data value, and 1 when the heights have one (level 0 alone).
    float4 values;
    // `sunHorizonMain`: the unit direction towards the sun across the grid (columns east, rows
    // south), and the maxima's levels.
    float4 sun;
}
[[vk::push_constant]] ConstantBuffer<SunHorizonConstants> c;

// No ground, and below any horizon's tangent.
static const float kNoHeight = -1e30;
static const float kNone = -100.0;
// Each step this much of the distance marched, at least a cell.
static const float kGrowth = 0.015;

[shader("compute")]
[numthreads(8, 8, 1)]
void heightMaxMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= c.size.z || id.y >= c.size.w) {
        return;
    }
    // Level 0 copies the heights cell for cell; the others halve the finer level.
    let copy = c.size.x == c.size.z && c.size.y == c.size.w;
    var highest = kNoHeight;
    let first = copy ? int2(id.xy) : int2(id.xy) * 2;
    var last = first;
    if (!copy) {
        last = first + int2(1);
        // The odd cell a finer level leaves at its far edges joins the last texel.
        if (id.x + 1 == c.size.z && (c.size.x & 1u) != 0) {
            last.x += 1;
        }
        if (id.y + 1 == c.size.w && (c.size.y & 1u) != 0) {
            last.y += 1;
        }
        last = min(last, int2(c.size.xy) - 1);
    }
    for (int y = first.y; y <= last.y; ++y) {
        for (int x = first.x; x <= last.x; ++x) {
            let h = finer.Load(int3(x, y, 0));
            if (isnan(h) || (c.values.y > 0.5 && h == c.values.x)) {
                continue;
            }
            highest = max(highest, h);
        }
    }
    target[id.xy] = highest;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void sunHorizonMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= c.size.z || id.y >= c.size.w) {
        return;
    }
    let scale = c.values.x;
    let cellSize = c.values.y;
    let levels = uint(c.sun.z);
    let size = float2(c.size.xy);
    let centre = (float2(id.xy) + 0.5) * scale - 0.5;
    let at = clamp(int2(floor(centre + 0.5)), int2(0), int2(c.size.xy) - 1);
    let base = maxima.Load(int3(at, 0));
    if (base <= 0.5 * kNoHeight) {
        horizon[id.xy] = float2(kNoHeight, 0.0);
        return;
    }
    let direction = c.sun.xy;
    var highest = kNone;
    var occluder = float2(kNoHeight, 0.0);
    var distance = 1.0;
    let far = 1.5 * max(size.x, size.y);
    while (distance < far) {
        let step = max(1.0, distance * kGrowth);
        let middle = distance + 0.5 * step;
        let p = centre + direction * middle;
        if (any(p < float2(-0.5)) || any(p > size - 0.5)) {
            break;
        }
        // A level whose texels are as wide as a step: consecutive steps' texels cover the way.
        let level = min(uint(floor(log2(step))), levels - 1);
        let texel = int2(floor((p + 0.5) / float(1u << level)));
        let h = maxima.Load(int3(texel, int(level)));
        if (h > 0.5 * kNoHeight) {
            let tangent = (h - base) / (middle * cellSize);
            if (tangent > highest) {
                highest = tangent;
                occluder = float2(h, dot(centre, direction) + middle);
            }
        }
        distance += step;
    }
    horizon[id.xy] = occluder;
}
