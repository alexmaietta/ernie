// SPDX-License-Identifier: BSD-3-Clause
// Terrain horizon maps for dome occlusion: for each texel of a grid coarser than the heights by
// `scale` cells, the tangent of the horizon's elevation (before vertical exaggeration) towards
// eight azimuths, 45 degrees apart from east counter-clockwise, seen from four heights above the
// ground (`HorizonConstants::levels`; the first the ground itself): layers 2 l and 2 l + 1 hold
// level l's azimuths 0 to 135 degrees and 180 to 315. Each march samples the heights at distances
// growing geometrically from a cell to the raster's size, so near relief and far ridges both
// count; texels with no horizon above the nadir hold -100. Shading turns them into the sky's
// visibility and bent normal over terrain and over what stands on it (shading.slang's
// `terrainOcclusion`).
//
// References:
// - N. L. Max, "Horizon mapping: shadows for bump-mapped surfaces", The Visual Computer 4(2),
//   1988. https://doi.org/10.1007/BF01905559
// - P.-P. Sloan and M. F. Cohen, "Interactive Horizon Mapping", Eurographics Workshop on
//   Rendering 2000.

[[vk::binding(0, 0)]] Texture2D<float> heights;
[[vk::image_format("rgba16f")]] [[vk::binding(1, 0)]] RWTexture2DArray<float4> horizon;

struct HorizonConstants {
    // Heights' width and height, the horizon grid's width and height.
    uint4 size;
    // Cells per horizon texel, the cell size in world units, the no-data value and 1 when the
    // heights have one.
    float4 values;
    // The heights above the ground the levels see from, in height units (before exaggeration).
    float4 levels;
}
[[vk::push_constant]] ConstantBuffer<HorizonConstants> c;

static const uint kSteps = 48;
static const uint kLevels = 4;
static const float kNone = -100.0;

// The height at cell `cell`, and whether it is one (inside the raster, not no-data).
bool heightAt(float2 cell, out float height) {
    let at = int2(floor(cell + 0.5));
    height = 0.0;
    if (any(at < 0) || at.x >= int(c.size.x) || at.y >= int(c.size.y)) {
        return false;
    }
    height = heights.Load(int3(at, 0));
    return !(c.values.w > 0.5 && height == c.values.z) && !isnan(height);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void horizonMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= c.size.z || id.y >= c.size.w) {
        return;
    }
    let scale = c.values.x;
    let cell = c.values.y;
    let centre = (float2(id.xy) + 0.5) * scale - 0.5;
    float ground;
    if (!heightAt(centre, ground)) {
        for (uint layer = 0; layer < 2 * kLevels; ++layer) {
            horizon[uint3(id.xy, layer)] = float4(kNone);
        }
        return;
    }
    // Distances from one cell to the raster's size, growing by `ratio` per step.
    let far = float(max(c.size.x, c.size.y));
    let ratio = pow(far, 1.0 / float(kSteps - 1));
    for (uint level = 0; level < kLevels; ++level) {
        let base = ground + c.levels[level];
        float tangents[8];
        for (uint k = 0; k < 8; ++k) {
            let azimuth = float(k) * 0.25 * 3.14159265;
            // East is +column, north -row.
            let direction = float2(cos(azimuth), -sin(azimuth));
            var highest = kNone;
            var distance = 1.0;
            for (uint step = 0; step < kSteps; ++step) {
                float height;
                if (heightAt(centre + direction * distance, height)) {
                    highest = max(highest, (height - base) / (distance * cell));
                } else if (distance > 1.5 * far) {
                    break;
                }
                distance *= ratio;
            }
            tangents[k] = highest;
        }
        horizon[uint3(id.xy, 2 * level)] =
            float4(tangents[0], tangents[1], tangents[2], tangents[3]);
        horizon[uint3(id.xy, 2 * level + 1)] =
            float4(tangents[4], tangents[5], tangents[6], tangents[7]);
    }
}
