// SPDX-License-Identifier: BSD-3-Clause
// A planet Terrain's horizons towards each distant light casting shadows, for their shadows over
// the whole body whatever the distance (shading.slang's `planetHorizonVisibility`), where the
// cascades' texels grow coarse: terrain_sun_horizon.slang's maps on the cube sphere's faces.
//
// `planetHeightMaxMain` builds a chain of the face heights' maxima, a layer per face: level 0 the
// heights (not-a-number as 0, as the planet draws it), each level after the highest of the 2 x 2
// points under a texel and the one past them (a face's grid of points is a cell larger than its
// cells), so that every point between two texels' corners counts in both. `planetHorizonMain`
// then marches from each texel of a light's six layers of the sun horizon map, at most a data
// cell across, along the great circle towards the light (in the body's frame, across faces),
// reading the maxima at steps a few per cent of the angle apart from the level as wide as a
// step, to the farthest angle a point of the body can rise above another's horizon; and keeps
// the point setting the horizon's elevation over the texel's point (the ellipsoid's normal its
// up), its height and its angle along the great circle, so that shading finds the horizon from
// each receiver's own place and height. A height of kNoHeight with no light to march towards.
//
// References:
// - V. Timonen and J. Westerholm, "Scalable Height Field Self-Shadowing", Computer Graphics Forum
//   29(2):723-731, 2010. https://doi.org/10.1111/j.1467-8659.2009.01642.x
// - E. Mazarico, G. A. Neumann, D. E. Smith, M. T. Zuber and M. H. Torrence, "Illumination
//   conditions of the lunar polar regions using LOLA topography", Icarus 211(2):1066-1081, 2011
//   (horizons from topography towards the Sun, for lunar illumination).
import cube_sphere;

// `planetHeightMaxMain`: the finer level (the heights, for level 0), a layer per face.
[[vk::binding(0, 0)]] Texture2DArray<float> finer;
// `planetHeightMaxMain`: the level written.
[[vk::image_format("r32f")]] [[vk::binding(1, 0)]] RWTexture2DArray<float> target;
// `planetHorizonMain`: every level of the maxima, and the sun horizon maps (x the height, y the
// angle along).
[[vk::binding(2, 0)]] Texture2DArray<float> maxima;
[[vk::image_format("rg32f")]] [[vk::binding(3, 0)]] RWTexture2DArray<float2> horizon;

struct PlanetHorizonConstants {
    // The finer level's (or the heights') points along a face and the target's;
    // `planetHorizonMain`: data cells along a face, map texels along one, the maxima's levels and
    // the light's first layer.
    uint4 size;
    // xyz: towards the light, body-fixed; w: the farthest angle marched, radians.
    float4 sun;
    // x: the largest radius, y: the vertical exaggeration.
    float4 body;
    // xyz: (radius / a_i)^2 for the ellipsoid's semi-axes a_i.
    float4 shape;
}
[[vk::push_constant]] ConstantBuffer<PlanetHorizonConstants> c;

static const float kNoHeight = -1e30;
// Each step this much of the angle marched, at least a data cell.
static const float kGrowth = 0.015;
static const float kHalfPi = 1.57079633;

[shader("compute")]
[numthreads(8, 8, 1)]
void planetHeightMaxMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= c.size.y || id.y >= c.size.y) {
        return;
    }
    let copy = c.size.x == c.size.y;
    let first = copy ? int2(id.xy) : int2(id.xy) * 2;
    let last = copy ? first : min(first + int2(2), int2(int(c.size.x) - 1));
    var highest = kNoHeight;
    for (int y = first.y; y <= last.y; ++y) {
        for (int x = first.x; x <= last.x; ++x) {
            let h = finer.Load(int4(x, y, int(id.z), 0));
            highest = max(highest, isnan(h) ? 0.0 : h);
        }
    }
    target[id] = highest;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void planetHorizonMain(uint3 id : SV_DispatchThreadID) {
    let cells = c.size.x;
    let texels = c.size.y;
    if (id.x >= texels || id.y >= texels) {
        return;
    }
    let face = id.z;
    let layer = int(c.size.w + face);
    let radius = c.body.x;
    let exaggeration = c.body.y;
    let shape = c.shape.xyz;
    let levels = c.size.z;
    let st = (float2(id.xy) + 0.5) / float(texels) * 2.0 - 1.0;
    let d0 = cubeDirection(face, st);
    // The texel's point, at the heights' nearest point.
    let g0 = clamp(int2(floor((st * 0.5 + 0.5) * float(cells) + 0.5)), int2(0), int2(int(cells)));
    let h0 = maxima.Load(int4(g0, int(face), 0));
    let p0 = ellipsoidPoint(d0, h0 * exaggeration, radius, shape);
    let up0 = normalize(d0 * shape);
    // Towards the light along the sphere of directions.
    let s = c.sun.xyz;
    let across = s - dot(s, d0) * d0;
    if (dot(across, across) < 1e-12) {
        horizon[int3(id.xy, layer)] = float2(kNoHeight, 0.0);
        return;
    }
    let towards = normalize(across);
    let cellAngle = kHalfPi / float(cells);
    var highest = -2.0;
    var occluder = float2(kNoHeight, 0.0);
    var angle = cellAngle;
    while (angle < c.sun.w) {
        let step = max(cellAngle, angle * kGrowth);
        let middle = angle + 0.5 * step;
        let d = normalize(cos(middle) * d0 + sin(middle) * towards);
        float2 at;
        let atFace = cubeFaceOf(d, at);
        // A level whose texels are as wide as a step.
        let level = min(uint(max(floor(log2(step / cellAngle)), 0.0)), levels - 1);
        let side = int(((cells + 1u) >> level)) - 1;
        let g = (at * 0.5 + 0.5) * float(cells) / float(1u << level);
        let texel = clamp(int2(floor(g)), int2(0), int2(max(side, 0)));
        let h = maxima.Load(int4(texel, int(atFace), int(level)));
        let q = ellipsoidPoint(d, h * exaggeration, radius, shape);
        let sine = dot(normalize(q - p0), up0);
        if (sine > highest) {
            highest = sine;
            occluder = float2(h, middle);
        }
        angle += step;
    }
    horizon[int3(id.xy, layer)] = occluder;
}
