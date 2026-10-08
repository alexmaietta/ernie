// Planet resources and the cube-sphere CDLOD grid vertex, used by the planet's shadow and colour
// passes. Besides the frame (bindings 0 to 4, frame.slang) and material (5 to 9, surface.slang)
// the passes bind
//   10 PlanetUniforms, 11 heights, 12 LOD patches, 13 height gradients, 14 patch anchors.
__exported import frame;
import brdf;

public struct PlanetUniforms {
    // Radius, vertical exaggeration, min height, max height.
    public float4 body;
    // Grid quads along a face, colour mode, tint patches by LOD level, grid quads along a patch.
    public uint4 dims;
    // xyz: eye position that drives LOD morphing, relative to the frame's origin; w: unused.
    public float4 lodEye;
    // Morph start distance and 1 / (morph end - morph start) per LOD level, two levels per float4.
    public float4 morph[8];
    // Material, prim id.
    public uint4 ids;
}

[[vk::binding(10, 0)]] public ConstantBuffer<PlanetUniforms> p;
// One layer per cube face (`CubeSphereLod`) of (faces cells + 1) squared vertex heights.
[[vk::binding(11, 0)]] public Texture2DArray<float> heights;
// Per patch: origin column, origin row, LOD level (vertex spacing 2^level cells), face.
[[vk::binding(12, 0)]] public StructuredBuffer<uint4> patches;
// Per vertex and face, the height's slope along the directions of increasing column and row,
// before exaggeration; filterable, unlike `heights`.
[[vk::binding(13, 0)]] public Texture2DArray<float2> gradients;
// Per patch: xyz the radius-high point above its origin vertex relative to the frame's origin,
// computed in double on the host, so vertices are small offsets from it.
[[vk::binding(14, 0)]] public StructuredBuffer<float4> anchors;

public struct FaceFrame {
    public float3 normal;
    public float3 u;
    public float3 v;
}

// The faces of `CubeSphereLod::FaceDirection`.
public FaceFrame faceFrame(uint face) {
    let normals = float3[6](float3(1, 0, 0), float3(0, 1, 0), float3(-1, 0, 0), float3(0, -1, 0),
                            float3(0, 0, 1), float3(0, 0, -1));
    let us = float3[6](float3(0, 1, 0), float3(-1, 0, 0), float3(0, -1, 0), float3(1, 0, 0),
                       float3(0, 1, 0), float3(0, 1, 0));
    let vs = float3[6](float3(0, 0, -1), float3(0, 0, -1), float3(0, 0, -1), float3(0, 0, -1),
                       float3(1, 0, 0), float3(-1, 0, 0));
    FaceFrame f;
    f.normal = normals[face];
    f.u = us[face];
    f.v = vs[face];
    return f;
}

// Face coordinate (-1 to 1) of grid line `index`.
public float2 faceCoords(float2 index) {
    return index * (2.0 / float(p.dims.x)) - 1.0;
}

// Unit direction of face point `st`, the equiangular cube-sphere mapping.
public float3 faceDirection(uint face, float2 st) {
    let f = faceFrame(face);
    let ab = tan(0.25 * PI * st);
    return normalize(f.normal + ab.x * f.u + ab.y * f.v);
}

// The unit direction at grid point `g` of `face` less that at grid point `origin`, accurate when
// the two are close although both directions are near unit length: tangent and length
// differences come from the small difference of their angles rather than from subtraction.
float3 directionOffset(uint face, int2 g, int2 origin, out float3 direction) {
    let f = faceFrame(face);
    let st = faceCoords(float2(g));
    let sta = faceCoords(float2(origin));
    let delta = float2(g - origin) * (2.0 / float(p.dims.x));
    let angle = 0.25 * PI * st;
    let angleA = 0.25 * PI * sta;
    let ab = tan(angle);
    let abA = tan(angleA);
    // tan x - tan y = sin(x - y) / (cos x cos y)
    let dab = sin(0.25 * PI * delta) / (cos(angle) * cos(angleA));
    let point = f.normal + ab.x * f.u + ab.y * f.v;
    let pointA = f.normal + abA.x * f.u + abA.y * f.v;
    let length0 = length(point);
    let lengthA = length(pointA);
    let squares = dab.x * (ab.x + abA.x) + dab.y * (ab.y + abA.y);
    let inverse = -squares / (length0 * lengthA * (length0 + lengthA));
    direction = point / length0;
    return (dab.x * f.u + dab.y * f.v) / length0 + pointA * inverse;
}

public float heightAt(uint face, int2 g) {
    return heights.Load(int4(g, int(face), 0));
}

float2 morphRange(uint level) {
    let pair = p.morph[level / 2];
    return level % 2 == 0 ? pair.xy : pair.zw;
}

// Grid point `g` of a patch anchored at `anchor` (its origin vertex `origin`), relative to the
// frame's origin.
float3 surfacePoint(uint face, int2 g, int2 origin, float3 anchor, out float h) {
    float3 direction;
    let offset = directionOffset(face, g, origin, direction);
    h = heightAt(face, g);
    return anchor + p.body.x * offset + (h * p.body.y) * direction;
}

public struct PlanetVertex {
    public float3 world;
    public float height;
    public float2 grid;  // the vertex's grid point on its face
    public uint face;
    public uint corner;
    public uint level;
}

// One vertex of a cube-sphere LOD patch (`CubeSphereLod`): two triangles per grid quad, six
// vertices per quad, no vertex or index buffers. Within the last part of its level's range each
// odd vertex moves in 3D towards the midpoint of its two neighbours on the next coarser level
// (along an edge, the edge's; for vertices odd in both directions, those on the quad diagonal
// from (1, 0) to (0, 1)), so at morph 1 the patch is exactly the coarser mesh and edges between
// faces, whose grids may run opposite ways, match.
public PlanetVertex planetVertex(uint vertexIndex, uint instanceIndex) {
    let corners = uint2[6](uint2(0, 0), uint2(1, 0), uint2(0, 1), uint2(1, 0), uint2(1, 1),
                           uint2(0, 1));
    let tile = patches[draw.first + instanceIndex];
    let anchor = anchors[draw.first + instanceIndex].xyz;
    let cells = p.dims.w;
    let quad = vertexIndex / 6;
    let corner = vertexIndex % 6;
    let local = uint2(quad % cells, quad / cells) + corners[corner];
    let spacing = int(1u << tile.z);
    let origin = int2(tile.xy);
    let g = origin + int2(local) * spacing;
    let odd = int2(local & uint2(1, 1));

    float h;
    let world = surfacePoint(tile.w, g, origin, anchor, h);
    var target = world;
    var hTarget = h;
    if (odd.x != 0 || odd.y != 0) {
        let step = odd.x != 0 && odd.y != 0 ? int2(spacing, -spacing) : odd * spacing;
        float h0;
        float h1;
        target = 0.5 * (surfacePoint(tile.w, g + step, origin, anchor, h0) +
                        surfacePoint(tile.w, g - step, origin, anchor, h1));
        hTarget = 0.5 * (h0 + h1);
    }
    let range = morphRange(tile.z);
    let k = clamp((distance(p.lodEye.xyz, world) - range.x) * range.y, 0.0, 1.0);

    PlanetVertex v;
    v.world = lerp(world, target, k);
    v.height = lerp(h, hTarget, k);
    v.grid = float2(g);
    v.face = tile.w;
    v.corner = corner;
    v.level = tile.z;
    return v;
}

// The surface normal at grid point `grid` of `face` from the gradient texture rather than from
// the mesh, so shading does not change as levels morph; filtered over the pixel's footprint
// through the gradients' mips (fragment shaders only: it takes screen derivatives). `up` is the
// sphere's normal there.
public float3 planetNormal(uint face, float2 grid, out float3 up) {
    let f = faceFrame(face);
    up = faceDirection(face, faceCoords(grid));
    let tu = normalize(f.u - up * dot(up, f.u));
    let tv = normalize(f.v - up * dot(up, f.v));
    let size = float(p.dims.x + 1);
    let slope = gradients.Sample(clampSampler, float3((grid + 0.5) / size, float(face))) *
                p.body.y;
    // The gradient from its components along the two grid directions, which meet at an angle.
    let c = dot(tu, tv);
    let gradient = ((slope.x - c * slope.y) * tu + (slope.y - c * slope.x) * tv) /
                   max(1.0 - c * c, 1e-6);
    return normalize(up - gradient);
}
