// SPDX-License-Identifier: BSD-3-Clause
// Point clouds drawn as ray-cast spheres (UsdGeomPoints): each point a quad facing the viewer that
// covers its sphere's silhouette, whose fragments intersect their ray with the sphere and write
// the hit's depth, so the spheres are exact at any distance and sort against everything else.
// Draw `draw.first` of `draws`; instance `i` of the draw is point `i`, six vertices each.
//
// Bindings: frame 0 to 4 (frame.slang), material 5 to 9 (surface.slang), 10 positions (two halves
// for motion blur), 11 widths, 12 colours, 13 the draws, 15 the visible chunks of culled draws.
//
// Culling on the GPU (`cullMain`, `PointsDrawable::RecordCull`): each job, a cloud in one raster
// or shadow view, tests its points in chunks of 64 against that view's frustum, taken from its
// view-projection where the GPU keeps it (the views' Uniforms, the light views' matrices), and
// lists the chunks with any point in view; its indirect draw then draws 64 instances per chunk
// listed, which the vertex stages map back to points (`pointOf`), those past the cloud's end
// drawing nothing.
//
// Reference: C. Sigg, T. Weyrich, M. Botsch and M. Gross, "GPU-Based Ray-Casting of Quadratic
// Surfaces", Eurographics Symposium on Point-Based Graphics, 59-65, 2006.
import frame;
import surface;
import virtual_shadow_caster;

static const uint kHasWidths = 1;
static const uint kHasColors = 2;

struct PointsDraw {
    // The cloud's transform relative to the frame's origin, and at the last frame.
    float4x4 model;
    float4x4 previous;
    // rgb: the points' colour without colours of their own; a: the constant width.
    float4 tint;
    // x: the half of the positions holding them now, y: at the last frame (points, not floats),
    // z: flags, w: the largest scale of `model`.
    uint4 info;
    // Material, prim id, light link group, points.
    uint4 ids;
}

static const uint kChunk = 64;

[[vk::binding(10, 0)]] StructuredBuffer<float> positions;
[[vk::binding(11, 0)]] StructuredBuffer<float> widths;
[[vk::binding(12, 0)]] StructuredBuffer<float> colors;
[[vk::binding(13, 0)]] StructuredBuffer<PointsDraw> draws;
[[vk::binding(15, 0)]] StructuredBuffer<uint> visibleChunks;

// The point instance `instance` of the draw draws: itself, or for a culled draw (`instanceOffset`
// 1 + where its visible chunks start) the point of its chunk listed; not `live` past the end.
uint pointOf(uint instance, out bool live) {
    live = true;
    if (draw.instanceOffset == 0) {
        return instance;
    }
    let chunk = visibleChunks[draw.instanceOffset - 1 + instance / kChunk];
    let point = chunk * kChunk + instance % kChunk;
    live = point < draws[draw.first].ids.w;
    return point;
}

// Off every view: a vertex of a point that is not drawn.
static const float4 kNowhere = float4(2.0, 2.0, 0.5, 1.0);

float3 load3(StructuredBuffer<float> buffer, uint i) {
    return float3(buffer[3 * i], buffer[3 * i + 1], buffer[3 * i + 2]);
}

// Point `point`'s centre relative to the origin, at `motion` of the way from the last frame (0) to
// this one (1), and its radius.
float4 sphereOf(PointsDraw d, uint point, float motion) {
    let now = mul(d.model, float4(load3(positions, d.info.x + point), 1.0)).xyz;
    var centre = now;
    if (motion != 1.0) {
        let was = mul(d.previous, float4(load3(positions, d.info.y + point), 1.0)).xyz;
        centre = lerp(was, now, motion);
    }
    let width = (d.info.z & kHasWidths) != 0 ? widths[point] : d.tint.a;
    return float4(centre, 0.5 * width * asfloat(d.info.w));
}

// The quad corners of two triangles, in [-1, 1]^2.
float2 cornerOf(uint vertexIndex) {
    let corners = float2[6](float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, -1),
                            float2(1, 1), float2(-1, 1));
    return corners[vertexIndex];
}

struct VertexOutput {
    float4 position : SV_Position;
    float3 world : WORLD;
    nointerpolation float4 sphere : SPHERE;
    nointerpolation float3 color : COLOR;
    nointerpolation uint point : POINT;
}

struct LayeredVertexOutput {
    float4 position : SV_Position;
    float3 world : WORLD;
    nointerpolation float4 sphere : SPHERE;
    nointerpolation float3 color : COLOR;
    nointerpolation uint point : POINT;
    uint layer : SV_RenderTargetArrayIndex;
}

struct PointsOutput {
    float4 normal : SV_Target0;
    float4 albedo : SV_Target1;
    uint4 ids : SV_Target2;
    float depth : SV_Depth;
}

[shader("vertex")]
LayeredVertexOutput vertexMain(uint vertexIndex : SV_VertexID, uint instance : SV_InstanceID) {
    let d = draws[draw.first];
    bool live;
    let point = pointOf(instance, live);
    if (!live) {
        LayeredVertexOutput none = {};
        none.position = kNowhere;
        none.layer = u.flags.z;
        return none;
    }
    let sphere = sphereOf(d, point, u.motion.x);
    // A quad facing the eye in the plane through the centre, as wide as the silhouette seen
    // from the eye projects onto that plane: r d / sqrt(d^2 - r^2); facing along the view, r
    // wide, for orthographic views (Uniforms.flags.w).
    let orthographic = u.flags.w != 0;
    let toEye = u.eye.xyz - sphere.xyz;
    let distance = length(toEye);
    let forward = orthographic ? -u.forward.xyz : toEye / max(distance, 1e-20);
    let helper = abs(forward.z) < 0.9 ? float3(0, 0, 1) : float3(1, 0, 0);
    let right = normalize(cross(helper, forward));
    let up = cross(forward, right);
    let r = sphere.w;
    let inside = !orthographic && distance <= r * 1.0001;
    let half = orthographic ? r
               : inside     ? 0.0
                            : r * distance / sqrt(distance * distance - r * r);
    let corner = cornerOf(vertexIndex);
    let world = sphere.xyz + (corner.x * right + corner.y * up) * half;
    LayeredVertexOutput out;
    out.layer = u.flags.z;
    out.position = projectWorld(world);
    out.world = world;
    out.sphere = sphere;
    out.color = (d.info.z & kHasColors) != 0 ? load3(colors, point) : d.tint.rgb;
    out.point = point;
    return out;
}

// The pixel's ray against the sphere, the nearer hit: from the eye, or along the view through
// the fragment for orthographic views. False when it misses.
bool sphereHit(VertexOutput in, out float3 hit) {
    let orthographic = u.flags.w != 0;
    let origin = orthographic ? in.world : u.eye.xyz;
    let direction = orthographic ? u.forward.xyz : normalize(in.world - origin);
    let offset = origin - in.sphere.xyz;
    let b = dot(direction, offset);
    let h = b * b - (dot(offset, offset) - in.sphere.w * in.sphere.w);
    hit = origin + (-b - sqrt(max(h, 0.0))) * direction;
    return h >= 0.0;
}

// What the G-buffer pass writes of the sphere where the pixel's ray meets it (`hit`); shared by
// `fragmentMain` and the minimal passes, since an entry point calling another hangs Slang's
// entry-point parameter lowering.
PointsOutput shadePoint(VertexOutput in, out float3 hit) {
    let d = draws[draw.first];
    if (!sphereHit(in, hit)) {
        discard;
    }
    let normal = normalize(hit - in.sphere.xyz);
    // A pushbroom's rows see from poses along its strip; the ray from its envelope's eye is
    // close for spheres small beside the strip's distance to them.
    let clip = projectWorld(hit);
    PointsOutput out;
    out.normal = float4(normal, 1.0);
    out.albedo = float4(surfaceAlbedoScale(hit), cascadeTint(in.color, hit));
    out.ids = uint4(d.ids.x, d.ids.y, in.point, packSurfaceFlags(linkGroupFlags(d.ids.z), 0.0));
    out.depth = saturate(clip.z / clip.w);
    return out;
}

[shader("fragment")]
PointsOutput fragmentMain(VertexOutput in) {
    float3 hit;
    return shadePoint(in, hit);
}

// Minimal shading (`minimalShade`) of what `fragmentMain` writes, at the sphere's depth.
struct PointsMinimalOutput {
    float4 color : SV_Target0;
    uint4 ids : SV_Target1;
    float depth : SV_Depth;
}

[shader("fragment")]
PointsMinimalOutput minimalFragmentMain(VertexOutput in) {
    float3 hit;
    let point = shadePoint(in, hit);
    GBuffer g;
    g.normal = point.normal;
    g.albedo = point.albedo;
    g.ids = point.ids;
    let shaded = minimalShade(g, hit);
    PointsMinimalOutput out;
    out.color = shaded.color;
    out.ids = shaded.ids;
    out.depth = point.depth;
    return out;
}

// Minimal shading without colour (kMinimalNone): the sphere's ids and normal at its depth, with
// no material evaluated.
[shader("fragment")]
PointsMinimalOutput idsFragmentMain(VertexOutput in) {
    let d = draws[draw.first];
    float3 hit;
    if (!sphereHit(in, hit)) {
        discard;
    }
    let clip = projectWorld(hit);
    PointsMinimalOutput out;
    out.color = float4(0.0, 0.0, 0.0, 1.0);
    out.ids = uint4(d.ids.y, in.point, packOctahedral(normalize(hit - in.sphere.xyz)), 0u);
    out.depth = saturate(clip.z / clip.w);
    return out;
}

// The MotionVector var's pass (`RasterDrawable::RecordMotion`): the moving points again over
// the G-buffer's depth, from vertexMain, each pixel's motion from where its point of the sphere
// was at the last frame (moved with the centre).
[[vk::binding(14, 0)]] ConstantBuffer<MotionUniforms> motionView;

struct MotionOutput {
    float4 motion : SV_Target0;
    float depth : SV_Depth;
}

[shader("fragment")]
MotionOutput motionFragment(VertexOutput in) {
    let d = draws[draw.first];
    float3 hit;
    if (!sphereHit(in, hit)) {
        discard;
    }
    let before = hit - in.sphere.xyz + sphereOf(d, in.point, 0.0).xyz;
    let clip = projectWorld(hit);
    MotionOutput out;
    out.motion = float4(
        pixelMotion(motionView, in.position.xy, mul(motionView.previousViewProj, float4(before, 1.0))),
        0.0, 0.0);
    out.depth = saturate(clip.z / clip.w);
    return out;
}

struct ShadowVertexOutput {
    float4 position : SV_Position;
    float2 corner : CORNER;
    nointerpolation float4 sphere : SPHERE;
}

// The shadow vertex stage's outputs: the fragment stage's inputs and the shadow map's layer.
struct ShadowVertexLayered {
    float4 position : SV_Position;
    float2 corner : CORNER;
    nointerpolation float4 sphere : SPHERE;
    uint layer : SV_RenderTargetArrayIndex;
}

// The light's view direction away from it, and two directions across it, from its
// view-projection: the gradients of clip z, x and y (exact for orthographic maps).
float3 lightAxis(uint row) {
    let m = shadowViewProj();
    return normalize(float3(m[row][0], m[row][1], m[row][2]));
}

[shader("vertex")]
ShadowVertexLayered shadowVertex(uint vertexIndex : SV_VertexID, uint instance : SV_InstanceID) {
    let d = draws[draw.first];
    bool live;
    let point = pointOf(instance, live);
    if (!live) {
        ShadowVertexLayered none = {};
        none.position = kNowhere;
        none.layer = shadowPosition(float3(0.0)).layer;
        return none;
    }
    // Where the point is at the pass's sub-frame (Uniforms.motion, 1 without motion blur).
    let sphere = sphereOf(d, point, u.motion.x);
    let corner = cornerOf(vertexIndex);
    let world = sphere.xyz + (corner.x * lightAxis(0) + corner.y * lightAxis(1)) * sphere.w;
    let at = shadowPosition(world);
    ShadowVertexLayered out;
    out.position = at.position;
    out.corner = corner;
    out.sphere = sphere;
    out.layer = at.layer;
    return out;
}

[shader("fragment")]
float shadowFragment(ShadowVertexOutput in) : SV_Depth {
    let rho2 = dot(in.corner, in.corner);
    if (rho2 > 1.0) {
        discard;
    }
    // The sphere's surface facing the light over this texel.
    let across = in.corner.x * lightAxis(0) + in.corner.y * lightAxis(1);
    let front = in.sphere.xyz + in.sphere.w * (across - sqrt(1.0 - rho2) * lightAxis(2));
    let clip = mul(shadowViewProj(), float4(front, 1.0));
    return clip.z / clip.w;
}

// A virtual shadow map's level (`Shadows::virtualMap`): the sphere's surface facing the light
// over this texel, written through its pages. Its depth is reversed, growing towards the light,
// so that surface lies along +`lightAxis(2)`.
[shader("fragment")]
void virtualShadowFragment(ShadowVertexOutput in) {
    let rho2 = dot(in.corner, in.corner);
    if (rho2 > 1.0) {
        discard;
    }
    let across = in.corner.x * lightAxis(0) + in.corner.y * lightAxis(1);
    let front = in.sphere.xyz + in.sphere.w * (across + sqrt(1.0 - rho2) * lightAxis(2));
    let clip = mul(shadowViewProj(), float4(front, 1.0));
    writeVirtualShadow(float4(in.position.xy, clip.z / clip.w, 1.0));
}

// Culling: each job's view (x 0 a raster view, its Uniforms in `cullViews`, `draw.records`
// float4s apart; 1 a light view, its matrix in `cullLights`), y the view's index, z where its
// visible chunks start; the jobs' indirect draws (vertices, instances, first vertex, first
// instance), whose instances start at 0.
[[vk::binding(16, 0)]] StructuredBuffer<uint4> cullJobs;
[[vk::binding(17, 0)]] StructuredBuffer<float4> cullViews;
[[vk::binding(18, 0)]] StructuredBuffer<float4> cullLights;
[[vk::binding(19, 0)]] RWStructuredBuffer<uint> visibleOut;
[[vk::binding(20, 0)]] RWStructuredBuffer<uint> cullDraws;

groupshared uint chunkSeen;

// One workgroup per chunk of draw `draw.first`'s `draw.instanceOffset` points (x, past chunk
// `draw.view`) and job (y, from `draw.indexOffset`): a chunk with any point whose sphere meets the
// view's frustum is listed, and the job's draw takes its 64 instances. Shadow passes clamp depth,
// so light views test their sides only.
//
// Reference: G. Gribb and K. Hartmann, "Fast Extraction of Viewing Frustum Planes from the
// World-View-Projection Matrix", 2001 (the planes from the matrix's rows).
[shader("compute")]
[numthreads(64, 1, 1)]
void cullMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    let chunk = draw.view + group.x;
    let jobIndex = draw.indexOffset + group.y;
    let job = cullJobs[jobIndex];
    if (lane == 0) {
        chunkSeen = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    let point = chunk * kChunk + lane;
    if (point < draw.instanceOffset) {
        // The view-projection's columns.
        float4 c[4];
        for (uint k = 0; k < 4; ++k) {
            c[k] = job.x == 0 ? cullViews[job.y * draw.records + k] : cullLights[4 * job.y + k];
        }
        float4 m[4];
        for (uint r = 0; r < 4; ++r) {
            m[r] = float4(c[0][r], c[1][r], c[2][r], c[3][r]);
        }
        // Clip space x and y in [-w, w], z in [0, w] (with reverse-Z, far to near).
        let planes = float4[6](m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1], m[2],
                               m[3] - m[2]);
        let sphere = sphereOf(draws[draw.first], point, 1.0);
        var inside = true;
        for (uint p = 0; p < (job.x == 0 ? 6 : 4); ++p) {
            let scale = max(length(planes[p].xyz), 1e-30);
            inside = inside && dot(planes[p].xyz, sphere.xyz) + planes[p].w >= -sphere.w * scale;
        }
        if (inside) {
            InterlockedOr(chunkSeen, 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0 && chunkSeen != 0) {
        uint listed;
        InterlockedAdd(cullDraws[4 * jobIndex + 1], kChunk, listed);
        visibleOut[job.z + listed / kChunk] = chunk;
    }
}
