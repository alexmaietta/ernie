// SPDX-License-Identifier: BSD-3-Clause
// Point clouds drawn as ray-cast spheres (UsdGeomPoints): each point a quad facing the viewer that
// covers its sphere's silhouette, whose fragments intersect their ray with the sphere and write
// the hit's depth, so the spheres are exact at any distance and sort against everything else.
// Draw `draw.first` of `draws`; instance `i` of the draw is point `i`, six vertices each.
//
// Bindings: frame 0 to 4 (frame.slang), material 5 to 9 (surface.slang), 10 positions (two halves
// for motion blur), 11 widths, 12 colours, 13 the draws.
//
// Reference: C. Sigg, T. Weyrich, M. Botsch and M. Gross, "GPU-Based Ray-Casting of Quadratic
// Surfaces", Eurographics Symposium on Point-Based Graphics, 59-65, 2006.
import frame;
import surface;

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
    // Material, prim id.
    uint4 ids;
}

[[vk::binding(10, 0)]] StructuredBuffer<float> positions;
[[vk::binding(11, 0)]] StructuredBuffer<float> widths;
[[vk::binding(12, 0)]] StructuredBuffer<float> colors;
[[vk::binding(13, 0)]] StructuredBuffer<PointsDraw> draws;

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
LayeredVertexOutput vertexMain(uint vertexIndex : SV_VertexID, uint point : SV_InstanceID) {
    let d = draws[draw.first];
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

[shader("fragment")]
PointsOutput fragmentMain(VertexOutput in) {
    let d = draws[draw.first];
    float3 hit;
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
    out.ids = uint4(d.ids.x, d.ids.y, in.point, 0u);
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
ShadowVertexLayered shadowVertex(uint vertexIndex : SV_VertexID, uint point : SV_InstanceID) {
    let d = draws[draw.first];
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
