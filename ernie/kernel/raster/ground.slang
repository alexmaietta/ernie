// SPDX-License-Identifier: BSD-3-Clause
// Ground plane G-buffer pass: an infinite horizontal plane at z = height, drawn with a full-screen
// triangle. Each fragment intersects its camera ray with the plane and writes the hit's depth, so
// other drawables sort against it; rays that miss (or hit behind the camera) are discarded, and
// hits past the far plane are pulled just inside it so the plane reaches the horizon.
//
// An optional checkerboard modulates the material's albedo, box-filtered over the pixel's
// footprint so it fades to its mean colour with distance instead of aliasing (Quilez, "Filtering
// the Checkerboard Pattern", https://iquilezles.org/articles/checkerfiltering/).
//
// Bindings: frame 0 to 4 (frame.slang), material 5 to 9 (surface.slang), 10 GroundUniforms.
import frame;
import projection;
import surface;

struct GroundUniforms {
    float4x4 invViewProj;
    // Height (relative to the frame's origin), checker square size, checker on, unused.
    float4 plane;
    // xy: the checkerboard's phase at the frame's origin, in squares modulo its two-square
    // period, so it stays fixed to the world.
    float4 checkerPhase;
    // Linear checker colours, rgb.
    float4 colour0;
    float4 colour1;
    // Image width and height in pixels.
    float4 viewport;
    // Material, prim id, light link group.
    uint4 ids;
    float4 padding[6];  // to 256 bytes, the views' stride
}

// Every view's uniforms, by the view drawn (`DrawConstants::view`).
[[vk::binding(10, 0)]] StructuredBuffer<GroundUniforms> grounds;

property GroundUniforms g {
    get {
        return grounds[draw.view];
    }
}

struct VertexOutput {
    float4 position : SV_Position;
}

// The vertex stage's outputs: the fragment stage's inputs (`VertexOutput`, in the same order)
// and the view's layer of the G-buffer, for one layered pass over every view
// (Uniforms.flags.z).
struct LayeredVertexOutput {
    float4 position : SV_Position;
    uint layer : SV_RenderTargetArrayIndex;
}

struct GroundOutput {
    float4 normal : SV_Target0;
    float4 albedo : SV_Target1;
    uint4 ids : SV_Target2;
    float depth : SV_Depth;
}

[shader("vertex")]
LayeredVertexOutput vertexMain(uint vertexIndex : SV_VertexID) {
    let uv = float2(float((vertexIndex << 1) & 2), float(vertexIndex & 2));
    LayeredVertexOutput out;
    out.layer = u.flags.z;
    out.position = float4(uv * 2.0 - 1.0, 0.5, 1.0);
    return out;
}

// Box-filtered checkerboard of unit squares at `p`: 0 on one colour, 1 on the other, blending to
// 0.5 as the filter footprint grows past a square.
float filteredChecker(float2 p) {
    let w = max(abs(ddx(p)), abs(ddy(p))) + 1e-4;
    let i = 2.0 * (abs(frac((p - 0.5 * w) * 0.5) - 0.5) - abs(frac((p + 0.5 * w) * 0.5) - 0.5)) / w;
    return 0.5 - 0.5 * i.x * i.y;
}

// The point where a pixel's camera ray meets the plane, and how far along the ray (`t`, negative
// when it misses).
float3 groundPoint(VertexOutput in, out float t, out float3 origin, out float3 direction) {
    let uv = in.position.xy / g.viewport.xy;
    let ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    // A pushbroom's strip is in the frame's uniforms (and its pixels' rays from different poses).
    cameraRayThrough(kPushbroomCamera ? u.viewProj : g.invViewProj, u.strip, ndc, origin,
                     direction);
    let valid = abs(direction.z) > 1e-12;
    t = valid ? (g.plane.x - origin.z) / direction.z : -1.0;
    return origin + direction * max(t, 0.0);
}

// The plane where a pixel's camera ray meets it, as the G-buffer's fragment writes it; shared by
// `fragmentMain` and `minimalFragmentMain`, since an entry point calling another hangs Slang's
// entry-point parameter lowering (translateEntryPointInParamToBorrow, Slang 2026.19).
GroundOutput shadeGround(VertexOutput in) {
    float t;
    float3 origin;
    float3 direction;
    let world = groundPoint(in, t, origin, direction);

    // Everything that takes screen-space derivatives runs before the discard.
    let detail = sampleDetail(detailCoords(world.xy, kProjectXy));
    var modulation = detail.albedo;
    if (g.plane.z > 0.0) {
        let blend = filteredChecker(world.xy / max(g.plane.y, 1e-6) + g.checkerPhase.xy);
        modulation = modulation * lerp(g.colour0.rgb, g.colour1.rgb, blend);
    }
    if (!(t > 0.0)) {
        discard;
    }

    let clip = projectWorld(world);
    GroundOutput out;
    out.normal = float4(normalize(float3(-detail.slope, 1.0)), 1.0);
    out.albedo = float4(surfaceAlbedoScale(world), cascadeTint(modulation, world));
    out.ids = uint4(g.ids.x, g.ids.y, 0,
                    packSurfaceFlags(linkGroupFlags(g.ids.z), detail.slopeVariance));
    // Reverse-Z: kept above the cleared 0 out to the horizon, past any far plane.
    out.depth = clamp(clip.z / clip.w, 1.17549435e-38, 1.0);
    return out;
}

[shader("fragment")]
GroundOutput fragmentMain(VertexOutput in) {
    return shadeGround(in);
}

// Minimal shading (`minimalShade`) of what `fragmentMain` writes, at its depth.
struct GroundMinimalOutput {
    float4 color : SV_Target0;
    uint4 ids : SV_Target1;
    float depth : SV_Depth;
}

// Minimal shading without colour (kMinimalNone): the plane's ids and normal at its depth, with
// no material evaluated.
[shader("fragment")]
GroundMinimalOutput idsFragmentMain(VertexOutput in) {
    let uv = in.position.xy / g.viewport.xy;
    let ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float3 origin;
    float3 direction;
    cameraRayThrough(kPushbroomCamera ? u.viewProj : g.invViewProj, u.strip, ndc, origin,
                     direction);
    let t = abs(direction.z) > 1e-12 ? (g.plane.x - origin.z) / direction.z : -1.0;
    if (!(t > 0.0)) {
        discard;
    }
    let clip = projectWorld(origin + direction * t);
    GroundMinimalOutput out;
    out.color = float4(0.0, 0.0, 0.0, 1.0);
    out.ids = uint4(g.ids.y, 0u, packOctahedral(float3(0.0, 0.0, 1.0)), 0u);
    // Reverse-Z: kept above the cleared 0 out to the horizon, past any far plane.
    out.depth = clamp(clip.z / clip.w, 1.17549435e-38, 1.0);
    return out;
}

[shader("fragment")]
GroundMinimalOutput minimalFragmentMain(VertexOutput in) {
    let ground = shadeGround(in);
    GBuffer g;
    g.normal = ground.normal;
    g.albedo = ground.albedo;
    g.ids = ground.ids;
    float t;
    float3 origin;
    float3 direction;
    let shaded = minimalShade(g, groundPoint(in, t, origin, direction));
    GroundMinimalOutput out;
    out.color = shaded.color;
    out.ids = shaded.ids;
    out.depth = ground.depth;
    return out;
}
