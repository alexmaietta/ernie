// SPDX-License-Identifier: BSD-3-Clause
// Per-frame resources every raster pass shares, at fixed bindings of its push-descriptor set:
//   0 every view's Uniforms (view and light; `u` is the drawn view's), 1 Shadows (a shadow pass's
//   light view, the cascades' splits), 2 the light shadow maps, 3 shadow comparison sampler, 4
//   clamping linear sampler;
// the per-draw push constants; and reflectance helpers. Materials take
// bindings 5 to 9 (surface.slang) and each drawable's own resources start at 10.
//
// Passes place world points in the image with `projectWorld`, from module `projection`: a frame
// camera's view-projection (projection_pinhole.slang), or a pushbroom's strip
// (projection_pushbroom.slang), whichever the pass was compiled with.
import projection;

public struct Uniforms {
    // The view-projection; a pushbroom's strip in its place (`PushbroomStrip::Packed`).
    public float4x4 viewProj;
    // xyz: camera forward, w: dot(eye, forward); view depth = dot(world, forward) - w.
    public float4 forward;
    // xyz: camera position, w: unused.
    public float4 eye;
    // x: wireframe (0 off, 1 overlay, 2 hidden line), y unused, z: the view's layer of the render
    // targets, which G-buffer vertex shaders draw into (SV_RenderTargetArrayIndex), w: 1 when the
    // projection is orthographic, every ray along `forward` (`towardsViewer`).
    public uint4 flags;
    // Minimal shading's distant lights (`minimalShade`), the first four this view sees: xyz each
    // unit direction towards it, w its share of what the brightest gives a white surface facing it
    // (0 past the last).
    public float4 minimalLights[4];
    // x: the view's sub-frame time as the share of the way from the last frame's transforms (0)
    // to this one's (1), which moving meshes interpolate for motion blur; 1 without. yzw: minimal
    // shading's ambient light per RGB channel, as a share of what the brightest distant light
    // gives a white surface facing it.
    public float4 motion;
    // A pushbroom's intrinsics (`PushbroomStrip::Intrinsics`); unused by frame cameras.
    public float4 strip;
    // Minimal shading (`minimalShade`): rgb the constant colour, w the mode (kMinimal*).
    public float4 minimal;
    // Minimal shading's shadow of `minimalLights[0]` (`Shadows::minimal`) in this view: [0] xy
    // scale and zw offset of its map's x and y clip coordinates, [1] x scale and y offset of its
    // depth, z the slot of its world's map past the first (a batch's), w 1 when it shadows the
    // view.
    public float4 minimalShadow[2];
}

// Clip coordinates of world point `world` in the view.
public float4 projectWorld(float3 world) {
    return cameraProject(u.viewProj, u.strip, world);
}

// Per view, for the MotionVector var's passes (motion_vectors.slang and drawables' motion
// pipelines): kept apart from Uniforms, so frames without the var upload nothing more. 256 bytes,
// so one view's binds as a uniform buffer and the views as an array.
public struct MotionUniforms {
    // The view's camera at the last frame, relative to this frame's origin.
    public float4x4 previousViewProj;
    // This view's inverse view-projection, for points from depth.
    public float4x4 inverseViewProj;
    // x, y: the raster image's width and height; z: 1 when motion applies (frame cameras), 0 for
    // zero motion (pushbroom).
    public float4 size;
    float4 padding[7];
}

// Pixel motion from the last frame's clip position `previous` to this frame's pixel `pixel`
// (raster pixels, y down), for view `m`; zero behind the last frame's camera.
public float2 pixelMotion(MotionUniforms m, float2 pixel, float4 previous) {
    if (m.size.z < 0.5 || previous.w <= 1e-6) {
        return float2(0.0);
    }
    let ndc = previous.xy / previous.w;
    let before = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * m.size.xy;
    return pixel - before;
}

// Per draw: the light's view-projection for shadow passes, the pass's first item (terrain patch
// or mesh draw) and, for instanced draws, the first instance.
public struct DrawConstants {
    public float4x4 lightViewProj;
    public uint first;
    // Instanced mesh draws: the first of the draw's instances drawn, such as one world's.
    public uint instanceOffset;
    // Mesh draws: where the level of detail drawn starts among the mesh's indices; 0 for the
    // full mesh.
    public uint indexOffset;
    // Mesh multi-draws: 1 + the first of the per-draw records (mesh.slang's binding 19) that
    // replace `first`, `instanceOffset` and `indexOffset`, one per draw by its draw index; 0 to
    // use those three.
    public uint records;
    // Shadow passes: 1 to take the light's view-projection from the frame's Shadows block
    // (`s.lightViewProj[0]`, which the GPU may have fitted: a light posed on the GPU) instead of
    // `lightViewProj`.
    public uint shadowFromFrame;
    // The view drawn: its Uniforms in `views` (`u`).
    public uint view;
    // Shadow passes: the layer of the layered shadow map drawn into (SV_RenderTargetArrayIndex),
    // so one pass draws every world's map.
    public uint shadowLayer;
    // To 96 bytes, as uints: a uint2 would align to 8.
    uint padding0;
}

// A shadow pass's light view (`lightViewProj[0]`, `DrawConstants::shadowFromFrame`); for the
// G-buffer and minimal passes, the view's cascade splits and minimal shading's main light's maps
// (its cascades over a single view, or a batch's shared map, which each view's world's places
// with `Uniforms::minimalShadow`).
public struct Shadows {
    public float4x4 lightViewProj[4];
    // Far view depth of each cascade.
    public float4 splits;
    // World size of one shadow map texel in each cascade.
    public float4 texelWorld;
    // World depth spanned by each cascade's [0, 1] depth.
    public float4 depthWorld;
    // Filter radius (texels), normal offset (texels), cascade count, enabled.
    public float4 params;
    // Minimal shading's main light's maps in `shadowMap`: x the first one's layer, y the share of a
    // layer's width they take (in its corner), z how many (its cascades; 1 for one map), w 1 with
    // maps.
    public float4 minimal;
    // x: tint cascades.
    public uint4 debug;
    // 16 Poisson disk samples in the unit disk, two per float4.
    public float4 poisson[8];
}

// Every view's Uniforms, so a pass binds them once for all its views and each draw names its
// view (`DrawConstants::view`).
[[vk::binding(0, 0)]] public StructuredBuffer<Uniforms> views;
[[vk::binding(1, 0)]] public ConstantBuffer<Shadows> s;
[[vk::binding(2, 0)]] public Texture2DArray<float> shadowMap;
[[vk::binding(3, 0)]] public SamplerComparisonState shadowSampler;
[[vk::binding(4, 0)]] public SamplerState clampSampler;
[[vk::push_constant]] public ConstantBuffer<DrawConstants> draw;

// The Uniforms of the view drawn.
public property Uniforms u {
    get {
        return views[draw.view];
    }
}

// The view-projection of the shadow pass drawing (`DrawConstants::shadowFromFrame`).
public float4x4 shadowViewProj() {
    return draw.shadowFromFrame != 0 ? s.lightViewProj[0] : draw.lightViewProj;
}

// A shadow pass's vertex: in the light's clip space, in the layer the draw names
// (`DrawConstants::shadowLayer`), so one layered pass draws every world's map.
public struct ShadowPosition {
    public float4 position : SV_Position;
    public uint layer : SV_RenderTargetArrayIndex;
}

public ShadowPosition shadowPosition(float3 world) {
    ShadowPosition out;
    out.position = mul(shadowViewProj(), float4(world, 1.0));
    out.layer = draw.shadowLayer;
    return out;
}

public float viewDepth(float3 world) {
    return dot(world, u.forward.xyz) - u.forward.w;
}

// The unit direction from `world` back along the ray that sees it: towards the eye, or against
// the view direction for an orthographic projection.
public float3 towardsViewer(float3 world) {
    return u.flags.w != 0 ? -u.forward.xyz : normalize(u.eye.xyz - world);
}

public uint cascadeIndex(float depth) {
    let count = uint(s.params.z);
    for (uint c = 0; c < count; ++c) {
        if (depth < s.splits[c]) {
            return c;
        }
    }
    return count;
}

// `color` tinted by shadow cascade when cascade debugging is on.
public float3 cascadeTint(float3 color, float3 world) {
    if (s.debug.x == 1 && s.params.w > 0.0) {
        let tints = float3[5](float3(1.0, 0.55, 0.55), float3(0.55, 1.0, 0.55),
                              float3(0.55, 0.65, 1.0), float3(1.0, 1.0, 0.5), float3(1.0, 1.0, 1.0));
        return color * tints[cascadeIndex(viewDepth(world))];
    }
    return color;
}

// The fraction of minimal shading's main light (`Uniforms::minimalLights[0]`) reaching `world`,
// whose geometric normal is `normal`: its map (the cascade holding the point, or the view's
// world's), offset along the normal against acne, compared in four hardware 2x2 taps half a texel
// apart; 1 off the map or without one. No soft shadows: minimal shading is for throughput.
//
// Reference: D. Holbert, "Saying 'Goodbye' to Shadow Acne", GDC 2011 poster (normal offset).
public float minimalShadow(float3 world, float3 normal) {
    if (u.minimalShadow[1].w == 0.0 || s.minimal.w == 0.0) {
        return 1.0;
    }
    let count = uint(s.minimal.z);
    let cascade = count > 1 ? min(cascadeIndex(viewDepth(world)), count - 1) : 0;
    let nDotL = saturate(dot(normal, u.minimalLights[0].xyz));
    let offset = normal * s.params.y * s.texelWorld[cascade] * max(1.0 - nDotL, 0.25);
    let clip = mul(s.lightViewProj[cascade], float4(world + offset, 1.0));
    // A batch view's world's map: the shared one's clip space, scaled and offset.
    let place = u.minimalShadow[0];
    let ndc = clip.xy / clip.w * place.xy + place.zw;
    let depth = min(clip.z / clip.w * u.minimalShadow[1].x + u.minimalShadow[1].y, 0.99999994);
    let uv = float2(0.5 * ndc.x + 0.5, 0.5 - 0.5 * ndc.y);
    if (any(uv < float2(0.0)) || any(uv > float2(1.0))) {
        return 1.0;
    }
    uint width;
    uint height;
    uint layers;
    shadowMap.GetDimensions(width, height, layers);
    let texel = 1.0 / float(width);
    let corner = s.minimal.y;
    let at = uv * corner;
    let layer = s.minimal.x + float(cascade) + u.minimalShadow[1].z;
    var lit = 0.0;
    for (uint i = 0; i < 4; ++i) {
        let tap = float2((i & 1u) != 0 ? 0.5 : -0.5, (i & 2u) != 0 ? 0.5 : -0.5) * texel;
        lit += shadowMap.SampleCmpLevelZero(
            shadowSampler,
            float3(clamp(at + tap, float2(0.0), float2(corner - 0.5 * texel)), layer), depth);
    }
    return lit * 0.25;
}

// What raster passes write for the spectral shading pass (see `RenderTargets`):
//   normal: unit shading normal, w 1 (the lights' shadows are the shading pass's);
//   albedo: x scale on the material's reflectance (albedo map over its luminance), yzw linear RGB
//           modulation of it (detail albedo, tints, vertex colours, debug overlays);
//   ids: material, prim id, instance id, flags (kUnlit: light without shading or shadows).
public static const uint kUnlit = 1;

public struct GBuffer {
    public float4 normal : SV_Target0;
    public float4 albedo : SV_Target1;
    public uint4 ids : SV_Target2;
}

// Whether the wireframe is hidden line: surfaces are drawn unlit (kUnlit) in the material's
// reflectance, with their triangle edges.
public bool hiddenLine() {
    return u.flags.x == 2;
}

// `color` with triangle edges drawn over it when the wireframe is on, from barycentric
// coordinates; in hidden line, the edges over white instead of `color`.
public float3 wireframe(float3 color, float3 barycentric) {
    if (u.flags.x == 0) {
        return color;
    }
    let edge = smoothstep(float3(0.0), 1.5 * fwidth(barycentric), barycentric);
    let line = 1.0 - min(edge.x, min(edge.y, edge.z));
    return lerp(hiddenLine() ? float3(1.0) : color, float3(0.05, 0.05, 0.06), line);
}
