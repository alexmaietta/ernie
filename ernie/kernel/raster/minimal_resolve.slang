// SPDX-License-Identifier: BSD-3-Clause
// Minimal shading's outputs (`MinimalTargets`): from a chunk of views' attachments, each held
// render var's texels, written into its linear buffer over every view (view, row, column), as
// the full pipeline writes them (sensor.slang's geometry vars): the background at infinity, with
// no normal, prim or instance, and in LdrColor the dome light where camera rays see it (its map
// at their direction, as bright as against a white surface facing the sun). One thread per pixel
// of each view of the chunk.
//
// Bindings: 0 colour (sRGB RGBA8), 1 ids (prim id, instance id, octahedral normal, flags; all
// ones for the background), 2 depth ([0, 1]); 3 the views' Uniforms (frame.slang) and 4 their
// ShadingUniforms (shading.slang), as bytes, which GPU poses may have patched; 5 LdrColor,
// 6 DistanceToImagePlane, 7 DistanceToCamera, 8 Normal, 9 PrimId and 10 InstanceId, each written
// only when held (`constants.holds`), 11 the dome's map (linear RGB, longitude across).

struct ResolveConstants {
    uint4 size;    // width, height, the chunk's views, its first view
    // Held vars (bits in binding order from 5), the Uniforms' stride, the shading views' blocks'
    // (shading.slang's ShadingView, each starting with its inverse view-projection) and where they
    // start, after the shared ShadingUniforms.
    uint4 layout;
    // Rows of the rotation from world directions to the dome's frame.
    float4 toDome[3];
    // The dome's luminance against a white surface facing the sun (0 when unseen), 1 when its map
    // shapes it, and the map's width and height.
    float4 sky;
}

[[vk::binding(0, 0)]] Texture2DArray<float4> color;
[[vk::binding(1, 0)]] Texture2DArray<uint4> ids;
[[vk::binding(2, 0)]] Texture2DArray<float> depth;
[[vk::binding(3, 0)]] ByteAddressBuffer uniforms;
[[vk::binding(4, 0)]] ByteAddressBuffer shading;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> ldrColor;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> distanceToImagePlane;
[[vk::binding(7, 0)]] RWStructuredBuffer<float> distanceToCamera;
[[vk::binding(8, 0)]] RWStructuredBuffer<float4> normalOut;
[[vk::binding(9, 0)]] RWStructuredBuffer<uint> primId;
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> instanceId;
[[vk::binding(11, 0)]] Texture2D<float4> domeMap;
[[vk::push_constant]] ConstantBuffer<ResolveConstants> constants;

static const uint kBackground = 0xFFFFFFFFu;
static const uint kHoldsLdrColor = 1u << 0;
static const uint kHoldsImagePlane = 1u << 1;
static const uint kHoldsCamera = 1u << 2;
static const uint kHoldsNormal = 1u << 3;
static const uint kHoldsPrimId = 1u << 4;
static const uint kHoldsInstanceId = 1u << 5;

// Byte offsets: Uniforms' forward, eye and flags.
static const uint kForward = 64;
static const uint kEye = 80;
static const uint kFlags = 96;

float3 load3(ByteAddressBuffer buffer, uint at) {
    return asfloat(buffer.Load3(at));
}

// The inverse view-projection stored by glm column, as a matrix.
float4x4 inverseViewProj(uint at) {
    let c0 = asfloat(shading.Load4(at));
    let c1 = asfloat(shading.Load4(at + 16));
    let c2 = asfloat(shading.Load4(at + 32));
    let c3 = asfloat(shading.Load4(at + 48));
    return float4x4(c0.x, c1.x, c2.x, c3.x, c0.y, c1.y, c2.y, c3.y, c0.z, c1.z, c2.z, c3.z,
                    c0.w, c1.w, c2.w, c3.w);
}

float3 unpackOctahedral(uint packed) {
    let p = float2(int2(int(packed << 16) >> 16, int(packed) >> 16)) / 32767.0;
    var n = float3(p, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * select(n.xy >= 0.0, float2(1.0), float2(-1.0));
    }
    return normalize(n);
}

float3 srgbEncode(float3 c) {
    let x = saturate(c);
    return select(x <= 0.0031308, 12.92 * x, 1.055 * pow(x, 1.0 / 2.4) - 0.055);
}

// The dome map at texel `at`, wrapping in longitude.
float3 domeTexel(int2 at, int2 size) {
    let wrapped = int2((at.x % size.x + size.x) % size.x, clamp(at.y, 0, size.y - 1));
    return max(domeMap.Load(int3(wrapped, 0)).rgb, float3(0.0));
}

// The dome as minimal shading shows it in world direction `d`: its map bilinearly, in the units of
// a white surface facing the sun.
float3 domeAt(float3 d) {
    let sky = constants.sky;
    if (sky.y == 0.0) {
        return float3(sky.x);
    }
    let local = float3(dot(constants.toDome[0].xyz, d), dot(constants.toDome[1].xyz, d),
                       dot(constants.toDome[2].xyz, d));
    var u = atan2(local.y, local.x) / (2.0 * 3.14159265);
    u = u < 0.0 ? u + 1.0 : u;
    let v = acos(clamp(local.z, -1.0, 1.0)) / 3.14159265;
    let size = int2(sky.zw);
    let p = float2(u, v) * sky.zw - 0.5;
    let base = int2(floor(p));
    let f = p - float2(base);
    let rgb = lerp(lerp(domeTexel(base, size), domeTexel(base + int2(1, 0), size), f.x),
                   lerp(domeTexel(base + int2(0, 1), size), domeTexel(base + int2(1, 1), size), f.x),
                   f.y);
    return rgb * sky.x;
}

uint packUnorm(float4 c) {
    let q = uint4(round(saturate(c) * 255.0));
    return q.x | (q.y << 8) | (q.z << 16) | (q.w << 24);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void resolveMain(uint3 id : SV_DispatchThreadID) {
    let width = constants.size.x;
    let height = constants.size.y;
    if (id.x >= width || id.y >= height || id.z >= constants.size.z) {
        return;
    }
    let view = constants.size.w + id.z;
    let texel = int4(int2(id.xy), int(id.z), 0);
    let index = (view * height + id.y) * width + id.x;
    let holds = constants.layout.x;
    let ids4 = ids.Load(texel);
    let background = ids4.x == kBackground;
    if ((holds & kHoldsLdrColor) != 0) {
        var ldr = color.Load(texel);
        if (background && constants.sky.x > 0.0) {
            // The dome along the pixel's ray: from the near plane (reverse-Z depth 1) to twice its
            // distance (1/2).
            let ndc = float2((float(id.x) + 0.5) / float(width) * 2.0 - 1.0,
                             1.0 - (float(id.y) + 0.5) / float(height) * 2.0);
            let toWorld = inverseViewProj(constants.layout.w + view * constants.layout.z);
            let near = mul(toWorld, float4(ndc, 1.0, 1.0));
            let beyond = mul(toWorld, float4(ndc, 0.5, 1.0));
            let d = normalize(beyond.xyz / beyond.w - near.xyz / near.w);
            ldr = float4(srgbEncode(domeAt(d)), 1.0);
        }
        ldrColor[index] = packUnorm(ldr);
    }
    if ((holds & (kHoldsImagePlane | kHoldsCamera)) != 0) {
        float imagePlane = asfloat(0x7F800000u);  // infinity
        float camera = imagePlane;
        if (!background) {
            let u = view * constants.layout.y;
            let ndc = float3((float(id.x) + 0.5) / float(width) * 2.0 - 1.0,
                             1.0 - (float(id.y) + 0.5) / float(height) * 2.0, depth.Load(texel));
            let clip = mul(inverseViewProj(constants.layout.w + view * constants.layout.z),
                           float4(ndc, 1.0));
            let world = clip.xyz / clip.w;
            let eye = load3(uniforms, u + kEye);
            imagePlane = dot(world - eye, load3(uniforms, u + kForward));
            // An orthographic view's rays start on its image plane, which is then the camera.
            camera = uniforms.Load(u + kFlags + 12) != 0 ? imagePlane : length(world - eye);
        }
        if ((holds & kHoldsImagePlane) != 0) {
            distanceToImagePlane[index] = imagePlane;
        }
        if ((holds & kHoldsCamera) != 0) {
            distanceToCamera[index] = camera;
        }
    }
    if ((holds & kHoldsNormal) != 0) {
        normalOut[index] = background ? float4(0.0) : float4(unpackOctahedral(ids4.z), 0.0);
    }
    if ((holds & kHoldsPrimId) != 0) {
        primId[index] = background ? 0u : ids4.x;
    }
    if ((holds & kHoldsInstanceId) != 0) {
        instanceId[index] = background ? 0u : ids4.y;
    }
}
