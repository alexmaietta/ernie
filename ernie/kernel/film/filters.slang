// Sensor filters: compute kernels a render product runs in order on its spectral radiance (see
// `SensorFilter` in render_targets.h). They share one push-descriptor layout over the frame's
// images, so a chain is just the order of kernels; each image a chain writes is a render var.
//
// One dispatch filters every view of a set of render targets, view `z` of the dispatch in their
// layer `z` (a group of `bands` layers of the spectral images) with the uniforms of view
// `firstView + z`.
//
// Bindings: 0 every view's SensorUniforms; inputs 1 spectral radiance (raster pixels),
// 2 G-buffer normal, 3 G-buffer ids, 4 depth; outputs 5 SensorRadiance (sensor pixels, one layer
// per band), 6 Electrons, 7 RawDN, 8 HdrColor, 9 LdrColor, 10 DistanceToImagePlane,
// 11 DistanceToCamera, 12 Normal, 13 PrimId, 14 InstanceId; every image one layer per view.
//
// References:
// - D. C. Brown, "Decentering Distortion of Lenses", Photogrammetric Engineering 32(3):444-462,
//   1966 (lens distortion, in OpenCV's convention).
// - B. K. P. Horn, "Robot Vision", MIT Press, 1986 (cos^4 falloff of image irradiance).
// - EMVA Standard 1288 Release 4.0 Linear, 2021 (the linear sensor model: quantum efficiency,
//   shot, read and dark noise, PRNU, DSNU, gain).
// - D. E. Knuth, "The Art of Computer Programming", vol. 2, 3rd ed., 1997, section 3.4.1
//   (Poisson sampling by multiplying uniforms).
// - IEC 61966-2-1:1999 (the sRGB transfer function).

static const uint kBackground = 0xFFFFFFFF;

struct SensorUniforms {
    float4x4 invViewProj;
    // xyz: camera position.
    float4 eye;
    // xyz: unit viewing direction.
    float4 forward;
    // Focal length in sensor pixels, principal point x and y (pixels from the top-left corner),
    // overscan: the raster image's focal length is the sensor's divided by it.
    float4 intrinsics;
    // Sensor width, height, band count, channel count.
    uint4 size;
    // Brown-Conrady k1, k2, k3, p1.
    float4 distortion;
    // p2, cos^4 vignetting (1 on), photographic exposure scale, exposure time (s).
    float4 lens;
    // Full well (e-), read noise (e-), dark current (e-/s), PRNU (fraction).
    float4 sensor;
    // DSNU (e-), electrons per DN, black level (DN), largest DN.
    float4 adc;
    // Noise on (1), seed, frame seed (FrameSeed of the seed, stream and frame), and the geometry
    // vars the lens writes: bits 0 to 5 for DistanceToImagePlane, DistanceToCamera, Normal, PrimId,
    // InstanceId and SpectralRadiance; the others stand in placeholders.
    uint4 noise;
    // Linear display colour of the background in LdrColor.
    float4 clear;
    // Per band: weights from radiance to linear sRGB (cd/m^2), with the luminous efficacy, the
    // CIE observer and Bradford adaptation from the equal-energy white folded in.
    float4 colour[64];
    // Per band: mean electrons per unit radiance for each channel.
    float4 qe[64];
    // To the views' stride in the buffer, 2304 bytes.
    float4 padding[2];
}

struct FilterConstants {
    // x: the first view's uniforms, yzw unused.
    uint4 view;
}

[[vk::binding(0, 0)]] StructuredBuffer<SensorUniforms> sensorViews;
[[vk::push_constant]] ConstantBuffer<FilterConstants> pass;
[[vk::binding(1, 0)]] Texture2DArray<float> spectral;
[[vk::binding(2, 0)]] Texture2DArray<float4> gNormal;
[[vk::binding(3, 0)]] Texture2DArray<uint4> gIds;
[[vk::binding(4, 0)]] Texture2DArray<float> gDepth;
[[vk::image_format("r32f")]] [[vk::binding(5, 0)]] RWTexture2DArray<float> sensorRadiance;
[[vk::image_format("rgba32f")]] [[vk::binding(6, 0)]] RWTexture2DArray<float4> electrons;
[[vk::image_format("rgba16ui")]] [[vk::binding(7, 0)]] RWTexture2DArray<uint4> rawDn;
[[vk::image_format("rgba32f")]] [[vk::binding(8, 0)]] RWTexture2DArray<float4> hdrColor;
[[vk::image_format("rgba8")]] [[vk::binding(9, 0)]] RWTexture2DArray<float4> ldrColor;
[[vk::image_format("r32f")]] [[vk::binding(10, 0)]] RWTexture2DArray<float> distanceToImagePlane;
[[vk::image_format("r32f")]] [[vk::binding(11, 0)]] RWTexture2DArray<float> distanceToCamera;
[[vk::image_format("rgba32f")]] [[vk::binding(12, 0)]] RWTexture2DArray<float4> normalOut;
[[vk::image_format("r32ui")]] [[vk::binding(13, 0)]] RWTexture2DArray<uint> primId;
[[vk::image_format("r32ui")]] [[vk::binding(14, 0)]] RWTexture2DArray<uint> instanceId;

// The uniforms of the view this thread filters (`inside` sets it, first in every kernel).
static uint filteredView;
#define cam sensorViews[filteredView]

bool inside(uint3 id) {
    filteredView = pass.view.x + id.z;
    return id.x < cam.size.x && id.y < cam.size.y;
}

// Pixels without a prim (prim id 0) are background.
bool isBackground(uint3 pixel) {
    return primId[pixel] == 0;
}

float2 distort(float2 p) {
    let k = cam.distortion;
    let p2 = cam.lens.x;
    let r2 = dot(p, p);
    let radial = 1.0 + r2 * (k.x + r2 * (k.y + r2 * k.z));
    return float2(p.x * radial + 2.0 * k.w * p.x * p.y + p2 * (r2 + 2.0 * p.x * p.x),
                  p.y * radial + k.w * (r2 + 2.0 * p.y * p.y) + 2.0 * p2 * p.x * p.y);
}

// The inverse of `distort` by fixed-point iteration.
float2 undistort(float2 distorted) {
    var p = distorted;
    for (int i = 0; i < 12; ++i) {
        p += distorted - distort(p);
    }
    return p;
}

// Lens: maps each sensor pixel through the inverse distortion to the undistorted raster image,
// resamples spectral radiance there (bilinearly) with cos^4 vignetting, and the geometry outputs
// (nearest). Sensor pixels whose rays leave the raster image are background.
// Bit of `cam.noise.w` set when the frame holds SpectralRadiance.
static const uint kHoldsSpectral = 32u;

// Writes the geometry vars the frame holds (`cam.noise.w`); the others are placeholders.
void writeGeometry(uint3 pixel, float imagePlane, float camera, float4 normal, uint prim,
                   uint instance) {
    let held = cam.noise.w;
    if ((held & 1u) != 0) {
        distanceToImagePlane[pixel] = imagePlane;
    }
    if ((held & 2u) != 0) {
        distanceToCamera[pixel] = camera;
    }
    if ((held & 4u) != 0) {
        normalOut[pixel] = normal;
    }
    if ((held & 8u) != 0) {
        primId[pixel] = prim;
    }
    if ((held & 16u) != 0) {
        instanceId[pixel] = instance;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lensMain(uint3 id : SV_DispatchThreadID) {
    if (!inside(id)) {
        return;
    }
    let f = cam.intrinsics.x;
    let distorted = float2((float(id.x) + 0.5 - cam.intrinsics.y) / f,
                           (cam.intrinsics.z - (float(id.y) + 0.5)) / f);
    let p = undistort(distorted);
    let rasterF = f / cam.intrinsics.w;
    let raster = float2(cam.intrinsics.y + p.x * rasterF, cam.intrinsics.z - p.y * rasterF) - 0.5;
    let size = float2(cam.size.xy);
    let bands = cam.size.z;
    let infinity = asfloat(0x7f800000u);
    let outside = any(raster < float2(-0.5)) || any(raster > size - 0.5);
    let nearest = int2(clamp(floor(raster + 0.5), float2(0.0), size - 1.0));
    let layer = int(id.z);
    let ids = gIds.Load(int4(nearest, layer, 0));
    if (outside || ids.x == kBackground) {
        for (uint band = 0; band < bands && (cam.noise.w & kHoldsSpectral) != 0; ++band) {
            sensorRadiance[int3(id.xy, id.z * bands + band)] = 0.0;
        }
        writeGeometry(id, infinity, infinity, float4(0.0), 0, 0);
        return;
    }
    let base = int2(floor(clamp(raster, float2(0.0), size - 1.0)));
    let next = min(base + 1, int2(size) - 1);
    let w = clamp(raster - float2(base), float2(0.0), float2(1.0));
    var falloff = 1.0;
    if (cam.lens.y > 0.5) {
        let cosine2 = 1.0 / (1.0 + dot(p, p));
        falloff = cosine2 * cosine2;
    }
    // Shading writes only the pixels it shades: background taps count as no radiance, without
    // clearing the raster radiance first.
    let litA = gIds.Load(int4(base.x, base.y, layer, 0)).x != kBackground;
    let litB = gIds.Load(int4(next.x, base.y, layer, 0)).x != kBackground;
    let litC = gIds.Load(int4(base.x, next.y, layer, 0)).x != kBackground;
    let litD = gIds.Load(int4(next.x, next.y, layer, 0)).x != kBackground;
    for (uint band = 0; band < bands && (cam.noise.w & kHoldsSpectral) != 0; ++band) {
        let slice = int(id.z * bands + band);
        let a = litA ? spectral.Load(int4(base.x, base.y, slice, 0)) : 0.0;
        let b = litB ? spectral.Load(int4(next.x, base.y, slice, 0)) : 0.0;
        let c = litC ? spectral.Load(int4(base.x, next.y, slice, 0)) : 0.0;
        let d = litD ? spectral.Load(int4(next.x, next.y, slice, 0)) : 0.0;
        let value = lerp(lerp(a, b, w.x), lerp(c, d, w.x), w.y);
        sensorRadiance[int3(id.xy, slice)] = value * falloff;
    }
    let ndc = float3((float(nearest.x) + 0.5) / size.x * 2.0 - 1.0,
                     1.0 - (float(nearest.y) + 0.5) / size.y * 2.0,
                     gDepth.Load(int4(nearest, layer, 0)));
    let clip = mul(cam.invViewProj, float4(ndc, 1.0));
    let world = clip.xyz / clip.w;
    writeGeometry(id, dot(world - cam.eye.xyz, cam.forward.xyz), length(world - cam.eye.xyz),
                  float4(normalize(gNormal.Load(int4(nearest, layer, 0)).xyz), 0.0), ids.y, ids.z);
}

// QE: mean electrons per channel, sum over bands of radiance times the channel's per-band factor
// (camera equation and quantum efficiency), plus dark current over the exposure, clipped at the
// full well.
[shader("compute")]
[numthreads(8, 8, 1)]
void qeMain(uint3 id : SV_DispatchThreadID) {
    if (!inside(id)) {
        return;
    }
    var e = float4(0.0);
    for (uint band = 0; band < cam.size.z; ++band) {
        e += sensorRadiance[int3(id.xy, id.z * cam.size.z + band)] * cam.qe[band];
    }
    e += cam.sensor.z * cam.lens.w;
    let channels = cam.size.w;
    for (uint c = channels; c < 4; ++c) {
        e[c] = 0.0;
    }
    electrons[id] = min(e, float4(cam.sensor.x));
}

uint3 pcg3d(uint3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> 16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// Uniform in (0, 1].
float uniform01(uint bits) {
    return (float(bits >> 8) + 1.0) / 16777216.0;
}

// Standard normal from two uniforms (Box-Muller).
float gaussian(uint2 bits) {
    let u1 = uniform01(bits.x);
    let u2 = uniform01(bits.y);
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}

// Poisson with mean `mean`: Knuth's product of uniforms below 30, else a normal approximation.
float poisson(float mean, uint3 seed) {
    if (mean <= 0.0) {
        return 0.0;
    }
    if (mean < 30.0) {
        let limit = exp(-mean);
        var product = 1.0;
        var count = 0;
        var state = seed;
        do {
            state = pcg3d(state);
            product *= uniform01(state.x);
            count += 1;
        } while (product > limit && count < 200);
        return float(count - 1);
    }
    let bits = pcg3d(seed);
    return max(0.0, floor(mean + sqrt(mean) * gaussian(bits.xy) + 0.5));
}

// Noise: per-pixel fixed patterns (PRNU gain and DSNU offset, from the seed alone) and temporal
// photon shot noise and read noise (from the frame seed), clipped at the full well.
[shader("compute")]
[numthreads(8, 8, 1)]
void noiseMain(uint3 id : SV_DispatchThreadID) {
    if (!inside(id) || cam.noise.x == 0) {
        return;
    }
    var e = electrons[id];
    for (uint c = 0; c < cam.size.w; ++c) {
        let pattern = pcg3d(uint3(id.x, id.y, cam.noise.y * 4u + c));
        let gain = 1.0 + cam.sensor.w * gaussian(pattern.xy);
        let offset = cam.adc.x * gaussian(pcg3d(pattern).xy);
        let temporal = uint3(id.x ^ (cam.noise.z * 9781u), id.y + cam.noise.z * 6271u,
                             cam.noise.y * 4u + c + 977u);
        let shot = poisson(e[c] * gain, temporal);
        let read = cam.sensor.y * gaussian(pcg3d(temporal + 1u).xy);
        e[c] = clamp(shot + offset + read, 0.0, cam.sensor.x);
    }
    electrons[id] = e;
}

// ADC: digital numbers, electrons over the ISO-dependent electrons per DN plus the black level,
// rounded and clipped to the ADC range.
[shader("compute")]
[numthreads(8, 8, 1)]
void adcMain(uint3 id : SV_DispatchThreadID) {
    if (!inside(id)) {
        return;
    }
    let e = electrons[id];
    let dn = clamp(floor(e / cam.adc.y + cam.adc.z + 0.5), float4(0.0), float4(cam.adc.w));
    var value = uint4(dn);
    for (uint c = cam.size.w; c < 4; ++c) {
        value[c] = 0;
    }
    rawDn[id] = value;
}

// CIE colour: linear sRGB in cd/m^2 from the sensor radiance, through the CIE 1931 observer and
// Bradford adaptation from the equal-energy white to D65, so flat spectra are neutral.
[shader("compute")]
[numthreads(8, 8, 1)]
void cieMain(uint3 id : SV_DispatchThreadID) {
    if (!inside(id)) {
        return;
    }
    var rgb = float3(0.0);
    for (uint band = 0; band < cam.size.z; ++band) {
        rgb += sensorRadiance[int3(id.xy, id.z * cam.size.z + band)] * cam.colour[band].xyz;
    }
    hdrColor[id] = float4(rgb, 1.0);
}

float encodeSrgb(float linear) {
    return linear <= 0.0031308 ? 12.92 * linear : 1.055 * pow(linear, 1.0 / 2.4) - 0.055;
}

// Exposure: the photographic output, HDR colour times USD's linear exposure scale, clipped to
// [0, 1] and sRGB-encoded; background pixels show the clear colour.
[shader("compute")]
[numthreads(8, 8, 1)]
void exposureMain(uint3 id : SV_DispatchThreadID) {
    if (!inside(id)) {
        return;
    }
    let linear = isBackground(id) ? cam.clear.rgb : clamp(hdrColor[id].rgb * cam.lens.z, 0.0, 1.0);
    ldrColor[id] = float4(encodeSrgb(linear.r), encodeSrgb(linear.g), encodeSrgb(linear.b),
                             1.0);
}
