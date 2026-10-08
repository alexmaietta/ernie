// The sensor: turns a frame's raster spectral radiance into its render vars through a chain of
// filters (`SensorFilter`). A planner (sensor_plan.cpp) fuses the chain into one compute kernel per
// plan that imports this module: a spatial head (the lens) resamples the raster radiance into each
// sensor pixel and streams its bands; band filters apply to each band, integrators fold the bands
// into channels (electrons, linear colour), and pixel filters run on those, in chain order, in
// registers. Only the vars a frame holds are written, so intermediates never touch memory.
//
// Custom filters (`SensorFilterModel`) implement `IBandFilter` or `IPixelFilter`, in modules that
// import this one and have no `module` declaration; their `Params` struct holds one float per
// declared parameter, filled from their prims' inputs:* attributes. The built-in filters (`Lens`,
// `Qe`, `Noise`, `Adc`, `Cie`, `Exposure`) read the view's SensorUniforms instead.
//
// One dispatch filters every view of a set of render targets, view `z` of the dispatch in their
// layer `z` (a group of `bands` layers of the spectral images) with the uniforms of view
// `firstView + z`.
//
// Bindings: 0 every view's SensorUniforms; inputs 1 spectral radiance (raster pixels),
// 2 G-buffer normal, 3 G-buffer ids, 4 depth; outputs 5 SpectralRadiance (sensor pixels, one layer
// per band), 6 Electrons, 7 RawDN, 8 HdrColor, 9 LdrColor, 10 DistanceToImagePlane,
// 11 DistanceToCamera, 12 Normal, 13 PrimId, 14 InstanceId, every image one layer per view;
// 15 the custom filters' parameters; for spatial filters, the images between kernels:
// 16 and 17 the spectrum and the channels (a layer per channel domain per view) a kernel reads,
// 18 and 19 those it writes for the next.
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
// - Chrono::Sensor's filter graphs, which this chain follows: A. Elmquist et al., "Methods and
//   Models for Simulating Autonomous Vehicle Sensors", IEEE T-IV 5(4), 2020.
//   https://doi.org/10.1109/TIV.2020.3003524

module sensor;

static const uint kBackground = 0xFFFFFFFF;

public struct SensorUniforms {
    public float4x4 invViewProj;
    // xyz: camera position.
    public float4 eye;
    // xyz: unit viewing direction.
    public float4 forward;
    // Focal length in sensor pixels, principal point x and y (pixels from the top-left corner),
    // overscan: the raster image's focal length is the sensor's divided by it.
    public float4 intrinsics;
    // Sensor width, height, band count, channel count.
    public uint4 size;
    // Brown-Conrady k1, k2, k3, p1.
    public float4 distortion;
    // p2, cos^4 vignetting (1 on), photographic exposure scale, exposure time (s).
    public float4 lens;
    // Full well (e-), read noise (e-), dark current (e-/s), PRNU (fraction).
    public float4 sensor;
    // DSNU (e-), electrons per DN, black level (DN), largest DN.
    public float4 adc;
    // Noise on (1), seed, frame seed (FrameSeed of the seed, stream and frame), frame number.
    public uint4 noise;
    // Linear display colour of the background in LdrColor (rgb); w: 1 when the background holds
    // radiance (a visible dome or stars, `skyMain`), which then shows in its place.
    public float4 clear;
    // Per band: weights from radiance to linear sRGB (cd/m^2), with the luminous efficacy, the
    // CIE observer and Bradford adaptation from the equal-energy white folded in.
    public float4 colour[64];
    // Per band: mean electrons per unit radiance for each channel.
    public float4 qe[64];
    // First band's lower wavelength and band width (nm), the lens's f-number for depth of field
    // (0: a pinhole), unused.
    public float4 range;
    // The f-number diffraction takes, the pixel pitch (um), the focal length (mm), the focus
    // distance (scene units; 0 is infinity). The views' stride in the buffer is 2304 bytes.
    public float4 optics;
}

public struct SensorConstants {
    // x: the first view's uniforms, yzw unused.
    public uint4 view;
}

[[vk::binding(0, 0)]] StructuredBuffer<SensorUniforms> sensorViews;
[[vk::push_constant]] ConstantBuffer<SensorConstants> pass;
[[vk::binding(1, 0)]] Texture2DArray<float> rasterRadiance;
[[vk::binding(2, 0)]] Texture2DArray<float4> gNormal;
[[vk::binding(3, 0)]] Texture2DArray<uint4> gIds;
[[vk::binding(4, 0)]] Texture2DArray<float> gDepth;
[[vk::image_format("r32f")]] [[vk::binding(5, 0)]] RWTexture2DArray<float> spectralRadiance;
[[vk::image_format("rgba32f")]] [[vk::binding(6, 0)]] RWTexture2DArray<float4> electrons;
[[vk::image_format("rgba16ui")]] [[vk::binding(7, 0)]] RWTexture2DArray<uint4> rawDn;
[[vk::image_format("rgba32f")]] [[vk::binding(8, 0)]] RWTexture2DArray<float4> hdrColor;
[[vk::image_format("rgba8")]] [[vk::binding(9, 0)]] RWTexture2DArray<float4> ldrColor;
[[vk::image_format("r32f")]] [[vk::binding(10, 0)]] RWTexture2DArray<float> distanceToImagePlane;
[[vk::image_format("r32f")]] [[vk::binding(11, 0)]] RWTexture2DArray<float> distanceToCamera;
[[vk::image_format("rgba32f")]] [[vk::binding(12, 0)]] RWTexture2DArray<float4> normalOut;
[[vk::image_format("r32ui")]] [[vk::binding(13, 0)]] RWTexture2DArray<uint> primId;
[[vk::image_format("r32ui")]] [[vk::binding(14, 0)]] RWTexture2DArray<uint> instanceId;
// The custom filters' parameters, packed in chain order (`SensorPlan::params`).
[[vk::binding(15, 0)]] public StructuredBuffer<float> filterParams;
// Between the kernels a spatial filter splits a chain into: each band at each sensor pixel (a
// group of `bands` layers per view), and each channel domain's values (`kChannel*` layers per
// view), as one kernel left them for the spatial filters heading the next.
[[vk::binding(16, 0)]] Texture2DArray<float> spectrumIn;
[[vk::binding(17, 0)]] Texture2DArray<float4> channelsIn;
[[vk::image_format("r32f")]] [[vk::binding(18, 0)]] RWTexture2DArray<float> spectrumOut;
[[vk::image_format("rgba32f")]] [[vk::binding(19, 0)]] RWTexture2DArray<float4> channelsOut;

// A view's layers of the channel images: one per domain after the spectrum.
public static const uint kChannelElectrons = 0;
public static const uint kChannelDigital = 1;
public static const uint kChannelRgb = 2;
public static const uint kChannelDisplay = 3;
public static const uint kChannelDomains = 4;

// The render vars a plan writes, one bit each in `kRenderVarNames` order (sensor_filters.h).
public static const uint kHoldsSpectral = 1u << 0;
public static const uint kHoldsElectrons = 1u << 1;
public static const uint kHoldsRawDn = 1u << 2;
public static const uint kHoldsHdrColor = 1u << 3;
public static const uint kHoldsLdrColor = 1u << 4;
public static const uint kHoldsDistanceToImagePlane = 1u << 5;
public static const uint kHoldsDistanceToCamera = 1u << 6;
public static const uint kHoldsNormal = 1u << 7;
public static const uint kHoldsPrimId = 1u << 8;
public static const uint kHoldsInstanceId = 1u << 9;

// One sensor pixel of one view: where it is, and the view's uniforms.
public struct PixelContext {
    // x, y: the sensor pixel; z: the view's layer in the dispatch's render targets.
    public uint3 id;
    // The view's index in the uniforms.
    public uint view;

    public SensorUniforms uniforms() {
        return sensorViews[view];
    }

    public uint bands() {
        return sensorViews[view].size.z;
    }

    // Layer `band` of this view's group in the spectral images.
    public int slice(uint band) {
        return int(id.z * sensorViews[view].size.z + band);
    }

    // The centre of band `band`, in nanometres.
    public float wavelength(uint band) {
        let range = sensorViews[view].range;
        return range.x + (float(band) + 0.5) * range.y;
    }

    // The camera's optics: the f-number diffraction takes (the lens's, else the exposure's),
    // the lens's own for depth of field (0: a pinhole, all in focus), the sensor's pixel pitch in
    // micrometres, the focal length in millimetres and the focus distance in scene units (0:
    // infinity).
    public float diffractionFNumber() {
        return sensorViews[view].optics.x;
    }
    public float lensFNumber() {
        return sensorViews[view].range.z;
    }
    public float pixelPitchMicrons() {
        return sensorViews[view].optics.y;
    }
    public float focalLengthMm() {
        return sensorViews[view].optics.z;
    }
    public float focusDistance() {
        return sensorViews[view].optics.w;
    }

    // The frame's number and the exposure time in seconds, for temporal filters.
    public uint frame() {
        return sensorViews[view].noise.w;
    }
    public float exposureTime() {
        return sensorViews[view].lens.w;
    }

    // A uniform number in (0, 1] for this pixel and `salt`, the same every frame (from the
    // sensor's seed alone): for fixed patterns, such as hot pixels.
    public float fixedRandom(uint salt) {
        let seed = sensorViews[view].noise.y;
        return uniform01(pcg3d(uint3(id.x, id.y, seed * 4u + salt * 2654435761u)).x);
    }

    // A uniform number in (0, 1] for this pixel and `salt`, new every frame (from the frame
    // seed, which also differs per view): for temporal noise.
    public float random(uint salt) {
        let frame = sensorViews[view].noise.z;
        return uniform01(
            pcg3d(uint3(id.x ^ (frame * 9781u), id.y + frame * 6271u, salt * 2654435761u + 977u))
                .x);
    }
}

// The pixel at dispatch thread `id`, or false past the image.
public bool beginPixel(uint3 id, out PixelContext pixel) {
    pixel.id = id;
    pixel.view = pass.view.x + id.z;
    let size = sensorViews[pixel.view].size;
    return id.x < size.x && id.y < size.y;
}

// The values filters pass along a pixel's chain, one field per domain.
public struct SensorPixel {
    public float4 electrons;  // per channel (Qe onward)
    public uint4 digital;     // DN per channel (Adc)
    public float3 rgb;        // linear sRGB, cd/m^2 (Cie onward)
    public float4 display;    // sRGB-encoded, alpha 1 (Exposure)
    public bool background;   // no prim at the pixel (prim id 0), or its ray leaves the image
}

// A filter on each band's radiance before integration, such as a transmission filter; band
// `band` is centred on `pixel.wavelength(band)` nanometres.
public interface IBandFilter {
    associatedtype Params;
    static float apply(float radiance, uint band, PixelContext pixel, Params params);
}

// A filter on one pixel's values of its domain after integration, in chain order.
public interface IPixelFilter {
    associatedtype Params;
    static void apply(inout SensorPixel value, PixelContext pixel, Params params);
}

// Folds the bands into channels: `accumulate` once per band in order, then `finish`.
public interface IIntegrator {
    static void accumulate(inout SensorPixel value, uint band, float radiance, PixelContext pixel);
    static void finish(inout SensorPixel value, PixelContext pixel);
}

// A filter that reads whole images and heads a fused kernel: it produces each band of its pixel
// for the filters after it.
public interface ISpatialSource {
    static This at(PixelContext pixel);
    float band(uint band, PixelContext pixel);
}

// What a spatial filter reads: its domain at every sensor pixel of its view, as the chain left it
// before the filter. Pixels outside the image read as the nearest edge pixel.
public struct SpatialImage {
    // The domain's layer among the channel images (`kChannel*`); unused for the spectrum.
    public uint channel;

    // The sensor image's width and height.
    public int2 size(PixelContext pixel) {
        return int2(sensorViews[pixel.view].size.xy);
    }

    // Band `band` of the spectrum at sensor pixel `at`.
    public float band(PixelContext pixel, int2 at, uint band) {
        let p = clamp(at, int2(0), size(pixel) - 1);
        return spectrumIn.Load(int4(p, pixel.slice(band), 0));
    }

    // The domain's values at sensor pixel `at`: electrons, digital numbers (as floats), linear
    // colour (xyz) or display colour.
    public float4 values(PixelContext pixel, int2 at) {
        let p = clamp(at, int2(0), size(pixel) - 1);
        return channelsIn.Load(int4(p, int(pixel.id.z * kChannelDomains + channel), 0));
    }
}

// A spatial filter on the spectrum: each band of its pixel from the bands of the pixels around
// it, before the integrators, such as a point-spread blur or chromatic aberration.
public interface ISpatialBandFilter {
    associatedtype Params;
    static float band(uint band, PixelContext pixel, SpatialImage input, Params params);
}

// A spatial filter on another domain: its pixel's values of the domain (the `SensorPixel` field
// it sets) from those around it, such as blooming on electrons or glare on linear colour.
public interface ISpatialPixelFilter {
    associatedtype Params;
    static void apply(inout SensorPixel value, PixelContext pixel, SpatialImage input,
                      Params params);
}

// Stores band `band` of the pixel's spectrum for the next kernel's spatial filters.
public void storeSpectrum(PixelContext pixel, uint band, float radiance) {
    spectrumOut[int3(pixel.id.xy, pixel.slice(band))] = radiance;
}

// Stores the pixel's values of channel domain `channel` for the next kernel's spatial filters.
public void storeChannel(SensorPixel value, PixelContext pixel, uint channel) {
    float4 stored;
    switch (channel) {
    case kChannelElectrons:
        stored = value.electrons;
        break;
    case kChannelDigital:
        stored = float4(value.digital);
        break;
    case kChannelRgb:
        stored = float4(value.rgb, 1.0);
        break;
    default:
        stored = value.display;
        break;
    }
    channelsOut[int3(pixel.id.xy, pixel.id.z * kChannelDomains + channel)] = stored;
}

float2 distort(SensorUniforms cam, float2 p) {
    let k = cam.distortion;
    let p2 = cam.lens.x;
    let r2 = dot(p, p);
    let radial = 1.0 + r2 * (k.x + r2 * (k.y + r2 * k.z));
    return float2(p.x * radial + 2.0 * k.w * p.x * p.y + p2 * (r2 + 2.0 * p.x * p.x),
                  p.y * radial + k.w * (r2 + 2.0 * p.y * p.y) + 2.0 * p2 * p.x * p.y);
}

// The inverse of `distort` by fixed-point iteration.
float2 undistort(SensorUniforms cam, float2 distorted) {
    var p = distorted;
    for (int i = 0; i < 12; ++i) {
        p += distorted - distort(cam, p);
    }
    return p;
}

// The raster image position (pixels, before rounding) that sensor pixel `at` of `cam` sees
// through the inverse lens distortion.
float2 rasterPositionOf(SensorUniforms cam, int2 at) {
    let f = cam.intrinsics.x;
    let distorted = float2((float(at.x) + 0.5 - cam.intrinsics.y) / f,
                           (cam.intrinsics.z - (float(at.y) + 0.5)) / f);
    let k = cam.distortion;
    let undistorted =
        all(k == float4(0.0)) && cam.lens.x == 0.0 ? distorted : undistort(cam, distorted);
    let rasterF = f / cam.intrinsics.w;
    return float2(cam.intrinsics.y + undistorted.x * rasterF,
                  cam.intrinsics.z - undistorted.y * rasterF) - 0.5;
}

// The distance from the camera, in scene units, of what sensor pixel `at` of the pixel's view
// sees: infinity for the background. For spatial filters such as depth of field.
public float sceneDistance(PixelContext pixel, int2 at) {
    let cam = pixel.uniforms();
    let size = float2(cam.size.xy);
    let raster = rasterPositionOf(cam, clamp(at, int2(0), int2(cam.size.xy) - 1));
    let nearest = int2(clamp(floor(raster + 0.5), float2(0.0), size - 1.0));
    let layer = int(pixel.id.z);
    if (gIds.Load(int4(nearest, layer, 0)).x == kBackground) {
        return asfloat(0x7f800000u);
    }
    let ndc = float3((float(nearest.x) + 0.5) / size.x * 2.0 - 1.0,
                     1.0 - (float(nearest.y) + 0.5) / size.y * 2.0,
                     gDepth.Load(int4(nearest, layer, 0)));
    let clip = mul(cam.invViewProj, float4(ndc, 1.0));
    return length(clip.xyz / clip.w - cam.eye.xyz);
}

// Lens: maps each sensor pixel through the inverse distortion to the undistorted raster image,
// resamples spectral radiance there (bilinearly) with cos^4 vignetting, and the geometry outputs
// (nearest). Sensor pixels whose rays leave the raster image are background.
public struct Lens : ISpatialSource {
    int3 base;   // the bilinear footprint's first texel (x, y, layer)
    int2 next;   // and its last
    float2 weight;
    float falloff;
    // Which of the four taps hold radiance: shading writes the prims' pixels, and the sky the
    // background's when it has any; the others are zero.
    bool4 lit;
    bool background;
    bool dark;  // no radiance: the ray leaves the image, or meets a background without any
    float imagePlane;
    float camera;
    float4 normal;
    uint prim;
    uint instance;

    public static Lens at(PixelContext pixel) {
        let cam = pixel.uniforms();
        let id = pixel.id;
        let f = cam.intrinsics.x;
        let distorted = float2((float(id.x) + 0.5 - cam.intrinsics.y) / f,
                               (cam.intrinsics.z - (float(id.y) + 0.5)) / f);
        let p = undistort(cam, distorted);
        let rasterF = f / cam.intrinsics.w;
        let raster =
            float2(cam.intrinsics.y + p.x * rasterF, cam.intrinsics.z - p.y * rasterF) - 0.5;
        let size = float2(cam.size.xy);
        let infinity = asfloat(0x7f800000u);
        let outside = any(raster < float2(-0.5)) || any(raster > size - 0.5);
        let nearest = int2(clamp(floor(raster + 0.5), float2(0.0), size - 1.0));
        let layer = int(id.z);
        let ids = gIds.Load(int4(nearest, layer, 0));
        Lens lens;
        let sky = cam.clear.w > 0.5;
        lens.background = outside || ids.x == kBackground;
        lens.dark = outside || (lens.background && !sky);
        lens.base = int3(0, 0, layer);
        lens.next = int2(0);
        lens.weight = float2(0.0);
        lens.falloff = 1.0;
        lens.lit = bool4(false);
        lens.imagePlane = infinity;
        lens.camera = infinity;
        lens.normal = float4(0.0);
        lens.prim = 0;
        lens.instance = 0;
        if (lens.dark) {
            return lens;
        }
        let base = int2(floor(clamp(raster, float2(0.0), size - 1.0)));
        let next = min(base + 1, int2(size) - 1);
        lens.base = int3(base, layer);
        lens.next = next;
        lens.weight = clamp(raster - float2(base), float2(0.0), float2(1.0));
        if (cam.lens.y > 0.5) {
            let cosine2 = 1.0 / (1.0 + dot(p, p));
            lens.falloff = cosine2 * cosine2;
        }
        lens.lit = sky ? bool4(true)
                       : bool4(gIds.Load(int4(base.x, base.y, layer, 0)).x != kBackground,
                               gIds.Load(int4(next.x, base.y, layer, 0)).x != kBackground,
                               gIds.Load(int4(base.x, next.y, layer, 0)).x != kBackground,
                               gIds.Load(int4(next.x, next.y, layer, 0)).x != kBackground);
        if (lens.background) {
            return lens;  // the sky's radiance, at infinity with no prim
        }
        let ndc = float3((float(nearest.x) + 0.5) / size.x * 2.0 - 1.0,
                         1.0 - (float(nearest.y) + 0.5) / size.y * 2.0,
                         gDepth.Load(int4(nearest, layer, 0)));
        let clip = mul(cam.invViewProj, float4(ndc, 1.0));
        let world = clip.xyz / clip.w;
        lens.imagePlane = dot(world - cam.eye.xyz, cam.forward.xyz);
        lens.camera = length(world - cam.eye.xyz);
        lens.normal = float4(normalize(gNormal.Load(int4(nearest, layer, 0)).xyz), 0.0);
        lens.prim = ids.y;
        lens.instance = ids.z;
        return lens;
    }

    public float band(uint band, PixelContext pixel) {
        if (dark) {
            return 0.0;
        }
        let slice = pixel.slice(band);
        let a = lit.x ? rasterRadiance.Load(int4(base.x, base.y, slice, 0)) : 0.0;
        let b = lit.y ? rasterRadiance.Load(int4(next.x, base.y, slice, 0)) : 0.0;
        let c = lit.z ? rasterRadiance.Load(int4(base.x, next.y, slice, 0)) : 0.0;
        let d = lit.w ? rasterRadiance.Load(int4(next.x, next.y, slice, 0)) : 0.0;
        let value = lerp(lerp(a, b, weight.x), lerp(c, d, weight.x), weight.y);
        return value * falloff;
    }

    // Writes the geometry vars of `holds` (kHolds* bits) and marks the background.
    public void finish(inout SensorPixel value, PixelContext pixel, uint holds) {
        value.background = prim == 0;
        let id = pixel.id;
        if ((holds & kHoldsDistanceToImagePlane) != 0) {
            distanceToImagePlane[id] = imagePlane;
        }
        if ((holds & kHoldsDistanceToCamera) != 0) {
            distanceToCamera[id] = camera;
        }
        if ((holds & kHoldsNormal) != 0) {
            normalOut[id] = normal;
        }
        if ((holds & kHoldsPrimId) != 0) {
            primId[id] = prim;
        }
        if ((holds & kHoldsInstanceId) != 0) {
            instanceId[id] = instance;
        }
    }
}

// Writes band `band` of SpectralRadiance.
public void writeSpectral(PixelContext pixel, uint band, float radiance) {
    spectralRadiance[int3(pixel.id.xy, pixel.slice(band))] = radiance;
}

// QE: mean electrons per channel, sum over bands of radiance times the channel's per-band factor
// (camera equation and quantum efficiency), plus dark current over the exposure, clipped at the
// full well.
public struct Qe : IIntegrator {
    public static void accumulate(inout SensorPixel value, uint band, float radiance,
                                  PixelContext pixel) {
        value.electrons += radiance * sensorViews[pixel.view].qe[band];
    }

    public static void finish(inout SensorPixel value, PixelContext pixel) {
        let cam = pixel.uniforms();
        var e = value.electrons;
        e += cam.sensor.z * cam.lens.w;
        let channels = cam.size.w;
        for (uint c = channels; c < 4; ++c) {
            e[c] = 0.0;
        }
        value.electrons = min(e, float4(cam.sensor.x));
    }
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
public struct Noise {
    public static void apply(inout SensorPixel value, PixelContext pixel) {
        let cam = pixel.uniforms();
        if (cam.noise.x == 0) {
            return;
        }
        let id = pixel.id;
        var e = value.electrons;
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
        value.electrons = e;
    }
}

// ADC: digital numbers, electrons over the ISO-dependent electrons per DN plus the black level,
// rounded and clipped to the ADC range.
public struct Adc {
    public static void apply(inout SensorPixel value, PixelContext pixel) {
        let cam = pixel.uniforms();
        let dn = clamp(floor(value.electrons / cam.adc.y + cam.adc.z + 0.5), float4(0.0),
                       float4(cam.adc.w));
        var digital = uint4(dn);
        for (uint c = cam.size.w; c < 4; ++c) {
            digital[c] = 0;
        }
        value.digital = digital;
    }
}

// CIE colour: linear sRGB in cd/m^2 from the sensor radiance, through the CIE 1931 observer and
// Bradford adaptation from the equal-energy white to D65, so flat spectra are neutral.
public struct Cie : IIntegrator {
    public static void accumulate(inout SensorPixel value, uint band, float radiance,
                                  PixelContext pixel) {
        value.rgb += radiance * sensorViews[pixel.view].colour[band].xyz;
    }

    public static void finish(inout SensorPixel value, PixelContext pixel) {}
}

float encodeSrgb(float linear) {
    return linear <= 0.0031308 ? 12.92 * linear : 1.055 * pow(linear, 1.0 / 2.4) - 0.055;
}

// Exposure: the photographic output, HDR colour times USD's linear exposure scale, clipped to
// [0, 1] and sRGB-encoded; background pixels show the clear colour unless the sky fills them.
public struct Exposure {
    public static void apply(inout SensorPixel value, PixelContext pixel) {
        let cam = pixel.uniforms();
        let linear = value.background && cam.clear.w < 0.5
                         ? cam.clear.rgb
                         : clamp(value.rgb * cam.lens.z, 0.0, 1.0);
        value.display =
            float4(encodeSrgb(linear.r), encodeSrgb(linear.g), encodeSrgb(linear.b), 1.0);
    }
}

// Writes the integrated and filtered vars of `holds` (kHolds* bits).
public void writeOutputs(SensorPixel value, PixelContext pixel, uint holds) {
    let id = pixel.id;
    if ((holds & kHoldsElectrons) != 0) {
        electrons[id] = value.electrons;
    }
    if ((holds & kHoldsRawDn) != 0) {
        rawDn[id] = value.digital;
    }
    if ((holds & kHoldsHdrColor) != 0) {
        hdrColor[id] = float4(value.rgb, 1.0);
    }
    if ((holds & kHoldsLdrColor) != 0) {
        ldrColor[id] = value.display;
    }
}
