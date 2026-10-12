// SPDX-License-Identifier: BSD-3-Clause
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
// Bindings: 0 the SensorUniforms, shared then per view (`sensorData`); inputs 1 spectral radiance
// (raster pixels), 3 G-buffer ids; outputs 2 DegreeOfLinearPolarization, 4
// AngleOfLinearPolarization, 5 SpectralRadiance (sensor pixels, one layer per band), 6 Electrons,
// 7 RawDN, 8 HdrColor, 9 LdrColor, 10 DistanceToImagePlane, 11 DistanceToCamera, 12 Normal,
// 13 PrimId, 14 InstanceId, 15 LinearStokes, every image one layer per view; for spatial
// filters, the images between kernels:
// 16 and 17 the spectrum and the channels (a layer per channel domain per view) a kernel reads,
// 18 and 19 those it writes for the next; 20 to 27 the AOV slots; 28 and 29 the temporal filters'
// state from the previous frame and for the next (17 is sampled linearly for pyramids); 30 the
// MotionVector var; 31 per raster pixel, its motion (xy) and motion blur's coverage (z). The
// G-buffer normal and depth come from the bindless heap (bindless.slang), and the custom
// filters' parameters through their address, both in the push constants: devices push at most
// 32 descriptors.
//
// The lens head's projection comes from module `lensProjection`, which the renderer generates for
// the camera's lens (`SensorPlan::lens`) from lens.slang's projections.
//
// References:
// - EMVA Standard 1288 Release 4.0 Linear, 2021 (the linear sensor model: quantum efficiency,
//   shot, read and dark noise, PRNU, DSNU, gain).
// - D. E. Knuth, "The Art of Computer Programming", vol. 2, 3rd ed., 1997, section 3.4.1
//   (Poisson sampling by multiplying uniforms).
// - IEC 61966-2-1:1999 (the sRGB transfer function).
// - Chrono::Sensor's filter graphs, which this chain follows: A. Elmquist et al., "Methods and
//   Models for Simulating Autonomous Vehicle Sensors", IEEE T-IV 5(4), 2020.
//   https://doi.org/10.1109/TIV.2020.3003524

module sensor;

import bindless;
import cube_faces;
import lens;
import lensProjection;
import projection;

static const uint kBackground = 0xFFFFFFFF;

// Distant lights whose stray light the sensor adds, such as the sun (renderer.cpp's).
public static const uint kStraySources = 4;

// The uniforms every view shares, as binding 0 holds them (renderer.cpp's SensorUniforms); its
// per-view fields are unused. Read in place (`sensorShared`): a copy whose arrays are indexed at
// run time stays whole in each thread's memory.
public struct SensorShared {
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
    // The lens's distortion: Brown-Conrady k1, k2, k3, p1, or the fisheye's k1 to k4.
    public float4 distortion;
    // Brown-Conrady p2, natural vignetting (1 on), photographic exposure scale, exposure time (s).
    public float4 lens;
    // Full well (e-), read noise (e-), dark current (e-/s), PRNU (fraction).
    public float4 sensor;
    // DSNU (e-), electrons per DN, black level (DN), largest DN.
    public float4 adc;
    // Noise on (1), seed, frame seed (FrameSeed of the seed, stream and frame), frame number.
    public uint4 noise;
    // Linear display colour of the background in LdrColor (rgb); w: 1 when the background holds
    // radiance (a visible dome or stars, `skyMain`), which then shows in its place; 2 with motion
    // blur and no sky, when every pixel holds radiance and the clear colour shows by the share of
    // the exposure no prim covered (binding 31).
    public float4 clear;
    // Per band: weights from radiance to linear sRGB (cd/m^2), with the luminous efficacy, the
    // CIE observer and Bradford adaptation from the equal-energy white folded in.
    public float4 colour[64];
    // Per band: mean electrons per unit radiance for each channel.
    public float4 qe[64];
    // First band's lower wavelength and band width (nm), the lens's f-number for depth of field
    // (0: a pinhole), 1 when the projection is orthographic (distances to the camera are along the
    // view direction).
    public float4 range;
    // The f-number diffraction takes, the pixel pitch (um), the focal length (mm), the focus
    // distance (scene units; 0 is infinity).
    public float4 optics;
    // Per band: photons a pixel collects over the exposure per unit radiance, before quantum
    // efficiency, four per float4.
    public float4 photons[16];
    // A camera rendered on cube faces (`kLensCube`): their shared projection's inverse, clip
    // space to the face's camera frame; and x the faces per view, y their focal length in raster
    // pixels, zw their principal point.
    public float4x4 faceInvProj;
    public float4 faces;
    // A pushbroom's (`kPushbroomCamera`) lens: Brown-Conrady p1 and p2 (its k1 to k3 and line's
    // normalised y are `distortion`'s); zw unused.
    public float4 pushbroom;
    // Distant lights for stray light: xyz the unit direction towards each in the camera's frame
    // (x right, y up, the camera looking down -z), w 1 for a light, 0 past the last
    // (`PixelContext.sourceInCamera`). In the buffer, in the world's frame.
    public float4 sources[kStraySources];
    // The camera's polarizer: x its angle (radians, from the image's +x towards +y), y 1 for a
    // mosaic (`polarizerAngle`), z and w its principal transmittances along and across its axis;
    // k1 = k2 = 1 without one.
    public float4 polarizer;
    float4 facesPadding[5];
    // Per band, two to a float4 (`chromaticOf`): the lens's image scale (lateral colour) and focal
    // length shift in millimetres (axial colour).
    public float4 chromatic[32];
    // Per light of `sources` and band, four bands to a float4 and 16 float4 per light: the
    // radiance whose image puts its irradiance at normal incidence on the focal plane,
    // E 4 N^2 / (pi T) (`PixelContext.sourceFocalRadiance`). The struct is 4352 bytes
    // (`kSensorSharedBytes`).
    public float4 sourceRadiance[16 * kStraySources];
}

// A view's uniforms, as the filters take them: `SensorShared`'s fields without its per-band
// arrays (read those from `sensorShared`), and the view's own camera and frame seed. Small
// enough to keep in registers.
public struct SensorUniforms {
    public float4x4 invViewProj;
    public float4 eye;
    public float4 forward;
    public float4 intrinsics;
    public uint4 size;
    public float4 distortion;
    public float4 lens;
    public float4 sensor;
    public float4 adc;
    public uint4 noise;
    public float4 clear;
    public float4 range;
    public float4 optics;
    public float4x4 faceInvProj;
    public float4 faces;
    public float4 pushbroom;
    public float4 polarizer;
}

// Band `band`'s image scale (x) and focal length shift in millimetres (y) through `cam`'s lens.
public float2 chromaticOf(SensorUniforms cam, uint band) {
    // From the buffer: indexing a copy's array would keep the copy in memory.
    let pair = sensorShared[0].chromatic[band / 2];
    return (band & 1u) == 0u ? pair.xy : pair.zw;
}

public struct SensorConstants {
    // x: the first view's uniforms, y: the AOV slots held (a bit each), z: 1 when the state holds
    // the product's previous frame, w: state slots per view.
    public uint4 view;
    // A lens of rays (`kLensTable`): x their first float in `params`, y the pixels between
    // them, z and w their columns and rows.
    public uint4 table;
    // x: the layers from a raster radiance layer to its Stokes Q, and on to U, for a polarizer
    // (0 without them); yzw unused.
    public uint4 stokes;
    // The custom filters' parameters, packed in chain order (`SensorPlan::params`), then a lens
    // of rays' table.
    public float* params;
    // The G-buffer normal's and depth's slots in the bindless heap's `heapTextureArrays`.
    public uint normalSlot;
    public uint depthSlot;
}

// Binding 0 holds the uniforms every view shares, a SensorUniforms whose per-view fields are
// unused, then each view's block (renderer.cpp's SensorView): its inverse view-projection and
// camera-to-world pose (glm's column-major order), eye, forward, distortion (a pushbroom's line
// differs per view) and frame seed (x).
[[vk::binding(0, 0)]] ByteAddressBuffer sensorData;
// The same buffer, its shared uniforms as element 0: reading a field through it reads that field
// alone.
[[vk::binding(0, 0)]] StructuredBuffer<SensorShared> sensorShared;
[[vk::push_constant]] ConstantBuffer<SensorConstants> pass;

static const uint kSensorSharedBytes = 4352;
static const uint kSensorViewBytes = 192;

// The column-major matrix at `address` of the sensor's buffer.
float4x4 loadMatrix(uint address) {
    return transpose(float4x4(asfloat(sensorData.Load4(address)),
                              asfloat(sensorData.Load4(address + 16)),
                              asfloat(sensorData.Load4(address + 32)),
                              asfloat(sensorData.Load4(address + 48))));
}

// The uniforms every view shares.
SensorUniforms sharedUniforms() {
    SensorUniforms cam;
    cam.invViewProj = sensorShared[0].invViewProj;
    cam.eye = sensorShared[0].eye;
    cam.forward = sensorShared[0].forward;
    cam.intrinsics = sensorShared[0].intrinsics;
    cam.size = sensorShared[0].size;
    cam.distortion = sensorShared[0].distortion;
    cam.lens = sensorShared[0].lens;
    cam.sensor = sensorShared[0].sensor;
    cam.adc = sensorShared[0].adc;
    cam.noise = sensorShared[0].noise;
    cam.clear = sensorShared[0].clear;
    cam.range = sensorShared[0].range;
    cam.optics = sensorShared[0].optics;
    cam.faceInvProj = sensorShared[0].faceInvProj;
    cam.faces = sensorShared[0].faces;
    cam.pushbroom = sensorShared[0].pushbroom;
    cam.polarizer = sensorShared[0].polarizer;
    return cam;
}

// The byte address of view `view`'s block.
uint viewBlock(uint view) {
    return kSensorSharedBytes + view * kSensorViewBytes;
}

// View `view`'s uniforms: the shared ones with its own camera and frame seed. Its stray light
// sources stay in the world's frame (`PixelContext.sourceInCamera` turns them). Nothing may
// index an array of the copy at run time, or the whole copy stays in memory: read those from
// `sensorShared`.
SensorUniforms viewUniforms(uint view) {
    var cam = sharedUniforms();
    let base = viewBlock(view);
    cam.invViewProj = loadMatrix(base);
    cam.eye = asfloat(sensorData.Load4(base + 128));
    cam.forward = asfloat(sensorData.Load4(base + 144));
    cam.distortion = asfloat(sensorData.Load4(base + 160));
    cam.noise.z = sensorData.Load(base + 176);
    return cam;
}
[[vk::binding(1, 0)]] Texture2DArray<float> rasterRadiance;
[[vk::image_format("r32f")]] [[vk::binding(2, 0)]] RWTexture2DArray<float> degreeOfPolarization;
[[vk::binding(3, 0)]] Texture2DArray<uint4> gIds;
[[vk::image_format("r32f")]] [[vk::binding(4, 0)]] RWTexture2DArray<float> angleOfPolarization;
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
[[vk::image_format("rgba32f")]] [[vk::binding(15, 0)]] RWTexture2DArray<float4> linearStokes;

// The G-buffer's normal and depth at texel `texel` of layer `layer`.
float4 gNormalAt(int2 texel, int layer) {
    return heapTextureArrays[pass.normalSlot].Load(int4(texel, layer, 0));
}
float gDepthAt(int2 texel, int layer) {
    return heapTextureArrays[pass.depthSlot].Load(int4(texel, layer, 0)).x;
}

// Custom filters' parameter `i` (`SensorConstants::params`).
public float filterParam(uint i) {
    return pass.params[i];
}
// Between the kernels a spatial filter splits a chain into: each band at each sensor pixel (a
// group of `bands` layers per view), and each channel domain's values (`kChannel*` layers per
// view), as one kernel left them for the spatial filters heading the next.
[[vk::binding(16, 0)]] Texture2DArray<float> spectrumIn;
// With linear filtering for reading the channel images' pyramids (`SpatialImage.valuesAt`).
[[vk::binding(17, 0)]] Sampler2DArray<float4> channelsIn;
[[vk::image_format("r32f")]] [[vk::binding(18, 0)]] RWTexture2DArray<float> spectrumOut;
[[vk::image_format("rgba32f")]] [[vk::binding(19, 0)]] RWTexture2DArray<float4> channelsOut;
// The product's AOV slots (`kAovSlotNames`), one layer per view, which filters write with
// `writeOutput`.
[[vk::image_format("rgba32f")]] [[vk::binding(20, 0)]] RWTexture2DArray<float4> aov0;
[[vk::image_format("rgba32f")]] [[vk::binding(21, 0)]] RWTexture2DArray<float4> aov1;
[[vk::image_format("rgba32f")]] [[vk::binding(22, 0)]] RWTexture2DArray<float4> aov2;
[[vk::image_format("rgba32f")]] [[vk::binding(23, 0)]] RWTexture2DArray<float4> aov3;
[[vk::image_format("rgba32f")]] [[vk::binding(24, 0)]] RWTexture2DArray<float4> aov4;
[[vk::image_format("rgba32f")]] [[vk::binding(25, 0)]] RWTexture2DArray<float4> aov5;
[[vk::image_format("rgba32f")]] [[vk::binding(26, 0)]] RWTexture2DArray<float4> aov6;
[[vk::image_format("rgba32f")]] [[vk::binding(27, 0)]] RWTexture2DArray<float4> aov7;
// Temporal filters' state (`readState`, `writeState`): what they stored on the product's previous
// frame and what they store on this one, `pass.view.w` slots per view.
[[vk::image_format("rgba32f")]] [[vk::binding(28, 0)]] RWTexture2DArray<float4> statePrevious;
[[vk::image_format("rgba32f")]] [[vk::binding(29, 0)]] RWTexture2DArray<float4> stateNext;
// The MotionVector var (sensor pixels).
[[vk::image_format("rg32f")]] [[vk::binding(30, 0)]] RWTexture2DArray<float2> motionVector;
// Per raster pixel (a layer per view): xy, with the MotionVector var, its motion since the last
// frame in raster pixels (motion_vectors.slang and the drawables' motion passes); z, with motion
// blur (`clear.w` 2), the share of its exposure prims covered (motion.slang). One image, as
// devices push at most 32 descriptors.
[[vk::image_format("rgba32f")]] [[vk::binding(31, 0)]] RWTexture2DArray<float4> gRaster;

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
public static const uint kHoldsMotionVector = 1u << 10;
public static const uint kHoldsDegreeOfLinearPolarization = 1u << 11;
public static const uint kHoldsAngleOfLinearPolarization = 1u << 12;
public static const uint kHoldsLinearStokes = 1u << 13;
static const uint kHoldsPolarization =
    kHoldsDegreeOfLinearPolarization | kHoldsAngleOfLinearPolarization | kHoldsLinearStokes;

// One sensor pixel of one view: where it is, and the view's uniforms.
public struct PixelContext {
    // x, y: the sensor pixel; z: the view's layer in the dispatch's render targets.
    public uint3 id;
    // The view's index in the uniforms.
    public uint view;
    // The AOV slot of the running filter's output 0 (the plan sets it before each filter); its
    // output k is slot outputBase + k.
    public uint outputBase;
    // Slots the product holds, a bit each, so writes to others are skipped.
    public uint heldOutputs;
    // The state slot of the running filter's state 0 (the plan sets it before each filter).
    public uint stateBase;

    public SensorUniforms uniforms() {
        return viewUniforms(view);
    }

    public uint bands() {
        return sensorShared[0].size.z;
    }

    // Layer `band` of this view's group in the spectral images.
    public int slice(uint band) {
        return int(id.z * sensorShared[0].size.z + band);
    }

    // The centre of band `band`, in nanometres.
    public float wavelength(uint band) {
        let range = sensorShared[0].range;
        return range.x + (float(band) + 0.5) * range.y;
    }

    // The camera's optics: the f-number diffraction takes (the lens's, else the exposure's),
    // the lens's own for depth of field (0: a pinhole, all in focus), the sensor's pixel pitch in
    // micrometres, the focal length in millimetres and the focus distance in scene units (0:
    // infinity).
    public float diffractionFNumber() {
        return sensorShared[0].optics.x;
    }
    public float lensFNumber() {
        return sensorShared[0].range.z;
    }
    public float pixelPitchMicrons() {
        return sensorShared[0].optics.y;
    }
    public float focalLengthMm() {
        return sensorShared[0].optics.z;
    }
    public float focusDistance() {
        return sensorShared[0].optics.w;
    }

    // Distant light `source` (below `kStraySources`), for stray light: the unit direction towards
    // it in the camera's frame (x right, y up, the camera looking down -z), w 1 for a light and 0
    // for none; and per band the radiance whose image puts its irradiance at normal incidence on
    // the focal plane (through the camera equation E = pi T L / (4 N^2) on axis), so a share k of
    // that irradiance reaching the sensor reads as k times this radiance.
    public float4 sourceInCamera(uint source) {
        // From the world's frame through the view's pose: world to camera is its rotation
        // transposed.
        let pose = loadMatrix(viewBlock(view) + 64);
        let toCamera = transpose(float3x3(pose[0].xyz, pose[1].xyz, pose[2].xyz));
        let world = sensorShared[0].sources[source];
        let direction = mul(toCamera, world.xyz);
        let unit = length(direction) > 0.0 ? normalize(direction) : direction;
        return float4(unit, world.w);
    }
    public float sourceFocalRadiance(uint source, uint band) {
        return sensorShared[0].sourceRadiance[16 * source + band / 4][band % 4];
    }

    // Photons collected over the exposure per unit spectral radiance in band `band` (the camera
    // equation, before quantum efficiency), for integrators with their own QE; and the dark
    // current's electrons over the exposure.
    public float photonsPerRadiance(uint band) {
        return sensorShared[0].photons[band / 4][band % 4];
    }
    public float darkElectrons() {
        return sensorShared[0].sensor.z * sensorShared[0].lens.w;
    }
    // The sensor's channel count (1 mono, 3 RGB, 4 RGB-NIR).
    public uint channels() {
        return sensorShared[0].size.w;
    }

    // The frame's number and the exposure time in seconds, for temporal filters.
    public uint frame() {
        return sensorShared[0].noise.w;
    }
    public float exposureTime() {
        return sensorShared[0].lens.w;
    }

    // A uniform number in (0, 1] for this pixel and `salt`, the same every frame (from the
    // sensor's seed alone): for fixed patterns, such as hot pixels.
    public float fixedRandom(uint salt) {
        let seed = sensorShared[0].noise.y;
        return uniform01(pcg3d(uint3(id.x, id.y, seed * 4u + salt * 2654435761u)).x);
    }

    // A standard normal number for `key` and `salt`, the same every frame (from the sensor's seed
    // alone), shared by every pixel with that key: for patterns of parts of the sensor, such as
    // a key of (column, 0) for each detector of a line scanner.
    public float fixedNormal(uint2 key, uint salt) {
        let seed = sensorShared[0].noise.y;
        let bits = pcg3d(uint3(key.x * 7919u + 13u, key.y * 104729u + 7u,
                               seed * 4u + salt * 2654435761u + 1u));
        return gaussian(bits.xy);
    }

    // A uniform number in (0, 1] for this pixel and `salt`, new every frame (from the frame
    // seed, which also differs per view): for temporal noise.
    public float random(uint salt) {
        let frame = sensorData.Load(viewBlock(view) + 176);
        return uniform01(
            pcg3d(uint3(id.x ^ (frame * 9781u), id.y + frame * 6271u, salt * 2654435761u + 977u))
                .x);
    }
}

// The pixel at dispatch thread `id`, or false past the image.
public bool beginPixel(uint3 id, out PixelContext pixel) {
    pixel.id = id;
    pixel.view = pass.view.x + id.z;
    pixel.outputBase = 0;
    pixel.heldOutputs = pass.view.y;
    pixel.stateBase = 0;
    let size = sensorShared[0].size;
    return id.x < size.x && id.y < size.y;
}

// The values filters pass along a pixel's chain, one field per domain.
public struct SensorPixel {
    public float4 electrons;  // per channel (Qe onward)
    public uint4 digital;     // DN per channel (Adc)
    public float3 rgb;        // linear sRGB, cd/m^2 (Cie onward)
    public float4 display;    // sRGB-encoded, alpha 1 (Exposure)
    public bool background;   // no prim at the pixel (prim id 0), or its ray leaves the image
    // The share of the exposure a prim covered the pixel: 0 or 1, or between with motion blur.
    public float coverage;
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

// A custom integrator, in place of `qe` (domain electrons: it sets `value.electrons`, typically
// radiance times `pixel.photonsPerRadiance(band)` times its own quantum efficiency, plus
// `pixel.darkElectrons()` in `finish`) or of `cie` (domain rgb: it sets `value.rgb`).
public interface ICustomIntegrator {
    associatedtype Params;
    static void accumulate(inout SensorPixel value, uint band, float radiance, PixelContext pixel,
                           Params params);
    static void finish(inout SensorPixel value, PixelContext pixel, Params params);
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
        return int2(sensorShared[0].size.xy);
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

    // For filters with a pyramid (`SensorFilterModel::pyramid`): the domain's values bilinearly
    // at `uv` (0 to 1 across the image) of pyramid level `level` (0 the image, each level half
    // the size, each texel the mean of what it covers; fractional levels blend two), clamped at
    // the edges. Level k is a box blur of about 2^k pixels, so wide kernels take a few taps.
    public float4 valuesAt(PixelContext pixel, float2 uv, float level) {
        return channelsIn.SampleLevel(float3(uv, float(pixel.id.z * kChannelDomains + channel)),
                                      level);
    }

    // The pyramid's levels (1 without one).
    public uint levels() {
        uint width = 0;
        uint height = 0;
        uint elements = 0;
        uint levelCount = 0;
        channelsIn.GetDimensions(0, width, height, elements, levelCount);
        return levelCount;
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

// Where sensor pixel `at` of `cam` (output view `view` of the pass) looks in the raster images.
struct RasterTap {
    float2 raster;     // the raster image position, pixels before rounding
    int layer;         // the raster layer: the view's, or its cube face's
    float2 distorted;  // the pixel's normalised image coordinates
    float2 pinhole;    // its pinhole image point (a cube camera's is unused)
    float3 ray;        // its lens ray in the camera's frame (a cube camera's)
}

// A lens of rays' ray at sensor position `pixel` (pixels from the top left corner): the table's
// rays (`pass.table`, in `pass.params`) interpolated bilinearly, normalised.
float3 tableRayAt(float2 pixel) {
    let columns = pass.table.z;
    let rows = pass.table.w;
    let at = clamp(pixel / float(pass.table.y), float2(0.0), float2(columns - 1, rows - 1));
    let x0 = min(uint(at.x), columns - 2);
    let y0 = min(uint(at.y), rows - 2);
    let t = at - float2(x0, y0);
    float3 corner[4];
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        let index = pass.table.x + 3 * ((y0 + i / 2) * columns + x0 + i % 2);
        corner[i] = float3(pass.params[index], pass.params[index + 1], pass.params[index + 2]);
    }
    return normalize(lerp(lerp(corner[0], corner[1], t.x), lerp(corner[2], corner[3], t.x), t.y));
}

// Through the lens's projection onto the pinhole raster image; for a camera rendered on cube
// faces (`kLensCube`), along the lens's ray onto the face it leaves through, which the view's
// faces' consecutive layers hold. A lens of rays (`kLensTable`) gives the ray from its table.
RasterTap rasterTapOf(SensorUniforms cam, int2 at, int view) {
    return rasterTapAt(cam, float2(at) + 0.5, view);
}

// A pushbroom's tap (`kPushbroomCamera`): sensor position `position`'s point of the detector line,
// normalised (x, the line's y), through the lens (lens.slang's `Pushbroom`) to its pinhole point,
// at the raster row of the position's own time, the raster x of the pinhole point's along the
// line. A pinhole point off the line (`pinhole.y` not the line's) is seen at another time, which
// `retimed` finds from the raster's depth. With lateral colour, the band's image is `scale`
// times the reference's about the principal point, so the point is the line's over `scale`.
RasterTap pushbroomTapAt(SensorUniforms cam, float2 position, int view, float scale) {
    let f = cam.intrinsics.x;
    RasterTap tap;
    tap.distorted = float2((position.x - cam.intrinsics.y) / f, cam.distortion.w) / scale;
    tap.pinhole = Pushbroom.undistort(tap.distorted, cam.distortion, cam.pushbroom.xy);
    tap.raster = float2(cam.intrinsics.y + tap.pinhole.x * f / cam.intrinsics.w, position.y) - 0.5;
    tap.layer = view;
    tap.ray = normalize(float3(tap.pinhole, -1.0));
    return tap;
}

// The intrinsics of a pushbroom's raster strip (`PushbroomStrip::Intrinsics`): its raster focal
// length and principal point in NDC x, and its detector line.
float4 stripIntrinsics(SensorUniforms cam) {
    let width = float(cam.size.x);
    return float4(2.0 * cam.intrinsics.x / cam.intrinsics.w / width,
                  2.0 * cam.intrinsics.y / width - 1.0, cam.distortion.w, 0.0);
}

// A pushbroom's tap `tap` of sensor row position `row` (pixels from the top), moved onto the raster
// row that sees what it sees when its pinhole point lies off the detector line (radial distortion
// of a line off the axis, tangential terms, lateral colour): the raster images the line's plane
// of view, row by row, and the pixel's ray from its row's pose leaves that plane. The point along
// the ray as far as what the raster shows about the tap (its depth, from the texel nearest it,
// three times so it settles onto the surface there) is projected through the strip
// (`cameraProject`); rows past the strip's ends hold the nearest end's. Over the background
// (no depth) the ray reaches the far plane.
RasterTap retimed(SensorUniforms cam, RasterTap tap, float row) {
    if (!(abs(tap.pinhole.y - cam.distortion.w) * cam.intrinsics.x > 1e-3)) {
        return tap;
    }
    let size = float2(cam.size.xy);
    let strip = stripIntrinsics(cam);
    // The pixel's ray: its row's pose, through its pinhole point (the strip's line moved to it).
    float3 origin;
    float3 direction;
    cameraRayThrough(cam.invViewProj, float4(strip.xy, tap.pinhole.y, 0.0),
                     float2(tap.pinhole.x * strip.x + strip.y, 1.0 - 2.0 * row / size.y), origin,
                     direction);
    var raster = tap.raster;
    for (int i = 0; i < 3; ++i) {
        let texel = int2(clamp(floor(raster + 0.5), float2(0.0), size - 1.0));
        // Background at the far plane, depth 0 in reverse-Z.
        let depth = gIds.Load(int4(texel, tap.layer, 0)).x == kBackground
                        ? 0.0
                        : gDepthAt(texel, tap.layer);
        let ndc = float2((float(texel.x) + 0.5) / size.x * 2.0 - 1.0,
                         1.0 - (float(texel.y) + 0.5) / size.y * 2.0);
        let seen = cameraPoint(cam.invViewProj, strip, ndc, depth);
        let along = dot(seen - origin, direction) / dot(direction, direction);
        let clip = cameraProject(cam.invViewProj, strip, origin + direction * along);
        raster = float2((clip.x / clip.w + 1.0) * 0.5 * size.x,
                        (1.0 - clip.y / clip.w) * 0.5 * size.y) - 0.5;
    }
    tap.raster = float2(raster.x, clamp(raster.y, 0.0, size.y - 1.0));
    return tap;
}

// The tap of what sensor position `position` sees: `rasterTapAt`'s, a pushbroom's `retimed`.
// Reads the G-buffer, which the overlay passes do not bind.
RasterTap sceneTapAt(SensorUniforms cam, float2 position, int view) {
    let tap = rasterTapAt(cam, position, view);
    return kPushbroomCamera ? retimed(cam, tap, position.y) : tap;
}

// As `rasterTapOf`, at sensor position `position` (pixels from the top left corner).
RasterTap rasterTapAt(SensorUniforms cam, float2 position, int view) {
    if (kPushbroomCamera) {
        return pushbroomTapAt(cam, position, view, 1.0);
    }
    let f = cam.intrinsics.x;
    RasterTap tap;
    tap.distorted = float2((position.x - cam.intrinsics.y) / f,
                           (cam.intrinsics.z - position.y) / f);
    if (kLensTable) {
        tap.ray = tableRayAt(position);
    } else if (kLensCube) {
        tap.ray = lensRayOf(tap.distorted, cam.distortion, cam.lens.x);
    }
    if (kLensCube) {
        let face = cubeFaceOf(tap.ray, int(cam.faces.x));
        let plane = cubeFacePlane(face, tap.ray);
        tap.raster = float2(cam.faces.z + plane.x * cam.faces.y,
                            cam.faces.w - plane.y * cam.faces.y) - 0.5;
        tap.layer = view * int(cam.faces.x) + face;
        tap.pinhole = float2(0.0);
        return tap;
    }
    tap.pinhole = kLensTable
                      ? tap.ray.xy / max(-tap.ray.z, 1e-6)
                      : lensPinholeOf(tap.distorted, cam.distortion, cam.lens.x);
    let rasterF = f / cam.intrinsics.w;
    tap.raster = float2(cam.intrinsics.y + tap.pinhole.x * rasterF,
                        cam.intrinsics.z - tap.pinhole.y * rasterF) - 0.5;
    tap.layer = view;
    tap.ray = normalize(float3(tap.pinhole, -1.0));
    return tap;
}

// Motion `moved` in a cube face's pixels, of what sensor position `position` of view `view` sees,
// in sensor pixels: through the inverse of the lens's local map from sensor to face pixels, its
// steps one sensor pixel along x and y (back instead where a step crosses onto another face).
// Zero where both cross.
float2 faceMotionToSensor(SensorUniforms cam, float2 position, int view, float2 moved) {
    let tap = rasterTapAt(cam, position, view);
    float2 columns[2];
    for (int axis = 0; axis < 2; ++axis) {
        let step = axis == 0 ? float2(1.0, 0.0) : float2(0.0, 1.0);
        var other = rasterTapAt(cam, position + step, view);
        var sign = 1.0;
        if (other.layer != tap.layer) {
            other = rasterTapAt(cam, position - step, view);
            sign = -1.0;
        }
        if (other.layer != tap.layer) {
            return float2(0.0);
        }
        columns[axis] = sign * (other.raster - tap.raster);
    }
    // The map's matrix has columns d(face) / d(sensor x) and d(face) / d(sensor y).
    let det = columns[0].x * columns[1].y - columns[1].x * columns[0].y;
    if (abs(det) < 1e-12) {
        return float2(0.0);
    }
    return float2(columns[1].y * moved.x - columns[1].x * moved.y,
                  -columns[0].y * moved.x + columns[0].x * moved.y) /
           det;
}

// The distances of what raster texel `nearest` of layer `layer` sees, at the end of tap `tap`.
void distancesOf(SensorUniforms cam, RasterTap tap, int2 nearest, out float imagePlane,
                 out float camera) {
    let size = float2(cam.size.xy);
    let ndc = float3((float(nearest.x) + 0.5) / size.x * 2.0 - 1.0,
                     1.0 - (float(nearest.y) + 0.5) / size.y * 2.0,
                     gDepthAt(nearest, tap.layer));
    if (kLensCube) {
        // From the face's own frame, at the camera: along the ray, and along the view's axis
        // (negative behind the image plane, for rays past 90 degrees).
        let clip = mul(cam.faceInvProj, float4(ndc, 1.0));
        camera = length(clip.xyz / clip.w);
        imagePlane = camera * -tap.ray.z;
        return;
    }
    distancesAt(cam, ndc, imagePlane, camera);
}

// The distance from the camera, in scene units, of what sensor pixel `at` of the pixel's view
// sees: infinity for the background. For spatial filters such as depth of field.
public float sceneDistance(PixelContext pixel, int2 at) {
    let cam = pixel.uniforms();
    let size = float2(cam.size.xy);
    let tap = sceneTapAt(cam, float2(clamp(at, int2(0), int2(cam.size.xy) - 1)) + 0.5,
                         int(pixel.id.z));
    let nearest = int2(clamp(floor(tap.raster + 0.5), float2(0.0), size - 1.0));
    if (gIds.Load(int4(nearest, tap.layer, 0)).x == kBackground) {
        return asfloat(0x7f800000u);
    }
    float imagePlane;
    float camera;
    distancesOf(cam, tap, nearest, imagePlane, camera);
    return camera;
}

// The distances of what raster normalised device coordinates `ndc` (with [0, 1] depth z) see:
// along the view from the image plane, and from the camera. An orthographic view's camera is its
// image plane; a pushbroom's is its row's pose (its strip in `invViewProj`; its raster focal
// length, principal point and detector line make its intrinsics, `PushbroomStrip::Intrinsics`).
void distancesAt(SensorUniforms cam, float3 ndc, out float imagePlane, out float camera) {
    if (kPushbroomCamera) {
        let strip = stripIntrinsics(cam);
        float3 origin;
        float3 direction;  // one unit of view depth long
        cameraRayThrough(cam.invViewProj, strip, ndc.xy, origin, direction);
        camera = length(cameraPoint(cam.invViewProj, strip, ndc.xy, ndc.z) - origin);
        imagePlane = camera / length(direction);
        return;
    }
    let clip = mul(cam.invViewProj, float4(ndc, 1.0));
    let world = clip.xyz / clip.w;
    imagePlane = dot(world - cam.eye.xyz, cam.forward.xyz);
    // An orthographic view's rays start on its image plane, which is then the camera.
    camera = cam.range.w != 0.0 ? imagePlane : length(world - cam.eye.xyz);
}

// Lens: maps each sensor pixel through the lens's projection (`lensPinholeOf`) to the pinhole
// raster image, resamples spectral radiance there (bilinearly) with the lens's natural vignetting
// (`lensFalloff`), and the geometry outputs
// (nearest). Sensor pixels whose rays leave the raster image are background. The geometry
// outputs (PrimId and the like) are point samples of the raster texel nearest each ray, so where
// the raster's grid drifts against the sensor's (a pushbroom's overscanned line) they land within
// half a pixel.
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
    float coverage;
    int3 nearest;  // the raster texel (x, y, layer) the geometry vars come from
    bool outside;  // the ray leaves the raster image
    float imagePlane;
    float camera;
    float4 normal;
    uint prim;
    uint instance;

    public static Lens at(PixelContext pixel) {
        let cam = pixel.uniforms();
        let id = pixel.id;
        let tap = sceneTapAt(cam, float2(id.xy) + 0.5, int(id.z));
        let distorted = tap.distorted;
        let p = tap.pinhole;
        let raster = tap.raster;
        let size = float2(cam.size.xy);
        let infinity = asfloat(0x7f800000u);
        let outside = any(raster < float2(-0.5)) || any(raster > size - 0.5);
        let nearest = int2(clamp(floor(raster + 0.5), float2(0.0), size - 1.0));
        let layer = tap.layer;
        let ids = gIds.Load(int4(nearest, layer, 0));
        Lens lens;
        // With motion blur every pixel holds radiance, the sub-frames' sum, as with a sky.
        let motion = cam.clear.w > 1.5;
        let sky = cam.clear.w > 0.5;
        lens.background = outside || ids.x == kBackground;
        lens.dark = outside || (lens.background && !sky);
        lens.coverage = outside           ? 0.0
                        : motion          ? gRaster[uint3(uint2(nearest), uint(layer))].z
                        : lens.background ? 0.0
                                          : 1.0;
        lens.base = int3(0, 0, layer);
        lens.nearest = int3(nearest, layer);
        lens.outside = outside;
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
            if (kLensTable) {
                // From the rays: the aperture's foreshortening |cos theta| times the pixel's
                // solid angle over the axis's 1 / f^2, cos^4 for a pinhole's.
                let at = float2(id.xy) + 0.5;
                let dx = tableRayAt(at + float2(0.5, 0.0)) - tableRayAt(at - float2(0.5, 0.0));
                let dy = tableRayAt(at + float2(0.0, 0.5)) - tableRayAt(at - float2(0.0, 0.5));
                let f = cam.intrinsics.x;
                lens.falloff = abs(tap.ray.z) * length(cross(dx, dy)) * f * f;
            } else {
                lens.falloff = lensFalloff(distorted, p, cam.distortion);
            }
        }
        lens.lit = sky ? bool4(true)
                       : bool4(gIds.Load(int4(base.x, base.y, layer, 0)).x != kBackground,
                               gIds.Load(int4(next.x, base.y, layer, 0)).x != kBackground,
                               gIds.Load(int4(base.x, next.y, layer, 0)).x != kBackground,
                               gIds.Load(int4(next.x, next.y, layer, 0)).x != kBackground);
        if (lens.background) {
            return lens;  // the sky's radiance, at infinity with no prim
        }
        distancesOf(cam, tap, nearest, lens.imagePlane, lens.camera);
        lens.normal = float4(normalize(gNormalAt(nearest, layer).xyz), 0.0);
        lens.prim = ids.y;
        lens.instance = ids.z;
        return lens;
    }

    public float band(uint band, PixelContext pixel) {
        if (kLensChromatic) {
            return chromaticBand(band, pixel);
        }
        if (dark) {
            return 0.0;
        }
        // The band's layer in the raster's group: the view's, or its cube face's.
        let slice = int(uint(base.z) * pixel.bands() + band);
        let intensity = resampled(slice);
        let stokes = int(pass.stokes.x);
        if (stokes == 0) {
            return intensity * falloff;
        }
        return analysed(intensity, resampled(slice + stokes), resampled(slice + 2 * stokes),
                        pixel) *
               falloff;
    }

    // Layer `slice` of the raster radiance, bilinearly between the four held texels of `at`.
    float resampled(int slice) {
        let a = lit.x ? rasterRadiance.Load(int4(base.x, base.y, slice, 0)) : 0.0;
        let b = lit.y ? rasterRadiance.Load(int4(next.x, base.y, slice, 0)) : 0.0;
        let c = lit.z ? rasterRadiance.Load(int4(base.x, next.y, slice, 0)) : 0.0;
        let d = lit.w ? rasterRadiance.Load(int4(next.x, next.y, slice, 0)) : 0.0;
        return lerp(lerp(a, b, weight.x), lerp(c, d, weight.x), weight.y);
    }

    // The radiance the camera's polarizer passes of linear Stokes (`intensity`, `q`, `u`),
    // measured from the image's +x towards +y: a partial polarizer's Mueller matrix, with
    // principal transmittances k1 along its axis and k2 across it, gives
    // ((k1 + k2) I + (k1 - k2) (Q cos 2 t + U sin 2 t)) / 2 at its angle t here.
    //
    // Reference: E. Collett, "Field Guide to Polarization", SPIE FG05, 2005.
    static float analysed(float intensity, float q, float u, PixelContext pixel) {
        let polarizer = sensorShared[0].polarizer;
        let t = polarizerAngle(pixel);
        return 0.5 * ((polarizer.z + polarizer.w) * intensity +
                      (polarizer.z - polarizer.w) * (q * cos(2.0 * t) + u * sin(2.0 * t)));
    }

    // The polarizer's angle at the pixel: its own, or a mosaic's for its place in its 2 x 2 cell,
    // 90, 45 degrees over 135, 0 (Sony's IMX250MZR).
    static float polarizerAngle(PixelContext pixel) {
        let polarizer = sensorShared[0].polarizer;
        if (polarizer.y < 0.5) {
            return polarizer.x;
        }
        let row = pixel.id.y % 2;
        let column = pixel.id.x % 2;
        let degrees = row == 0 ? (column == 0 ? 90.0 : 45.0) : (column == 0 ? 135.0 : 0.0);
        return radians(degrees);
    }

    // Lateral colour (`kLensChromatic`): band `band`'s image is scaled by s about the principal
    // point, so this pixel sees what the reference wavelength's image holds at its position over
    // s; that tap, resampled as `at` does, with its own coverage, so a band can reach an object
    // the reference misses (the fringes). A pushbroom's pixel sees the reference's at its point
    // of the line over s (`pushbroomTapAt`): along the line, and for a line off the axis off it,
    // at another time (`retimed`).
    float chromaticBand(uint band, PixelContext pixel) {
        let cam = pixel.uniforms();
        let scale = chromaticOf(cam, band).x;
        let principal = cam.intrinsics.yz;
        let at = float2(pixel.id.xy) + 0.5;
        let tap = kPushbroomCamera
                      ? retimed(cam, pushbroomTapAt(cam, at, int(pixel.id.z), scale), at.y)
                      : rasterTapAt(cam, principal + (at - principal) / scale, int(pixel.id.z));
        let size = float2(cam.size.xy);
        if (any(tap.raster < float2(-0.5)) || any(tap.raster > size - 0.5)) {
            return 0.0;
        }
        let first = int2(floor(clamp(tap.raster, float2(0.0), size - 1.0)));
        let last = min(first + 1, int2(size) - 1);
        let w = clamp(tap.raster - float2(first), float2(0.0), float2(1.0));
        let sky = cam.clear.w > 0.5;
        let slice = int(uint(tap.layer) * pixel.bands() + band);
        let intensity = heldBilinear(first, last, w, tap.layer, slice, sky);
        let stokes = int(pass.stokes.x);
        if (stokes == 0) {
            return intensity * falloff;
        }
        return analysed(intensity, heldBilinear(first, last, w, tap.layer, slice + stokes, sky),
                        heldBilinear(first, last, w, tap.layer, slice + 2 * stokes, sky), pixel) *
               falloff;
    }

    // Slice `slice` of layer `layer`'s held raster radiance (`heldRadiance`), bilinearly at
    // weights `w` between texels `first` and `last`.
    static float heldBilinear(int2 first, int2 last, float2 w, int layer, int slice, bool sky) {
        let a = heldRadiance(first, layer, slice, sky);
        let b = heldRadiance(int2(last.x, first.y), layer, slice, sky);
        let c = heldRadiance(int2(first.x, last.y), layer, slice, sky);
        let d = heldRadiance(last, layer, slice, sky);
        return lerp(lerp(a, b, w.x), lerp(c, d, w.x), w.y);
    }

    // The raster radiance of texel `texel` in layer `layer`'s band slice `slice`: zero for a
    // background without a sky, which shading leaves unwritten.
    static float heldRadiance(int2 texel, int layer, int slice, bool sky) {
        let held = sky || gIds.Load(int4(texel, layer, 0)).x != kBackground;
        return held ? rasterRadiance.Load(int4(texel, slice, 0)) : 0.0;
    }

    // Writes the geometry vars of `holds` (kHolds* bits) and marks the background.
    public void finish(inout SensorPixel value, PixelContext pixel, uint holds) {
        value.background = prim == 0;
        value.coverage = coverage;
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
        if ((holds & kHoldsMotionVector) != 0) {
            // Raster pixels to sensor pixels: the raster image is drawn `overscan` times wider,
            // so a raster pixel spans `overscan` sensor pixels; a cube camera's face pixels
            // through the lens's local map onto its face.
            let raster = outside ? float2(0.0) : gRaster[uint3(nearest)].xy;
            let cam = pixel.uniforms();
            motionVector[id] = kLensCube
                                   ? faceMotionToSensor(cam, float2(id.xy) + 0.5, int(id.z), raster)
                                   : raster * cam.intrinsics.w;
        }
        if ((holds & kHoldsPolarization) != 0) {
            writePolarization(pixel, holds);
        }
    }

    // Writes the polarization vars of `holds`: the light reaching the pixel before the camera's
    // polarizer, its linear Stokes summed over the bands, resampled at the reference band's tap
    // (lateral colour aside), and its degree sqrt(S1^2 + S2^2) / S0 and angle atan2(S2, S1) / 2
    // from the image's +x towards +y.
    //
    // Reference: E. Collett, "Field Guide to Polarization", SPIE FG05, 2005.
    void writePolarization(PixelContext pixel, uint holds) {
        var stokes = float3(0.0);
        let offset = int(pass.stokes.x);
        if (!dark && offset != 0) {
            for (uint band = 0; band < pixel.bands(); ++band) {
                let slice = int(uint(base.z) * pixel.bands() + band);
                stokes += float3(resampled(slice), resampled(slice + offset),
                                 resampled(slice + 2 * offset));
            }
            stokes *= falloff;
        }
        let id = pixel.id;
        if ((holds & kHoldsDegreeOfLinearPolarization) != 0) {
            degreeOfPolarization[id] = stokes.x > 0.0 ? saturate(length(stokes.yz) / stokes.x) : 0.0;
        }
        if ((holds & kHoldsAngleOfLinearPolarization) != 0) {
            let angle = 0.5 * atan2(stokes.z, stokes.y);
            angleOfPolarization[id] = angle < 0.0 ? angle + 3.14159265 : angle;
        }
        if ((holds & kHoldsLinearStokes) != 0) {
            linearStokes[id] = float4(stokes, 0.0);
        }
    }
}

// Temporal filters: whether the product rendered a frame before this one with the same chain,
// resolution and views, whose state `readState` then returns.
public bool hasHistory(PixelContext pixel) {
    return pass.view.z != 0;
}

// The state's layer: by the view's index in the whole batch (`pixel.view`), as batches larger
// than one set of render targets record in chunks whose layers start again at 0.
int stateLayer(PixelContext pixel, uint slot) {
    return int(pixel.view * pass.view.w + pixel.stateBase + slot);
}

// What the running filter stored in its state slot `slot` (its model's `states`) at this pixel
// on the product's previous frame; zeros without history (`hasHistory`).
public float4 readState(PixelContext pixel, uint slot) {
    if (pass.view.z == 0) {
        return float4(0.0);
    }
    return statePrevious[int3(pixel.id.xy, stateLayer(pixel, slot))];
}

// Stores `value` in the running filter's state slot `slot` for the product's next frame, which
// reads it with `readState`. A filter writes every slot it reads, every frame.
public void writeState(PixelContext pixel, uint slot, float4 value) {
    stateNext[int3(pixel.id.xy, stateLayer(pixel, slot))] = value;
}

// A curve parameter (`SensorFilterParam::size` above 1) at `t` in [0, 1]: linear between its N
// evenly spaced samples, values[0] at 0 and values[N - 1] at 1, held beyond them.
// Value `i` of a table parameter (`SensorFilterParam::table`) whose
// Params field is `start`.
public float paramTable(float start, uint i) {
    return pass.params[uint(start) + i];
}

public float curveAt<let N : int>(float values[N], float t) {
    let at = saturate(t) * float(N - 1);
    let lower = min(int(at), N - 2);
    return lerp(values[lower], values[lower + 1], at - float(lower));
}

// Writes `value` to output `output` of the running filter (its model's `outputs`, in order), the
// AOV a render var asks for as "aov:<name>"; nothing when the product does not hold it.
public void writeOutput(PixelContext pixel, uint output, float4 value) {
    let slot = pixel.outputBase + output;
    if (slot >= 8u || ((pixel.heldOutputs >> slot) & 1u) == 0) {
        return;
    }
    let id = pixel.id;
    switch (slot) {
    case 0:
        aov0[id] = value;
        break;
    case 1:
        aov1[id] = value;
        break;
    case 2:
        aov2[id] = value;
        break;
    case 3:
        aov3[id] = value;
        break;
    case 4:
        aov4[id] = value;
        break;
    case 5:
        aov5[id] = value;
        break;
    case 6:
        aov6[id] = value;
        break;
    default:
        aov7[id] = value;
        break;
    }
}

// Writes band `band` of SpectralRadiance.
public void writeSpectral(PixelContext pixel, uint band, float radiance) {
    spectralRadiance[int3(pixel.id.xy, pixel.slice(band))] = radiance;
}

// QE: mean electrons per channel, sum over bands of radiance times the channel's per-band factor
// (camera equation and quantum efficiency), plus dark current over the exposure. Not yet clipped
// at the full well: filters on electrons see the charge beyond it, which blooming spills to the
// neighbours; the Electrons var and the ADC clip it (`fullWell`).
public struct Qe : IIntegrator {
    public static void accumulate(inout SensorPixel value, uint band, float radiance,
                                  PixelContext pixel) {
        value.electrons += radiance * sensorShared[0].qe[band];
    }

    public static void finish(inout SensorPixel value, PixelContext pixel) {
        let cam = pixel.uniforms();
        var e = value.electrons;
        e += cam.sensor.z * cam.lens.w;
        let channels = cam.size.w;
        for (uint c = channels; c < 4; ++c) {
            e[c] = 0.0;
        }
        value.electrons = e;
    }
}

// The full well in electrons: what a pixel holds, the Electrons var and the ADC clipping there.
public float fullWell(PixelContext pixel) {
    return pixel.uniforms().sensor.x;
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
// photon shot noise and read noise (from the frame seed), at least zero; charge beyond the full
// well stays for filters after it (see `Qe`).
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
            e[c] = max(shot + offset + read, 0.0);
        }
        value.electrons = e;
    }
}

// ADC: digital numbers of the electrons clipped at the full well, over the ISO-dependent
// electrons per DN plus the black level, rounded and clipped to the ADC range.
public struct Adc {
    public static void apply(inout SensorPixel value, PixelContext pixel) {
        let cam = pixel.uniforms();
        let e = min(value.electrons, float4(cam.sensor.x));
        let dn = clamp(floor(e / cam.adc.y + cam.adc.z + 0.5), float4(0.0),
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
        value.rgb += radiance * sensorShared[0].colour[band].xyz;
    }

    public static void finish(inout SensorPixel value, PixelContext pixel) {}
}

float encodeSrgb(float linear) {
    return linear <= 0.0031308 ? 12.92 * linear : 1.055 * pow(linear, 1.0 / 2.4) - 0.055;
}

// Exposure: the photographic output, HDR colour times USD's linear exposure scale, clipped to
// [0, 1] and sRGB-encoded; background pixels show the clear colour unless the sky fills them,
// and with motion blur, the clear colour shows by the share of the exposure no prim covered.
public struct Exposure {
    public static void apply(inout SensorPixel value, PixelContext pixel) {
        let cam = pixel.uniforms();
        let shown = clamp(value.rgb * cam.lens.z, 0.0, 1.0);
        let linear = cam.clear.w > 0.5 && cam.clear.w < 1.5
                         ? shown
                         : lerp(cam.clear.rgb, shown, value.coverage);
        value.display =
            float4(encodeSrgb(linear.r), encodeSrgb(linear.g), encodeSrgb(linear.b), 1.0);
    }
}

// Writes the integrated and filtered vars of `holds` (kHolds* bits).
public void writeOutputs(SensorPixel value, PixelContext pixel, uint holds) {
    let id = pixel.id;
    if ((holds & kHoldsElectrons) != 0) {
        electrons[id] = min(value.electrons, float4(pixel.uniforms().sensor.x));
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

// Overlays of a camera rendered on cube faces (`RasterRenderer::RecordOverlays`): each face's
// drawn over black (32) and over white (33), a layer per view and face as the raster images,
// and resampled through the lens onto LdrColor. Over black they give their colour, already
// weighted by their opacity; the difference over white gives what of the image below shows
// through, so LdrColor blends as when they draw onto it directly. A pass of its own, without a
// sensor plan: bindings 0 and 9 as the sensor kernels', 32 and 33, and their constants (a lens
// of rays reads `params`).
// With `pass.view.y` 1 it stores the two at 9 instead (colour over black, and what shows through
// as alpha), for the present pass to blend over another var (`RasterRenderer::Record`).
[[vk::binding(32, 0)]] Texture2DArray<float4> overlayOverBlack;
[[vk::binding(33, 0)]] Texture2DArray<float4> overlayOverWhite;

float4 overlayAt(Texture2DArray<float4> image, float2 raster, int layer, float2 size) {
    let p = clamp(raster, float2(0.0), size - 1.0);
    let base = int2(floor(p));
    let next = min(base + 1, int2(size) - 1);
    let t = p - float2(base);
    let top = lerp(image.Load(int4(base, layer, 0)), image.Load(int4(next.x, base.y, layer, 0)),
                   t.x);
    let bottom = lerp(image.Load(int4(base.x, next.y, layer, 0)),
                      image.Load(int4(next, layer, 0)), t.x);
    return lerp(top, bottom, t.y);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void overlayLensMain(uint3 id : SV_DispatchThreadID) {
    let cam = viewUniforms(pass.view.x + id.z);
    if (id.x >= cam.size.x || id.y >= cam.size.y) {
        return;
    }
    let size = float2(cam.size.xy);
    let tap = rasterTapOf(cam, int2(id.xy), int(id.z));
    if (any(tap.raster < float2(-0.5)) || any(tap.raster > size - 0.5)) {
        return;
    }
    let black = overlayAt(overlayOverBlack, tap.raster, tap.layer, size).rgb;
    let white = overlayAt(overlayOverWhite, tap.raster, tap.layer, size).rgb;
    let through = saturate(dot(white - black, float3(1.0 / 3.0)));
    if (pass.view.y == 1) {
        // For a presented var other than LdrColor: their colour over black and what shows
        // through, which present.slang's `presentOverlay` blends over the presented image.
        ldrColor[id] = float4(black, through);
        return;
    }
    let below = ldrColor[id];
    ldrColor[id] = float4(black + through * below.rgb, below.a);
}

// A camera rendered on cube faces with a rolling shutter (`RasterRenderer::RecordFaceRows`):
// each face texel's row of the sensor image, as the share of its height down to the row, for
// motion.slang to time the texel by. The rows of the sensor pixels whose lens taps read the
// texel, the least of them, as float bits (which order as uints for non-negative floats);
// 0xFFFFFFFF where none reads it. A layer per face, the same for every view. Passes of their
// own, as `overlayLensMain`: binding 0 as the sensor kernels', 34.
[[vk::image_format("r32ui")]] [[vk::binding(34, 0)]] RWTexture2DArray<uint> faceRows;

[shader("compute")]
[numthreads(8, 8, 1)]
void faceRowsClearMain(uint3 id : SV_DispatchThreadID) {
    uint width;
    uint height;
    uint layers;
    faceRows.GetDimensions(width, height, layers);
    if (id.x < width && id.y < height) {
        faceRows[id] = 0xFFFFFFFFu;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void faceRowsMain(uint3 id : SV_DispatchThreadID) {
    let cam = viewUniforms(pass.view.x);
    if (id.x >= cam.size.x || id.y >= cam.size.y) {
        return;
    }
    let size = float2(cam.size.xy);
    let tap = rasterTapOf(cam, int2(id.xy), 0);
    let row = asuint((float(id.y) + 0.5) / size.y);
    let p = clamp(tap.raster, float2(0.0), size - 1.0);
    let base = uint2(floor(p));
    let next = min(base + 1, uint2(size) - 1);
    uint previous;
    InterlockedMin(faceRows[uint3(base, tap.layer)], row, previous);
    InterlockedMin(faceRows[uint3(next.x, base.y, tap.layer)], row, previous);
    InterlockedMin(faceRows[uint3(base.x, next.y, tap.layer)], row, previous);
    InterlockedMin(faceRows[uint3(next, tap.layer)], row, previous);
}
