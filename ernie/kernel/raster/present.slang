// SPDX-License-Identifier: BSD-3-Clause
// Present pass: draws one of a frame's render vars into a colour attachment with a full-screen
// triangle, as a display image, decoding to linear last when the attachment encodes sRGB itself.
// Float vars bind at 0, integer vars at 1 and SpectralRadiance's bands at 2; each entry point
// reads only its own binding. `presentOutline` draws the selection's outline over the image from
// the PrimId var at 1 and the highlighted prims at 3.

struct PresentConstants {
    // x: 1 when the attachment has an sRGB format; y: how to show the var (kShow*); z: for
    // SpectralRadiance, the band shown, or 0xFFFFFFFF for their mean; w: the band count.
    uint4 flags;
    // Distances: x near and y far end of the logarithmic range. Sensor vars: z the value shown
    // as white. SpectralRadiance: w the display value per W m^-2 sr^-1 nm^-1.
    float4 range;
    // SpectralRadiance: the automatic exposure's scale on `range.w`, which the frame's exposure
    // pass wrote (meter.slang), read when `hasGain` is 1 (a flag, not a null test, which would
    // need 64-bit integers in the shader).
    float* gain;
    uint hasGain;
}

static const uint kShowLdr = 0;        // sRGB-encoded colour, as is.
static const uint kShowHdr = 1;        // Linear colour, clipped at 1.
static const uint kShowDistance = 2;   // Distance, white near to black far on a log scale.
static const uint kShowNormal = 3;     // World normal, xyz in [-1, 1] as rgb in [0, 1].
static const uint kShowLinear = 4;     // Linear rgb over range.z.
static const uint kShowId = 5;         // Integer id as a hashed colour, black for 0 (the
                                       // background, and instance 0).
static const uint kShowLinearUint = 6; // Integer rgb over range.z.
static const uint kShowMotion = 8;     // Motion (xy pixels): hue its direction, brightness its
                                       // length, full at 8 pixels.
static const uint kShowDegree = 9;     // A degree of polarization in [0, 1] as grey.
static const uint kShowAngle = 10;     // An angle of polarization in [0, pi) as hue.
static const uint kShowStokes = 11;    // Linear Stokes: hue its angle, brightness its degree.
static const uint kShowTemperature = 12;  // Kelvin from range.x to range.y in the ironbow
                                          // palette, black for 0 (none).

// The ironbow palette of thermal cameras at `t` in [0, 1]: black through indigo, magenta, red
// and orange to yellow and white.
float3 ironbow(float t) {
    static const float3 stops[6] = { float3(0.0, 0.0, 0.0),  float3(0.15, 0.0, 0.5),
                                     float3(0.6, 0.0, 0.6),  float3(0.9, 0.25, 0.1),
                                     float3(1.0, 0.75, 0.0), float3(1.0, 1.0, 0.9) };
    let position = saturate(t) * 5.0;
    let index = min(uint(position), 4u);
    return lerp(stops[index], stops[index + 1], position - float(index));
}

// A hue for polarization angle `angle`, periodic in pi.
float3 angleHue(float angle) {
    let hue = frac(angle / 3.14159265);
    return saturate(abs(frac(hue + float3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0);
}

// Unknown formats, so one declaration reads vars of any channel count and width.
[[vk::binding(0, 0)]] [[vk::image_format("unknown")]] Texture2D<float4> floatVar;
[[vk::binding(1, 0)]] [[vk::image_format("unknown")]] Texture2D<uint4> uintVar;
[[vk::binding(2, 0)]] Texture2DArray<float> spectralVar;
[[vk::push_constant]] ConstantBuffer<PresentConstants> constants;

struct VertexOutput {
    float4 position : SV_Position;
}

[shader("vertex")]
VertexOutput presentVertex(uint vertexIndex : SV_VertexID) {
    let uv = float2(float((vertexIndex << 1) & 2), float(vertexIndex & 2));
    VertexOutput out;
    out.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

float decodeSrgb(float encoded) {
    return encoded <= 0.04045 ? encoded / 12.92 : pow((encoded + 0.055) / 1.055, 2.4);
}

float encodeSrgb(float linear) {
    return linear <= 0.0031308 ? linear * 12.92 : 1.055 * pow(linear, 1.0 / 2.4) - 0.055;
}

// The attachment's value for display colour `encoded` (sRGB-encoded, in [0, 1]).
float4 display(float3 encoded) {
    let c = saturate(encoded);
    if (constants.flags.x == 1) {
        return float4(decodeSrgb(c.r), decodeSrgb(c.g), decodeSrgb(c.b), 1.0);
    }
    return float4(c, 1.0);
}

// The attachment's value for linear colour `linear`.
float4 displayLinear(float3 linear) {
    let c = saturate(linear);
    return display(float3(encodeSrgb(c.r), encodeSrgb(c.g), encodeSrgb(c.b)));
}

[shader("fragment")]
float4 presentFloat(VertexOutput in) : SV_Target {
    let value = floatVar.Load(int3(int2(in.position.xy), 0));
    switch (constants.flags.y) {
    case kShowHdr:
        return displayLinear(value.rgb);
    case kShowDistance: {
        let d = value.r;
        if (isinf(d) || !(d > 0.0)) {
            return display(float3(0.0));
        }
        let near = constants.range.x;
        let far = max(constants.range.y, near * 1.0001);
        let t = saturate(log(max(d, near) / near) / log(far / near));
        return display(float3(1.0 - t));
    }
    case kShowNormal:
        return all(value.xyz == 0.0) ? display(float3(0.0)) : display(value.xyz * 0.5 + 0.5);
    case kShowLinear:
        return displayLinear(value.rgb / constants.range.z);
    case kShowMotion: {
        // The usual optical-flow colour wheel: hue by direction, brightness by length.
        let hue = (atan2(-value.y, value.x) / 6.2831853 + 1.0) % 1.0;
        let rgb = saturate(abs(frac(hue + float3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0);
        return display(rgb * saturate(length(value.xy) / 8.0));
    }
    case kShowDegree:
        return display(float3(value.x));
    case kShowAngle:
        return display(angleHue(value.x));
    case kShowStokes: {
        let degree = value.x > 0.0 ? length(value.yz) / value.x : 0.0;
        return display(angleHue(0.5 * atan2(value.z, value.y)) * saturate(degree));
    }
    case kShowTemperature: {
        if (!(value.x > 0.0)) {
            return display(float3(0.0));
        }
        let low = constants.range.x;
        let high = max(constants.range.y, low + 1e-3);
        // A floor of 0.05 keeps the coolest surface off the background's black.
        return display(ironbow(0.05 + 0.95 * (value.x - low) / (high - low)));
    }
    default:
        return display(value.rgb);
    }
}

// A well-spread colour for id `id` (lowbias32 hash).
float3 idColor(uint id) {
    var h = id;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return 0.25 + 0.75 * float3(float(h & 0xffu), float((h >> 8) & 0xffu),
                                float((h >> 16) & 0xffu)) / 255.0;
}

// SpectralRadiance as grey: one band's spectral radiance, or the mean over the bands, times the
// exposure's gain, so a flat spectrum shows as bright as in the photographic image.
[shader("fragment")]
float4 presentSpectral(VertexOutput in) : SV_Target {
    let pixel = int2(in.position.xy);
    let bands = constants.flags.w;
    var radiance = 0.0;
    if (constants.flags.z < bands) {
        radiance = spectralVar.Load(int4(pixel, int(constants.flags.z), 0));
    } else {
        for (uint band = 0; band < bands; ++band) {
            radiance += spectralVar.Load(int4(pixel, int(band), 0));
        }
        radiance /= float(max(bands, 1u));
    }
    let gain = constants.hasGain != 0 ? constants.gain[0] : 1.0;
    return displayLinear(float3(radiance * constants.range.w * gain));
}

[shader("fragment")]
float4 presentUint(VertexOutput in) : SV_Target {
    let value = uintVar.Load(int3(int2(in.position.xy), 0));
    if (constants.flags.y == kShowId) {
        return value.x == 0 ? display(float3(0.0)) : display(idColor(value.x));
    }
    return displayLinear(float3(value.rgb) / constants.range.z);
}

// The highlighted prims (the viewer's selection): a bit per prim id, 32 to a word.
[[vk::binding(3, 0)]] StructuredBuffer<uint> highlighted;

bool isHighlighted(uint id) {
    let word = id / 32u;
    return id != 0 && word < constants.flags.z && ((highlighted[word] >> (id % 32u)) & 1u) != 0;
}

bool highlightedAt(int2 pixel, int2 size) {
    let p = clamp(pixel, int2(0), size - 1);
    return isHighlighted(uintVar.Load(int3(p, 0)).x);
}

// Omniverse Kit's selection outline from the PrimId var (bound at 1): pixels within flags.y pixels
// of the boundary between highlighted prims and the rest, on both sides, in the display colour
// range.rgb at opacity range.a, blended over the presented image.
[shader("fragment")]
float4 presentOutline(VertexOutput in) : SV_Target {
    uint width = 0;
    uint height = 0;
    uintVar.GetDimensions(width, height);
    let size = int2(int(width), int(height));
    let pixel = int2(in.position.xy);
    let centre = highlightedAt(pixel, size);
    let radius = int(constants.flags.y);
    var edge = false;
    for (int dy = -radius; dy <= radius && !edge; ++dy) {
        for (int dx = -radius; dx <= radius && !edge; ++dx) {
            if (dx * dx + dy * dy <= radius * radius &&
                highlightedAt(pixel + int2(dx, dy), size) != centre) {
                edge = true;
            }
        }
    }
    if (!edge) {
        discard;
    }
    var colour = display(constants.range.rgb);
    colour.a = constants.range.a;
    return colour;
}

// A cube camera's overlays over a presented var other than LdrColor, from sensor.slang's
// `overlayLensMain` resampling them through the lens (bound at 0): rgb their colour over black,
// their opacity already weighted in and sRGB-encoded as LdrColor holds it, and a the share of the
// image that shows through. Blended source over at opacity 1 - a, as the overlay pass draws onto
// the attachment.
[shader("fragment")]
float4 presentOverlay(VertexOutput in) : SV_Target {
    let value = floatVar.Load(int3(int2(in.position.xy), 0));
    let opacity = 1.0 - value.a;
    if (opacity <= 0.0) {
        discard;
    }
    var colour = display(value.rgb / opacity);
    colour.a = opacity;
    return colour;
}
