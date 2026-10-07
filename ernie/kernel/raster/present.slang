// Present pass: draws one of a frame's render vars into a colour attachment with a full-screen
// triangle, as a display image, decoding to linear last when the attachment encodes sRGB itself.
// Float vars bind at 0 and integer vars at 1; each entry point reads only its own binding.

struct PresentConstants {
    // x: 1 when the attachment has an sRGB format; y: how to show the var (kShow*).
    uint4 flags;
    // Distances: x near and y far end of the logarithmic range. Sensor vars: z the value shown
    // as white.
    float4 range;
}

static const uint kShowLdr = 0;        // sRGB-encoded colour, as is.
static const uint kShowHdr = 1;        // Linear colour, clipped at 1.
static const uint kShowDistance = 2;   // Distance, white near to black far on a log scale.
static const uint kShowNormal = 3;     // World normal, xyz in [-1, 1] as rgb in [0, 1].
static const uint kShowLinear = 4;     // Linear rgb over range.z.
static const uint kShowId = 5;         // Integer id as a hashed colour, black for 0 (the
                                       // background, and instance 0).
static const uint kShowLinearUint = 6; // Integer rgb over range.z.

// Unknown formats, so one declaration reads vars of any channel count and width.
[[vk::binding(0, 0)]] [[vk::image_format("unknown")]] Texture2D<float4> floatVar;
[[vk::binding(1, 0)]] [[vk::image_format("unknown")]] Texture2D<uint4> uintVar;
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

[shader("fragment")]
float4 presentUint(VertexOutput in) : SV_Target {
    let value = uintVar.Load(int3(int2(in.position.xy), 0));
    if (constants.flags.y == kShowId) {
        return value.x == 0 ? display(float3(0.0)) : display(idColor(value.x));
    }
    return displayLinear(float3(value.rgb) / constants.range.z);
}
