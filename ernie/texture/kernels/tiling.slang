// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.device.kernels.device;

static const float kHexScale = 3.46410162f;

float2 Rotate(float2 v, float angle) {
    const float c = cos(angle);
    const float s = sin(angle);
    return float2(c * v.x - s * v.y, s * v.x + c * v.y);
}

float2 HexHash(int2 vertex) {
    const float2 p = float2(vertex);
    const float2 r = float2(dot(p, float2(127.1f, 311.7f)), dot(p, float2(269.5f, 183.3f)));
    return frac(sin(r) * 43758.5453f);
}

float HexAngle(int2 vertex, float strength) {
    float angle = abs(float(vertex.x * vertex.y)) + abs(float(vertex.x + vertex.y)) + kPi;
    angle = fmod(angle, 2 * kPi);
    angle = angle > kPi ? angle - 2 * kPi : angle;
    return angle * strength;
}

struct Placement {
    float2 st;
    float weight;
    float2 axis;
    float footprint;
    float3 tangent;
    float3 bitangent;
    float angle;
}

interface IPlacementVisitor {
    [mutating]
    void Visit(Placement placement);
}

struct PlacementTransform {
    float angle;
    float2 scale;
    float2 offset;
}

struct PlacementSettings {
    PlacementTransform uv;
    PlacementTransform detail;
    bool fine;
    bool single;
    bool hex;
    float hexContrast;
    float hexRotation;
    float3 frameNormal;
}

Placement Framed(Placement placement, float3 normal) {
    Placement framed = placement;
    const float3 t = normalize(placement.tangent - dot(placement.tangent, normal) * normal);
    const float3 b = placement.bitangent - dot(placement.bitangent, normal) * normal -
                     dot(placement.bitangent, t) * t;
    framed.tangent = t;
    framed.bitangent = length(b) > 1e-8f ? normalize(b) : cross(normal, t);
    return framed;
}

Placement Transformed(Placement placement, PlacementTransform transform) {
    Placement result = placement;
    const float c = cos(transform.angle);
    const float s = sin(transform.angle);
    result.st = Rotate(placement.st * transform.scale, transform.angle) + transform.offset;
    result.axis = Rotate(placement.axis * transform.scale, transform.angle);
    result.footprint = placement.footprint * max(abs(transform.scale.x), abs(transform.scale.y));
    const float2 inverse = select(transform.scale != 0, 1 / transform.scale, float2(1));
    const float3 u = placement.tangent * inverse.x;
    const float3 v = placement.bitangent * inverse.y;
    result.tangent = normalize(u * c - v * s);
    result.bitangent = normalize(u * s + v * c);
    return result;
}

uint ProbeCount(PlacementSettings settings) {
    return settings.hex && !settings.single ? 3 : 1;
}

Placement Probe(Placement placement, PlacementSettings settings, uint i) {
    Placement mapped = Transformed(placement, settings.uv);
    if (settings.fine) {
        mapped = Framed(Transformed(mapped, settings.detail), settings.frameNormal);
    }
    if (!settings.hex) {
        return mapped;
    }
    const float2 scaled = mapped.st * kHexScale;
    const float2 skewed = float2(scaled.x - 0.57735027f * scaled.y, 1.15470054f * scaled.y);
    const int2 base = int2(floor(skewed));
    const float2 f = frac(skewed);
    const float z = 1 - f.x - f.y;
    const float s = z < 0 ? 1.0f : 0.0f;
    const float s2 = 2 * s - 1;
    const float3 w = float3(-z * s2, s - f.y * s2, s - f.x * s2);
    const float3 sharp = pow(max(w, float3(1e-6f)), float3(1 + 7 * settings.hexContrast));
    const uint strongest =
        sharp.x >= sharp.y && sharp.x >= sharp.z ? 0 : (sharp.y >= sharp.z ? 1 : 2);
    const uint pick = settings.single ? strongest : i;
    const int2 vertex = pick == 0   ? base + int2(int(s), int(s))
                        : pick == 1 ? base + int2(int(s), 1 - int(s))
                                    : base + int2(1 - int(s), int(s));
    const float2 corner = float2(vertex);
    const float2 center = float2(corner.x + 0.5f * corner.y, corner.y / 1.15470054f) / kHexScale;
    const float angle = HexAngle(vertex, settings.hexRotation);
    Placement probe = mapped;
    probe.st = Rotate(mapped.st - center, angle) + center + HexHash(vertex);
    probe.weight =
        settings.single ? mapped.weight : mapped.weight * sharp[i] / (sharp.x + sharp.y + sharp.z);
    probe.axis = Rotate(mapped.axis, angle);
    probe.angle = angle;
    return probe;
}

float3 TangentNormal(float4 texel, float3 scale, float3 bias, float strength, float angle) {
    float3 n = texel.xyz * scale + bias;
    n.xy = Rotate(n.xy, -angle);
    n.xy *= strength;
    return n;
}
