// SPDX-License-Identifier: BSD-3-Clause
// The G-buffer's normal target (RGBA16F, frame.slang's `GBuffer.normal`), which the raster passes
// write and the shading pass and sensor read: xy the unit shading normal in octahedral
// coordinates, zw the surface's temperature in kelvin as a half-precision pair, the value rounded
// to half and its remainder, whose sum keeps it to about 1e-4 K (0: its material's own).
//
// Reference: Z. H. Cigolle, S. Donow, D. Evangelakos, M. Mara, M. McGuire and Q. Meyer, "A Survey
// of Efficient Representations for Independent Unit Vectors", JCGT 3(2), 2014.
// http://jcgt.org/published/0003/02/01/

// Unit vector `n` as octahedral coordinates in [-1, 1]^2.
public float2 octahedralEncode(float3 n) {
    let p = n.xy / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-12);
    return n.z < 0.0 ? (1.0 - abs(p.yx)) * select(p >= 0.0, float2(1.0), float2(-1.0)) : p;
}

public float3 octahedralDecode(float2 p) {
    var n = float3(p, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * select(n.xy >= 0.0, float2(1.0), float2(-1.0));
    }
    return normalize(n);
}

public float4 packNormal(float3 n, float kelvin = 0.0) {
    // Rounded to half's 10 mantissa bits in 32-bit integers (f32tof16 would need 16-bit
    // capabilities), so the target stores it exactly.
    let high = asfloat((asuint(kelvin) + 0x1000u) & 0xFFFFE000u);
    return float4(octahedralEncode(n), high, kelvin - high);
}

public float3 unpackNormal(float4 packed) {
    return octahedralDecode(packed.xy);
}

// The temperature a raster pass wrote, or 0 for the material's own.
public float unpackTemperature(float4 packed) {
    return packed.z + packed.w;
}

// The G-buffer's tangent target (RG16 unorm, `RenderTargets::kGBufferFormats[3]`), bound only
// where a material reads it (anisotropic reflection): the surface's tangent, along which its
// texture coordinates' u grows, in octahedral coordinates kept off 0, which stands for none (a
// surface without texture coordinates, or a pass that has no tangents).
public float2 packTangent(float3 t) {
    let length2 = dot(t, t);
    if (!(length2 > 1e-20)) {
        return float2(0.0);
    }
    return (octahedralEncode(t * rsqrt(length2)) * 0.5 + 0.5) * (65534.0 / 65535.0) +
           1.0 / 65535.0;
}

// The tangent `packTangent` wrote, or 0 for none.
public float3 unpackTangent(float2 packed) {
    if (all(packed < 0.5 / 65535.0)) {
        return float3(0.0);
    }
    return octahedralDecode(((packed - 1.0 / 65535.0) * (65535.0 / 65534.0)) * 2.0 - 1.0);
}
