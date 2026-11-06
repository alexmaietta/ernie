// SPDX-License-Identifier: BSD-3-Clause
struct Push {
    uint* pixels;
    uint2 size;
    float2 origin;
    float2 extent;
    float opacity;
    uint blended;
};

[vk::push_constant]
ConstantBuffer<Push> push;

[shader("vertex")]
float4 VertexMain(uint vertex: SV_VulkanVertexID) : SV_Position {
    const float2 corner = float2(float((vertex << 1) & 2), float(vertex & 2));
    return float4(corner * 2.0 - 1.0, 0.0, 1.0);
}

[shader("fragment")]
float4 FragmentMain(float4 position: SV_Position) : SV_Target {
    const float2 local = (position.xy - push.origin) / push.extent;
    const uint2 texel = min(uint2(local * float2(push.size)), push.size - 1);
    const uint packed = push.pixels[texel.y * push.size.x + texel.x];
    const float3 color =
        float3(float(packed & 0xff), float((packed >> 8) & 0xff), float((packed >> 16) & 0xff));
    const float alpha = push.blended != 0 ? float(packed >> 24) / 255.0 * push.opacity : 1.0;
    return float4(color / 255.0, alpha);
}
