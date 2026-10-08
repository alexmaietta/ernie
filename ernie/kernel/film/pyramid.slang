// A level of an image pyramid for spatial sensor filters that read far around a pixel (`glare`):
// each texel the mean of the texels of the level before it covers, every layer (view and channel
// domain) at once. Sources need not be twice the size: each texel averages every source texel it
// overlaps.
//
// Bindings: 0 the source level (all layers), 1 the destination level.
//
// Reference: L. Williams, "Pyramidal Parametrics", SIGGRAPH '83, 1-11, 1983.
// https://doi.org/10.1145/964967.801126

struct PyramidConstants {
    // Source width and height, destination width and height.
    uint4 size;
}

[[vk::binding(0, 0)]] Texture2DArray<float4> source;
[[vk::binding(1, 0)]] [[vk::image_format("rgba32f")]] RWTexture2DArray<float4> destination;
[[vk::push_constant]] ConstantBuffer<PyramidConstants> constants;

[shader("compute")]
[numthreads(8, 8, 1)]
void pyramidMain(uint3 id : SV_DispatchThreadID) {
    let size = constants.size;
    if (id.x >= size.z || id.y >= size.w) {
        return;
    }
    let first = uint2(id.x * size.x / size.z, id.y * size.y / size.w);
    let last = min(uint2(((id.x + 1) * size.x + size.z - 1) / size.z,
                         ((id.y + 1) * size.y + size.w - 1) / size.w),
                   size.xy);
    var sum = float4(0.0);
    var count = 0.0;
    for (uint y = first.y; y < last.y; ++y) {
        for (uint x = first.x; x < last.x; ++x) {
            sum += source.Load(int4(int(x), int(y), int(id.z), 0));
            count += 1.0;
        }
    }
    destination[id] = sum / max(count, 1.0);
}
