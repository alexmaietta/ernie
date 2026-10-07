// Hierarchical depth (Hi-Z) for occlusion culling: one level of a depth pyramid of a batch of
// views (one layer each) from the level before, or level 0 from the depth buffer. Each texel
// keeps the farthest depth of the source texels it covers (depth grows away from the camera), so
// a box whose nearest depth lies beyond the farthest depth below its footprint is hidden.
// Sources need not be twice the size: each texel takes every source texel it overlaps.
//
// Bindings: 0 the source level (all layers), 1 the destination level.
//
// References:
// - N. Greene, M. Kass and G. Miller, "Hierarchical Z-Buffer Visibility", SIGGRAPH '93, 231-238,
//   1993. https://doi.org/10.1145/166117.166147
// - U. Haar and S. Aaltonen, "GPU-Driven Rendering Pipelines", SIGGRAPH 2015 Advances in
//   Real-Time Rendering in Games course.
//   https://advances.realtimerendering.com/s2015/aaltonenhaar_siggraph2015_combined_final_footer_220dpi.pdf
// - A. Kapoulkine, "niagara", depth pyramid reduction. https://github.com/zeux/niagara

struct HzbConstants {
    // Source width and height, destination width and height.
    uint4 size;
}

[[vk::binding(0, 0)]] Texture2DArray<float> source;
[[vk::binding(1, 0)]] [[vk::image_format("r32f")]] RWTexture2DArray<float> destination;
[[vk::push_constant]] ConstantBuffer<HzbConstants> constants;

[shader("compute")]
[numthreads(8, 8, 1)]
void hzbMain(uint3 id : SV_DispatchThreadID) {
    let size = constants.size;
    if (id.x >= size.z || id.y >= size.w) {
        return;
    }
    // The source texels this texel overlaps: [first, last) on each axis.
    let first = uint2(id.x * size.x / size.z, id.y * size.y / size.w);
    let last = min(uint2(((id.x + 1) * size.x + size.z - 1) / size.z,
                         ((id.y + 1) * size.y + size.w - 1) / size.w),
                   size.xy);
    var farthest = 0.0;
    for (uint y = first.y; y < last.y; ++y) {
        for (uint x = first.x; x < last.x; ++x) {
            farthest = max(farthest, source.Load(int4(int(x), int(y), int(id.z), 0)));
        }
    }
    destination[id] = farthest;
}
