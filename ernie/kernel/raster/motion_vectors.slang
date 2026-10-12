// SPDX-License-Identifier: BSD-3-Clause
// The MotionVector var's camera pass: every raster pixel's motion since the last frame from the
// camera's alone, reprojecting the point its depth gives with the last frame's view-projection. The
// sky keeps the cleared reverse-Z depth 0: the far plane, or with an infinite one the point at
// infinity, kept homogeneous (w = 0) so it moves with the camera's turn alone. Drawables that moved then draw their own motion over it
// (`RasterDelegate::RecordMotion`).
//
// Bindings: 0 the G-buffer depth (a layer per view), 1 the motion written (raster pixels, xy of
// `RenderTargets::Raster`), 2 the views' MotionUniforms, from view `firstView`.
//
// Reference: D. Nehab, P. V. Sander, J. Lawrence, N. Tatarchuk and J. R. Isidoro, "Accelerating
// Real-Time Shading with Reverse Reprojection Caching", Graphics Hardware 2007, 25-35.

// As frame.slang's (its bindings are this pass's own, so it does not import frame).
struct MotionUniforms {
    float4x4 previousViewProj;
    float4x4 inverseViewProj;
    float4 size;
    float4 padding[7];
}

struct CameraMotionConstants {
    // x: the first view's index in `views`.
    uint4 view;
}

[[vk::binding(0, 0)]] Texture2DArray<float> depth;
// `RenderTargets::Raster`: motion in xy, motion blur's coverage in z, kept.
[[vk::binding(1, 0)]] [[vk::image_format("rgba32f")]] RWTexture2DArray<float4> motion;
[[vk::binding(2, 0)]] StructuredBuffer<MotionUniforms> views;
[[vk::push_constant]] ConstantBuffer<CameraMotionConstants> constants;

[shader("compute")]
[numthreads(8, 8, 1)]
void cameraMotionMain(uint3 id : SV_DispatchThreadID) {
    let m = views[constants.view.x + id.z];
    if (id.x >= uint(m.size.x) || id.y >= uint(m.size.y)) {
        return;
    }
    let pixel = float2(id.xy) + 0.5;
    let ndc = float3(pixel.x / m.size.x * 2.0 - 1.0, 1.0 - pixel.y / m.size.y * 2.0,
                     depth.Load(int4(int3(id), 0)));
    let unprojected = mul(m.inverseViewProj, float4(ndc, 1.0));
    let world = unprojected.w < 0.0 ? -unprojected : unprojected;
    let previous = mul(m.previousViewProj, world);
    // As frame.slang's pixelMotion.
    var moved = float2(0.0);
    if (m.size.z > 0.5 && previous.w > 1e-6) {
        let before = previous.xy / previous.w;
        moved = pixel - float2(before.x * 0.5 + 0.5, 0.5 - before.y * 0.5) * m.size.xy;
    }
    motion[id] = float4(moved, motion[id].zw);
}
