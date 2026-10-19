// SPDX-License-Identifier: BSD-3-Clause
/// @file camera.slang
/// @brief Cameras traced one ray a pixel through their lenses, every view in one dispatch: a
/// headlit colour into a bindless image, and each pixel's distance, world normal, instance, prim,
/// albedo and RGBA8 colour into buffers kernels, CUDA or PyTorch read. The views and parameters
/// are shared with C++ (rt/camera_pass.cpp).
#ifdef __cplusplus
#pragma once
#include "ernie/shaders/rt/ray_tracer.slang"
#include "ernie/shaders/sensors/lens.slang"
namespace ernie::shaders {
#else
import affine;
import bindless;
import ray_tracer;
import buffer_ref;
import lens;
import spectrum;
import transform;
#endif

/// No colour image: its bindless index left out.
static const uint kNoImage = 0xffffffff;
/// What a pixel that sees nothing gets in `instances` and `prims`.
static const uint kNoInstance = 0xffffffff;

/// A view.
struct View {
    /// Camera to world, relative to the render origin, unless `node` names the camera's
    /// transform-graph slot; the camera looks down -z, +y up.
    float4 rows[3];
    Lens lens;
    uint2 size;
    /// What the colour image shows where nothing is hit, as stored.
    float3 background;
    /// The colour image's bindless index, or `kNoImage`.
    uint colour;
    /// The camera's slot in the transform graph's worlds, read in place of `rows`; or `kNone`.
    uint node;
    /// The pixels of the outputs before the view's own: views stack in shared outputs.
    uint first;
    /// Per pixel, row by row: the hit distance (infinity for none), the world normal's x, y and
    /// z, the instance row, the prim's slot, `kBands` albedo bands, and the colour image's RGBA8
    /// bytes; each null for no output.
    BufferRef<float> distances;
    BufferRef<float> normals;
    BufferRef<uint> instances;
    BufferRef<uint> prims;
    BufferRef<float> albedos;
    BufferRef<uint> ldr;
};

struct CameraParameters {
    TracerParameters tracer;
    BufferRef<View> views;
    /// The transform graph's worlds, which views with a node read their pose from.
    BufferRef<float4> worlds;
    /// Each instance row's albedo, three linear RGB floats, and its prim's slot.
    BufferRef<float> colours;
    BufferRef<uint> instancePrims;
    uint count;
};

#ifdef __cplusplus
}  // namespace ernie::shaders
#else

/// The light a surface facing away from the eye still gets, of what one facing it gets.
static const float kAmbient = 0.15;

/// @return The light a pixel sees: the surface's albedo lit from the eye; differentiable in the
/// albedo and the normal.
[Differentiable]
Spectrum Shade(Spectrum albedo, float3 normal, no_diff float3 direction) {
    float lit = abs(dot(normal, direction));
    return albedo.Scaled(lerp(kAmbient, 1.0, lit));
}

/// @return A view's ray through the middle of a pixel, from the camera's pose `toWorld`.
RayDesc PixelRay(View view, float3x4 toWorld, uint2 pixel) {
    LensRay ray = Unproject(view.lens, float2(pixel) + 0.5);
    return { mul(toWorld, float4(ray.origin, 1)), 0,
             normalize(mul(toWorld, float4(ray.direction, 0))), 1e30 };
}

/// @return A view's camera to world: its node's world, or its own rows.
float3x4 ViewPose(View view, BufferRef<float4> worlds) {
    Affine pose;
    pose.rows = view.rows;
    if (view.node != kNone) {
        pose = LoadWorld(worlds, view.node);
    }
    return float3x4(pose.rows[0], pose.rows[1], pose.rows[2]);
}

/// @brief Writes the colour a pixel shows to the view's image and RGBA8 buffer, as it asks.
void WriteColour(View view, uint2 pixel, uint at, Hit hit, Spectrum light) {
    float3 shown = view.background;
    if (hit.found) {
        // Stored as UNORM: encoded for display roughly as sRGB would.
        shown = pow(saturate(ToLinearSrgb(light)), 1 / 2.2);
    }
    if (view.colour != kNoImage) {
        images[NonUniformResourceIndex(view.colour)][pixel] = float4(shown, 1);
    }
    if (!view.ldr.IsNull()) {
        uint3 bytes = uint3(round(saturate(shown) * 255));
        view.ldr[at] = bytes.r | bytes.g << 8 | bytes.b << 16 | 0xff000000u;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID, uniform CameraParameters parameters) {
    if (id.z >= parameters.count) {
        return;
    }
    View view = parameters.views[id.z];
    uint2 pixel = id.xy;
    if (any(pixel >= view.size)) {
        return;
    }
    RayDesc ray = PixelRay(view, ViewPose(view, parameters.worlds), pixel);
    RayTracer tracer = RayTracer(parameters.tracer);
    Hit hit = tracer.TraceClosest(ray, 0xff);
    float3 normal = float3(0);
    Spectrum albedo = Spectrum(0);
    if (hit.found) {
        normal = tracer.WorldNormal(hit);
        uint colour = 3 * hit.instance;
        albedo = Uplift(float3(parameters.colours[colour], parameters.colours[colour + 1],
                               parameters.colours[colour + 2]));
    }

    uint at = view.first + pixel.y * view.size.x + pixel.x;
    if (view.colour != kNoImage || !view.ldr.IsNull()) {
        WriteColour(view, pixel, at, hit, Shade(albedo, normal, ray.Direction));
    }
    if (!view.distances.IsNull()) {
        view.distances[at] = hit.found ? hit.t : 1.0 / 0.0;
    }
    if (!view.normals.IsNull()) {
        view.normals[3 * at] = normal.x;
        view.normals[3 * at + 1] = normal.y;
        view.normals[3 * at + 2] = normal.z;
    }
    if (!view.instances.IsNull()) {
        view.instances[at] = hit.found ? hit.instance : kNoInstance;
    }
    if (!view.prims.IsNull()) {
        view.prims[at] = hit.found ? parameters.instancePrims[hit.instance] : kNoInstance;
    }
    if (!view.albedos.IsNull()) {
        for (uint band = 0; band < kBands; ++band) {
            view.albedos[kBands * at + band] = albedo.values[band];
        }
    }
}

#endif
