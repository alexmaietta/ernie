// SPDX-License-Identifier: BSD-3-Clause
/// @file camera.slang
/// @brief Pinhole cameras traced one ray a pixel, every view in one dispatch: a headlit colour
/// into a bindless image, and each pixel's distance, world normal, instance, albedo and RGBA8
/// colour into buffers kernels, CUDA or PyTorch read. The views and parameters are shared with
/// C++ (rt/camera_pass.cpp).
#ifdef __cplusplus
#pragma once
#include "ernie/shaders/rt/ray_tracer.slang"
namespace ernie::shaders {
#else
import bindless;
import ray_tracer;
import ref;
import spectrum;
#endif

/// No colour image: its bindless index left out.
static const uint kNoImage = 0xffffffff;
/// What a pixel that sees nothing gets in `instances`.
static const uint kNoInstance = 0xffffffff;

/// A view.
struct View {
    /// Camera to world, relative to the render origin; the camera looks down -z, +y up.
    float4 rows[3];
    /// The half extents at unit distance: tan(fov / 2) times the aspect, and tan(fov / 2).
    float2 halfExtent;
    uint2 size;
    /// What the colour image shows where nothing is hit, as stored.
    float3 background;
    /// The colour image's bindless index, or `kNoImage`.
    uint colour;
    /// Per pixel, row by row: the hit distance (infinity for none), the world normal's x, y and
    /// z, the instance row, `kBands` albedo bands, and the colour image's RGBA8 bytes; each null
    /// for no output.
    Ref<float> distances;
    Ref<float> normals;
    Ref<uint> instances;
    Ref<float> albedos;
    Ref<uint> ldr;
};

struct CameraParameters {
    TracerParameters tracer;
    Ref<View> views;
    /// Each instance row's albedo, three linear RGB floats.
    Ref<float> colours;
    uint count;
};

#ifdef __cplusplus
}  // namespace ernie::shaders
#else

/// The light a surface facing away from the eye still gets, of what one facing it gets.
static const float kAmbient = 0.15;

/// @return The light a pixel sees: the surface's albedo lit from the eye.
Spectrum Shade(Spectrum albedo, float3 normal, float3 direction) {
    float lit = abs(dot(normal, direction));
    return albedo.Scaled(lerp(kAmbient, 1.0, lit));
}

/// @return A view's ray through the middle of a pixel.
RayDesc PixelRay(View view, uint2 pixel) {
    float2 ndc = (float2(pixel) + 0.5) / float2(view.size) * 2 - 1;
    float3 direction = normalize(float3(ndc.x * view.halfExtent.x, -ndc.y * view.halfExtent.y, -1));
    float3x4 toWorld = float3x4(view.rows[0], view.rows[1], view.rows[2]);
    return { mul(toWorld, float4(0, 0, 0, 1)), 0, mul(toWorld, float4(direction, 0)), 1e30 };
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
    RayDesc ray = PixelRay(view, pixel);
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

    uint at = pixel.y * view.size.x + pixel.x;
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
    if (!view.albedos.IsNull()) {
        for (uint band = 0; band < kBands; ++band) {
            view.albedos[kBands * at + band] = albedo.values[band];
        }
    }
}

#endif
