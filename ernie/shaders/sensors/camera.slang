// SPDX-License-Identifier: BSD-3-Clause
/// @file camera.slang
/// @brief Pinhole cameras traced one ray a pixel, every view in one dispatch: a headlit colour
/// into a bindless image, and each pixel's distance, world normal, instance and albedo into
/// buffers kernels, CUDA or PyTorch read.
import bindless;
import ray_tracer;
import spectrum;

/// No output: an image index or address left out.
static const uint kNoImage = 0xffffffff;
/// What a pixel that sees nothing gets in `instances`.
static const uint kNoInstance = 0xffffffff;

/// A view, as camera_pass.cpp's `ViewRecord`.
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
    /// z, the instance row, `kBands` albedo bands, and the colour image's RGBA8 bytes; each 0 for
    /// no output.
    float* distances;
    float* normals;
    uint* instances;
    float* albedos;
    uint* ldr;
};

struct Parameters {
    TracerParameters tracer;
    View* views;
    /// Each instance row's albedo, three linear RGB floats.
    float* colours;
    uint count;
};
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

/// @return The light a pixel sees: the surface's albedo lit from the eye.
Spectrum Shade(Spectrum albedo, float3 normal, float3 direction) {
    float lit = abs(dot(normal, direction));
    return albedo.Scaled(0.15 + 0.85 * lit);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.z >= parameters.count) {
        return;
    }
    View view = parameters.views[id.z];
    uint2 pixel = id.xy;
    if (any(pixel >= view.size)) {
        return;
    }
    float2 ndc = (float2(pixel) + 0.5) / float2(view.size) * 2 - 1;
    float3 direction = normalize(float3(ndc.x * view.halfExtent.x, -ndc.y * view.halfExtent.y, -1));
    float3x4 toWorld = float3x4(view.rows[0], view.rows[1], view.rows[2]);
    RayDesc ray = { mul(toWorld, float4(0, 0, 0, 1)), 0, mul(toWorld, float4(direction, 0)), 1e30 };

    RayTracer tracer = RayTracer(parameters.tracer);
    Hit hit = tracer.TraceClosest(ray, 0xff);
    float3 normal = float3(0);
    Spectrum albedo = Spectrum(0);
    if (hit.found) {
        normal = tracer.WorldNormal(hit);
        float* colour = parameters.colours + 3 * hit.instance;
        albedo = Uplift(float3(colour[0], colour[1], colour[2]));
    }

    uint at = pixel.y * view.size.x + pixel.x;
    if (view.colour != kNoImage || view.ldr != nullptr) {
        float3 shown = view.background;
        if (hit.found) {
            // Stored as UNORM: encoded for display roughly as sRGB would.
            shown = pow(saturate(ToLinearSrgb(Shade(albedo, normal, ray.Direction))), 1 / 2.2);
        }
        if (view.colour != kNoImage) {
            images[NonUniformResourceIndex(view.colour)][pixel] = float4(shown, 1);
        }
        if (view.ldr != nullptr) {
            uint3 bytes = uint3(round(saturate(shown) * 255));
            view.ldr[at] = bytes.r | bytes.g << 8 | bytes.b << 16 | 0xff000000u;
        }
    }
    if (view.distances != nullptr) {
        view.distances[at] = hit.found ? hit.t : 1.0 / 0.0;
    }
    if (view.normals != nullptr) {
        view.normals[3 * at] = normal.x;
        view.normals[3 * at + 1] = normal.y;
        view.normals[3 * at + 2] = normal.z;
    }
    if (view.instances != nullptr) {
        view.instances[at] = hit.found ? hit.instance : kNoInstance;
    }
    if (view.albedos != nullptr) {
        for (int band = 0; band < kBands; ++band) {
            view.albedos[kBands * at + band] = albedo.values[band];
        }
    }
}
