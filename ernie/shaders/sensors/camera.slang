// SPDX-License-Identifier: BSD-3-Clause
/// @file camera.slang
/// @brief A pinhole camera traced one ray a pixel: a headlit colour into a bindless image, and
/// each pixel's distance, world normal and instance into buffers kernels, CUDA or PyTorch read.
import bindless;
import ray_tracer;

/// No output: an image index or address left out.
static const uint kNoImage = 0xffffffff;
/// What a pixel that sees nothing gets in `instances`.
static const uint kNoInstance = 0xffffffff;

/// As camera_pass.cpp's `CameraParameters`.
struct Parameters {
    TracerParameters tracer;
    /// Camera to world, relative to the render origin; the camera looks down -z, +y up.
    float4 rows[3];
    /// The view's half extents at unit distance: tan(fov / 2) times the aspect, and tan(fov / 2).
    float2 halfExtent;
    uint2 size;
    /// The colour image's bindless index, or `kNoImage`.
    uint colour;
    uint pad;
    /// Per pixel, row by row: the hit distance (infinity for none), the world normal's x, y and
    /// z, and the instance row; each 0 for no output.
    float* distances;
    float* normals;
    uint* instances;
};
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

/// A plain look: grey surfaces lit from the eye, over a sky that darkens to the horizon.
float3 Shade(Hit hit, float3 normal, float3 direction) {
    if (!hit.found) {
        return lerp(float3(0.35, 0.38, 0.42), float3(0.55, 0.65, 0.8), saturate(direction.z));
    }
    float lit = abs(dot(normal, direction));
    return float3(0.8) * (0.15 + 0.85 * lit);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint2 pixel = id.xy;
    if (any(pixel >= parameters.size)) {
        return;
    }
    float2 ndc = (float2(pixel) + 0.5) / float2(parameters.size) * 2 - 1;
    float3 view = normalize(float3(ndc.x * parameters.halfExtent.x,
                                   -ndc.y * parameters.halfExtent.y, -1));
    float3x4 toWorld = float3x4(parameters.rows[0], parameters.rows[1], parameters.rows[2]);
    RayDesc ray = { mul(toWorld, float4(0, 0, 0, 1)), 0, mul(toWorld, float4(view, 0)), 1e30 };

    RayTracer tracer = RayTracer(parameters.tracer);
    Hit hit = tracer.TraceClosest(ray, 0xff);
    float3 normal = float3(0);
    if (hit.found) {
        normal = tracer.WorldNormal(hit);
    }

    uint at = pixel.y * parameters.size.x + pixel.x;
    if (parameters.colour != kNoImage) {
        // The viewport's image is UNORM: encode for display as sRGB roughly would.
        float3 colour = pow(Shade(hit, normal, ray.Direction), 1 / 2.2);
        images[NonUniformResourceIndex(parameters.colour)][pixel] = float4(colour, 1);
    }
    if (parameters.distances != nullptr) {
        parameters.distances[at] = hit.found ? hit.t : 1.0 / 0.0;
    }
    if (parameters.normals != nullptr) {
        parameters.normals[3 * at] = normal.x;
        parameters.normals[3 * at + 1] = normal.y;
        parameters.normals[3 * at + 2] = normal.z;
    }
    if (parameters.instances != nullptr) {
        parameters.instances[at] = hit.found ? hit.instance : kNoInstance;
    }
}
