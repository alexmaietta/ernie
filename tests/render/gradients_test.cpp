// SPDX-License-Identifier: BSD-3-Clause
/// @file gradients_test.cpp
/// @brief The differentiable models: they compile in differentiable form, and the gradients the
/// compiler derives match central finite differences.
#include "ernie/device/arena.h"
#include "ernie/device/kernel.h"
#include "ernie/device/staging.h"
#include "ernie/render/rt/camera_pass.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <vector>

namespace ernie {
namespace {

/// Writes the gradient of the summed shaded bands with respect to the albedo's bands and the
/// normal: the compiler's (6 floats), then central differences' (6 floats).
const ShaderModule kShadeGradients{"shade_gradients", R"(
import buffer_ref;
import camera;
import spectrum;

[Differentiable]
float Loss(Spectrum albedo, float3 normal, no_diff float3 direction) {
    Spectrum light = Shade(albedo, normal, direction);
    return light.values[0] + light.values[1] + light.values[2];
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uniform BufferRef<float> gradients) {
    Spectrum albedo;
    albedo.values = { 0.2, 0.5, 0.8 };
    float3 normal = float3(0.3, -0.4, 0.8);
    float3 direction = normalize(float3(0.1, -0.2, -1));

    var albedoPair = diffPair(albedo);
    var normalPair = diffPair(normal);
    bwd_diff(Loss)(albedoPair, normalPair, direction, 1.0);
    for (uint i = 0; i < 3; ++i) {
        gradients[i] = albedoPair.d.values[i];
        gradients[3 + i] = normalPair.d[i];
    }

    const float h = 1e-2;
    for (uint i = 0; i < 3; ++i) {
        Spectrum up = albedo;
        Spectrum down = albedo;
        up.values[i] += h;
        down.values[i] -= h;
        gradients[6 + i] = (Loss(up, normal, direction) - Loss(down, normal, direction)) / (2 * h);
        float3 offset = float3(0);
        offset[i] = h;
        gradients[9 + i] = (Loss(albedo, normal + offset, direction) -
                            Loss(albedo, normal - offset, direction)) / (2 * h);
    }
}
)"};

/// Writes the gradient of a weighted sum of a pinhole ray's direction with respect to the pixel
/// coordinate: the compiler's (2 floats), then central differences' (2 floats).
const ShaderModule kLensGradients{"lens_gradients", R"(
import buffer_ref;
import lens;

[Differentiable]
float Loss(no_diff Lens lens, float2 pixel) {
    return dot(Unproject(lens, pixel).direction, float3(0.3, -0.7, 0.2));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uniform BufferRef<float> gradients) {
    Lens lens = { kPinhole, float2(40, 38), float2(16, 12), 0.05, { 0 } };
    float2 pixel = float2(5.5, 20.25);
    var pair = diffPair(pixel);
    bwd_diff(Loss)(lens, pair, 1.0);
    const float h = 1e-2;
    for (uint i = 0; i < 2; ++i) {
        gradients[i] = pair.d[i];
        float2 offset = float2(0);
        offset[i] = h;
        gradients[2 + i] = (Loss(lens, pixel + offset) - Loss(lens, pixel - offset)) / (2 * h);
    }
}
)"};

/// @return What a gradient kernel writes: `count` floats.
std::vector<float> RunGradients(DeviceContext& context, const ShaderModule& test,
                                std::size_t count) {
    std::vector<ShaderModule> modules(CameraPass::Modules().begin(), CameraPass::Modules().end());
    modules.push_back(test);
    const ComputeKernel kernel(context, ShaderCompiler().Spirv(modules, "main"), "gradients",
                               sizeof(shaders::BufferRef<float>));
    Arenas arenas(context, MemoryUse::Device, "gradients", 1 << 12);
    const BufferRange range = arenas.Allocate(count * sizeof(float));
    const Ticket written = context.Main().Submit([&](CommandList commands) {
        kernel.Dispatch(commands, {1, 1, 1}, shaders::BufferRef<float>{arenas.Address(range)});
    });
    Readback readback(context, context.Main(), arenas.Storage(range), range.offset, range.size,
                      std::span(&written, 1));
    return readback.Values<float>();
}

TEST(Gradients, PinholeRaysMatchFiniteDifferences) {
    ERNIE_CONTEXT(context);
    const std::vector<float> gradients = RunGradients(*context, kLensGradients, 4);
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(gradients[i], gradients[2 + i], 1e-4f) << "pixel axis " << i;
        EXPECT_NE(gradients[i], 0.0f) << "the ray turns as the pixel moves";
    }
    test::ExpectClean(*context);
}

TEST(Gradients, ShadingMatchesFiniteDifferences) {
    ERNIE_CONTEXT(context);
    const std::vector<float> gradients = RunGradients(*context, kShadeGradients, 12);
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(gradients[i], gradients[6 + i], 1e-3f)
            << (i < 3 ? "albedo band " : "normal component ") << i % 3;
    }
    EXPECT_GT(gradients[0], 0.0f) << "brighter albedo, more light";
    test::ExpectClean(*context);
}

}  // namespace
}  // namespace ernie
