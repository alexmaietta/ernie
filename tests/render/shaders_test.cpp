// SPDX-License-Identifier: BSD-3-Clause
/// @file shaders_test.cpp
/// @brief The embedded shaders: the structs they share with C++ lay out alike in both, the
/// kernels that need no ray tracing stay portable to WebGPU, and the ray tracer compiles without a
/// device.
#include "ernie/device/kernel.h"
#include "ernie/render/rt/camera_pass.h"
#include "ernie/render/rt/ray_tracer.h"
#include "ernie/render/transform_graph.h"
#include "ernie/shaders/device/bindless.h"
#include "ernie/shaders/device/buffer_ref.h"
#include "ernie/shaders/rt/write_instances.h"
#include "ernie/shaders/sampling/random.h"
#include "ernie/shaders/sensors/camera.h"
#include "ernie/shaders/spectral/spectrum.h"

#include "ernie/util/random.h"

#include "tests/render/common.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstddef>
#include <vector>

namespace ernie {
namespace {

/// Writes the size of each struct shared with C++, as kernels lay them out, in `kExpected`'s order.
const ShaderModule kSizes{"sizes", R"(
import camera;
import geometry;
import lens;
import ray_tracer;
import buffer_ref;
import transform;
import write_instances;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uniform BufferRef<uint> sizes) {
    uint values[] = { sizeof(GeometryRecord), sizeof(TracerParameters), sizeof(SplitRows),
                      sizeof(Op), sizeof(LocalParameters), sizeof(ComposeParameters),
                      sizeof(Instance), sizeof(WriteParameters), sizeof(Lens), sizeof(View),
                      sizeof(CameraParameters) };
    for (uint i = 0; i < values.getCount(); ++i) {
        sizes[i] = values[i];
    }
}
)"};

/// The bytes a kernel's parameters use: Slang's scalar layout has no tail padding there.
template <typename T> constexpr std::uint32_t kUsedBytes = offsetof(T, count) + sizeof(T::count);

const std::vector<std::uint32_t> kExpected{
    sizeof(shaders::GeometryRecord),
    sizeof(shaders::TracerParameters),
    sizeof(shaders::SplitRows),
    sizeof(shaders::Op),
    kUsedBytes<shaders::LocalParameters>,
    kUsedBytes<shaders::ComposeParameters>,
    sizeof(shaders::Instance),
    kUsedBytes<shaders::WriteParameters>,
    sizeof(shaders::Lens),
    sizeof(shaders::View),
    kUsedBytes<shaders::CameraParameters>,
};

/// @return Every module a kernel importing the camera and the instance writer needs, in order.
std::vector<ShaderModule> SharedModules() {
    std::vector<ShaderModule> modules(CameraPass::Modules().begin(), CameraPass::Modules().end());
    modules.push_back(shaders::kWriteInstances);
    return modules;
}

TEST(Shaders, SharedStructsLayOutAsInCpp) {
    ERNIE_CONTEXT(context);
    std::vector<ShaderModule> modules = SharedModules();
    modules.push_back(kSizes);
    const ComputeKernel sizes(*context, ShaderCompiler().Spirv(modules, "main"), "sizes",
                              sizeof(shaders::BufferRef<std::uint32_t>));
    Arenas arenas(*context, MemoryUse::Device, "sizes", 1 << 12);
    const BufferRange range = arenas.Allocate(kExpected.size() * sizeof(std::uint32_t));
    const Ticket written = context->Main().Submit([&](VkCommandBuffer commands) {
        sizes.Dispatch(commands, {1, 1, 1},
                       shaders::BufferRef<std::uint32_t>{arenas.Address(range)});
    });
    Readback readback(*context, context->Main(), arenas.Storage(range), range.offset, range.size,
                      std::span(&written, 1));
    EXPECT_EQ(readback.Values<std::uint32_t>(), kExpected);
    test::ExpectClean(*context);
}

/// Writes each key's hashed bits and Sobol samples, as `ExpectedSamples` lists them.
const ShaderModule kSamples{"samples", R"(
import random;
import buffer_ref;
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uniform BufferRef<uint> out, uniform uint count) {
    if (id.x >= count) {
        return;
    }
    SampleKey key = { FrameSeed(id.x, id.x * 31u, id.x / 64u), id.x % 3u, id.x * 7919u,
                      id.x % 64u };
    for (uint dimension = 0; dimension < 8; ++dimension) {
        out[(id.x * 8 + dimension) * 2] = SampleBits(key, dimension);
        out[(id.x * 8 + dimension) * 2 + 1] = asuint(SampleSobol(key, dimension));
    }
}
)"};

/// @return What `kSamples` writes for `count` keys, from the host's functions.
std::vector<std::uint32_t> ExpectedSamples(std::uint32_t count) {
    std::vector<std::uint32_t> values;
    for (std::uint32_t i = 0; i < count; ++i) {
        const SampleKey key{FrameSeed(i, i * 31u, i / 64u), i % 3u, i * 7919u, i % 64u};
        for (std::uint32_t dimension = 0; dimension < 8; ++dimension) {
            values.push_back(SampleBits(key, dimension));
            values.push_back(std::bit_cast<std::uint32_t>(SampleSobol(key, dimension)));
        }
    }
    return values;
}

TEST(Shaders, RandomNumbersMatchTheHostsBitForBit) {
    ERNIE_CONTEXT(context);
    const std::array modules{shaders::kBufferRef, shaders::kRandom, kSamples};
    struct Parameters {
        shaders::BufferRef<std::uint32_t> out;
        std::uint32_t count;
    };
    const ComputeKernel samples(*context, ShaderCompiler().Spirv(modules, "main"), "samples",
                                std::uint32_t{sizeof(Parameters)});
    constexpr std::uint32_t kCount = 256;
    const std::vector<std::uint32_t> expected = ExpectedSamples(kCount);
    Arenas arenas(*context, MemoryUse::Device, "samples", 1 << 16);
    const BufferRange range = arenas.Allocate(expected.size() * sizeof(std::uint32_t));
    const Ticket written = context->Main().Submit([&](VkCommandBuffer commands) {
        samples.Dispatch(commands, {kCount / 64, 1, 1},
                         Parameters{{arenas.Address(range)}, kCount});
    });
    Readback readback(*context, context->Main(), arenas.Storage(range), range.offset, range.size,
                      std::span(&written, 1));
    EXPECT_EQ(readback.Values<std::uint32_t>(), expected);
    test::ExpectClean(*context);
}

// WebGPU has no ray queries, but everything else stays in the subset WGSL takes.
TEST(Shaders, TheTransformKernelsCompileForWebGpu) {
    const ShaderCompiler compiler;
    for (const char* entry : {"Local", "Compose"}) {
        EXPECT_NE(compiler.Wgsl(TransformGraph::WorldModules(), entry).find("@compute"),
                  std::string::npos)
            << entry;
    }
}

// Without a device that ray traces, the tests that trace are skipped; this still compiles them.
TEST(Shaders, TheRayTracerCompilesIntoAKernel) {
    std::vector<ShaderModule> modules(RayTracer::Modules().begin(), RayTracer::Modules().end());
    modules.push_back(test::kTrace);
    const std::vector<std::uint32_t> spirv = ShaderCompiler().Spirv(modules, "main");
    ASSERT_FALSE(spirv.empty());
    EXPECT_EQ(spirv.front(), 0x07230203u);
}

}  // namespace
}  // namespace ernie
