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
#include "ernie/shaders/rt/write_instances.h"
#include "ernie/shaders/sensors/camera.h"
#include "ernie/shaders/spectral/spectrum.h"

#include "tests/render/common.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace ernie {
namespace {

/// Writes the size of each struct shared with C++, as kernels lay them out, in `kExpected`'s order.
const ShaderModule kSizes{"sizes", R"(
import camera;
import geometry;
import ray_tracer;
import ref;
import transform;
import write_instances;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uniform Ref<uint> sizes) {
    uint values[] = { sizeof(GeometryRecord), sizeof(TracerParameters), sizeof(SplitRows),
                      sizeof(Op), sizeof(LocalParameters), sizeof(ComposeParameters),
                      sizeof(Instance), sizeof(WriteParameters), sizeof(View),
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
    sizeof(shaders::View),
    kUsedBytes<shaders::CameraParameters>,
};

/// @return Every module a kernel importing the camera and the instance writer needs, in order.
std::vector<ShaderModule> SharedModules() {
    std::vector<ShaderModule> modules(RayTracer::Modules().begin(), RayTracer::Modules().end());
    for (const ShaderModule& module : TransformGraph::WorldModules().subspan(1)) {
        modules.push_back(module);
    }
    modules.insert(modules.end(), {shaders::kBindless, shaders::kSpectrum, shaders::kCamera,
                                   shaders::kWriteInstances});
    return modules;
}

TEST(Shaders, SharedStructsLayOutAsInCpp) {
    ERNIE_CONTEXT(context);
    std::vector<ShaderModule> modules = SharedModules();
    modules.push_back(kSizes);
    const ComputeKernel sizes(*context, ShaderCompiler().Spirv(modules, "main"), "sizes",
                              sizeof(shaders::Ref<std::uint32_t>));
    Arenas arenas(*context, MemoryUse::Device, "sizes", 1 << 12);
    const BufferRange range = arenas.Allocate(kExpected.size() * sizeof(std::uint32_t));
    const Ticket written = context->Main().Submit([&](VkCommandBuffer commands) {
        sizes.Dispatch(commands, {1, 1, 1}, shaders::Ref<std::uint32_t>{arenas.Address(range)});
    });
    Readback readback(*context, context->Main(), arenas.Storage(range), range.offset, range.size,
                      std::span(&written, 1));
    EXPECT_EQ(readback.Values<std::uint32_t>(), kExpected);
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
