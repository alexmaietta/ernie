// SPDX-License-Identifier: BSD-3-Clause
/// @file ray_tracer_test.cpp
/// @brief The tracing modules compile into a kernel that traces through `IRayTracer`.
#include "ernie/render/ray_tracer.h"

#include <gtest/gtest.h>

#include <vector>

namespace ernie {
namespace {

const ShaderModule kTrace{"trace", R"(
import ray_tracer;
struct Parameters { uint64_t scene; GeometryRecord* records; float* depths; };
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

float Depth<T : IRayTracer>(T tracer, RayDesc ray) {
    Hit hit = tracer.TraceClosest(ray, 0xff);
    return hit.found && !tracer.TraceAny(ray, 0xff) ? -1 : hit.t;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    HardwareRayQuery tracer = { parameters.scene, parameters.records };
    RayDesc ray = { float3(id.x, 0, 10), 0, float3(0, 0, -1), 100 };
    parameters.depths[id.x] = Depth(tracer, ray);
}
)"};

TEST(RayTracer, ModulesCompileIntoATracingKernel) {
    const ShaderCompiler compiler;
    std::vector<ShaderModule> modules(RayTracerModules().begin(), RayTracerModules().end());
    modules.push_back(kTrace);
    const std::vector<std::uint32_t> spirv = compiler.Spirv(modules, "main");
    ASSERT_FALSE(spirv.empty());
    EXPECT_EQ(spirv.front(), 0x07230203u);
}

}  // namespace
}  // namespace ernie
