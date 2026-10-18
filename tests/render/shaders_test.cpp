// SPDX-License-Identifier: BSD-3-Clause
/// @file shaders_test.cpp
/// @brief The embedded shaders: the constants and layouts they share with C++ match it, and the
/// ray tracer compiles into a kernel without a device.
#include "ernie/device/kernel.h"
#include "ernie/device/staging.h"
#include "ernie/render/geometry.h"
#include "ernie/render/rt/ray_tracer.h"
#include "ernie/render/transform_graph.h"
#include "ernie/shaders/scene/geometry.h"
#include "ernie/shaders/scene/transform_ops.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

namespace ernie {
namespace {

/// Writes each value the C++ side mirrors, in the order `kExpected` lists them.
const ShaderModule kShared{"shared", R"(
import geometry;
import transform_ops;
struct Parameters { uint* values; };
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    uint values[] = { kTriangles, kSphere, kCube, kCylinder, kCone, kCapsule, kPlane,
                      sizeof(GeometryRecord), kConstant, kTranslate, kScale, kRotateX, kRotateXYZ,
                      kOrient, kInvert, kFloatsShift, kMaxOps, kNone };
    for (uint i = 0; i < values.getCount(); ++i) {
        parameters.values[i] = values[i];
    }
}
)"};

using Graph = TransformGraph;
constexpr std::uint32_t Kind(GeometryKind kind) {
    return static_cast<std::uint32_t>(kind);
}
const std::vector<std::uint32_t> kExpected{
    Kind(GeometryKind::Triangles),
    Kind(GeometryKind::Sphere),
    Kind(GeometryKind::Cube),
    Kind(GeometryKind::Cylinder),
    Kind(GeometryKind::Cone),
    Kind(GeometryKind::Capsule),
    Kind(GeometryKind::Plane),
    sizeof(GeometryRecord),
    Graph::kConstantOp,
    Graph::OpKind(XformOpKind::Translate),
    Graph::OpKind(XformOpKind::Scale),
    Graph::OpKind(XformOpKind::RotateX),
    Graph::OpKind(XformOpKind::RotateXYZ),
    Graph::OpKind(XformOpKind::Orient),
    Graph::kInvertFlag,
    Graph::kFloatsShift,
    Graph::kMaxOps,
    Graph::kNone,
};

TEST(Shaders, ConstantsAndLayoutsMatchTheirCppMirrors) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    const std::vector<ShaderModule> modules{shaders::kGeometry, shaders::kTransformOps, kShared};
    const ComputeKernel shared(*context, compiler.Spirv(modules, "main"), "shared",
                               sizeof(VkDeviceAddress));
    Arenas arenas(*context, MemoryUse::Device, "values", 1 << 12);
    const BufferRange range = arenas.Allocate(kExpected.size() * sizeof(std::uint32_t));
    const Ticket written = context->Main().Submit([&](VkCommandBuffer commands) {
        shared.Dispatch(commands, {1, 1, 1}, arenas.Address(range));
    });
    Readback readback(*context, context->Main(), arenas.Storage(range), range.offset, range.size,
                      std::span(&written, 1));
    std::vector<std::uint32_t> values(kExpected.size());
    std::memcpy(values.data(), readback.Bytes().data(), range.size);
    EXPECT_EQ(values, kExpected) << "geometry.slang and transform_ops.slang against their C++";
    test::ExpectClean(*context);
}

TEST(Shaders, TheRayTracerCompilesIntoAKernel) {
    std::vector<ShaderModule> modules(RayTracer::Modules().begin(), RayTracer::Modules().end());
    modules.push_back({"trace", R"(
import ray_tracer;
struct Parameters { TracerParameters tracer; float* depths; };
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    RayTracer tracer = RayTracer(parameters.tracer);
    RayDesc ray = { float3(id.x, 0, 10), 0, float3(0, 0, -1), 100 };
    Hit hit = tracer.TraceClosest(ray, 0xff);
    bool blocked = tracer.TraceAny(ray, 0xff);
    parameters.depths[id.x] = hit.found && blocked ? tracer.WorldNormal(hit).z * hit.t : -1;
}
)"});
    const std::vector<std::uint32_t> spirv = ShaderCompiler().Spirv(modules, "main");
    ASSERT_FALSE(spirv.empty());
    EXPECT_EQ(spirv.front(), 0x07230203u);
}

}  // namespace
}  // namespace ernie
