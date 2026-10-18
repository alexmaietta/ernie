// SPDX-License-Identifier: BSD-3-Clause
/// @file kernel_test.cpp
/// @brief Tests of Slang compilation and compute kernels: a kernel written in Slang fills an
/// arena range through its device address, under validation.
#include "ernie/device/arena.h"
#include "ernie/device/kernel.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/device/staging.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <cstring>
#include <stdexcept>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;

/// Writes index × scale into `count` words at an address.
const ShaderModule kFill{"fill", R"(
struct Parameters {
    uint* data;
    uint count;
    uint scale;
};
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x < parameters.count) {
        parameters.data[id.x] = id.x * parameters.scale;
    }
}
)"};

struct FillParameters {
    VkDeviceAddress data;
    std::uint32_t count;
    std::uint32_t scale;
};

TEST(ShaderCompiler, CompilesSlangToSpirvAndWgsl) {
    const ShaderCompiler compiler;
    const std::vector<std::uint32_t> spirv = compiler.Spirv(std::span(&kFill, 1), "main");
    ASSERT_FALSE(spirv.empty());
    EXPECT_EQ(spirv.front(), 0x07230203u) << "SPIR-V's magic number";

    // WGSL has no pointers, so the WebGPU path reads bound buffers.
    const ShaderModule bound{"double", R"(
RWStructuredBuffer<float> values;
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) { values[id.x] *= 2.0; }
)"};
    EXPECT_NE(compiler.Wgsl(std::span(&bound, 1), "main").find("@compute"), std::string::npos);
}

TEST(ShaderCompiler, ErrorsCarrySlangsDiagnostics) {
    const ShaderCompiler compiler;
    const ShaderModule broken{"broken", "void main() { undefined(); }"};
    try {
        compiler.Spirv(std::span(&broken, 1), "main");
        ADD_FAILURE() << "compiled";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("undefined"), std::string::npos) << error.what();
    }
    EXPECT_THROW(compiler.Spirv(std::span(&kFill, 1), "nowhere"), std::invalid_argument);
}

TEST(ComputeKernel, FillsAnArenaRangeThroughItsAddress) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    const ComputeKernel fill(*context, compiler.Spirv(std::span(&kFill, 1), "main"), "fill",
                             sizeof(FillParameters));
    Arenas arenas(*context, MemoryUse::Device, "values", 1 << 16);
    arenas.Allocate(36);
    const std::uint32_t count = 1000;
    const BufferRange range = arenas.Allocate(count * 4);
    const Ticket filled = context->Main().Submit([&](VkCommandBuffer commands) {
        fill.Dispatch(commands, {(count + 63) / 64, 1, 1},
                      FillParameters{arenas.Address(range), count, 3});
    });
    EXPECT_THROW(context->Main().Submit(
                     [&](VkCommandBuffer commands) { fill.Dispatch(commands, {1, 1, 1}); }),
                 std::invalid_argument)
        << "its parameters are required";

    Readback readback(*context, context->Main(), arenas.Storage(range), range.offset, range.size,
                      std::span(&filled, 1));
    std::vector<std::uint32_t> values(count);
    std::memcpy(values.data(), readback.Bytes().data(), range.size);
    for (std::uint32_t i = 0; i < count; ++i) {
        ASSERT_EQ(values[i], 3 * i) << i;
    }
    ExpectClean(*context);
}

TEST(ComputeKernel, ParametersComeInWholeWords) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    const std::vector<std::uint32_t> spirv = compiler.Spirv(std::span(&kFill, 1), "main");
    EXPECT_THROW(ComputeKernel(*context, spirv, "fill", 6), std::invalid_argument);
    EXPECT_THROW(ComputeKernel(*context, spirv, "fill", 4096), DeviceError);
    ExpectClean(*context);
}

}  // namespace
}  // namespace ernie
