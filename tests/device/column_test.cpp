// SPDX-License-Identifier: BSD-3-Clause
/// @file column_test.cpp
/// @brief Tests of device columns: growth by device copies, and a step of many poses that stays on
/// the device and moves no bulk bytes through the host.
#include "ernie/device/column.h"
#include "ernie/device/staging.h"

#include "tests/device/columns.h"
#include "tests/device/common.h"
#include "tests/device/pose_step.h"

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;
using test::ReadFloats;

TEST(DeviceColumn, RowsKeepTheirShapeAndValuesAcrossResizes) {
    ERNIE_CONTEXT(context);
    Arenas arenas(*context, MemoryUse::Device, "columns", 1 << 16);
    DeviceColumn column(arenas, "points", DType::Float32, {3}, 100);
    EXPECT_EQ(column.RowBytes(), 12u);
    std::vector<float> points(300);
    std::iota(points.begin(), points.end(), 0.0f);
    Uploader uploader(*context, context->Main());
    uploader.Write(column.Storage(), column.Range().offset, std::span<const float>(points));
    const Ticket written = uploader.Flush();

    const Ticket grown = column.Resize(101, context->Main(), std::span(&written, 1));
    EXPECT_EQ(column.Rows(), 101u);
    EXPECT_EQ(column.Capacity(), 200u) << "the capacity doubles";
    EXPECT_EQ(ReadFloats(*context, column, grown, 100), points);
    const VkDeviceAddress address = column.Address();
    EXPECT_EQ(column.Resize(200, context->Main()), Ticket()) << "nothing to copy within capacity";
    EXPECT_EQ(column.Address(), address);
    const Ticket large = column.Resize(5000, context->Main());
    EXPECT_EQ(column.Capacity(), 5000u);
    EXPECT_EQ(ReadFloats(*context, column, large, 100), points);
    column.Resize(10, context->Main());
    EXPECT_EQ(column.Bytes(), 120u);
    EXPECT_EQ(ReadFloats(*context, column),
              std::vector<float>(points.begin(), points.begin() + 30));
    ExpectClean(*context);
}

TEST(DeviceColumn, StepsOfManyPosesMoveNoBulkBytesThroughTheHost) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    const std::uint32_t instances = 16'384;
    test::PoseStep poses(*context, compiler, instances);

    const Transfers before = context->Transfers();
    Ticket stepped;
    for (std::uint32_t step = 1; step <= 10; ++step) {
        stepped = poses.Step(step);
    }
    stepped.Wait();
    EXPECT_EQ(context->Transfers(), before) << "the steps stay on the device";

    const std::vector<float> transforms = ReadFloats(*context, poses.Transforms());
    EXPECT_EQ(context->Transfers().readBack - before.readBack, instances * 48u)
        << "only the explicit readback is counted";
    for (std::uint32_t instance : {0u, 1u, 127u, 128u, 9999u, instances - 1}) {
        const std::array<float, 12> expected = test::PoseStep::Expected(instance, 10);
        for (std::size_t i = 0; i < 12; ++i) {
            ASSERT_NEAR(transforms[instance * 12u + i], expected[i], 1e-5) << instance << ", " << i;
        }
    }
    ExpectClean(*context);
}

}  // namespace
}  // namespace ernie
