// SPDX-License-Identifier: BSD-3-Clause
/// @file column_test.cpp
/// @brief Tests of device columns: growth by device copies, and an RL step that keeps its poses
/// on the device and moves no bulk bytes through the host.
#include "ernie/device/column.h"
#include "ernie/device/staging.h"

#include "tests/device/common.h"
#include "tests/device/rl_step.h"

#include <gtest/gtest.h>

#include <cstring>
#include <numeric>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;

std::vector<float> Read(DeviceContext& context, const DeviceColumn& column, std::uint64_t rows,
                        std::span<const Ticket> waits) {
    Readback readback(context, context.Main(), column.Storage(), column.Range().offset,
                      rows * column.RowBytes(), waits);
    std::vector<float> values(rows * column.RowBytes() / sizeof(float));
    std::memcpy(values.data(), readback.Bytes().data(), values.size() * sizeof(float));
    return values;
}

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

    const Ticket grown = column.Resize(5000, context->Main(), std::span(&written, 1));
    EXPECT_EQ(column.Rows(), 5000u);
    EXPECT_EQ(Read(*context, column, 100, std::span(&grown, 1)), points);
    const Ticket shrunk = column.Resize(10, context->Main());
    EXPECT_EQ(Read(*context, column, 10, std::span(&shrunk, 1)),
              std::vector<float>(points.begin(), points.begin() + 30));
    column.Resize(0, context->Main());
    context->WaitIdle();
    EXPECT_LE(arenas.Used(), 1u) << "only the empty column's byte is left";
    ExpectClean(*context);
}

TEST(DeviceColumn, AnRlStepMovesNoBulkBytesThroughTheHost) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    const std::uint32_t worlds = 16'384;
    test::RlStep rl(*context, compiler, worlds);

    const Transfers before = context->Transfers();
    Ticket stepped;
    for (std::uint32_t step = 1; step <= 10; ++step) {
        stepped = rl.Step(step);
    }
    stepped.Wait();
    EXPECT_EQ(context->Transfers(), before) << "the steps stay on the device";

    const std::vector<float> transforms = Read(*context, rl.Transforms(), worlds, {});
    EXPECT_EQ(context->Transfers().readBack - before.readBack, worlds * 48u)
        << "only the explicit readback is counted";
    for (std::uint32_t world : {0u, 1u, 127u, 128u, 9999u, worlds - 1}) {
        const std::array<float, 12> expected = test::RlStep::Expected(world, 10);
        for (std::size_t i = 0; i < 12; ++i) {
            ASSERT_NEAR(transforms[world * 12u + i], expected[i], 1e-5) << world << ", " << i;
        }
    }
    ExpectClean(*context);
}

}  // namespace
}  // namespace ernie
