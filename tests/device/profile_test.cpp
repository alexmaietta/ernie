// SPDX-License-Identifier: BSD-3-Clause
/// @file profile_test.cpp
/// @brief Tests of GPU zones: timestamps around commands arrive in Ernie's profiler, nested and
/// on the steady clock, under validation.
#include "ernie/device/buffer.h"
#include "ernie/device/profile.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string_view>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;

TEST(GpuZone, TimesCommandsIntoTheFrameThatRecordedThem) {
    ERNIE_CONTEXT(context);
    if (!context->Capabilities().calibratedTimestamps) {
        GTEST_SKIP() << "The device cannot time zones";
    }
    const std::uint64_t size = std::uint64_t{64} << 20;
    const Buffer buffer(*context, size, MemoryUse::Device, "filled",
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    profiler::Clear();
    profiler::SetGpuEnabled(true);
    profiler::BeginFrame();
    const std::uint64_t recorded = profiler::CurrentFrame();
    context->Main()
        .Submit([&](VkCommandBuffer commands) {
            ERNIE_GPU_ZONE(*context, commands, "Fill");
            vkCmdFillBuffer(commands, buffer.Handle(), size / 2, size / 2, 1);
            {
                ERNIE_GPU_ZONE(*context, commands, std::string("Fill half").c_str());
                vkCmdFillBuffer(commands, buffer.Handle(), 0, size / 2, 2);
            }
        })
        .Wait();
    context->Collect();
    profiler::BeginFrame();
    profiler::SetGpuEnabled(false);
    const std::uint64_t collected = profiler::Now();

    const std::vector<profiler::Frame> frames = profiler::Frames();
    const auto frame = std::ranges::find(frames, recorded, &profiler::Frame::index);
    ASSERT_NE(frame, frames.end());
    const auto find = [&](std::string_view name) {
        return std::ranges::find_if(frame->gpu,
                                    [&](const profiler::Zone& zone) { return name == zone.name; });
    };
    const auto outer = find("Fill");
    const auto inner = find("Fill half");
    ASSERT_NE(outer, frame->gpu.end());
    ASSERT_NE(inner, frame->gpu.end());
    EXPECT_EQ(outer->depth, 0u);
    EXPECT_EQ(inner->depth, 1u);
    EXPECT_LT(outer->start, outer->end) << "filling 64 MiB takes time";
    EXPECT_LE(outer->start, inner->start);
    EXPECT_GE(outer->end, inner->end);
    // On the steady clock: after recording began and before the results were read, give or take
    // the calibration's deviation.
    const std::uint64_t slack = 1'000'000;
    EXPECT_GE(outer->start + slack, frame->start);
    EXPECT_LE(outer->end, collected + slack);
    ExpectClean(*context);
}

TEST(GpuZone, RecordsNothingWhileDisabled) {
    ERNIE_CONTEXT(context);
    profiler::Clear();
    profiler::SetGpuEnabled(false);
    profiler::BeginFrame();
    context->Main()
        .Submit([&](VkCommandBuffer commands) { ERNIE_GPU_ZONE(*context, commands, "Idle"); })
        .Wait();
    context->Collect();
    profiler::BeginFrame();
    for (const profiler::Frame& frame : profiler::Frames()) {
        EXPECT_TRUE(frame.gpu.empty());
    }
    ExpectClean(*context);
}

}  // namespace
}  // namespace ernie
