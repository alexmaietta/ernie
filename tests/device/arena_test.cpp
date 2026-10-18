// SPDX-License-Identifier: BSD-3-Clause
/// @file arena_test.cpp
/// @brief Tests of arenas: ranges by offset, reuse once the GPU is done, growth and large ranges.
#include "ernie/device/arena.h"
#include "ernie/device/staging.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;

TEST(Arenas, RangesAreAlignedAndApart) {
    ERNIE_CONTEXT(context);
    Arenas arenas(*context, MemoryUse::Device, "geometry", 1 << 16);
    const BufferRange a = arenas.Allocate(100);
    const BufferRange b = arenas.Allocate(100, 256);
    EXPECT_EQ(a.arena, b.arena);
    EXPECT_EQ(b.offset % 256, 0u);
    EXPECT_TRUE(b.offset >= a.offset + a.size || a.offset >= b.offset + b.size) << "no overlap";
    EXPECT_EQ(arenas.Address(b), arenas.Storage(b).Address() + b.offset);
    EXPECT_EQ(arenas.Used(), 200u);
    ExpectClean(*context);
}

TEST(Arenas, FreedRangesComeBackOnlyOnceTheGpuIsDone) {
    ERNIE_CONTEXT(context);
    Arenas arenas(*context, MemoryUse::Device, "geometry", 1 << 16);
    const BufferRange range = arenas.Allocate(1 << 16);
    const Ticket work = context->Main().Submit([&](VkCommandBuffer commands) {
        vkCmdFillBuffer(commands, arenas.Storage(range).Handle(), range.offset, range.size, 1);
    });
    arenas.Free(range);
    EXPECT_EQ(arenas.Used(), std::uint64_t{1} << 16) << "still in use until its work is done";
    context->Main().Wait(work);
    context->Collect();
    EXPECT_EQ(arenas.Used(), 0u);
    EXPECT_EQ(arenas.Allocate(1 << 16).arena, range.arena) << "the arena is reused";
    EXPECT_EQ(arenas.Count(), 1u);
    ExpectClean(*context);
}

TEST(Arenas, FullArenasAreJoinedAndLargeRangesGetTheirOwn) {
    ERNIE_CONTEXT(context);
    Arenas arenas(*context, MemoryUse::Device, "geometry", 4096);
    arenas.Allocate(4096);
    EXPECT_EQ(arenas.Allocate(16).arena, 1u) << "a full arena is joined by another";
    const BufferRange large = arenas.Allocate(1 << 20);
    EXPECT_EQ(arenas.Storage(large).Size() >= large.size, true);
    EXPECT_THROW(arenas.Allocate(context->MaxBufferSize() + 1), DeviceError);
    ExpectClean(*context);
}

TEST(Arenas, RangesUploadAndReadBackThroughTheirArena) {
    ERNIE_CONTEXT(context);
    Arenas arenas(*context, MemoryUse::Device, "values", 1 << 16);
    arenas.Allocate(12);
    const BufferRange range = arenas.Allocate(64 * 4);
    const std::vector<float> values(64, 2.5f);
    Uploader uploader(*context, context->Main());
    uploader.Write(arenas.Storage(range), range.offset, std::span(values));
    const Ticket uploaded = uploader.Flush();
    Readback readback(*context, context->Main(), arenas.Storage(range), range.offset, range.size,
                      std::span(&uploaded, 1));
    EXPECT_EQ(std::memcmp(readback.Bytes().data(), values.data(), range.size), 0);
    ExpectClean(*context);
}

}  // namespace
}  // namespace ernie
