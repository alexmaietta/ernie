// SPDX-License-Identifier: BSD-3-Clause
/// @file external_test.cpp
/// @brief Tests of sharing memory and timelines: a second context on the same device stands in
/// for CUDA, importing a column's arena and waiting on the queue that wrote it.
#include "ernie/device/column.h"
#include "ernie/device/staging.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <memory>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;

/// Skips the test unless the context's device shares `capability`.
#define ERNIE_REQUIRE_SHARING(context, capability)                                                 \
    if (!(context)->Capabilities().capability) {                                                   \
        GTEST_SKIP() << (context)->Info().name << " has no " #capability;                          \
    }

TEST(ExternalMemory, AnotherContextReadsAColumnAfterTheQueueThatWroteIt) {
    ERNIE_CONTEXT(writer);
    ERNIE_REQUIRE_SHARING(writer, externalMemory);
    ERNIE_CONTEXT(reader, writer->Info().uuid);
    Arenas arenas(*writer, MemoryUse::Shared, "shared", 1 << 16);
    arenas.Allocate(256);
    const DeviceColumn column(arenas, "temperatures", DType::UInt32, {}, 1000);
    ASSERT_GT(column.Range().offset, 0u) << "the column sits inside its arena";
    const Ticket written = writer->Main().Submit([&](VkCommandBuffer commands) {
        vkCmdFillBuffer(commands, column.Storage().Handle(), column.Range().offset,
                        column.Range().size, 0x2a);
    });

    // The arena is imported once; the column is reached at its offset, as CUDA would.
    const Buffer imported(*reader, column.Storage().Export(), column.Storage().Size(), "imported");
    // The reader waits on the writer's timeline where it is shared, else the host waits for it.
    std::unique_ptr<Timeline> writes;
    Ticket ready;
    if (writer->Capabilities().externalTimeline) {
        writes = std::make_unique<Timeline>(reader->Device(), "writer's main timeline",
                                            writer->Main().Timeline().Export());
        ready = writes->At(written.value);
    } else {
        written.Wait();
    }
    Readback readback(*reader, reader->Main(), imported, column.Range().offset, column.Range().size,
                      std::span(&ready, 1));
    std::vector<std::uint32_t> values(1000);
    std::memcpy(values.data(), readback.Bytes().data(), column.Range().size);
    EXPECT_EQ(values, std::vector<std::uint32_t>(1000, 0x2a));
    reader->WaitIdle();
    ExpectClean(*writer);
    ExpectClean(*reader);
}

TEST(ExternalMemory, QueuesWaitOnTimelinesAnotherApiSignals) {
    ERNIE_CONTEXT(context);
    ERNIE_REQUIRE_SHARING(context, externalTimeline);
    ERNIE_CONTEXT(other, context->Info().uuid);
    // Signalled outside any queue, as CUDA signals a timeline Ernie exported to it.
    const Timeline producer(context->Device(), "producer", true);
    const Timeline imported(other->Device(), "producer", producer.Export());
    const Ticket work = context->Main().Submit([](VkCommandBuffer) {}, std::array{producer.At(5)});
    EXPECT_FALSE(work.Done()) << "held until the producer reaches 5";
    imported.Signal(5);
    work.Wait();
    EXPECT_EQ(producer.Reached(), 5u);
    ExpectClean(*context);
    ExpectClean(*other);
}

TEST(ExternalMemory, OnlySharedBuffersExport) {
    ERNIE_CONTEXT(context);
    const Buffer buffer(*context, 64, MemoryUse::Device, "private");
    EXPECT_THROW(static_cast<void>(buffer.Export()), DeviceError);
    const Timeline timeline(context->Device(), "private");
    EXPECT_THROW(static_cast<void>(timeline.Export()), DeviceError);
    ExpectClean(*context);
}

}  // namespace
}  // namespace ernie
