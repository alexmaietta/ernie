// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/commands.h"

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/util/error.h"
#include "support/buffers.h"
#include "support/devices.h"

namespace ernie::device {
namespace {

using test::Download;
using test::ForEachDevice;

std::string Kinds(const CommandBuffer& commands) {
    std::string kinds;
    for (const Command& command : commands.Commands()) {
        kinds += std::visit(
            Overloaded{
                [](const CopyCommand&) { return 'C'; }, [](const FillCommand&) { return 'F'; },
                [](const UpdateCommand&) { return 'U'; }, [](const BarrierCommand&) { return 'B'; },
                [](const ZoneCommand&) { return 'Z'; }, [](const auto&) { return '?'; }},
            command);
    }
    return kinds;
}

TEST(CommandsTest, RecordInOrderWithZones) {
    ForEachDevice([](Device& device) {
        const Ref<Buffer> buffer = device.CreateBuffer({.bytes = 256});
        const uint32_t word = 5;
        CommandEncoder encoder = device.GetQueue().CreateEncoder();
        {
            const ZoneScope zone(encoder, profiler::ZoneId::Named("Test/Commands"));
            encoder.Fill(*buffer, 1);
            encoder.Barrier();
            encoder.Update(buffer->Slice(0, 4), &word);
            encoder.Copy(buffer->Slice(0, 16), buffer->Slice(16, 16));
        }
        const CommandBuffer commands = encoder.Finish();
        EXPECT_EQ(Kinds(commands), "ZBFBUBCZ");
        EXPECT_EQ(commands.Zones().size(), 1u);
        EXPECT_FALSE(commands.Reusable());
        EXPECT_TRUE(device.GetQueue().CreateEncoder().Finish().Empty());

        CommandEncoder concurrent = device.GetQueue().CreateEncoder();
        concurrent.Concurrent(true);
        concurrent.Fill(buffer->Slice(0, 16), 1);
        concurrent.Fill(buffer->Slice(16, 16), 2);
        concurrent.Barrier();
        concurrent.Fill(buffer->Slice(32, 16), 3);
        concurrent.Concurrent(false);
        concurrent.Fill(buffer->Slice(48, 16), 4);
        EXPECT_EQ(Kinds(concurrent.Finish()), "BFFBFBF");
    });
}

TEST(CommandsTest, ReusableBuffersRunEachTimeTheyAreSubmitted) {
    ForEachDevice([](Device& device) {
        Queue& queue = device.GetQueue();
        const Ref<Buffer> counter = device.CreateBuffer({.label = "counter"}, std::vector{0u});
        const Ref<Buffer> one = device.CreateBuffer({.label = "one"}, std::vector{1u});
        CommandEncoder encoder = queue.CreateEncoder();
        encoder.Copy(*counter, *one);
        encoder.Fill(*counter, 9);
        const CommandBuffer reused = encoder.Finish({.reusable = true});
        EXPECT_TRUE(reused.Reusable());

        queue.BeginBatch();
        queue.Submit(reused);
        queue.Submit(reused);
        const Operation batch = queue.EndBatch();
        batch.Wait();
        EXPECT_TRUE(batch.Ready());
        EXPECT_EQ(Download<uint32_t>(device, *one, 1), std::vector{9u});

        const Operation mark = queue.Mark();
        auto retired = std::make_shared<int>(1);
        const std::weak_ptr<int> watch = retired;
        queue.Retire(std::move(retired), mark);
        bool used = false;
        queue.Exclusive([&] { used = true; });
        queue.WaitIdle();
        EXPECT_TRUE(used);
        EXPECT_TRUE(mark.Ready());
        EXPECT_GE(queue.Completed(), mark.value);
        EXPECT_TRUE(watch.expired());
    });
}

TEST(CommandsTest, ReusableBuffersInABatchRunBetweenTheWorkAroundThem) {
    ForEachDevice([](Device& device) {
        Queue& queue = device.GetQueue();
        const Ref<Buffer> source = device.CreateBuffer({.label = "source"}, std::vector{0u});
        const Ref<Buffer> copied = device.CreateBuffer({.label = "copied"}, std::vector{0u});
        CommandEncoder recorder = queue.CreateEncoder();
        recorder.Copy(*source, *copied);
        const CommandBuffer reused = recorder.Finish({.reusable = true});

        for (const uint32_t value : {3u, 7u}) {
            queue.BeginBatch();
            CommandEncoder before = queue.CreateEncoder();
            before.Fill(*source, value);
            queue.Submit(before.Finish());
            queue.Submit(reused);
            CommandEncoder after = queue.CreateEncoder();
            after.Fill(*source, 0);
            queue.Submit(after.Finish());
            queue.EndBatch().Wait();
            EXPECT_EQ(Download<uint32_t>(device, *copied, 1), std::vector{value});
            EXPECT_EQ(Download<uint32_t>(device, *source, 1), std::vector{0u});
        }
    });
}

TEST(CommandsTest, RejectUpdatesTooLargeToInline) {
    ForEachDevice([](Device& device) {
        const Ref<Buffer> buffer =
            device.CreateBuffer({.bytes = CommandEncoder::kMaxUpdateBytes + 4});
        const std::vector<std::byte> data(CommandEncoder::kMaxUpdateBytes + 4);
        CommandEncoder encoder = device.GetQueue().CreateEncoder();
        EXPECT_THROW(encoder.Update(*buffer, data.data()), Error);
        EXPECT_TRUE(encoder.Finish().Empty());
    });
}

}
}
