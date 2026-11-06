// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/buffer.h"

#include <cstdint>
#include <cstring>
#include <numeric>
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

std::vector<uint32_t> Counting(uint32_t count, uint32_t first = 0) {
    std::vector<uint32_t> values(count);
    std::iota(values.begin(), values.end(), first);
    return values;
}

TEST(BufferTest, ComesInEveryHostVisibleKind) {
    ForEachDevice([](Device& device) {
        for (const Memory memory : {Memory::Device, Memory::Upload, Memory::Readback}) {
            SCOPED_TRACE(int(memory));
            const Ref<Buffer> buffer =
                device.CreateBuffer({.bytes = 1000, .memory = memory, .label = "kind"});
            EXPECT_EQ(buffer->GetMemory(), memory);
            EXPECT_GE(buffer->Bytes(), 1000u);
            EXPECT_EQ(memory != Memory::Device, buffer->Mapped() != nullptr);
            EXPECT_FALSE(buffer->Virtual());
        }
        const Ref<Buffer> usage = device.CreateBuffer(
            {.bytes = 64, .usage = BufferUsage::Indirect | BufferUsage::AccelBuildInput});
        EXPECT_TRUE(Has(usage->Usage(), BufferUsage::Indirect));
        EXPECT_FALSE(Has(usage->Usage(), BufferUsage::Interop));
    });
}

TEST(BufferTest, SlicesStayInsideTheirBuffer) {
    ForEachDevice([](Device& device) {
        const Ref<Buffer> buffer = device.CreateBuffer({.bytes = 1024});
        const BufferRange middle = buffer->Slice(256, 512);
        EXPECT_EQ(middle.Offset(), 256u);
        EXPECT_EQ(middle.Bytes(), 512u);
        EXPECT_EQ(middle.Address(), buffer->Address() + 256);
        const BufferRange inner = middle.Slice(128);
        EXPECT_EQ(inner.Offset(), 384u);
        EXPECT_EQ(inner.Bytes(), 384u);
        EXPECT_EQ(&inner.GetBuffer(), buffer.Get());
        EXPECT_THROW(buffer->Slice(1025), Error);
        EXPECT_THROW(middle.Slice(0, 513), Error);
    });
}

TEST(BufferTest, TransfersEveryWay) {
    ForEachDevice([](Device& device) {
        Queue& queue = device.GetQueue();
        const TrafficStats before = device.Traffic();
        const Ref<Buffer> source = device.CreateBuffer({.label = "source"}, Counting(256));
        const Ref<Buffer> target = device.CreateBuffer({.bytes = 1024, .label = "target"});

        const std::vector<uint32_t> written = Counting(16, 1000);
        queue.Write(target->Slice(64, 64), written.data());
        const BufferRange staged = queue.Upload(written.data(), 64);
        const uint32_t patch[4] = {7, 7, 7, 7};

        CommandEncoder encoder = queue.CreateEncoder();
        encoder.Copy(source->Slice(0, 64), target->Slice(0, 64));
        encoder.Fill(target->Slice(128, 64), 0xABu);
        encoder.Copy(staged, target->Slice(192, 64));
        encoder.Update(target->Slice(256, 16), patch);
        encoder.Fill(target->Slice(272), 0);
        queue.Submit(encoder.Finish());

        std::vector<uint32_t> expected = Counting(16);
        expected.insert(expected.end(), written.begin(), written.end());
        expected.insert(expected.end(), 16, 0xABu);
        expected.insert(expected.end(), written.begin(), written.end());
        expected.insert(expected.end(), 4, 7u);
        expected.resize(256, 0);
        EXPECT_EQ(Download<uint32_t>(device, *target, 256), expected);

        const device::Download download = queue.ReadAsync(target->Slice(64, 64));
        download.Wait();
        EXPECT_TRUE(download.Ready());
        ASSERT_EQ(download.Bytes().size(), 64u);
        EXPECT_EQ(std::memcmp(download.Bytes().data(), written.data(), 64), 0);

        const TrafficStats moved = device.Traffic() - before;
        EXPECT_GT(moved.uploadBytes, 0u);
        EXPECT_GT(moved.readbackBytes, 0u);
        EXPECT_GT(moved.submissions, 0u);
    });
}

TEST(BufferTest, CountsMemoryWhileAlive) {
    ForEachDevice([](Device& device) {
        const uint64_t before = device.MemoryUsage().deviceBytes;
        {
            const Ref<Buffer> buffer = device.CreateBuffer({.bytes = 1 << 20, .label = "counted"});
            const MemoryStats stats = device.MemoryUsage();
            EXPECT_GE(stats.deviceBytes, before + (1 << 20));
            bool listed = false;
            for (const MemoryUse& use : stats.uses) {
                listed = listed || use.label == "counted";
            }
            EXPECT_TRUE(listed);
        }
        device.GetQueue().WaitIdle();
        EXPECT_EQ(device.MemoryUsage().deviceBytes, before);
    });
}

}
}
