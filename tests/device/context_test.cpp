// SPDX-License-Identifier: BSD-3-Clause
/// @file context_test.cpp
/// @brief Tests of the device context, queues and tickets, buffers and staging, under validation.
#include "ernie/device/staging.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;

std::vector<std::uint32_t> Counting(std::size_t count) {
    std::vector<std::uint32_t> values(count);
    std::iota(values.begin(), values.end(), 1u);
    return values;
}

TEST(DeviceContext, ChoosesASupportedDeviceWithItsQueues) {
    ERNIE_CONTEXT(context);
    EXPECT_TRUE(context->Info().supported);
    EXPECT_EQ(context->Info().uuid.size(), 32u) << "a 16-byte UUID in hex";
    EXPECT_FALSE(context->Queues().empty());
    EXPECT_EQ(&context->Main(), context->Queues().front());
    EXPECT_THROW(DeviceContext({.device = "no such device"}), DeviceError);
    ExpectClean(*context);
}

TEST(DeviceContext, TicketsOrderSubmissionsAndRecycleCommandBuffers) {
    ERNIE_CONTEXT(context);
    Queue& queue = context->Main();
    EXPECT_TRUE(Ticket().Done()) << "a default ticket is done";
    Ticket last;
    for (int i = 0; i < 8; ++i) {
        last = queue.Submit([](VkCommandBuffer) {}, std::span(&last, 1));
    }
    last.Wait();
    EXPECT_TRUE(last.Done());
    EXPECT_EQ(queue.Last(), last);
    ExpectClean(*context);
}

TEST(DeviceContext, UploadsAndReadbacksRoundTrip) {
    ERNIE_CONTEXT(context);
    const std::vector<std::uint32_t> values = Counting(1000);
    const Buffer device(*context, values.size() * 4, MemoryUse::Device, "values");
    EXPECT_NE(device.Address(), 0u) << "every buffer has a device address";
    EXPECT_TRUE(device.Mapped().empty());

    Uploader uploader(*context, context->Main(), 1024);
    uploader.Write(device, 0, std::span(values).first(500));
    uploader.Write(device, 2000, std::span(values).subspan(500));
    EXPECT_EQ(uploader.Pending(), 4000u) << "staged across chunks, copied at flush";
    const Ticket uploaded = uploader.Flush();
    EXPECT_EQ(uploader.Pending(), 0u);

    Readback readback(*context, context->Main(), device, 0, device.Size(), std::span(&uploaded, 1));
    const std::span<const std::byte> bytes = readback.Bytes();
    ASSERT_EQ(bytes.size(), values.size() * 4);
    EXPECT_EQ(std::memcmp(bytes.data(), values.data(), bytes.size()), 0);
    ExpectClean(*context);
}

TEST(DeviceContext, MappedDestinationsAreWrittenInPlace) {
    ERNIE_CONTEXT(context);
    const Buffer mapped(*context, 64, MemoryUse::Upload, "constants");
    ASSERT_FALSE(mapped.Mapped().empty());
    Uploader uploader(*context, context->Main());
    const std::vector<std::uint32_t> values = Counting(16);
    uploader.Write(mapped, 0, std::span(values));
    EXPECT_EQ(uploader.Pending(), 0u) << "no staging";
    EXPECT_EQ(std::memcmp(mapped.Mapped().data(), values.data(), 64), 0);
    EXPECT_THROW(uploader.Write(mapped, 60, std::span(values)), std::out_of_range);
    ExpectClean(*context);
}

TEST(DeviceContext, RetiredObjectsLiveUntilTheirWorkIsDone) {
    ERNIE_CONTEXT(context);
    auto buffer = std::make_shared<Buffer>(*context, 256, MemoryUse::Device, "retired");
    const std::weak_ptr<Buffer> watch = buffer;
    const Ticket work = context->Main().Submit([&](VkCommandBuffer commands) {
        vkCmdFillBuffer(commands, buffer->Handle(), 0, VK_WHOLE_SIZE, 7);
    });
    context->Retire(std::move(buffer));
    EXPECT_FALSE(watch.expired()) << "held while its work may run";
    work.Wait();
    context->Collect();
    EXPECT_TRUE(watch.expired());
    ExpectClean(*context);
}

TEST(DeviceContext, TwoContextsKeepTheirWorkApart) {
    ERNIE_CONTEXT(first);
    // A second device where there is one (lavapipe beside a GPU), else the same device again.
    const std::vector<VulkanDeviceInfo> devices = ListVulkanDevices();
    const auto other = std::ranges::find_if(devices, [&](const VulkanDeviceInfo& info) {
        return info.supported && info.uuid != first->Info().uuid;
    });
    ERNIE_CONTEXT(second, other != devices.end() ? other->uuid : first->Info().uuid);

    const std::vector<std::uint32_t> a = Counting(64);
    const std::vector<std::uint32_t> b(64, 9);
    std::vector<std::unique_ptr<Readback>> reads;
    for (const auto& [context, values] :
         {std::pair{first.get(), &a}, std::pair{second.get(), &b}}) {
        const Buffer buffer(*context, 256, MemoryUse::Device, "values");
        Uploader uploader(*context, context->Main());
        uploader.Write(buffer, 0, std::span(*values));
        const Ticket uploaded = uploader.Flush();
        Readback readback(*context, context->Main(), buffer, 0, 256, std::span(&uploaded, 1));
        EXPECT_EQ(std::memcmp(readback.Bytes().data(), values->data(), 256), 0);
    }
    const Ticket firsts = first->Main().Submit([](VkCommandBuffer) {});
    if (second->Device() != first->Device()) {
        EXPECT_THROW(second->Main().Submit([](VkCommandBuffer) {}, std::span(&firsts, 1)),
                     DeviceError)
            << "a ticket names its device";
    }
    firsts.Wait();
    ExpectClean(*first);
    ExpectClean(*second);
}

}  // namespace
}  // namespace ernie
