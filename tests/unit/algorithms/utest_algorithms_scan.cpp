// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/algorithms/scan.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/util/error.h"
#include "support/buffers.h"
#include "support/devices.h"

namespace ernie::algorithms {
namespace {

using device::Buffer;
using device::CommandEncoder;
using device::Device;
using test::Download;
using test::ForEachDevice;
using test::Random;
using test::Upload;

constexpr uint32_t kUntouched = 0xDEADBEEFu;

std::vector<uint32_t> ExclusiveSum(const std::vector<uint32_t>& input, uint32_t count) {
    std::vector<uint32_t> sums(input.size(), kUntouched);
    std::exclusive_scan(input.begin(), input.begin() + count, sums.begin(), 0u);
    return sums;
}

TEST(ScanTest, MatchesTheHostAcrossTileAndLevelBoundaries) {
    constexpr uint32_t kCounts[] = {0, 1, 1023, 1024, 1025, 1024 * 1024 + 1};
    ForEachDevice([&](Device& device) {
        const Scan scan(device, kCounts[std::size(kCounts) - 1]);
        for (const uint32_t count : kCounts) {
            SCOPED_TRACE(count);
            const std::vector<uint32_t> input = Random<uint32_t>(count, count);
            const Ref<Buffer> in = Upload(device, input);
            const Ref<Buffer> out = Upload(device, std::vector<uint32_t>(count, kUntouched));

            CommandEncoder encoder = device.GetQueue().CreateEncoder();
            scan.ExclusiveSum(encoder, *in, *out, count);
            scan.ExclusiveSum(encoder, *in, *in, count);
            device.GetQueue().Submit(encoder.Finish());

            const std::vector<uint32_t> expected = ExclusiveSum(input, count);
            EXPECT_EQ(Download<uint32_t>(device, *out, count), expected);
            EXPECT_EQ(Download<uint32_t>(device, *in, count), expected);
        }
    });
}

TEST(ScanTest, StopsAtTheDeviceCount) {
    constexpr uint32_t kCapacity = 5000;
    ForEachDevice([&](Device& device) {
        const Scan scan(device, kCapacity);
        const std::vector<uint32_t> input = Random<uint32_t>(kCapacity, 7);
        const Ref<Buffer> in = Upload(device, input);
        for (const uint32_t live : {0u, 1u, 2049u, kCapacity, kCapacity + 5}) {
            SCOPED_TRACE(live);
            const Ref<Buffer> out = Upload(device, std::vector<uint32_t>(kCapacity, kUntouched));
            const Ref<Buffer> count = Upload(device, std::vector<uint32_t>{live});

            CommandEncoder encoder = device.GetQueue().CreateEncoder();
            scan.ExclusiveSum(encoder, *in, *out, *count, kCapacity);
            device.GetQueue().Submit(encoder.Finish());

            EXPECT_EQ(Download<uint32_t>(device, *out, kCapacity),
                      ExclusiveSum(input, std::min(live, kCapacity)));
        }
    });
}

TEST(ScanTest, RejectsWhatItCannotScan) {
    ForEachDevice([](Device& device) {
        EXPECT_THROW(Scan(device, Scan::kMaxItems + 1), Error);

        const Scan scan(device, 100);
        const Ref<Buffer> small = device.CreateBuffer({.bytes = 64});
        const Ref<Buffer> large = device.CreateBuffer({.bytes = 400});
        CommandEncoder encoder = device.GetQueue().CreateEncoder();
        EXPECT_THROW(scan.ExclusiveSum(encoder, *large, *large, 101), Error);
        EXPECT_THROW(scan.ExclusiveSum(encoder, *small, *large, 17), Error);
        EXPECT_THROW(scan.ExclusiveSum(encoder, *large, *small, 17), Error);
        EXPECT_THROW(scan.ExclusiveSum(encoder, *large, *large, *small, 101), Error);
        EXPECT_THROW(scan.ExclusiveSum(encoder, *small, *small, *small, 17), Error);
        EXPECT_TRUE(encoder.Finish().Empty());
    });
}

}
}
