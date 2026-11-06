// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/algorithms/radix_sort.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
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

template <class Key> std::vector<Key> KeysWithRepeats(uint32_t count, uint32_t seed) {
    const std::vector<Key> repeated = Random<Key>(64, seed);
    std::vector<Key> keys = Random<Key>(count, seed + 1);
    std::mt19937 random(seed);
    for (uint32_t i = 0; i < count; i += 2) {
        keys[i] = repeated[random() % repeated.size()];
    }
    return keys;
}

template <class Key> struct Pairs {
    std::vector<Key> keys;
    std::vector<uint32_t> values;

    bool operator==(const Pairs&) const = default;
};

template <class Key> Pairs<Key> StableSort(const Pairs<Key>& pairs, uint32_t count, uint32_t bits) {
    const Key mask = bits == 8 * sizeof(Key) ? ~Key(0) : (Key(1) << bits) - 1;
    std::vector<uint32_t> order(count);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return (pairs.keys[a] & mask) < (pairs.keys[b] & mask);
    });

    Pairs<Key> sorted = pairs;
    for (uint32_t i = 0; i < count; ++i) {
        sorted.keys[i] = pairs.keys[order[i]];
        sorted.values[i] = pairs.values[order[i]];
    }
    return sorted;
}

struct SortCase {
    uint32_t capacity;
    uint32_t count;
    std::optional<uint32_t> bits;
    bool countOnDevice = false;
};

template <class Key>
void ExpectSorted(Device& device, const RadixSort& sort, const SortCase& sortCase) {
    const auto [capacity, count, bits, countOnDevice] = sortCase;
    SCOPED_TRACE(testing::Message() << sizeof(Key) << "-byte keys, count " << count << ", bits "
                                    << bits.value_or(0) << (countOnDevice ? ", on device" : ""));

    Pairs<Key> pairs{KeysWithRepeats<Key>(capacity, count), std::vector<uint32_t>(capacity)};
    std::iota(pairs.values.begin(), pairs.values.end(), 0u);
    const Ref<Buffer> keys = Upload(device, pairs.keys);
    const Ref<Buffer> values = Upload(device, pairs.values);
    const Ref<Buffer> live = Upload(device, std::vector<uint32_t>{count});

    CommandEncoder encoder = device.GetQueue().CreateEncoder();
    if (countOnDevice) {
        sort.Sort(encoder, *keys, *values, *live, capacity, bits);
    } else {
        sort.Sort(encoder, *keys, *values, count, bits);
    }
    device.GetQueue().Submit(encoder.Finish());

    const Pairs<Key> sorted{Download<Key>(device, *keys, capacity),
                            Download<uint32_t>(device, *values, capacity)};
    const Pairs<Key> expected =
        StableSort(pairs, std::min(count, capacity), bits.value_or(8 * sizeof(Key)));
    EXPECT_EQ(sorted.keys, expected.keys);
    EXPECT_EQ(sorted.values, expected.values);
}

template <class Key> void SortAcrossTileBoundaries(Device& device) {
    constexpr uint32_t kCounts[] = {0, 1, 1023, 1024, 1025, 100'000};
    const RadixSort sort(device, 100'000, sizeof(Key));
    for (const uint32_t count : kCounts) {
        ExpectSorted<Key>(device, sort, {.capacity = count, .count = count, .bits = {}});
    }
}

TEST(RadixSortTest, SortsStablyAcrossTileBoundaries) {
    ForEachDevice([](Device& device) {
        SortAcrossTileBoundaries<uint32_t>(device);
        SortAcrossTileBoundaries<uint64_t>(device);
    });
}

TEST(RadixSortTest, SortsByTheLowKeyBits) {
    constexpr uint32_t kCount = 5000;
    ForEachDevice([](Device& device) {
        const RadixSort narrow(device, kCount, 4);
        for (const uint32_t bits : {1u, 8u, 13u, 24u, 32u}) {
            ExpectSorted<uint32_t>(device, narrow,
                                   {.capacity = kCount, .count = kCount, .bits = bits});
        }

        const RadixSort wide(device, kCount, 8);
        for (const uint32_t bits : {20u, 40u, 64u}) {
            ExpectSorted<uint64_t>(device, wide,
                                   {.capacity = kCount, .count = kCount, .bits = bits});
        }
    });
}

TEST(RadixSortTest, LeavesItemsPastTheDeviceCountAlone) {
    constexpr uint32_t kCapacity = 5000;
    ForEachDevice([](Device& device) {
        const RadixSort narrow(device, kCapacity, 4);
        const RadixSort wide(device, kCapacity, 8);
        for (const uint32_t live : {0u, 1u, 3000u, kCapacity, kCapacity + 7}) {
            for (const uint32_t bits : {24u, 32u}) {
                ExpectSorted<uint32_t>(
                    device, narrow,
                    {.capacity = kCapacity, .count = live, .bits = bits, .countOnDevice = true});
            }
            ExpectSorted<uint64_t>(
                device, wide,
                {.capacity = kCapacity, .count = live, .bits = 40, .countOnDevice = true});
        }
    });
}

TEST(RadixSortTest, RejectsWhatItCannotSort) {
    ForEachDevice([](Device& device) {
        EXPECT_THROW(RadixSort(device, 100, 2), Error);
        EXPECT_THROW(RadixSort(device, RadixSort::kMaxItems + 1), Error);

        const RadixSort sort(device, 100, 4);
        const Ref<Buffer> small = device.CreateBuffer({.bytes = 64});
        const Ref<Buffer> large = device.CreateBuffer({.bytes = 800});
        CommandEncoder encoder = device.GetQueue().CreateEncoder();
        EXPECT_THROW(sort.Sort(encoder, *large, *large, 101), Error);
        EXPECT_THROW(sort.Sort(encoder, *large, *large, 100, 0), Error);
        EXPECT_THROW(sort.Sort(encoder, *large, *large, 100, 33), Error);
        EXPECT_THROW(sort.Sort(encoder, *small, *large, 17), Error);
        EXPECT_THROW(sort.Sort(encoder, *large, *small, 17), Error);
        EXPECT_THROW(sort.Sort(encoder, *large, *large, *small, 101), Error);
        EXPECT_THROW(sort.Sort(encoder, *small, *small, *small, 17), Error);
        EXPECT_TRUE(encoder.Finish().Empty());
    });
}

}
}
