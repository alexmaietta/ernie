// SPDX-License-Identifier: BSD-3-Clause
/// @file bytes_test.cpp
/// @brief Tests of typed loads and byte-order swaps.
#include "ernie/util/bytes.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>

namespace ernie {
namespace {

TEST(Bytes, LoadAsReadsUnalignedValues) {
    std::array<std::byte, 5> bytes{};
    const std::uint32_t value = 0x01020304u;
    std::memcpy(bytes.data() + 1, &value, 4);
    EXPECT_EQ(LoadAs<std::uint32_t>(bytes.data() + 1), value);
}

TEST(Bytes, ReadLittleEndianReadsInBoundsAndRejectsPastTheEnd) {
    const std::array<std::uint8_t, 6> bytes{0xff, 0x04, 0x03, 0x02, 0x01, 0xff};
    EXPECT_EQ(ReadLittleEndian<std::uint32_t>(std::span<const std::uint8_t>(bytes), 1),
              0x01020304u);
    EXPECT_EQ(ReadLittleEndian<std::uint16_t>(std::as_bytes(std::span(bytes)), 4), 0xff01u);
    EXPECT_THROW(ReadLittleEndian<std::uint32_t>(std::span<const std::uint8_t>(bytes), 3),
                 std::runtime_error);
    EXPECT_THROW(ReadLittleEndian<std::uint8_t>(std::span<const std::uint8_t>(bytes), 99),
                 std::runtime_error);
}

TEST(Bytes, FourCcPutsTheFirstCharacterLowest) {
    static_assert(FourCc("DDS ") == 0x20534444u);
    const std::uint32_t code = FourCc("DX10");
    EXPECT_EQ(std::memcmp(&code, "DX10", 4), 0);
}

TEST(Bytes, SwapByteOrderReversesEachValue) {
    std::array<std::byte, 5> bytes{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4},
                                   std::byte{5}};
    SwapByteOrder(bytes, 2);
    EXPECT_EQ(bytes, (std::array<std::byte, 5>{std::byte{2}, std::byte{1}, std::byte{4},
                                               std::byte{3}, std::byte{5}}))
        << "a partial value at the end is left";
}

TEST(Bytes, OnlyWideValuesInTheOtherOrderNeedSwapping) {
    constexpr bool big = std::endian::native == std::endian::big;
    EXPECT_FALSE(NeedsByteSwap(!big, 1));
    EXPECT_FALSE(NeedsByteSwap(big, 4));
    EXPECT_TRUE(NeedsByteSwap(!big, 4));
}

}  // namespace
}  // namespace ernie
