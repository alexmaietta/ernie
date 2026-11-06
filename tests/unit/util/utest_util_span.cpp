// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/span.h"

#include <array>
#include <cstdint>
#include <ranges>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

namespace ernie {
namespace {

static_assert(std::ranges::borrowed_range<Span<int>>);
static_assert(std::is_convertible_v<Span<int>, Span<const int>>);
static_assert(!std::is_convertible_v<Span<const int>, Span<int>>);
static_assert(!std::is_constructible_v<Span<int>, std::vector<int>&&>);
static_assert(std::is_constructible_v<Span<const int>, std::vector<int>&&>);
static_assert(std::is_same_v<decltype(Span(std::declval<int (&)[3]>())), Span<int>>);
static_assert(std::is_same_v<decltype(Span(std::declval<std::array<int, 3>&>())), Span<int>>);
static_assert(
    std::is_same_v<decltype(Span(std::declval<const std::array<int, 3>&>())), Span<const int>>);
static_assert(std::is_same_v<decltype(Span(std::declval<std::vector<float>&>())), Span<float>>);

TEST(SpanTest, IsEmptyByDefault) {
    const Span<int> empty;
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.data(), nullptr);
    EXPECT_EQ(empty.size(), 0u);
    EXPECT_EQ(empty.begin(), empty.end());
}

TEST(SpanTest, ViewsArraysPointersAndContainers) {
    int array[] = {1, 2, 3, 4};
    std::vector<int> vector{5, 6};
    const Span<int> fromArray(array);
    const Span<int> fromPointer(array + 1, 2);
    const Span<int> fromRange(array, array + 3);
    const Span<int> fromVector(vector);
    const Span<const int> fromTemporary(std::vector<int>{7});
    const Span<const int> widened = fromArray;

    EXPECT_EQ(fromArray.size(), 4u);
    EXPECT_EQ(fromPointer.front(), 2);
    EXPECT_EQ(fromRange.back(), 3);
    EXPECT_EQ(fromVector[1], 6);
    EXPECT_EQ(fromTemporary.size(), 1u);
    EXPECT_EQ(widened.data(), array);
    EXPECT_EQ(fromArray.size_bytes(), 4 * sizeof(int));
}

TEST(SpanTest, IteratesAndWritesThrough) {
    int array[] = {1, 2, 3};
    int sum = 0;
    for (int& value : Span<int>(array)) {
        sum += value;
        value *= 10;
    }
    EXPECT_EQ(sum, 6);
    EXPECT_EQ(array[2], 30);
}

TEST(SpanTest, SlicesFromEitherEnd) {
    int array[] = {1, 2, 3, 4, 5};
    const Span<int> span(array);
    EXPECT_EQ(span.first(2).back(), 2);
    EXPECT_EQ(span.last(2).front(), 4);
    EXPECT_EQ(span.subspan(1).size(), 4u);
    EXPECT_EQ(span.subspan(1, 2).back(), 3);
}

TEST(SpanTest, ConvertsToAndFromBytes) {
    const uint16_t words[] = {0x0102, 0x0304};
    const Span<const std::byte> bytes = AsBytes(Span<const uint16_t>(words));
    EXPECT_EQ(bytes.size(), 4u);
    EXPECT_EQ(Items<uint16_t>(bytes), (std::vector<uint16_t>{0x0102, 0x0304}));
    EXPECT_EQ(Items<uint32_t>(bytes.first(3)).size(), 0u);
    EXPECT_EQ(Items<uint16_t>(bytes.first(3)), (std::vector<uint16_t>{0x0102}));
}

}
}
