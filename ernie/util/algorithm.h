// SPDX-License-Identifier: BSD-3-Clause
/// @file algorithm.h
/// @brief Small range helpers.
#pragma once

#include <algorithm>
#include <functional>
#include <ranges>
#include <type_traits>
#include <vector>

namespace ernie {

/// @return `transform` of each element of `range`, in order.
template <std::ranges::input_range R, typename F> auto Map(const R& range, F&& transform) {
    using Out = std::decay_t<std::invoke_result_t<F&, std::ranges::range_reference_t<const R>>>;
    std::vector<Out> out;
    if constexpr (std::ranges::sized_range<R>) {
        out.reserve(std::ranges::size(range));
    }
    for (auto&& element : range) {
        out.push_back(std::invoke(transform, element));
    }
    return out;
}

/// @brief Sorts `values` and drops repeats, as a set of indices to visit in order.
template <typename T> void SortUnique(std::vector<T>& values) {
    std::ranges::sort(values);
    values.erase(std::ranges::unique(values).begin(), values.end());
}

}  // namespace ernie
