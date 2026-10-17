// SPDX-License-Identifier: BSD-3-Clause
/// @file numbers.h
/// @brief Plain number ranges accepted in place of glm vectors.
#pragma once

#include <glm/glm.hpp>

#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ernie {

/// @brief A contiguous range of numbers on the host.
/// A `std::array`, `std::vector`, `std::span` or C array, which
/// setters take in place of glm's types.
template <typename R>
concept NumberRange = std::ranges::contiguous_range<R> && std::ranges::sized_range<R> &&
                      std::is_arithmetic_v<std::ranges::range_value_t<R>> &&
                      !std::is_same_v<std::ranges::range_value_t<R>, bool>;

/// @brief Converts a range of `N` numbers to a vector.
/// @param numbers The numbers.
/// @return The numbers as a double-precision vector.
/// @throws std::invalid_argument If there are not exactly `N`.
template <glm::length_t N, NumberRange R> glm::vec<N, double> ToVector(const R& numbers) {
    const std::span values(numbers);
    if (values.size() != static_cast<std::size_t>(N)) {
        throw std::invalid_argument("Expected " + std::to_string(N) + " numbers, not " +
                                    std::to_string(values.size()));
    }
    glm::vec<N, double> vector(0.0);
    for (glm::length_t i = 0; i < N; ++i) {
        vector[i] = static_cast<double>(values[static_cast<std::size_t>(i)]);
    }
    return vector;
}

}  // namespace ernie
