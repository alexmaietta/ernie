// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <glm/glm.hpp>

#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ernie {

/// A contiguous range of numbers on the host: a `std::array`,
/// `std::vector`, `std::span` or C array of them, which setters take
/// in place of glm's types.
template <typename R>
concept NumberRange =
    std::ranges::contiguous_range<R> && std::ranges::sized_range<R> &&
    std::is_arithmetic_v<std::ranges::range_value_t<R>> &&
    !std::is_same_v<std::ranges::range_value_t<R>, bool>;

/// The `N` numbers of `numbers` as a vector.
/// @throws std::invalid_argument if they are not `N`.
template <glm::length_t N, NumberRange R>
glm::vec<N, double> VectorOf(const R& numbers) {
    const std::span values(numbers);
    if (values.size() != static_cast<std::size_t>(N)) {
        throw std::invalid_argument("Expected " + std::to_string(N) +
                                    " numbers, not " +
                                    std::to_string(values.size()));
    }
    glm::vec<N, double> vector(0.0);
    for (glm::length_t i = 0; i < N; ++i) {
        vector[i] =
            static_cast<double>(values[static_cast<std::size_t>(i)]);
    }
    return vector;
}

}  // namespace ernie
