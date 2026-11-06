// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#if __has_include(<version>)
#include <version>
#endif
#ifdef __cpp_lib_ranges
#include <ranges>
#endif
#ifdef __cpp_lib_span
#include <span>
#endif

namespace ernie {

template <class T> class Span {
    template <class Container>
    using Element = std::remove_pointer_t<decltype(std::data(std::declval<Container&>()))>;

    template <class Container>
    using FromContainer = std::enable_if_t<
        !std::is_same_v<std::remove_cv_t<std::remove_reference_t<Container>>, Span> &&
        (std::is_lvalue_reference_v<Container> || std::is_const_v<T>) &&
        std::is_convertible_v<Element<Container> (*)[], T (*)[]> &&
        std::is_convertible_v<decltype(std::size(std::declval<Container&>())), size_t>>;

  public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;

    constexpr Span() noexcept = default;

    constexpr Span(T* data, size_t size) noexcept : m_data(data), m_size(size) {}

    constexpr Span(T* first, T* last) noexcept
        : m_data(first), m_size(static_cast<size_t>(last - first)) {}

    template <size_t N> constexpr Span(T (&array)[N]) noexcept : m_data(array), m_size(N) {}

    template <class Container, class = FromContainer<Container>>
    constexpr Span(Container&& container) noexcept(noexcept(std::data(container)))
        : m_data(std::data(container)), m_size(std::size(container)) {}

    template <class U, class = std::enable_if_t<std::is_convertible_v<U (*)[], T (*)[]>>>
    constexpr Span(Span<U> other) noexcept : m_data(other.data()), m_size(other.size()) {}

    constexpr T* data() const noexcept {
        return m_data;
    }

    constexpr size_t size() const noexcept {
        return m_size;
    }

    constexpr size_t size_bytes() const noexcept {
        return m_size * sizeof(T);
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return m_size == 0;
    }

    constexpr T* begin() const noexcept {
        return m_data;
    }

    constexpr T* end() const noexcept {
        return m_data + m_size;
    }

    constexpr T& operator[](size_t index) const noexcept {
        return m_data[index];
    }

    constexpr T& front() const noexcept {
        return m_data[0];
    }

    constexpr T& back() const noexcept {
        return m_data[m_size - 1];
    }

    constexpr Span first(size_t count) const noexcept {
        return {m_data, count};
    }

    constexpr Span last(size_t count) const noexcept {
        return {m_data + m_size - count, count};
    }

    constexpr Span subspan(size_t offset, size_t count = size_t(-1)) const noexcept {
        return {m_data + offset, count == size_t(-1) ? m_size - offset : count};
    }

  private:
    T* m_data = nullptr;
    size_t m_size = 0;
};

template <class T, size_t N> Span(T (&)[N]) -> Span<T>;
template <class T, size_t N> Span(std::array<T, N>&) -> Span<T>;
template <class T, size_t N> Span(const std::array<T, N>&) -> Span<const T>;
template <class Container>
Span(Container&) -> Span<std::remove_pointer_t<decltype(std::data(std::declval<Container&>()))>>;

template <class T> Span<const std::byte> AsBytes(Span<T> span) noexcept {
    return {reinterpret_cast<const std::byte*>(span.data()), span.size_bytes()};
}

template <class T> std::vector<T> Items(Span<const std::byte> bytes) {
    std::vector<T> items(bytes.size() / sizeof(T));
    if (!items.empty()) {
        std::memcpy(items.data(), bytes.data(), items.size() * sizeof(T));
    }
    return items;
}

}

#ifdef __cpp_lib_ranges
template <class T> inline constexpr bool std::ranges::enable_borrowed_range<ernie::Span<T>> = true;
#endif
