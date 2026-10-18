// SPDX-License-Identifier: BSD-3-Clause
/// @file bytes.h
/// @brief Typed loads from raw bytes and byte-order swaps, for reading binary formats.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ernie {

/// @brief Reads a value from bytes that need not be aligned for it.
/// @tparam T A trivially copyable type.
/// @param bytes Where the value starts, in host byte order.
/// @return The value.
template <typename T> T LoadAs(const std::byte* bytes) {
    static_assert(std::is_trivially_copyable_v<T>);
    T value;
    std::memcpy(&value, bytes, sizeof(T));
    return value;
}

/// @brief Reads a little-endian value at an offset, as binary formats store them.
/// @tparam T    A trivially copyable type.
/// @param bytes The data, of any one-byte element type.
/// @param at    The value's offset in bytes.
/// @return The value in host order.
/// @throws std::runtime_error If the value runs past the data's end.
template <typename T, typename Byte, std::size_t Extent>
    requires(sizeof(Byte) == 1)
T ReadLittleEndian(std::span<const Byte, Extent> bytes, std::size_t at) {
    if (at > bytes.size() || sizeof(T) > bytes.size() - at) {
        throw std::runtime_error("Data ends before the value at byte " + std::to_string(at));
    }
    std::array<std::byte, sizeof(T)> value;
    std::memcpy(value.data(), bytes.data() + at, sizeof(T));
    if constexpr (std::endian::native == std::endian::big) {
        std::reverse(value.begin(), value.end());
    }
    return std::bit_cast<T>(value);
}

/// @brief A four-character code as formats store it, its first character in the lowest byte.
/// @param code The four characters.
/// @return The code as a little-endian 32-bit value.
constexpr std::uint32_t FourCc(const char (&code)[5]) {
    return static_cast<std::uint32_t>(static_cast<unsigned char>(code[0])) |
           static_cast<std::uint32_t>(static_cast<unsigned char>(code[1])) << 8 |
           static_cast<std::uint32_t>(static_cast<unsigned char>(code[2])) << 16 |
           static_cast<std::uint32_t>(static_cast<unsigned char>(code[3])) << 24;
}

/// @brief Whether values stored in one byte order need swapping on this host.
/// @param bigEndian Whether they are stored most significant byte first.
/// @param width     Their size in bytes.
/// @return True if they are wider than a byte and stored in the other order.
constexpr bool NeedsByteSwap(bool bigEndian, std::size_t width) {
    return width > 1 && bigEndian != (std::endian::native == std::endian::big);
}

/// @brief Reverses the bytes of each `width`-byte value in place, between byte orders.
/// @param data  The values, packed.
/// @param width Each value's size in bytes.
inline void SwapByteOrder(std::span<std::byte> data, std::size_t width) {
    for (std::size_t i = 0; i + width <= data.size(); i += width) {
        std::reverse(data.begin() + static_cast<std::ptrdiff_t>(i),
                     data.begin() + static_cast<std::ptrdiff_t>(i + width));
    }
}

}  // namespace ernie
