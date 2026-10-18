// SPDX-License-Identifier: BSD-3-Clause
/// @file text.h
/// @brief Small string helpers for parsing text formats.
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ernie {

/// @brief Removes leading and trailing whitespace.
/// @param text The text.
/// @return The text without surrounding spaces, tabs and line breaks.
inline std::string Trim(std::string_view text) {
    const auto space = [](unsigned char c) { return std::isspace(c) != 0; };
    const auto begin = std::find_if_not(text.begin(), text.end(), space);
    const auto end =
        std::find_if_not(text.rbegin(), std::make_reverse_iterator(begin), space).base();
    return std::string(begin, end);
}

/// @brief Converts ASCII letters to upper case.
/// @param text The text.
/// @return The text in capitals.
inline std::string ToUpper(std::string_view text) {
    std::string upper(text);
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return upper;
}

/// @brief Converts ASCII letters to lower case.
/// @param text The text.
/// @return The text in small letters.
inline std::string ToLower(std::string_view text) {
    std::string lower(text);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower;
}

/// @return `number` as short text: "170" or "0.01", not "170.000000".
inline std::string ToText(double number) {
    std::ostringstream text;
    text << number;
    return text.str();
}

/// @brief Tests for an ASCII identifier: a letter or underscore, then letters, digits and
/// underscores, whatever the locale. USD prim names and attribute name parts are identifiers.
/// @param text The text.
/// @return Whether it is one.
constexpr bool IsIdentifier(std::string_view text) {
    const auto letter = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
    };
    if (text.empty() || !letter(text.front())) {
        return false;
    }
    return std::all_of(text.begin(), text.end(),
                       [&](char c) { return letter(c) || (c >= '0' && c <= '9'); });
}

/// @brief Reads numbers separated by blanks, line breaks or commas, in turn.
class NumberReader {
  public:
    /// @param text   The numbers.
    /// @param format The format they belong to, for errors.
    NumberReader(std::string_view text, std::string format) : format_(std::move(format)) {
        std::string spaced(text);
        std::replace(spaced.begin(), spaced.end(), ',', ' ');
        stream_.str(std::move(spaced));
    }

    /// @return The next number.
    /// @throws std::invalid_argument If there are no more.
    double Next() {
        double value = 0.0;
        if (!(stream_ >> value)) {
            throw std::invalid_argument(format_ + " ends early");
        }
        return value;
    }

    /// @param count How many numbers to pass over.
    void Skip(std::size_t count) {
        Values(count);
    }

    /// @param count How many numbers to read.
    /// @return The next `count` numbers.
    std::vector<double> Values(std::size_t count) {
        std::vector<double> values(count);
        for (double& value : values) {
            value = Next();
        }
        return values;
    }

    /// @return The next number, as a count.
    /// @throws std::invalid_argument If it is negative or fractional.
    std::size_t Count() {
        const double value = Next();
        if (!(value >= 0.0) || value != std::floor(value)) {
            throw std::invalid_argument(format_ + " has a bad count");
        }
        return static_cast<std::size_t>(value);
    }

  private:
    std::string format_;
    std::istringstream stream_;
};

}  // namespace ernie
