// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "ernie/util/text.h"

namespace ernie {

std::string Environment(const char* name);

template <class T> struct Choice {
    std::string_view name;
    T value;
};

[[noreturn]] void ThrowUnknownChoice(const char* name, std::string_view value,
                                     const std::vector<std::string>& names);

template <class Choices>
auto EnvironmentChoice(const char* name, const Choices& choices)
    -> std::optional<std::remove_cvref_t<decltype(std::begin(choices)->value)>> {
    const std::string value = Lowercase(Environment(name));
    if (value.empty()) {
        return std::nullopt;
    }

    std::vector<std::string> names;
    for (const auto& choice : choices) {
        if (choice.name == value) {
            return choice.value;
        }
        names.emplace_back(choice.name);
    }
    ThrowUnknownChoice(name, value, names);
}

}
