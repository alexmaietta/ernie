// SPDX-License-Identifier: BSD-3-Clause
/// @file convert.h
/// @brief Conversions between `Value`s and column elements.
#pragma once

#include "ernie/scene/storage.h"
#include "ernie/scene/value.h"
#include "ernie/util/algorithm.h"

#include <cmath>
#include <optional>
#include <type_traits>
#include <vector>

namespace ernie {

/// @return A number of any width, or none.
inline std::optional<double> ToNumber(const Value& value) {
    if (const auto* d = value.Get<double>()) {
        return *d;
    }
    if (const auto* f = value.Get<float>()) {
        return *f;
    }
    if (const auto* i = value.Get<int>()) {
        return *i;
    }
    return std::nullopt;
}

/// @return A vector of N doubles from one of floats, doubles or (for two) ints, or none.
template <glm::length_t N> std::optional<glm::vec<N, double>> ToVector(const Value& value) {
    if (const auto* d = value.Get<glm::vec<N, double>>()) {
        return *d;
    }
    if (const auto* f = value.Get<glm::vec<N, float>>()) {
        return glm::vec<N, double>(*f);
    }
    if constexpr (N == 2) {
        if (const auto* i = value.Get<glm::ivec2>()) {
            return glm::dvec2(*i);
        }
    }
    return std::nullopt;
}

/// @return The numbers of a number, or the components of a vector; none for other values.
inline std::vector<double> ToNumbers(const Value& value) {
    if (const std::optional<double> number = ToNumber(value)) {
        return {*number};
    }
    const auto components = [](const auto& vector) {
        std::vector<double> numbers(static_cast<std::size_t>(vector.length()));
        for (glm::length_t i = 0; i < vector.length(); ++i) {
            numbers[static_cast<std::size_t>(i)] = vector[i];
        }
        return numbers;
    };
    if (const auto v = ToVector<2>(value)) {
        return components(*v);
    }
    if (const auto v = ToVector<3>(value)) {
        return components(*v);
    }
    if (const auto v = ToVector<4>(value)) {
        return components(*v);
    }
    return {};
}

/// @return `value` as a column element, interning tokens and paths; none if it is another kind.
template <typename T> std::optional<T> ToElement(PathTable& paths, const Value& value) {
    if constexpr (std::is_same_v<T, std::uint8_t>) {
        if (const auto* b = value.Get<bool>()) {
            return static_cast<std::uint8_t>(*b);
        }
        if (const auto* i = value.Get<int>(); i && (*i == 0 || *i == 1)) {
            return static_cast<std::uint8_t>(*i);
        }
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        if (const std::optional<double> n = ToNumber(value); n && std::trunc(*n) == *n) {
            return static_cast<std::int32_t>(*n);
        }
    } else if constexpr (std::is_same_v<T, double>) {
        return ToNumber(value);
    } else if constexpr (std::is_same_v<T, glm::dvec2> || std::is_same_v<T, glm::dvec3> ||
                         std::is_same_v<T, glm::dvec4>) {
        return ToVector<T::length()>(value);
    } else if constexpr (std::is_same_v<T, glm::dquat>) {
        if (const auto* q = value.Get<glm::dquat>()) {
            return *q;
        }
    } else if constexpr (std::is_same_v<T, glm::dmat4>) {
        if (const auto* m = value.Get<glm::dmat4>()) {
            return *m;
        }
    } else if constexpr (std::is_same_v<T, Token>) {
        if (const auto* text = value.Get<std::string>()) {
            return paths.Names().Intern(*text);
        }
    } else if constexpr (std::is_same_v<T, Target>) {
        if (const auto* path = value.Get<Path>()) {
            return Target{path->IsEmpty() ? kNoPath : paths.Intern(*path)};
        }
    } else if constexpr (std::is_same_v<T, std::vector<Target>>) {
        if (const auto* path = value.Get<Path>()) {
            return std::vector<Target>{{paths.Intern(*path)}};
        }
        if (const auto* list = value.Get<std::vector<Path>>()) {
            return Map(*list, [&](const Path& path) { return Target{paths.Intern(path)}; });
        }
    } else if constexpr (std::is_same_v<T, std::vector<float>>) {
        if (const auto* floats = value.Get<std::vector<float>>()) {
            return *floats;
        }
        if (const std::optional<double> n = ToNumber(value)) {
            return std::vector<float>{static_cast<float>(*n)};
        }
    } else if constexpr (std::is_same_v<T, std::vector<Token>>) {
        if (const auto* texts = value.Get<std::vector<std::string>>()) {
            return Map(*texts, [&](const std::string& text) { return paths.Names().Intern(text); });
        }
        // An empty list, as Python and JSON pass one.
        if (const auto* floats = value.Get<std::vector<float>>(); floats && floats->empty()) {
            return std::vector<Token>{};
        }
    }
    return std::nullopt;
}

/// @return A column element as a `Value`; tensors have none.
template <typename T> Value ToValue(const PathTable& paths, const T& element) {
    if constexpr (std::is_same_v<T, std::uint8_t>) {
        return element != 0;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return static_cast<int>(element);
    } else if constexpr (std::is_same_v<T, Token>) {
        return std::string(paths.Names().Text(element));
    } else if constexpr (std::is_same_v<T, Target>) {
        return element.path == kNoPath ? Path() : paths.ToPath(element.path);
    } else if constexpr (std::is_same_v<T, std::vector<Target>>) {
        return Map(element, [&](const Target& target) { return paths.ToPath(target.path); });
    } else if constexpr (std::is_same_v<T, std::vector<Token>>) {
        return Map(element, [&](Token token) { return std::string(paths.Names().Text(token)); });
    } else if constexpr (std::is_same_v<T, Tensor>) {
        return Value();
    } else {
        return element;
    }
}

/// @return The column type an undeclared attribute's first value makes, or none.
inline std::optional<ColumnType> InferColumnType(const Value& value) {
    return std::visit(
        [](const auto& v) -> std::optional<ColumnType> {
            using V = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<V, bool>) {
                return ColumnType::Bool;
            } else if constexpr (std::is_same_v<V, int>) {
                return ColumnType::Int;
            } else if constexpr (std::is_same_v<V, float> || std::is_same_v<V, double>) {
                return ColumnType::Double;
            } else if constexpr (std::is_same_v<V, glm::ivec2> || std::is_same_v<V, glm::vec2> ||
                                 std::is_same_v<V, glm::dvec2>) {
                return ColumnType::Vec2;
            } else if constexpr (std::is_same_v<V, glm::vec3> || std::is_same_v<V, glm::dvec3>) {
                return ColumnType::Vec3;
            } else if constexpr (std::is_same_v<V, glm::vec4> || std::is_same_v<V, glm::dvec4>) {
                return ColumnType::Vec4;
            } else if constexpr (std::is_same_v<V, glm::dquat>) {
                return ColumnType::Quat;
            } else if constexpr (std::is_same_v<V, glm::dmat4>) {
                return ColumnType::Matrix;
            } else if constexpr (std::is_same_v<V, std::string>) {
                return ColumnType::TokenValue;
            } else if constexpr (std::is_same_v<V, Path>) {
                return ColumnType::Relationship;
            } else if constexpr (std::is_same_v<V, std::vector<Path>>) {
                return ColumnType::Relationships;
            } else if constexpr (std::is_same_v<V, std::vector<float>>) {
                return ColumnType::FloatArray;
            } else if constexpr (std::is_same_v<V, std::vector<std::string>>) {
                return ColumnType::TokenArray;
            } else {
                return std::nullopt;
            }
        },
        value.Raw());
}

}  // namespace ernie
