// SPDX-License-Identifier: BSD-3-Clause
/// @file value.h
/// @brief One attribute value of any type, as JSON, Python and importers hand them to the scene.
#pragma once

#include "ernie/scene/path.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ernie {

/// @brief An attribute value: a number, vector, matrix, quaternion, token, path or array. The
/// scene converts it to the attribute's declared type, widening floats and ints.
class Value {
  public:
    using Data = std::variant<std::monostate, bool, int, float, double, glm::ivec2, glm::vec2,
                              glm::vec3, glm::vec4, glm::dvec2, glm::dvec3, glm::dvec4, glm::dmat4,
                              glm::dquat, std::string, Path, std::vector<float>, std::vector<Path>,
                              std::vector<std::string>>;

    Value() = default;
    template <typename T>
        requires std::is_constructible_v<Data, T> && (!std::is_same_v<std::decay_t<T>, Value>)
    Value(T&& value) : data_(std::forward<T>(value)) {}
    Value(std::string_view value) : data_(std::string(value)) {}
    Value(const char* value) : data_(std::string(value)) {}

    bool Empty() const {
        return std::holds_alternative<std::monostate>(data_);
    }
    template <typename T> bool Is() const {
        return std::holds_alternative<T>(data_);
    }
    template <typename T> const T* Get() const {
        return std::get_if<T>(&data_);
    }
    const Data& Raw() const {
        return data_;
    }
    bool operator==(const Value&) const = default;

  private:
    Data data_;
};

}  // namespace ernie
