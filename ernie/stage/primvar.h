// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "ernie/stage/schema/primvars.h"
#include "ernie/stage/schema/types.h"
#include "ernie/util/span.h"

namespace ernie {

enum class Interpolation : uint32_t;

template <class T> constexpr PrimvarType kPrimvarType = PrimvarType::Float;
template <> inline constexpr PrimvarType kPrimvarType<Float2> = PrimvarType::Float2;
template <> inline constexpr PrimvarType kPrimvarType<Float3> = PrimvarType::Float3;
template <> inline constexpr PrimvarType kPrimvarType<Float4> = PrimvarType::Float4;
template <> inline constexpr PrimvarType kPrimvarType<Double3> = PrimvarType::Float3;
template <> inline constexpr PrimvarType kPrimvarType<int32_t> = PrimvarType::Int;

struct Primvar {
    std::string name;
    PrimvarType type = PrimvarType::Float;
    Interpolation interpolation{};
    uint32_t elementSize = 1;
    std::vector<uint32_t> words;
    std::vector<int32_t> indices;

    template <class T>
    static Primvar Create(std::string name, Interpolation interpolation, Span<const T> values,
                          std::vector<int32_t> indices = {}) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, Float2> ||
                      std::is_same_v<T, Float3> || std::is_same_v<T, Float4> ||
                      std::is_same_v<T, double> || std::is_same_v<T, Double3> ||
                      std::is_same_v<T, int32_t>);
        Primvar primvar{std::move(name), kPrimvarType<T>, interpolation};
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, Double3>) {
            for (const T& value : values) {
                for (const double component :
                     std::bit_cast<std::array<double, sizeof(T) / 8>>(value)) {
                    primvar.words.push_back(std::bit_cast<uint32_t>(float(component)));
                }
            }
        } else {
            primvar.words.resize(values.size_bytes() / sizeof(uint32_t));
            std::memcpy(primvar.words.data(), values.data(), values.size_bytes());
        }
        primvar.indices = std::move(indices);
        return primvar;
    }

    template <class T>
    static Primvar Create(std::string name, Interpolation interpolation,
                          const std::vector<T>& values, std::vector<int32_t> indices = {}) {
        return Create(std::move(name), interpolation, Span<const T>(values), std::move(indices));
    }

    uint32_t ValueCount() const {
        return uint32_t(words.size() / (PrimvarComponents(type) * elementSize));
    }

    template <class T> std::vector<T> Values() const {
        std::vector<T> values(words.size() * sizeof(uint32_t) / sizeof(T));
        std::memcpy(static_cast<void*>(values.data()), words.data(), values.size() * sizeof(T));
        return values;
    }
};

}
