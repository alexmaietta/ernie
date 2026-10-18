// SPDX-License-Identifier: BSD-3-Clause
/// @file time_samples.cpp
/// @brief Bracketing time codes among samples, and interpolating between them.
#include "ernie/scene/time_samples.h"

#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <vector>

namespace ernie {
namespace {

template <typename T> std::vector<T> Mix(const T* a, const T* b, std::size_t count, double alpha) {
    std::vector<T> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = static_cast<T>(a[i] + (b[i] - a[i]) * alpha);
    }
    return out;
}

}  // namespace

SampleBracket Bracket(std::span<const TimeSample> samples, double time) {
    const auto above = std::ranges::upper_bound(samples, time, {}, &TimeSample::time);
    if (above == samples.begin()) {
        return {};
    }
    const auto lower = static_cast<std::size_t>(above - samples.begin()) - 1;
    if (above == samples.end() || time <= samples[lower].time) {
        return {lower, lower, 0.0};
    }
    const double t0 = samples[lower].time;
    return {lower, lower + 1, (time - t0) / (above->time - t0)};
}

Value Interpolate(const Value& a, const Value& b, double alpha) {
    if (alpha <= 0.0 || a.Raw().index() != b.Raw().index()) {
        return a;
    }
    return std::visit(
        [&](const auto& x) -> Value {
            using T = std::decay_t<decltype(x)>;
            const T& y = *b.Get<T>();
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                return static_cast<T>(x + (y - x) * alpha);
            } else if constexpr (std::is_same_v<T, glm::dquat>) {
                return glm::slerp(x, y, alpha);
            } else if constexpr (requires { T::length(); }) {
                // Vectors and matrices of floats or doubles; integer vectors hold.
                if constexpr (std::is_floating_point_v<typename T::value_type>) {
                    return x + (y - x) * static_cast<typename T::value_type>(alpha);
                } else {
                    return a;
                }
            } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                return x.size() == y.size() ? Value(Mix(x.data(), y.data(), x.size(), alpha)) : a;
            } else {
                return a;
            }
        },
        a.Raw());
}

std::optional<Tensor> Interpolate(const Tensor& a, const Tensor& b, double alpha) {
    const bool mixable = !a.Empty() && a.Location().kind == DeviceKind::Cpu &&
                         b.Location().kind == DeviceKind::Cpu && a.Type() == b.Type() &&
                         a.Shape() == b.Shape() &&
                         (a.Type() == DType::Float32 || a.Type() == DType::Float64);
    if (!mixable) {
        return std::nullopt;
    }
    const Tensor x = a.Contiguous();
    const Tensor y = b.Contiguous();
    const auto count = static_cast<std::size_t>(x.Elements());
    if (a.Type() == DType::Float32) {
        return Tensor::Copy<float>(Mix(x.As<float>(), y.As<float>(), count, alpha), a.Shape());
    }
    return Tensor::Copy<double>(Mix(x.As<double>(), y.As<double>(), count, alpha), a.Shape());
}

TimeSample Evaluate(std::span<const TimeSample> samples, double time) {
    const SampleBracket bracket = Bracket(samples, time);
    const TimeSample& lower = samples[bracket.lower];
    const TimeSample& upper = samples[bracket.upper];
    if (bracket.alpha == 0.0) {
        return lower;
    }
    if (!lower.tensor.Empty()) {
        return {time,
                {},
                Interpolate(lower.tensor, upper.tensor, bracket.alpha).value_or(lower.tensor)};
    }
    return {time, Interpolate(lower.value, upper.value, bracket.alpha), {}};
}

}  // namespace ernie
