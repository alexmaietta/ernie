// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/time_samples.h"

#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <thread>
#include <variant>

namespace ernie {

SampleBracket BracketOf(std::span<const TimeSample> samples,
                        double time) {
    SampleBracket bracket;
    const auto upper =
        std::upper_bound(samples.begin(), samples.end(), time,
                         [](double t, const TimeSample& sample) {
                             return t < sample.time;
                         });
    if (upper == samples.begin()) {
        return bracket;
    }
    if (upper == samples.end()) {
        bracket.lower = bracket.upper = samples.size() - 1;
        return bracket;
    }
    bracket.upper = static_cast<std::size_t>(upper - samples.begin());
    bracket.lower = bracket.upper - 1;
    const double t0 = samples[bracket.lower].time;
    const double t1 = samples[bracket.upper].time;
    if (time <= t0) {
        bracket.upper = bracket.lower;
        return bracket;
    }
    bracket.alpha = (time - t0) / (t1 - t0);
    return bracket;
}

Value Interpolate(const Value& a, const Value& b, double alpha) {
    if (alpha <= 0.0 || a.Raw().index() != b.Raw().index()) {
        return a;
    }
    return std::visit(
        [&](const auto& x) -> Value {
            using T = std::decay_t<decltype(x)>;
            const T& y = *b.Get<T>();
            if constexpr (std::is_same_v<T, float>) {
                return x + (y - x) * static_cast<float>(alpha);
            } else if constexpr (std::is_same_v<T, double>) {
                return x + (y - x) * alpha;
            } else if constexpr (std::is_same_v<T, glm::vec2> ||
                                 std::is_same_v<T, glm::vec3> ||
                                 std::is_same_v<T, glm::vec4>) {
                return glm::mix(x, y, static_cast<float>(alpha));
            } else if constexpr (std::is_same_v<T, glm::dvec2> ||
                                 std::is_same_v<T, glm::dvec3> ||
                                 std::is_same_v<T, glm::dvec4>) {
                return glm::mix(x, y, alpha);
            } else if constexpr (std::is_same_v<T, glm::dmat4>) {
                return x + (y - x) * alpha;
            } else if constexpr (std::is_same_v<T, glm::dquat>) {
                return glm::slerp(x, y, alpha);
            } else if constexpr (std::is_same_v<T,
                                                std::vector<float>>) {
                if (x.size() != y.size()) {
                    return a;
                }
                std::vector<float> out(x.size());
                for (std::size_t i = 0; i < x.size(); ++i) {
                    out[i] = x[i] + (y[i] - x[i]) *
                                        static_cast<float>(alpha);
                }
                return out;
            } else {
                return a;  // Held: bools, ints, tokens, paths.
            }
        },
        a.Raw());
}

namespace {

template <typename T>
Tensor Blend(const Tensor& a, const Tensor& b, double alpha) {
    const Tensor pa = a.Contiguous();
    const Tensor pb = b.Contiguous();
    const auto count = static_cast<std::size_t>(pa.Elements());
    std::vector<T> out(count);
    const T* x = pa.As<T>();
    const T* y = pb.As<T>();
    const T t = static_cast<T>(alpha);
    // Large arrays (a point cache) split over threads.
    const std::size_t threads = std::clamp<std::size_t>(
        count / (1u << 18), 1,
        std::max(1u, std::thread::hardware_concurrency()));
    const auto run = [&](std::size_t part) {
        const std::size_t begin = count * part / threads;
        const std::size_t end = count * (part + 1) / threads;
        for (std::size_t i = begin; i < end; ++i) {
            out[i] = x[i] + (y[i] - x[i]) * t;
        }
    };
    std::vector<std::thread> pool;
    for (std::size_t part = 1; part < threads; ++part) {
        pool.emplace_back(run, part);
    }
    run(0);
    for (std::thread& thread : pool) {
        thread.join();
    }
    return Tensor::Copy<T>(out, a.Shape());
}

}  // namespace

bool CanInterpolate(const Tensor& a, const Tensor& b) {
    return a.Location().kind == DeviceKind::Cpu &&
           b.Location().kind == DeviceKind::Cpu &&
           a.Type() == b.Type() && a.Shape() == b.Shape() &&
           (a.Type() == DType::Float32 || a.Type() == DType::Float64);
}

std::optional<Tensor> Interpolate(const Tensor& a, const Tensor& b,
                                  double alpha) {
    if (alpha <= 0.0 || !CanInterpolate(a, b)) {
        return std::nullopt;
    }
    if (a.Type() == DType::Float32) {
        return Blend<float>(a, b, alpha);
    }
    if (a.Type() == DType::Float64) {
        return Blend<double>(a, b, alpha);
    }
    return std::nullopt;
}

}  // namespace ernie
