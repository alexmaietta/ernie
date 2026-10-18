// SPDX-License-Identifier: BSD-3-Clause
/// @file time_samples.h
/// @brief An attribute's values at time codes, and its value between them, as USD interpolates
/// them.
///
/// References:
/// - Pixar, "UsdAttribute: Interpolation of Time Samples",
///   https://openusd.org/release/api/class_usd_attribute.html
#pragma once

#include "ernie/scene/value.h"
#include "ernie/util/tensor.h"

#include <cstddef>
#include <optional>
#include <span>

namespace ernie {

/// @brief One time sample: a value, or for an array attribute a tensor.
struct TimeSample {
    double time = 0.0;
    Value value;
    Tensor tensor;
};

/// @brief Where a time falls among samples: the samples either side, and how far between them
/// (0 at the lower). At or beyond the ends, and on a sample, both are that sample.
struct SampleBracket {
    std::size_t lower = 0;
    std::size_t upper = 0;
    double alpha = 0.0;
    bool operator==(const SampleBracket&) const = default;
};

/// @param samples In increasing time, at least one.
SampleBracket Bracket(std::span<const TimeSample> samples, double time);

/// @return `a` and `b` mixed `alpha` of the way: linearly for numbers, vectors, matrices (by
/// element, as USD does) and float arrays of one size, spherically for quaternions; `a`, held,
/// for the rest (bools, ints, tokens, paths) and for values of different kinds.
Value Interpolate(const Value& a, const Value& b, double alpha);
/// @return Two float CPU tensors of one shape and type mixed linearly, element by element; none
/// for any others, which hold the lower sample (a device's consumers mix them there).
std::optional<Tensor> Interpolate(const Tensor& a, const Tensor& b, double alpha);

/// @return The samples' value or tensor at `time`.
TimeSample Evaluate(std::span<const TimeSample> samples, double time);

}  // namespace ernie
