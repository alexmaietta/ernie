// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/value.h"
#include "ernie/util/tensor.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

/// Time samples: an attribute's values at time codes, and its value
/// at any time between them, as USD interpolates them (`UsdAttribute`
/// with linear interpolation, the stage default).
///
/// Reference: Pixar, "UsdAttribute: Interpolation of Time Samples",
/// https://openusd.org/release/api/class_usd_attribute.html
namespace ernie {

/// One time sample of an attribute: a value, or a tensor and the CUDA
/// stream it was written on (`Scene::Write`).
struct TimeSample {
    double time = 0.0;
    Value value;
    Tensor tensor;
    std::uintptr_t stream = 1;
    bool isTensor = false;
};

/// Where `time` falls among samples at `times` (increasing, not
/// empty): the samples either side and how far between them, 0 at the
/// lower; one sample (both the same, 0) at or outside the ends and on
/// a sample.
struct SampleBracket {
    std::size_t lower = 0;
    std::size_t upper = 0;
    double alpha = 0.0;
};
SampleBracket BracketOf(std::span<const TimeSample> samples,
                        double time);

/// `a` and `b` interpolated `alpha` of the way: linearly for numbers,
/// vectors, matrices (by element, as USD does) and float arrays of
/// one size, spherically for quaternions; `a` (held) for the rest
/// (bools, ints, tokens, paths) and for values of different types.
Value Interpolate(const Value& a, const Value& b, double alpha);

/// `a` and `b` interpolated linearly, element by element, when both
/// are float32 or float64 CPU tensors of one shape and type; none
/// otherwise (a device tensor is held, or blended by its consumer on
/// the device).
std::optional<Tensor> Interpolate(const Tensor& a, const Tensor& b,
                                  double alpha);

/// Whether `Interpolate` blends `a` and `b` (rather than holding
/// `a`).
bool CanInterpolate(const Tensor& a, const Tensor& b);

}  // namespace ernie
