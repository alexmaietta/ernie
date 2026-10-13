// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <span>

namespace ernie::bench {

/// The summary of a metric's samples, none dropped as outliers.
struct Summary {
    std::size_t count = 0;
    double mean = 0.0;
    double median = 0.0;
    double stdev = 0.0;  ///< Of the population of samples.
    double min = 0.0;
    double max = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
};

/// The `probability` quantile of `values`, interpolated linearly
/// between the nearest two.
double Percentile(std::span<const double> values, double probability);

Summary Summarize(std::span<const double> values);

/// The two-sided 95% quantile of Student's t with `df` degrees of
/// freedom.
double StudentT95(double df);

}  // namespace ernie::bench
