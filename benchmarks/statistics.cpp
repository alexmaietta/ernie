// SPDX-License-Identifier: BSD-3-Clause
#include "statistics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ernie::bench {

double Percentile(std::span<const double> values,
                  double probability) {
    if (values.empty()) {
        throw std::invalid_argument("A percentile needs a value");
    }
    std::vector<double> ordered(values.begin(), values.end());
    std::sort(ordered.begin(), ordered.end());
    const double position =
        static_cast<double>(ordered.size() - 1) * probability;
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight;
}

Summary Summarize(std::span<const double> values) {
    Summary summary;
    if (values.empty()) {
        return summary;
    }
    const auto n = static_cast<double>(values.size());
    summary.count = values.size();
    summary.mean =
        std::accumulate(values.begin(), values.end(), 0.0) / n;
    double squares = 0.0;
    for (const double value : values) {
        squares += (value - summary.mean) * (value - summary.mean);
    }
    summary.stdev = std::sqrt(squares / n);
    summary.min = *std::min_element(values.begin(), values.end());
    summary.max = *std::max_element(values.begin(), values.end());
    summary.median = summary.p50 = Percentile(values, 0.50);
    summary.p95 = Percentile(values, 0.95);
    summary.p99 = Percentile(values, 0.99);
    return summary;
}

double StudentT95(double df) {
    // Exact to three places for 1 to 30 degrees of freedom; beyond,
    // the normal quantile with its first correction term.
    static constexpr std::array<double, 30> kTable = {
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
        2.262,  2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
        2.110,  2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
        2.060,  2.056, 2.052, 2.048, 2.045, 2.042};
    if (!(df >= 1.0)) {
        return kTable[0];
    }
    if (df <= 30.0) {
        return kTable[static_cast<std::size_t>(std::floor(df)) - 1];
    }
    constexpr double z = 1.959964;
    return z + (z * z * z + z) / (4.0 * df);
}

}  // namespace ernie::bench
