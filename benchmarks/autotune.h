// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Autotuning: the largest value of a case's integer parameter
// (worlds, views...) whose run stays within limits, as Isaac Lab's
// camera benchmark finds how many cameras fit. Probes grow the value
// by a factor until one fails (a limit crossed, or the run failing,
// out of memory say), then bisect between the last that passed and
// it.

#include "runner.h"

#include <string>
#include <vector>

namespace ernie::bench {

/// A bound on a metric's mean over a probe's samples, as
/// "duration_ms<=16.7" or "views_per_s>=100000".
struct Limit {
    std::string metric;
    bool atMost = true;
    double value = 0.0;

    /// @throws std::invalid_argument for text not of that form.
    static Limit Parse(const std::string& text);
    std::string Text() const;
};

struct AutotuneOptions {
    std::string param;  ///< An integer parameter of the case's.
    int start = 0;      ///< 0: the case's own value.
    double factor = 2.0;
    int max = 1 << 20;
    /// The bisection's precision as a fraction of the value found.
    double precision = 0.05;
    /// Besides the run being valid; none given, the device's memory
    /// at most 90% of its budget.
    std::vector<Limit> limits;
    SamplingOverrides probe{.warmupSamples = 1, .measuredSamples = 5};
};

struct Probe {
    int value = 0;
    bool passed = false;
    std::string reason;  ///< Why it failed.
    std::map<std::string, double> means;
};

struct AutotuneResult {
    Case c;
    AutotuneOptions options;
    std::vector<Probe> probes;
    int best = 0;  ///< 0 when even the first probe failed.
    std::string limitedBy;
};

/// Autotunes `c` on `runner`, prints its probes, and writes
/// "autotune__<case>.json" to the run's directory with the best
/// value's full result beside it.
AutotuneResult Autotune(Runner& runner, const Case& c,
                        const AutotuneOptions& options);

}  // namespace ernie::bench
