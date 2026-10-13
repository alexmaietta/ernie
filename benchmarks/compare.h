// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Comparing runs, after Isaac Lab's benchmarking method: a metric's
// difference between a base and a head per case, with a 95%
// confidence interval (Welch's), called faster or slower only when
// the interval leaves out zero, else inconclusive. Several runs of
// each (separate processes, alternated) make the interval one over
// runs; one run of each, one over its samples, which shares the
// process's state and so understates the uncertainty.

#include <filesystem>
#include <string>
#include <vector>

namespace ernie::bench {

/// Prints the comparison of `metric` between the runs in `base` and
/// those in `head` (run directories); returns how many cases differ
/// conclusively for the worse.
int Compare(const std::vector<std::filesystem::path>& base,
            const std::vector<std::filesystem::path>& head,
            const std::string& metric);

}  // namespace ernie::bench
