// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// The runner's side of `Measure` and `Record` (bench.h): while a
// lifecycle phase or a sample runs, they write into its
// `Observations`.

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ernie::bench {

/// What `Measure` and `Record` collected in one phase or sample.
struct Observations {
    /// Each region's times, ms, by its "/"-joined name.
    std::map<std::string, std::vector<double>> regions;
    /// Each recorded metric's values and unit.
    std::map<std::string, std::pair<std::vector<double>, std::string>>
        records;
};

/// Makes `observations` the target of `Measure`, and of `Record` when
/// `records` (in measured samples), until destroyed.
class ActiveObservations {
  public:
    ActiveObservations(Observations& observations, bool records);
    ~ActiveObservations();
    ActiveObservations(const ActiveObservations&) = delete;
    ActiveObservations& operator=(const ActiveObservations&) = delete;

  private:
    Observations* previous_;
    bool previousRecords_;
    std::vector<std::string> previousStack_;
};

}  // namespace ernie::bench
