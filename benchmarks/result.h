// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// A case's result, and its document: one JSON file per case, schema
// "ernie.benchmark.result.v1", keeping every sample and region beside
// their summaries, after Isaac Sim's result v2.

#include "bench.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace ernie::bench {

inline constexpr const char* kResultSchema =
    "ernie.benchmark.result.v1";

/// One measured sample: `steps` steps, each `durationMs` on average,
/// and what was recorded over them by name (with
/// `CaseResult::units`).
struct Sample {
    int index = 0;
    int steps = 1;
    double durationMs = 0.0;
    double totalMs = 0.0;
    std::map<std::string, double> metrics;
};

/// A region's times in one lifecycle phase or sample.
struct RegionObservation {
    std::string name;
    std::string phase;  ///< "setup", "measurement", "validation"...
    int sample = -1;    ///< Its sample, or -1 outside sampling.
    std::size_t count = 0;
    double totalMs = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;
};

struct CaseResult {
    std::string runId;
    Case c;
    Definition definition;
    std::map<std::string, std::string>
        parameters;  ///< Every one's value.
    std::vector<Sample> samples;
    std::map<std::string, std::string> units;  ///< Each metric's.
    std::vector<RegionObservation> regions;
    std::map<std::string, double> lifecycleMs;  ///< setup_ms...
    std::vector<std::pair<std::string, double>>
        stats;  ///< The scene's.
    /// What a step renders (`Scenario::Units`); each is a metric
    /// "<name>_per_s".
    std::vector<Unit> rates;
    bool validated = false;
    std::string validationFailure;
    std::vector<std::string> errors;  ///< "phase: what".
    nlohmann::json environment;

    bool Valid() const {
        return errors.empty() && static_cast<int>(samples.size()) ==
                                     definition.measuredSamples;
    }

    /// The mean over samples of metric `name` ("duration_ms" among
    /// them), or NaN.
    double Mean(const std::string& name) const;
};

nlohmann::json Document(const CaseResult& result);

/// Writes `document` to `path` whole or not at all: a temporary file
/// beside it, renamed over it.
void WriteAtomically(const std::filesystem::path& path,
                     const nlohmann::json& document);

/// `name` as a file name: slashes as "__", "=" as "-".
std::string FileNameOf(const std::string& name);

}  // namespace ernie::bench
