// SPDX-License-Identifier: BSD-3-Clause
#include "compare.h"

#include "result.h"
#include "statistics.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <stdexcept>

namespace ernie::bench {

namespace {

/// A case's metric in one side's runs: each run's samples.
using Runs = std::vector<std::vector<double>>;

/// Each case's samples of `metric` in the results of `directories`.
std::map<std::string, Runs>
Load(const std::vector<std::filesystem::path>& directories,
     const std::string& metric, std::string& unit) {
    std::map<std::string, Runs> cases;
    for (const std::filesystem::path& directory : directories) {
        if (!std::filesystem::is_directory(directory)) {
            throw std::invalid_argument(directory.string() +
                                        " is not a run directory");
        }
        for (const auto& entry :
             std::filesystem::directory_iterator(directory)) {
            if (entry.path().extension() != ".json") {
                continue;
            }
            std::ifstream file(entry.path());
            const nlohmann::json document =
                nlohmann::json::parse(file, nullptr, false);
            if (document.is_discarded() ||
                document.value("schema", "") != kResultSchema ||
                !document["status"].value("valid", false)) {
                continue;
            }
            const auto& metrics = document["metrics"];
            if (!metrics.contains(metric)) {
                continue;
            }
            unit = metrics[metric].value("unit", "");
            std::vector<double> samples;
            for (const auto& sample : metrics[metric]["samples"]) {
                samples.push_back(sample["value"].get<double>());
            }
            cases[document["benchmark"]["case"].get<std::string>()]
                .push_back(std::move(samples));
        }
    }
    return cases;
}

struct Spread {
    double mean = 0.0;
    double variance = 0.0;  ///< Of the mean.
    double n = 0.0;
};

/// The mean and its variance: over the runs' means when there are
/// several runs, else over the one run's samples.
Spread SpreadOf(const Runs& runs) {
    std::vector<double> values;
    if (runs.size() > 1) {
        for (const std::vector<double>& run : runs) {
            values.push_back(Summarize(run).mean);
        }
    } else {
        values = runs.front();
    }
    const Summary s = Summarize(values);
    const auto n = static_cast<double>(values.size());
    // The sample variance (Bessel's), over n for the mean's.
    const double variance =
        n > 1.0 ? s.stdev * s.stdev * n / (n - 1.0) : 0.0;
    return {.mean = s.mean, .variance = variance / n, .n = n};
}

/// Whether larger values of `metric` are better: rates and fractions
/// of work done.
bool HigherIsBetter(const std::string& metric) {
    return metric.ends_with("_per_s");
}

}  // namespace

int Compare(const std::vector<std::filesystem::path>& base,
            const std::vector<std::filesystem::path>& head,
            const std::string& metric) {
    std::string unit;
    const std::map<std::string, Runs> before =
        Load(base, metric, unit);
    const std::map<std::string, Runs> after =
        Load(head, metric, unit);
    std::size_t width = 4;
    for (const auto& [name, runs] : before) {
        width = std::max(width, name.size());
    }
    const int w = static_cast<int>(width);
    std::printf(
        "%s (%s), %zu base and %zu head runs; 95%% intervals over "
        "%s\n\n",
        metric.c_str(), unit.c_str(), base.size(), head.size(),
        base.size() > 1 && head.size() > 1 ? "runs"
                                           : "one run's samples");
    std::printf("%-*s %11s %11s %9s %21s  %s\n", w, "case", "base",
                "head", "change", "95% interval", "verdict");
    int worse = 0;
    int compared = 0;
    for (const auto& [name, runs] : before) {
        const auto other = after.find(name);
        if (other == after.end()) {
            continue;
        }
        ++compared;
        const Spread a = SpreadOf(runs);
        const Spread b = SpreadOf(other->second);
        const double delta = b.mean - a.mean;
        const double se = std::sqrt(a.variance + b.variance);
        // Welch-Satterthwaite degrees of freedom.
        double df = 1.0;
        if (se > 0.0 && a.n > 1.0 && b.n > 1.0) {
            df = (a.variance + b.variance) *
                 (a.variance + b.variance) /
                 (a.variance * a.variance / (a.n - 1.0) +
                  b.variance * b.variance / (b.n - 1.0));
        }
        const double half = StudentT95(df) * se;
        const double low = delta - half;
        const double high = delta + half;
        std::string verdict = "inconclusive";
        if (low > 0.0 || high < 0.0) {
            const bool better =
                (delta > 0.0) == HigherIsBetter(metric);
            verdict = better ? "better" : "worse";
            worse += better ? 0 : 1;
        }
        const double relative =
            a.mean != 0.0 ? 100.0 * delta / a.mean : 0.0;
        char interval[64];
        std::snprintf(interval, sizeof(interval),
                      "[%+.1f%%, %+.1f%%]",
                      a.mean != 0.0 ? 100.0 * low / a.mean : 0.0,
                      a.mean != 0.0 ? 100.0 * high / a.mean : 0.0);
        std::printf("%-*s %11.4g %11.4g %+8.1f%% %21s  %s\n", w,
                    name.c_str(), a.mean, b.mean, relative, interval,
                    verdict.c_str());
    }
    if (compared == 0) {
        std::printf("No case has valid results with %s in both.\n",
                    metric.c_str());
    }
    if (base.size() < 3 || head.size() < 3) {
        std::printf("\nFor screening regressions, compare at least 3 "
                    "runs of each, "
                    "alternating base and head.\n");
    }
    return worse;
}

}  // namespace ernie::bench
