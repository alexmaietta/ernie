// SPDX-License-Identifier: BSD-3-Clause
#include "result.h"

#include "statistics.h"

#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ernie::bench {

namespace {

nlohmann::json SummaryJson(const std::vector<double>& values) {
    if (values.empty()) {
        return nlohmann::json::object();
    }
    const Summary s = Summarize(values);
    return {
        {"count", s.count}, {"mean", s.mean}, {"median", s.median},
        {"stdev", s.stdev}, {"min", s.min},   {"max", s.max},
        {"p50", s.p50},     {"p95", s.p95},   {"p99", s.p99}};
}

std::string NowUtc() {
    const std::time_t now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm utc{};
    gmtime_r(&now, &utc);
    char text[32];
    std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return text;
}

}  // namespace

double CaseResult::Mean(const std::string& name) const {
    double sum = 0.0;
    std::size_t count = 0;
    for (const Sample& sample : samples) {
        if (name == "duration_ms") {
            sum += sample.durationMs;
        } else if (const auto it = sample.metrics.find(name);
                   it != sample.metrics.end()) {
            sum += it->second;
        } else {
            continue;
        }
        ++count;
    }
    return count == 0 ? std::numeric_limits<double>::quiet_NaN()
                      : sum / static_cast<double>(count);
}

nlohmann::json Document(const CaseResult& result) {
    // Each metric's samples and summary.
    std::map<std::string, std::vector<std::pair<int, double>>> values;
    for (const Sample& sample : result.samples) {
        values["duration_ms"].emplace_back(sample.index,
                                           sample.durationMs);
        values["total_duration_ms"].emplace_back(sample.index,
                                                 sample.totalMs);
        for (const auto& [name, value] : sample.metrics) {
            values[name].emplace_back(sample.index, value);
        }
    }
    nlohmann::json metrics = nlohmann::json::object();
    for (const auto& [name, samples] : values) {
        nlohmann::json list = nlohmann::json::array();
        std::vector<double> plain;
        for (const auto& [index, value] : samples) {
            list.push_back({{"sample", index}, {"value", value}});
            plain.push_back(value);
        }
        const auto unit = result.units.find(name);
        metrics[name] = {
            {"unit", unit != result.units.end() ? unit->second : ""},
            {"samples", list},
            {"statistics", SummaryJson(plain)}};
    }

    // Each region's observations, and the summary of its measured
    // samples' totals and means.
    std::map<std::string, std::vector<const RegionObservation*>>
        grouped;
    for (const RegionObservation& region : result.regions) {
        grouped[region.name].push_back(&region);
    }
    nlohmann::json regions = nlohmann::json::object();
    for (const auto& [name, observations] : grouped) {
        nlohmann::json list = nlohmann::json::array();
        std::vector<double> totals;
        std::vector<double> means;
        for (const RegionObservation* o : observations) {
            const double mean =
                o->totalMs / static_cast<double>(o->count);
            list.push_back(
                {{"phase", o->phase},
                 {"sample", o->sample < 0
                                ? nlohmann::json(nullptr)
                                : nlohmann::json(o->sample)},
                 {"count", o->count},
                 {"total_ms", o->totalMs},
                 {"mean_ms", mean},
                 {"min_ms", o->minMs},
                 {"max_ms", o->maxMs}});
            if (o->sample >= 0) {
                totals.push_back(o->totalMs);
                means.push_back(mean);
            }
        }
        regions[name] = {{"unit", "ms"},
                         {"samples", list},
                         {"statistics",
                          {{"total_ms", SummaryJson(totals)},
                           {"mean_ms", SummaryJson(means)}}}};
    }

    std::vector<int> steps;
    for (const Sample& sample : result.samples) {
        if (std::find(steps.begin(), steps.end(), sample.steps) ==
            steps.end()) {
            steps.push_back(sample.steps);
        }
    }
    nlohmann::json stats = nlohmann::json::object();
    for (const auto& [name, value] : result.stats) {
        stats[name] = value;
    }
    nlohmann::json lifecycle = nlohmann::json::object();
    for (const auto& [name, ms] : result.lifecycleMs) {
        lifecycle[name] = ms;
    }
    lifecycle["validation"] = {
        {"performed", result.validated},
        {"passed",
         result.validated && result.validationFailure.empty()}};
    if (!result.validationFailure.empty()) {
        lifecycle["validation"]["reason"] = result.validationFailure;
    }
    const Definition& d = result.definition;
    return {
        {"schema", kResultSchema},
        {"generated_at", NowUtc()},
        {"run_id", result.runId},
        {"benchmark",
         {{"id", "ernie." + result.c.scenario},
          {"case", result.c.Name()},
          {"scenario", result.c.scenario},
          {"scope", NameOf(d.scope)},
          {"completion", NameOf(d.completion)},
          {"duration", d.duration},
          {"profiles", d.profiles},
          {"parameters", result.parameters}}},
        {"sampling",
         {{"warmup_samples", d.warmupSamples},
          {"measured_samples", d.measuredSamples},
          {"completed_samples", result.samples.size()},
          {"steps_per_sample", steps},
          {"calibrated", d.stepsPerSample == 0},
          {"min_sample_ms", d.minSampleMs}}},
        {"status",
         {{"valid", result.Valid()}, {"errors", result.errors}}},
        {"lifecycle", lifecycle},
        {"scene", stats},
        {"metrics", metrics},
        {"regions", regions},
        {"environment", result.environment},
    };
}

void WriteAtomically(const std::filesystem::path& path,
                     const nlohmann::json& document) {
    std::filesystem::create_directories(path.parent_path());
    const std::filesystem::path temporary =
        path.parent_path() /
        ("." + path.filename().string() + ".tmp");
    {
        std::ofstream file(temporary);
        if (!file) {
            throw std::runtime_error("Cannot write " +
                                     temporary.string());
        }
        file << document.dump(2) << "\n";
        if (!file.flush()) {
            throw std::runtime_error("Cannot write " +
                                     temporary.string());
        }
    }
    std::filesystem::rename(temporary, path);
}

std::string FileNameOf(const std::string& name) {
    std::string file;
    for (const char c : name) {
        file += c == '/'   ? std::string("__")
                : c == '=' ? std::string("-")
                           : std::string(1, c);
    }
    return file;
}

}  // namespace ernie::bench
