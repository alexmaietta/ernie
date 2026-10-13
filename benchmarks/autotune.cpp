// SPDX-License-Identifier: BSD-3-Clause
#include "autotune.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace ernie::bench {

namespace {

/// The metrics a probe reports.
const std::vector<std::string> kReported = {
    "duration_ms", "gpu_ms", "gpu_busy", "device_memory_used_mib",
    "device_memory_fraction"};

Probe RunProbe(Runner& runner, const Case& c,
               const AutotuneOptions& options, int value) {
    Case probe = c;
    probe.params[options.param] = std::to_string(value);
    const CaseResult result = runner.Run(probe, options.probe, false);
    Probe p{.value = value};
    std::vector<std::string> metrics = kReported;
    for (const Limit& limit : options.limits) {
        if (std::find(metrics.begin(), metrics.end(), limit.metric) ==
            metrics.end()) {
            metrics.push_back(limit.metric);
        }
    }
    for (const std::string& name : metrics) {
        const double mean = result.Mean(name);
        if (!std::isnan(mean)) {
            p.means[name] = mean;
        }
    }
    if (!result.Valid()) {
        p.reason = result.errors.empty() ? "too few samples"
                                         : result.errors.front();
        return p;
    }
    for (const Limit& limit : options.limits) {
        const auto mean = p.means.find(limit.metric);
        if (mean == p.means.end()) {
            p.reason = "no " + limit.metric + " to limit";
            return p;
        }
        if (limit.atMost ? mean->second > limit.value
                         : mean->second < limit.value) {
            char text[128];
            std::snprintf(text, sizeof(text), "%s %.4g, past %s",
                          limit.metric.c_str(), mean->second,
                          limit.Text().c_str());
            p.reason = text;
            return p;
        }
    }
    p.passed = true;
    return p;
}

}  // namespace

Limit Limit::Parse(const std::string& text) {
    for (const char* op : {"<=", ">="}) {
        const std::size_t at = text.find(op);
        if (at != std::string::npos && at > 0) {
            return {.metric = text.substr(0, at),
                    .atMost = op[0] == '<',
                    .value = std::stod(text.substr(at + 2))};
        }
    }
    throw std::invalid_argument(
        "A limit is METRIC<=VALUE or METRIC>=VALUE, not " + text);
}

std::string Limit::Text() const {
    char text[32];
    std::snprintf(text, sizeof(text), "%g", value);
    return metric + (atMost ? "<=" : ">=") + text;
}

AutotuneResult Autotune(Runner& runner, const Case& c,
                        const AutotuneOptions& given) {
    AutotuneResult result{.c = c, .options = given};
    AutotuneOptions& options = result.options;
    if (options.limits.empty()) {
        options.limits.push_back(
            Limit::Parse("device_memory_fraction<=0.9"));
    }
    const std::unique_ptr<Scenario> scenario =
        FindScenario(c.scenario).make();
    int value = options.start > 0
                    ? options.start
                    : scenario->GetInt(c, options.param);
    std::string limits;
    for (const Limit& limit : options.limits) {
        limits += (limits.empty() ? "" : ", ") + limit.Text();
    }
    std::printf("Autotuning %s of %s within %s\n",
                options.param.c_str(), c.Name().c_str(),
                limits.c_str());

    const auto probe = [&](int v) {
        Probe p = RunProbe(runner, c, options, v);
        std::printf("  %s=%-8d %s%s\n", options.param.c_str(), v,
                    p.passed ? "fits" : "fails: ", p.reason.c_str());
        std::fflush(stdout);
        result.probes.push_back(p);
        return p;
    };

    // Grow until a probe fails or the maximum passes.
    int passed = 0;
    int failed = 0;
    while (true) {
        const Probe p = probe(value);
        if (!p.passed) {
            failed = value;
            result.limitedBy = p.reason;
            break;
        }
        passed = value;
        if (value >= options.max) {
            result.limitedBy =
                "the maximum, " + std::to_string(options.max);
            break;
        }
        value = std::min(
            options.max,
            std::max(value + 1, static_cast<int>(std::ceil(
                                    value * options.factor))));
    }
    // Bisect between them.
    if (passed > 0 && failed > 0) {
        while (failed - passed >
               std::max(
                   1, static_cast<int>(passed * options.precision))) {
            const int mid = passed + (failed - passed) / 2;
            const Probe p = probe(mid);
            if (p.passed) {
                passed = mid;
            } else {
                failed = mid;
                result.limitedBy = p.reason;
            }
        }
    }
    result.best = passed;
    std::printf("  best %s=%d (limited by %s)\n\n",
                options.param.c_str(), passed,
                result.limitedBy.c_str());

    // The best value's full result, and the search.
    std::string bestFile;
    if (passed > 0) {
        Case best = c;
        best.params[options.param] = std::to_string(passed);
        runner.Run(best);
        bestFile = FileNameOf(best.Name()) + ".json";
    }
    nlohmann::json probes = nlohmann::json::array();
    for (const Probe& p : result.probes) {
        probes.push_back({{"value", p.value},
                          {"passed", p.passed},
                          {"reason", p.reason},
                          {"means", p.means}});
    }
    nlohmann::json limitTexts = nlohmann::json::array();
    for (const Limit& limit : options.limits) {
        limitTexts.push_back(limit.Text());
    }
    WriteAtomically(
        runner.RunDirectory() /
            ("autotune__" + FileNameOf(c.Name()) + ".json"),
        {{"schema", "ernie.benchmark.autotune.v1"},
         {"case", c.Name()},
         {"parameter", options.param},
         {"factor", options.factor},
         {"max", options.max},
         {"precision", options.precision},
         {"limits", limitTexts},
         {"probe_samples",
          {{"warmup", options.probe.warmupSamples.value_or(0)},
           {"measured", options.probe.measuredSamples.value_or(0)}}},
         {"probes", probes},
         {"best", passed},
         {"best_result", bestFile},
         {"limited_by", result.limitedBy}});
    return result;
}

}  // namespace ernie::bench
