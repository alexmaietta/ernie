// SPDX-License-Identifier: BSD-3-Clause
// ernie_benchmarks: runs a suite, or a sweep given on the command
// line, through the benchmark lifecycle (runner.h), autotunes a
// parameter, or compares runs; see benchmarks/README.md.

#include "autotune.h"
#include "bench.h"
#include "compare.h"
#include "runner.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ernie::bench;

constexpr const char* kHelp = R"(Usage: ernie_benchmarks [options]

Cases:
  --suite NAME            A registered suite (default smoke, unless a
                          --scenario is given)
  --scenario A,B          Scenarios at their defaults, in place of a suite
  --set NAME=V1,V2        A parameter's values: replaces that axis in every
                          case whose scenario has it, one case per value
                          (repeatable)
  --filter REGEX          Only the cases whose names match
  --list                  The scenarios and their parameters, presets,
                          output sets and suites
  --cases                 The cases that would run, without running them

Sampling (over each scenario's own):
  --warmup N              Untimed samples after setup
  --samples N             Measured samples
  --steps N               Steps per sample; 0 calibrates to 50 ms samples
  --completion MODE       fenced (each step waited for: latency) or
                          pipelined (a sample's last: throughput)
  --profiles A,B          process, gpu, or none

Autotuning:
  --autotune NAME         The largest value of integer parameter NAME
                          within the limits, for each case
  --limit METRIC<=VALUE   A limit on a metric's mean, such as
                          duration_ms<=16.7 or views_per_s>=1e5
                          (repeatable; default
                          device_memory_fraction<=0.9)
  --autotune-max N        The largest value tried (default 1048576)
  --autotune-factor F     The growth between probes (default 2)

Comparing:
  --compare BASE HEAD     Runs' directories, each a comma-separated list
                          of runs; prints each case's change, with a 95%
                          interval and a verdict, and exits 1 if any is
                          conclusively worse
  --metric NAME           The metric compared (default duration_ms)

Results:
  --output DIR            Where runs go (default $ERNIE_BENCHMARK_OUTPUT_DIR,
                          else ~/.cache/ernie/benchmarks); each run is a
                          directory of a JSON result per case, run.json
                          and summary.csv
  --label NAME            The run directory's suffix (default the suite)
  --images DIR            A contact sheet of each case's vars
  --device NAME           GPU to prefer, by part of its name
  --validation            The Vulkan validation layer, where installed
  --reuse-renderer        One renderer for every case, rather than a fresh
                          one each (later cases then inherit what earlier
                          ones allocated)
)";

std::vector<std::string> Split(const std::string& list) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= list.size()) {
        const std::size_t comma = list.find(',', start);
        const std::size_t end =
            comma == std::string::npos ? list.size() : comma;
        if (end > start) {
            parts.push_back(list.substr(start, end - start));
        }
        start = end + 1;
    }
    return parts;
}

void List() {
    std::printf("Scenarios:\n");
    for (const ScenarioInfo& info : Scenarios()) {
        const std::unique_ptr<Scenario> scenario = info.make();
        const Definition d =
            scenario->Describe({.scenario = info.name});
        std::printf("  %-14s %s (%s, %s)\n", info.name.c_str(),
                    info.description.c_str(),
                    std::string(NameOf(d.scope)).c_str(),
                    std::string(NameOf(d.completion)).c_str());
        for (const Param& param : scenario->Params()) {
            std::printf("      %-12s %-10s %s\n", param.name.c_str(),
                        param.defaultValue.c_str(),
                        param.doc.c_str());
        }
    }
    std::printf("\nPresets:\n");
    for (const PresetInfo& info : Presets()) {
        std::printf("  %-14s %s\n", info.name.c_str(),
                    info.description.c_str());
    }
    std::printf("\nOutput sets:\n");
    for (const OutputsInfo& info : OutputSets()) {
        std::string vars;
        for (const std::string& var : info.vars) {
            vars += (vars.empty() ? "" : ", ") + var;
        }
        std::printf("  %-14s %s: %s\n", info.name.c_str(),
                    info.description.c_str(), vars.c_str());
    }
    std::printf("\nSuites:\n");
    for (const SuiteInfo& info : Suites()) {
        std::printf("  %-14s %3zu cases  %s\n", info.name.c_str(),
                    info.cases().size(), info.description.c_str());
    }
}

bool HasParam(const Case& c, const std::string& name) {
    const std::vector<Param> params = ParamsOf(c.scenario);
    return std::any_of(
        params.begin(), params.end(),
        [&](const Param& p) { return p.name == name; });
}

/// `cases` with axis `name` set to each of `values` where their
/// scenario has it.
std::vector<Case> Override(const std::vector<Case>& cases,
                           const std::string& name,
                           const std::vector<std::string>& values) {
    std::vector<Case> result;
    for (const Case& c : cases) {
        if (!HasParam(c, name)) {
            Append(result, {c});
            continue;
        }
        for (const std::string& value : values) {
            Case more = c;
            more.params[name] = value;
            Append(result, {more});
        }
    }
    return result;
}

std::filesystem::path DefaultOutput() {
    if (const char* configured =
            std::getenv("ERNIE_BENCHMARK_OUTPUT_DIR")) {
        return configured;
    }
    const char* cache = std::getenv("XDG_CACHE_HOME");
    const char* home = std::getenv("HOME");
    const std::filesystem::path root =
        cache != nullptr
            ? std::filesystem::path(cache)
            : std::filesystem::path(home != nullptr ? home : ".") /
                  ".cache";
    return root / "ernie" / "benchmarks";
}

int Main(int argc, char** argv) {
    RunOptions options{.output = DefaultOutput()};
    options.commandLine.assign(argv, argv + argc);
    std::string suite;
    std::string filter;
    std::vector<std::string> scenarios;
    std::vector<std::pair<std::string, std::vector<std::string>>>
        sets;
    std::optional<AutotuneOptions> autotune;
    std::vector<std::string> limits;
    std::optional<int> autotuneMax;
    std::optional<double> autotuneFactor;
    std::vector<std::filesystem::path> base;
    std::vector<std::filesystem::path> head;
    std::string metric = "duration_ms";
    bool list = false;
    bool listCases = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(arg + " needs a value");
            }
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            std::printf("%s", kHelp);
            return 0;
        } else if (arg == "--suite") {
            suite = value();
        } else if (arg == "--scenario") {
            scenarios = Split(value());
        } else if (arg == "--set") {
            const std::string set = value();
            const std::size_t equals = set.find('=');
            if (equals == std::string::npos) {
                throw std::invalid_argument(
                    "--set takes NAME=VALUES, not " + set);
            }
            sets.emplace_back(set.substr(0, equals),
                              Split(set.substr(equals + 1)));
        } else if (arg == "--filter") {
            filter = value();
        } else if (arg == "--list") {
            list = true;
        } else if (arg == "--cases") {
            listCases = true;
        } else if (arg == "--warmup") {
            options.sampling.warmupSamples = std::stoi(value());
        } else if (arg == "--samples") {
            options.sampling.measuredSamples = std::stoi(value());
        } else if (arg == "--steps") {
            options.sampling.stepsPerSample = std::stoi(value());
        } else if (arg == "--completion") {
            options.sampling.completion = CompletionNamed(value());
        } else if (arg == "--profiles") {
            std::vector<std::string> profiles = Split(value());
            if (profiles == std::vector<std::string>{"none"}) {
                profiles.clear();
            }
            options.sampling.profiles = profiles;
        } else if (arg == "--autotune") {
            autotune = AutotuneOptions{.param = value()};
        } else if (arg == "--limit") {
            limits.push_back(value());
        } else if (arg == "--autotune-max") {
            autotuneMax = std::stoi(value());
        } else if (arg == "--autotune-factor") {
            autotuneFactor = std::stod(value());
        } else if (arg == "--compare") {
            for (const std::string& path : Split(value())) {
                base.emplace_back(path);
            }
            for (const std::string& path : Split(value())) {
                head.emplace_back(path);
            }
        } else if (arg == "--metric") {
            metric = value();
        } else if (arg == "--output") {
            options.output = value();
        } else if (arg == "--label") {
            options.label = value();
        } else if (arg == "--images") {
            options.images = value();
        } else if (arg == "--device") {
            options.device = value();
        } else if (arg == "--validation") {
            options.validation = true;
        } else if (arg == "--reuse-renderer") {
            options.reuseRenderer = true;
        } else {
            throw std::invalid_argument("Unknown option " + arg +
                                        "; see --help");
        }
    }
    if (list) {
        List();
        return 0;
    }
    if (!base.empty()) {
        return Compare(base, head, metric) > 0 ? 1 : 0;
    }

    std::vector<Case> cases;
    if (!scenarios.empty()) {
        if (!suite.empty()) {
            throw std::invalid_argument(
                "Give --suite or --scenario, not both");
        }
        for (const std::string& scenario : scenarios) {
            FindScenario(scenario);
            Append(cases, {Case{.scenario = scenario}});
        }
    } else {
        suite = suite.empty() ? "smoke" : suite;
        cases = FindSuite(suite).cases();
    }
    for (const auto& [name, values] : sets) {
        if (std::none_of(
                cases.begin(), cases.end(),
                [&](const Case& c) { return HasParam(c, name); })) {
            throw std::invalid_argument(
                "No case's scenario has a parameter " + name);
        }
        cases = Override(cases, name, values);
    }
    if (!filter.empty()) {
        const std::regex pattern(filter);
        std::erase_if(cases, [&](const Case& c) {
            return !std::regex_search(c.Name(), pattern);
        });
    }
    if (listCases) {
        for (const Case& c : cases) {
            std::printf("%s\n", c.Name().c_str());
        }
        return 0;
    }
    if (cases.empty()) {
        throw std::invalid_argument("No cases to run");
    }
    if (options.label == "run") {
        options.label = autotune         ? "autotune"
                        : !suite.empty() ? suite
                                         : "scenarios";
    }

    Runner runner(options);
    runner.Start();
    if (autotune) {
        for (const std::string& text : limits) {
            autotune->limits.push_back(Limit::Parse(text));
        }
        autotune->max = autotuneMax.value_or(autotune->max);
        autotune->factor = autotuneFactor.value_or(autotune->factor);
        for (const Case& c : cases) {
            if (!HasParam(c, autotune->param)) {
                throw std::invalid_argument("Scenario " + c.scenario +
                                            " has no parameter " +
                                            autotune->param);
            }
            Autotune(runner, c, *autotune);
        }
    } else {
        for (const Case& c : cases) {
            runner.Run(c);
        }
    }
    runner.Finish();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return Main(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ernie_benchmarks: %s\n", error.what());
        return 2;
    }
}
