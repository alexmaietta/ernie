// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Runs cases through one fixed lifecycle, after Isaac Sim's
// performance benchmarking runner:
//
// 0. a fresh renderer, unless reused;
// 1. setup: the scenario's `Build` and `Setup`, then a first, fenced
//    step (the renderer's first sync of the scene), each a region;
// 2. calibration, when the steps per sample are not given: doubling
//    them from 1 until a sample takes `minSampleMs`;
// 3. warm-up samples, untimed;
// 4. measured samples, each timed whole: its steps, each changing the
//    scene (`Scenario::Step`), submitted and their outputs mapped to
//    CUDA (as a learner reads them), waited for at each step
//    (fenced) or at the sample's end (pipelined);
// 5. validation: one more fenced step, its outputs checked by the
//    scenario (and drawn as contact sheets);
// 6. teardown.
//
// A phase that throws ends the case as invalid, its error kept; the
// result holds every sample, and each sample its metrics.

#include "bench.h"
#include "result.h"

#include "ernie/session/renderer.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ernie::bench {

/// What the command line overrides of every case's `Definition`.
struct SamplingOverrides {
    std::optional<int> warmupSamples;
    std::optional<int> measuredSamples;
    std::optional<int> stepsPerSample;
    std::optional<Completion> completion;
    std::optional<std::vector<std::string>> profiles;
};

struct RunOptions {
    /// Where runs go: a directory per run, "<UTC time>-<label>",
    /// holding a result per case, run.json and summary.csv.
    std::filesystem::path output;
    std::string label = "run";
    std::filesystem::path images;  ///< Contact sheets, when set.
    std::string device;
    bool validation = false;
    /// Renders every case with the one renderer, rather than a fresh
    /// one each, so later cases inherit what earlier ones allocated.
    bool reuseRenderer = false;
    SamplingOverrides sampling;
    std::vector<std::string> commandLine;
};

class Runner {
  public:
    explicit Runner(RunOptions options);
    ~Runner();

    /// Creates the renderer and the run's directory, and prints the
    /// device.
    void Start();

    /// Runs `c` through the lifecycle with `sampling` over the run's
    /// overrides, writes its result unless `save` is false, and keeps
    /// it for the summary.
    CaseResult Run(const Case& c,
                   const SamplingOverrides& sampling = {},
                   bool save = true);

    /// Prints a table of the results and writes run.json and
    /// summary.csv.
    void Finish() const;

    const std::filesystem::path& RunDirectory() const {
        return directory_;
    }

    Renderer& GetRenderer() {
        return *renderer_;
    }

  private:
    struct Execution;

    void Setup(Execution& e, CaseResult& result);
    void Measure(Execution& e, CaseResult& result);
    void Validate(Execution& e, CaseResult& result);
    void Teardown(Execution& e, CaseResult& result);
    double RunSteps(Execution& e, int steps);

    RunOptions options_;
    std::unique_ptr<Renderer> renderer_;
    std::filesystem::path directory_;
    std::string runId_;
    std::vector<CaseResult> results_;
};

/// Prints `results` as a table: per case, ms per step (mean, stdev
/// and p95), GPU ms and how busy that keeps the GPU, device memory,
/// setup time, and its rates.
void PrintTable(const std::vector<CaseResult>& results);

}  // namespace ernie::bench
