// SPDX-License-Identifier: BSD-3-Clause
#include "runner.h"

#include "images.h"
#include "instrument.h"
#include "metrics.h"
#include "statistics.h"

#include "ernie/device/compute_interop.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>

namespace ernie::bench {

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kMib = 1024.0 * 1024.0;

/// Metrics the runner records itself, and their units.
const std::map<std::string, std::string> kRunnerUnits = {
    {"duration_ms", "ms"},
    {"total_duration_ms", "ms"},
    {"steps_per_s", "steps/s"},
    {"process_cpu_utilization", "ratio"},
    {"process_rss_mib", "MiB"},
    {"device_memory_used_mib", "MiB"},
    {"device_memory_budget_mib", "MiB"},
    {"device_memory_fraction", "ratio"},
    {"renderer_memory_mib", "MiB"},
    {"triangles", "count"},
    {"gpu_ms", "ms"},
    {"gpu_busy", "ratio"},
    {"gpu_frames", "count"},
};

double MsSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() -
                                                     start)
        .count();
}

std::string ErrorOf(const char* phase, const std::exception& error) {
    return std::string(phase) + ": " + error.what();
}

/// A GPU zone's name as part of a metric name: lower case, "/"s kept,
/// runs of other characters as "_".
std::string MetricName(const std::string& zone) {
    std::string name;
    for (const char c : zone) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            name += static_cast<char>(
                std::tolower(static_cast<unsigned char>(c)));
        } else if (c == '/') {
            while (!name.empty() && name.back() == '_') {
                name.pop_back();
            }
            name += '/';
        } else if (!name.empty() && name.back() != '_' &&
                   name.back() != '/') {
            name += '_';
        }
    }
    while (!name.empty() && name.back() == '_') {
        name.pop_back();
    }
    return name;
}

void Retain(CaseResult& result, const Observations& observations,
            const std::string& phase, int sample) {
    for (const auto& [name, values] : observations.regions) {
        RegionObservation region{.name = name,
                                 .phase = phase,
                                 .sample = sample,
                                 .count = values.size()};
        region.minMs =
            *std::min_element(values.begin(), values.end());
        region.maxMs =
            *std::max_element(values.begin(), values.end());
        for (const double ms : values) {
            region.totalMs += ms;
        }
        result.regions.push_back(region);
    }
}

bool Has(const std::vector<std::string>& list,
         const std::string& item) {
    return std::find(list.begin(), list.end(), item) != list.end();
}

/// A run's id: its start time and random hex.
std::string NewRunId() {
    std::random_device device;
    char id[9];
    std::snprintf(id, sizeof(id), "%08x", device());
    return id;
}

std::string UtcStamp() {
    const std::time_t now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm utc{};
    gmtime_r(&now, &utc);
    char text[32];
    std::strftime(text, sizeof(text), "%Y%m%dT%H%M%SZ", &utc);
    return text;
}

/// `value` with a k, M or G suffix, three significant figures.
std::string Short(double value) {
    const char* suffix = "";
    for (const char* next : {"k", "M", "G"}) {
        if (std::abs(value) < 1000.0) {
            break;
        }
        value /= 1000.0;
        suffix = next;
    }
    char text[32];
    std::snprintf(text, sizeof(text),
                  value < 10.0    ? "%.2f%s"
                  : value < 100.0 ? "%.1f%s"
                                  : "%.0f%s",
                  value, suffix);
    return text;
}

std::string CsvField(const std::string& value) {
    if (value.find_first_of(",\"\n") == std::string::npos) {
        return value;
    }
    std::string quoted = "\"";
    for (const char c : value) {
        quoted += c == '"' ? std::string("\"\"") : std::string(1, c);
    }
    return quoted + "\"";
}

void AddName(std::vector<std::string>& names,
             const std::string& name) {
    if (!Has(names, name)) {
        names.push_back(name);
    }
}

/// A setup region's time in ms, or NaN.
double SetupMs(const CaseResult& result, const std::string& name) {
    for (const RegionObservation& region : result.regions) {
        if (region.phase == "setup" && region.name == name) {
            return region.totalMs;
        }
    }
    return std::nan("");
}

std::vector<double> Values(const CaseResult& result,
                           const std::string& name) {
    std::vector<double> values;
    for (const Sample& sample : result.samples) {
        if (name == "duration_ms") {
            values.push_back(sample.durationMs);
        } else if (const auto it = sample.metrics.find(name);
                   it != sample.metrics.end()) {
            values.push_back(it->second);
        }
    }
    return values;
}

}  // namespace

/// One case's scene and state while the lifecycle runs.
struct Runner::Execution {
    const Case& c;
    std::unique_ptr<Scenario> scenario;
    Scene scene;
    std::vector<Path> products;
    Path settings;
    Definition definition;
    int step = 0;
    int stepsPerSample = 1;
    bool map = false;
    /// Each measured sample's first and last profiler frames.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> frames;
};

Runner::Runner(RunOptions options) : options_(std::move(options)) {}

Runner::~Runner() = default;

void Runner::Start() {
    renderer_ = std::make_unique<Renderer>(
        RendererOptions{.device = options_.device,
                        .validation = options_.validation});
    runId_ = NewRunId();
    directory_ =
        options_.output / (UtcStamp() + "-" + options_.label);
    std::filesystem::create_directories(directory_);
    const bool cuda = renderer_->Delegate().Compute() != nullptr;
    const std::optional<DeviceMemory> memory =
        DeviceMemoryNow(renderer_->Context());
    std::printf("%s (%s), %s, CUDA interop %s, device memory %s\n",
                renderer_->DeviceName().c_str(),
                renderer_->Context().DriverDescription().c_str(),
                renderer_->Context().VulkanVersion().c_str(),
                cuda ? "on" : "off",
                memory ? (Short(memory->heapMib) + " MiB").c_str()
                       : "unknown (no VK_EXT_memory_budget)");
    if (!cuda) {
        std::printf("Without CUDA interop outputs are not mapped.\n");
    }
    std::printf("Results: %s\n\n", directory_.string().c_str());
}

double Runner::RunSteps(Execution& e, int steps) {
    const Clock::time_point start = Clock::now();
    for (int k = 0; k < steps; ++k) {
        NextFrame();
        e.scenario->Step(*renderer_, e.scene, e.c, e.step++);
        StepOperation operation =
            renderer_->StepAsync(e.scene, e.products, e.settings);
        if (e.map) {
            // On CUDA, as a learner reads them: the stream waits on
            // the GPU, the host does not.
            for (const auto& [path, product] : operation.Outputs()) {
                for (const auto& [name, var] : product.vars) {
                    RenderVarOutput output = var;
                    const Tensor tensor =
                        output.Map(MapDevice::Gpu).tensor;
                    KeepAlive(tensor.Data());
                }
            }
        }
        if (e.definition.completion == Completion::Fenced ||
            k + 1 == steps) {
            operation.Wait();
        }
    }
    return MsSince(start);
}

void Runner::Setup(Execution& e, CaseResult& result) {
    const Clock::time_point start = Clock::now();
    Observations observations;
    {
        ActiveObservations active(observations, false);
        {
            bench::Measure region("build");
            e.scenario->Build(e.scene, e.c);
        }
        {
            bench::Measure region("setup");
            e.scenario->Setup(*renderer_, e.scene, e.c);
        }
        e.products = e.scenario->Products(e.c);
        if (e.scene.Has(kPaths.settings)) {
            e.settings = kPaths.settings;
        }
        {
            // The renderer's first sync of the scene, waited for.
            bench::Measure region("first_step");
            const Completion completion = e.definition.completion;
            e.definition.completion = Completion::Fenced;
            RunSteps(e, 1);
            e.definition.completion = completion;
        }
    }
    Retain(result, observations, "setup", -1);
    result.lifecycleMs["setup_ms"] = MsSince(start);
}

void Runner::Measure(Execution& e, CaseResult& result) {
    const Definition& d = e.definition;
    const bool gpu = Has(d.profiles, "gpu");
    const bool process = Has(d.profiles, "process");

    // Calibration: steps per sample, doubling until a sample is long
    // enough to time.
    const Clock::time_point calibration = Clock::now();
    e.stepsPerSample = std::max(d.stepsPerSample, 1);
    if (d.stepsPerSample == 0) {
        while (RunSteps(e, e.stepsPerSample) < d.minSampleMs &&
               e.stepsPerSample < d.maxSteps) {
            e.stepsPerSample =
                std::min(e.stepsPerSample * 2, d.maxSteps);
        }
        result.lifecycleMs["calibration_ms"] = MsSince(calibration);
    }
    const Clock::time_point warmup = Clock::now();
    for (int s = 0; s < d.warmupSamples; ++s) {
        RunSteps(e, e.stepsPerSample);
    }
    result.lifecycleMs["warmup_ms"] = MsSince(warmup);

    if (gpu) {
        StartGpuTimes(
            static_cast<std::size_t>(e.stepsPerSample) *
                static_cast<std::size_t>(d.measuredSamples) +
            64);
    }
    const auto units = e.scenario->Units(e.c);
    result.rates = units;
    for (const Unit& unit : units) {
        result.units[unit.name + "_per_s"] = unit.name + "/s";
    }
    for (int s = 0; s < d.measuredSamples; ++s) {
        const ProcessSnapshot before =
            process ? ProcessNow() : ProcessSnapshot{};
        const std::uint64_t first = CurrentFrame() + 1;
        Observations observations;
        double ms = 0.0;
        {
            ActiveObservations active(observations, true);
            ms = RunSteps(e, e.stepsPerSample);
        }
        e.frames.emplace_back(first, CurrentFrame());
        const double seconds = ms / 1000.0;
        Sample sample{.index = s,
                      .steps = e.stepsPerSample,
                      .durationMs = ms / e.stepsPerSample,
                      .totalMs = ms};
        sample.metrics["steps_per_s"] = e.stepsPerSample / seconds;
        for (const Unit& unit : units) {
            sample.metrics[unit.name + "_per_s"] =
                unit.count / unit.scale * e.stepsPerSample / seconds;
        }
        if (process) {
            const ProcessSnapshot after = ProcessNow();
            sample.metrics["process_cpu_utilization"] =
                (after.cpuSeconds - before.cpuSeconds) / seconds;
            sample.metrics["process_rss_mib"] =
                std::max(before.rssMib, after.rssMib);
        }
        if (gpu) {
            if (const std::optional<DeviceMemory> memory =
                    DeviceMemoryNow(renderer_->Context())) {
                sample.metrics["device_memory_used_mib"] =
                    memory->usedMib;
                sample.metrics["device_memory_budget_mib"] =
                    memory->budgetMib;
                sample.metrics["device_memory_fraction"] =
                    memory->budgetMib > 0.0
                        ? memory->usedMib / memory->budgetMib
                        : 0.0;
            }
        }
        const RenderStats stats = renderer_->Stats();
        sample.metrics["renderer_memory_mib"] =
            static_cast<double>(stats.gpuBytes) / kMib;
        sample.metrics["triangles"] =
            static_cast<double>(stats.triangles);
        for (const auto& [name, recorded] : observations.records) {
            if (kRunnerUnits.contains(name)) {
                throw std::invalid_argument("Metric " + name +
                                            " is the runner's own");
            }
            const auto& [values, unit] = recorded;
            if (const auto kept = result.units.find(name);
                kept != result.units.end() && kept->second != unit) {
                throw std::invalid_argument("Metric " + name +
                                            " changed units");
            }
            result.units[name] = unit;
            double sum = 0.0;
            for (const double value : values) {
                sum += value;
            }
            sample.metrics[name] =
                sum / static_cast<double>(values.size());
        }
        Retain(result, observations, "measurement", s);
        result.samples.push_back(std::move(sample));
    }
}

void Runner::Validate(Execution& e, CaseResult& result) {
    const Clock::time_point start = Clock::now();
    NextFrame();
    e.scenario->Step(*renderer_, e.scene, e.c, e.step++);
    const RenderProductSetOutputs outputs =
        renderer_->Step(e.scene, e.products, e.settings);
    result.validated = true;
    result.validationFailure = e.scenario->Validate(e.c, outputs);
    if (!result.validationFailure.empty()) {
        result.errors.push_back("validation: " +
                                result.validationFailure);
    }
    if (!options_.images.empty()) {
        WriteSheets(options_.images, FileNameOf(e.c.Name()), outputs);
    }
    result.lifecycleMs["validation_ms"] = MsSince(start);
}

void Runner::Teardown(Execution& e, CaseResult& result) {
    const Clock::time_point start = Clock::now();
    e.scenario->Teardown(*renderer_, e.scene, e.c);
    result.lifecycleMs["teardown_ms"] = MsSince(start);
}

CaseResult Runner::Run(const Case& c,
                       const SamplingOverrides& sampling, bool save) {
    CaseResult result{.runId = runId_, .c = c};
    result.units = kRunnerUnits;
    // A renderer of its own, so no case inherits another's targets,
    // pools or caches; the old one goes first, with its memory.
    if (!options_.reuseRenderer || renderer_ == nullptr) {
        const Clock::time_point start = Clock::now();
        renderer_.reset();
        renderer_ = std::make_unique<Renderer>(
            RendererOptions{.device = options_.device,
                            .validation = options_.validation});
        result.lifecycleMs["renderer_ms"] = MsSince(start);
    }
    result.environment =
        Environment(*renderer_, options_.commandLine);
    Execution e{.c = c};
    try {
        e.scenario = FindScenario(c.scenario).make();
        Definition& d = e.definition = e.scenario->Describe(c);
        for (const SamplingOverrides* o :
             {static_cast<const SamplingOverrides*>(
                  &options_.sampling),
              &sampling}) {
            d.warmupSamples =
                o->warmupSamples.value_or(d.warmupSamples);
            d.measuredSamples =
                o->measuredSamples.value_or(d.measuredSamples);
            d.stepsPerSample =
                o->stepsPerSample.value_or(d.stepsPerSample);
            d.completion = o->completion.value_or(d.completion);
            d.profiles = o->profiles.value_or(d.profiles);
        }
        result.definition = d;
        for (const Param& param : e.scenario->Params()) {
            result.parameters[param.name] =
                e.scenario->Get(c, param.name);
        }
        for (const auto& [name, value] : c.params) {
            if (!result.parameters.contains(name)) {
                throw std::invalid_argument("Scenario " + c.scenario +
                                            " has no parameter " +
                                            name);
            }
        }
        e.map = renderer_->Delegate().Compute() != nullptr;
    } catch (const std::exception& error) {
        result.errors.push_back(ErrorOf("construct", error));
    }
    if (result.errors.empty()) {
        bool setUp = false;
        try {
            Setup(e, result);
            setUp = true;
            Measure(e, result);
        } catch (const std::exception& error) {
            result.errors.push_back(
                ErrorOf(setUp ? "measurement" : "setup", error));
        }
        if (result.errors.empty()) {
            try {
                Validate(e, result);
            } catch (const std::exception& error) {
                result.errors.push_back(ErrorOf("validation", error));
            }
        }
        // The samples' GPU times, now that their frames' zones have
        // arrived (the validation step collects the last).
        for (std::size_t s = 0;
             s < e.frames.size() && s < result.samples.size(); ++s) {
            const GpuTimes times =
                GpuTimesOf(e.frames[s].first, e.frames[s].second);
            if (times.frames == 0) {
                continue;
            }
            Sample& sample = result.samples[s];
            sample.metrics["gpu_ms"] = times.frameMs;
            sample.metrics["gpu_busy"] =
                times.frameMs / sample.durationMs;
            sample.metrics["gpu_frames"] =
                static_cast<double>(times.frames);
            for (const auto& [pass, ms] : times.passes) {
                const std::string name =
                    "gpu/" + MetricName(pass) + "_ms";
                sample.metrics[name] = ms;
                result.units[name] = "ms";
            }
        }
        StopGpuTimes();
        if (e.scenario != nullptr) {
            try {
                Teardown(e, result);
            } catch (const std::exception& error) {
                result.errors.push_back(ErrorOf("teardown", error));
            }
            result.stats = e.scenario->Stats();
        }
    }

    // A line per case as it finishes.
    if (result.Valid()) {
        const Summary s = Summarize(Values(result, "duration_ms"));
        std::printf("%-60s %8.3f ms/step  sd %6.3f  p95 %8.3f  (%zu "
                    "x %d steps)\n",
                    c.Name().c_str(), s.mean, s.stdev, s.p95, s.count,
                    result.samples.front().steps);
    } else {
        std::printf("%-60s invalid: %s\n", c.Name().c_str(),
                    result.errors.empty()
                        ? "too few samples"
                        : result.errors.front().c_str());
    }
    std::fflush(stdout);
    if (save) {
        WriteAtomically(directory_ / (FileNameOf(c.Name()) + ".json"),
                        Document(result));
        results_.push_back(result);
    }
    return result;
}

void PrintTable(const std::vector<CaseResult>& results) {
    if (results.empty()) {
        return;
    }
    std::size_t width = 4;
    for (const CaseResult& r : results) {
        width = std::max(width, r.c.Name().size());
    }
    const int w = static_cast<int>(width);
    std::printf("\n%-*s %9s %7s %9s %8s %8s %8s %7s  %s\n", w, "case",
                "ms/step", "sd", "p95", "GPU ms", "busy", "MiB",
                "setup s", "rates");
    for (const CaseResult& r : results) {
        if (!r.Valid()) {
            std::printf("%-*s  invalid: %s\n", w, r.c.Name().c_str(),
                        r.errors.empty() ? "too few samples"
                                         : r.errors.front().c_str());
            continue;
        }
        const Summary step = Summarize(Values(r, "duration_ms"));
        std::string rates;
        for (const Unit& unit : r.rates) {
            rates += Short(r.Mean(unit.name + "_per_s")) + " " +
                     unit.name + "/s  ";
        }
        const double gpuMs = r.Mean("gpu_ms");
        const double busy = r.Mean("gpu_busy");
        const double mib = r.Mean("device_memory_used_mib");
        const auto setup = r.lifecycleMs.find("setup_ms");
        std::printf(
            "%-*s %9.3f %7.3f %9.3f %8s %7s%% %8s %7.1f  %s\n", w,
            r.c.Name().c_str(), step.mean, step.stdev, step.p95,
            std::isnan(gpuMs) ? "-" : Short(gpuMs).c_str(),
            std::isnan(busy) ? "-" : Short(busy * 100.0).c_str(),
            std::isnan(mib) ? "-" : Short(mib).c_str(),
            setup == r.lifecycleMs.end() ? 0.0
                                         : setup->second / 1000.0,
            rates.c_str());
    }
}

void Runner::Finish() const {
    PrintTable(results_);
    if (results_.empty()) {
        return;
    }
    // summary.csv: a row per case, its parameters, the means (and the
    // step time's spread) of its metrics, its setup and scene.
    std::vector<std::string> params;
    std::vector<std::string> metrics;
    std::vector<std::string> stats;
    for (const CaseResult& r : results_) {
        for (const auto& [name, value] : r.parameters) {
            AddName(params, name);
        }
        for (const Sample& sample : r.samples) {
            for (const auto& [name, value] : sample.metrics) {
                AddName(metrics, name);
            }
        }
        for (const auto& [name, value] : r.stats) {
            AddName(stats, name);
        }
    }
    std::sort(metrics.begin(), metrics.end());
    std::ofstream csv(directory_ / "summary.csv");
    csv << "case,scenario";
    for (const std::string& name : params) {
        csv << "," << name;
    }
    csv << ",scope,completion,valid,samples,steps_per_sample,"
           "duration_ms,"
           "duration_ms_stdev,duration_ms_p50,duration_ms_p95,"
           "duration_ms_p99";
    for (const std::string& name : metrics) {
        csv << "," << name;
    }
    csv << ",setup_ms,setup_build_ms,setup_setup_ms,setup_first_step_"
           "ms";
    for (const std::string& name : stats) {
        csv << "," << name;
    }
    csv << ",error\n";
    for (const CaseResult& r : results_) {
        csv << CsvField(r.c.Name()) << "," << r.c.scenario;
        for (const std::string& name : params) {
            const auto it = r.parameters.find(name);
            csv << ","
                << (it == r.parameters.end() ? ""
                                             : CsvField(it->second));
        }
        const std::vector<double> steps = Values(r, "duration_ms");
        const Summary s = Summarize(steps);
        csv << "," << NameOf(r.definition.scope) << ","
            << NameOf(r.definition.completion) << ","
            << (r.Valid() ? 1 : 0) << "," << r.samples.size() << ","
            << (r.samples.empty() ? 0 : r.samples.front().steps);
        if (steps.empty()) {
            csv << ",,,,,";
        } else {
            csv << "," << s.mean << "," << s.stdev << "," << s.p50
                << "," << s.p95 << "," << s.p99;
        }
        for (const std::string& name : metrics) {
            const double mean = r.Mean(name);
            csv << ",";
            if (!std::isnan(mean)) {
                csv << mean;
            }
        }
        const auto setup = r.lifecycleMs.find("setup_ms");
        csv << ",";
        if (setup != r.lifecycleMs.end()) {
            csv << setup->second;
        }
        for (const char* region : {"build", "setup", "first_step"}) {
            const double ms = SetupMs(r, region);
            csv << ",";
            if (!std::isnan(ms)) {
                csv << ms;
            }
        }
        for (const std::string& name : stats) {
            const auto stat = std::find_if(
                r.stats.begin(), r.stats.end(),
                [&](const auto& kept) { return kept.first == name; });
            csv << ",";
            if (stat != r.stats.end()) {
                csv << stat->second;
            }
        }
        csv << ","
            << CsvField(r.errors.empty() ? "" : r.errors.front())
            << "\n";
    }

    // run.json: the run, its environment and its cases' files.
    nlohmann::json cases = nlohmann::json::array();
    for (const CaseResult& r : results_) {
        cases.push_back({{"case", r.c.Name()},
                         {"file", FileNameOf(r.c.Name()) + ".json"},
                         {"valid", r.Valid()}});
    }
    WriteAtomically(directory_ / "run.json",
                    {{"schema", "ernie.benchmark.run.v1"},
                     {"run_id", runId_},
                     {"label", options_.label},
                     {"environment", results_.front().environment},
                     {"cases", cases}});
    std::printf("\nWrote %s\n", directory_.string().c_str());
}

}  // namespace ernie::bench
