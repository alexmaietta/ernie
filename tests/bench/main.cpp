// SPDX-License-Identifier: BSD-3-Clause
#include <cstdio>
#include <ctime>
#include <exception>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

#include "bench.h"
#include "ernie/device/adapter.h"
#include "ernie/util/profiler.h"

namespace {

constexpr const char* kUsage =
    R"(Ernie's benchmarks. Ernie's own flags, then Google Benchmark's:

  --device NAME    Use the adapter whose alias is NAME or whose name contains it
  --save[=LABEL]   Write the results as JSON to tests/bench/data/<program>-<time>[-LABEL].json
  --all            Run every benchmark. Without it or --benchmark_filter, only the core set runs


Compare two saved runs with Google Benchmark's tools/compare.py:

  compare.py benchmarks base.json new.json

)";

std::string Timestamp() {
    const std::time_t now = std::time(nullptr);
    char text[32];
    std::strftime(text, sizeof(text), "%Y%m%d-%H%M%S", std::localtime(&now));
    return text;
}

}

int main(int argc, char** argv) {
    std::vector<std::string> flags = {argv[0]};
    std::optional<std::string> save;
    bool all = false;
    bool filtered = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--device") {
            if (i + 1 == argc) {
                std::fputs("--device needs a name\n", stderr);
                return 2;
            }
            ernie::bench::SelectAdapter(argv[++i]);
        } else if (flag == "--all") {
            all = true;
        } else if (flag == "--save") {
            save = "";
        } else if (flag.starts_with("--save=")) {
            save = "-" + std::string(flag.substr(7));
        } else {
            if (flag == "--help" || flag == "-h") {
                std::fputs(kUsage, stdout);
                for (const std::string& pattern : ernie::bench::CorePatterns()) {
                    std::printf("Core: %s\n", pattern.c_str());
                }
                std::fputs("\n", stdout);
            }
            filtered = filtered || flag.starts_with("--benchmark_filter");
            flags.emplace_back(flag);
        }
    }
    std::filesystem::path out;
    if (save) {
        const std::string program = std::filesystem::path(argv[0]).stem().string();
        out = std::filesystem::path(ERNIE_BENCH_DATA_DIR) /
              (program + "-" + Timestamp() + *save + ".json");
        std::filesystem::create_directories(out.parent_path());
        flags.push_back("--benchmark_out=" + out.string());
        flags.push_back("--benchmark_out_format=json");
    }
    const std::vector<std::string>& core = ernie::bench::CorePatterns();
    if (!all && !filtered && !core.empty()) {
        std::string filter;
        for (const std::string& pattern : core) {
            filter += (filter.empty() ? "" : "|") + pattern;
        }
        flags.push_back("--benchmark_filter=" + filter);
    }
    std::vector<char*> pointers;
    for (std::string& flag : flags) {
        pointers.push_back(flag.data());
    }
    int count = int(pointers.size());
    benchmark::Initialize(&count, pointers.data());
    if (benchmark::ReportUnrecognizedArguments(count, pointers.data())) {
        return 2;
    }

    ernie::profiler::SetEnabled(true);
    ernie::profiler::SetHistory(2);
    ernie::profiler::BeginFrame();
    try {
        const ernie::device::Device& device = ernie::bench::BenchDevice();
        const ernie::device::AdapterInfo& adapter = device.Adapter();
        benchmark::AddCustomContext("ernie_adapter", adapter.name);
        benchmark::AddCustomContext(
            "ernie_traversal",
            device.GetTraversal() == ernie::device::Traversal::Hardware ? "hardware" : "software");
        benchmark::AddCustomContext(
            "ernie_sampling",
            device.GetSampling() == ernie::device::Sampling::Hardware ? "hardware" : "software");
        benchmark::AddCustomContext("ernie_build", ERNIE_BENCH_BUILD_TYPE);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s, so nothing ran\n", error.what());
        return 0;
    }
    benchmark::RunSpecifiedBenchmarks();
    ernie::bench::Release();
    benchmark::Shutdown();
    if (save) {
        std::printf("Saved %s\n", out.string().c_str());
    }
    return 0;
}
