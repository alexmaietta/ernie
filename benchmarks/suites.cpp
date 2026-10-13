// SPDX-License-Identifier: BSD-3-Clause
// The registered suites. A suite is any list of cases; add one with
// ERNIE_BENCH_SUITE, usually from sweeps.

#include "bench.h"

namespace ernie::bench {

namespace {

/// The batch scenarios, simplest first.
const std::vector<std::string> kBatch = {
    "cartpole", "shapes", "tabletop", "house", "dense"};

}  // namespace

ERNIE_BENCH_SUITE("smoke",
                  "A quick check that every batch scenario runs", [] {
                      return Sweep{kBatch,
                                   {{"preset", {"minimal", "lit"}},
                                    {"resolution", Values({64})},
                                    {"worlds", Values({64})}}}
                          .Cases();
                  });

ERNIE_BENCH_SUITE(
    "scaling",
    "Resolution against worlds, minimal and lit shading, colour", [] {
        return Sweep{kBatch,
                     {{"preset", {"minimal", "lit"}},
                      {"resolution", Values({64, 128, 256})},
                      {"worlds", Values({256, 1024, 4096})}}}
            .Cases();
    });

ERNIE_BENCH_SUITE("outputs",
                  "Each output set at 128 px and 1024 worlds", [] {
                      return Sweep{
                          kBatch,
                          {{"preset", {"minimal", "lit"}},
                           {"outputs",
                            {"depth", "rgb", "rgbd", "gbuffer"}}}}
                          .Cases();
                  });

ERNIE_BENCH_SUITE(
    "presets", "Each preset at 128 px and 1024 worlds", [] {
        return Sweep{kBatch,
                     {{"preset",
                       {"minimal", "lit-noshadow", "lit", "native"}}}}
            .Cases();
    });

ERNIE_BENCH_SUITE("views", "Views per world, 1 to 8", [] {
    return Sweep{{"shapes", "tabletop"},
                 {{"preset", {"minimal", "lit"}},
                  {"views", Values({1, 2, 4, 8})}}}
        .Cases();
});

ERNIE_BENCH_SUITE(
    "envs",
    "Cartpole from 16 to 4096 worlds at 128 to 512 px, one view each",
    [] {
        return Sweep{{"cartpole"},
                     {{"preset", {"minimal", "lit"}},
                      {"resolution", Values({128, 256, 512})},
                      {"worlds", Values({16, 64, 256, 1024, 4096})}}}
            .Cases();
    });

ERNIE_BENCH_SUITE(
    "ladder",
    "Scenes of rising complexity at 1024 worlds, 64 to 256 px: "
    "shapes seen by 5 agents each, a depth-only tabletop, houses, "
    "dense rooms",
    [] {
        const std::vector<
            std::pair<std::string, std::vector<std::string>>>
            axes = {{"preset", {"minimal", "lit"}},
                    {"resolution", Values({64, 128, 256})}};
        std::vector<Case> cases;
        auto with = [&](std::string scenario, std::string name,
                        std::string value) {
            auto more = axes;
            more.push_back({std::move(name), {std::move(value)}});
            Append(cases, Sweep{{std::move(scenario)}, more}.Cases());
        };
        with("shapes", "views", "5");
        with("tabletop", "outputs", "depth");
        with("house", "outputs", "rgb");
        with("dense", "outputs", "rgb");
        return cases;
    });

ERNIE_BENCH_SUITE(
    "scene",
    "Syncing the renderer with 1k to 100k cubes: each kind of edit a "
    "step, against none",
    [] {
        return Sweep{{"scene_sync"},
                     {{"edit",
                       {"none", "colour", "xform", "parent", "batch",
                        "add", "walk"}},
                      {"prims", Values({1000, 10000, 100000})}}}
            .Cases();
    });

}  // namespace ernie::bench
