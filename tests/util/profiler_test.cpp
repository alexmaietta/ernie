// SPDX-License-Identifier: BSD-3-Clause
/// @file profiler_test.cpp
/// @brief Tests of the CPU zone profiler, its reports and their exports.
#include "ernie/util/profiler.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <string_view>

namespace ernie::profiler {
namespace {

void Work() {
    ERNIE_ZONE("Outer");
    {
        ERNIE_ZONE("Inner");
    }
}

TEST(Profiler, RecordsNestedZonesIntoTheFrameTheyEndIn) {
    Clear();
    SetCpuEnabled(true);
    BeginFrame();
    Work();
    BeginFrame();
    SetCpuEnabled(false);

    const std::vector<Frame> frames = Frames();
    ASSERT_FALSE(frames.empty());
    const std::vector<Zone>& zones = frames.back().cpu;
    const auto find = [&](std::string_view name) {
        return std::find_if(zones.begin(), zones.end(),
                            [&](const Zone& z) { return name == z.name; });
    };
    ASSERT_NE(find("Outer"), zones.end());
    ASSERT_NE(find("Inner"), zones.end());
    EXPECT_EQ(find("Outer")->depth, 0u);
    EXPECT_EQ(find("Inner")->depth, 1u);
    EXPECT_LE(find("Outer")->start, find("Inner")->start);
    EXPECT_GE(find("Outer")->end, find("Inner")->end);
}

TEST(Profiler, RecordsNothingWhileDisabled) {
    Clear();
    SetCpuEnabled(false);
    BeginFrame();
    Work();
    BeginFrame();
    for (const Frame& frame : Frames()) {
        EXPECT_TRUE(frame.cpu.empty());
    }
}

TEST(Profiler, InternReturnsOnePointerPerName) {
    const std::string name = "Dynamic";
    EXPECT_EQ(Intern(name), Intern("Dynamic"));
    EXPECT_STREQ(Intern(name), "Dynamic");
}

constexpr std::uint64_t kMs = 1'000'000;

/// A 10 ms frame: on the GPU, "Render" (4 ms) holding "Shadows"
/// (1 ms) and "Shading" (2 ms); on thread 0, "Update" (6 ms) holding
/// "Cull" (`cull` ms).
Frame MakeFrame(std::uint64_t index, std::uint64_t cull) {
    Frame frame;
    frame.index = index;
    frame.start = index * 10 * kMs;
    frame.end = frame.start + 10 * kMs;
    const std::uint64_t at = frame.start;
    frame.cpu = {{"Cull", at + kMs, at + kMs + cull * kMs, 0, 1},
                 {"Update", at, at + 6 * kMs, 0, 0}};
    frame.gpu = {{"Shadows", 100 * kMs, 101 * kMs, 0, 1},
                 {"Shading", 101 * kMs, 103 * kMs, 0, 1},
                 {"Render", 100 * kMs, 104 * kMs, 0, 0}};
    return frame;
}

TEST(ProfileReportTest, NestsZonesAndAveragesThemOverFrames) {
    const std::vector<Frame> frames = {MakeFrame(0, 1), MakeFrame(1, 3)};
    const Report report = Summarize(frames, {"Main"});
    EXPECT_EQ(report.gpuFrames, 2u);
    EXPECT_NEAR(report.gpuMs, 4.0, 1e-9);
    EXPECT_NEAR(report.cpuMs, 10.0, 1e-9);
    ASSERT_EQ(report.gpu.size(), 1u);
    const ReportNode& render = report.gpu[0];
    EXPECT_EQ(render.name, "Render");
    EXPECT_NEAR(render.percent, 100.0, 1e-9);
    ASSERT_EQ(render.children.size(), 2u);
    EXPECT_EQ(render.children[0].name, "Shadows") << "in the order run";
    EXPECT_NEAR(render.children[1].ms, 2.0, 1e-9);
    EXPECT_NEAR(render.children[1].percent, 50.0, 1e-9);

    ASSERT_EQ(report.cpu.size(), 1u);
    const ReportNode& main = report.cpu[0];
    EXPECT_EQ(main.name, "Main");
    EXPECT_NEAR(main.ms, 6.0, 1e-9);
    ASSERT_EQ(main.children.size(), 1u);
    ASSERT_EQ(main.children[0].children.size(), 1u);
    const ReportNode& cull = main.children[0].children[0];
    EXPECT_NEAR(cull.ms, 2.0, 1e-9) << "the mean of 1 and 3 ms";
    EXPECT_NEAR(cull.calls, 1.0, 1e-9);
    EXPECT_NEAR(cull.percent, 20.0, 1e-9);
}

TEST(ProfileReportTest, GpuZonesAverageOnlyOverFramesTheyReached) {
    Frame pending = MakeFrame(2, 1);
    pending.gpu.clear();
    const std::vector<Frame> frames = {MakeFrame(0, 1), pending};
    const Report report = Summarize(frames, {});
    EXPECT_EQ(report.gpuFrames, 1u);
    EXPECT_EQ(report.cpuFrames, 2u);
    EXPECT_NEAR(report.gpu[0].ms, 4.0, 1e-9);
    EXPECT_EQ(report.cpu[0].name, "Thread 0");
}

TEST(ProfileReportTest, RowsFilterByDepthAndTimeAndSortByTime) {
    const std::vector<Frame> frames = {MakeFrame(0, 1)};
    const Report report = Summarize(frames, {});
    const std::vector<ReportRow> tree = Rows(report.gpu, {});
    ASSERT_EQ(tree.size(), 3u);
    EXPECT_TRUE(tree[0].parent);
    EXPECT_EQ(tree[1].depth, 1u);

    const std::vector<ReportRow> outer = Rows(report.gpu, {0});
    ASSERT_EQ(outer.size(), 1u);
    EXPECT_FALSE(outer[0].parent) << "its children are not listed";

    EXPECT_EQ(Rows(report.gpu, {8, 1.5}).size(), 2u) << "Shadows takes less than 1.5 ms";

    const std::vector<ReportRow> byTime = Rows(report.gpu, {8, 0.0, true});
    ASSERT_EQ(byTime.size(), 3u);
    EXPECT_EQ(byTime[0].node->name, "Render");
    EXPECT_TRUE(byTime[0].parent) << "still nested";
    EXPECT_EQ(byTime[1].node->name, "Shading") << "siblings longest first";
    EXPECT_EQ(byTime[1].depth, 1u);
    EXPECT_EQ(byTime[2].node->name, "Shadows");
}

TEST(ProfileReportTest, ExportsTextJsonAndAChromeTrace) {
    const std::vector<Frame> frames = {MakeFrame(0, 1), MakeFrame(1, 1)};
    const Report report = Summarize(frames, {"Main"});
    const std::string text = ReportText(report, {});
    EXPECT_NE(text.find("    Shading"), std::string::npos);
    EXPECT_LT(text.find("GPU total"), text.find("Render")) << "the total heads its section";

    const nlohmann::json json = nlohmann::json::parse(ReportJson(report));
    EXPECT_EQ(json["gpu"]["zones"][0]["name"], "Render");
    EXPECT_EQ(json["gpu"]["zones"][0]["children"].size(), 2u);

    const nlohmann::json trace = nlohmann::json::parse(TraceJson(frames, {"Main"}));
    int render = 0;
    for (const nlohmann::json& event : trace["traceEvents"]) {
        if (event["name"] == "Render") {
            ++render;
            EXPECT_EQ(event["pid"], 2);
            EXPECT_NEAR(event["dur"].get<double>(), 4000.0, 1e-6);
        }
        if (event["name"] == "Cull" && event["ts"].get<double>() > 1e4) {
            EXPECT_NEAR(event["ts"].get<double>(), 11000.0, 1e-6)
                << "microseconds from the first frame";
        }
    }
    EXPECT_EQ(render, 2);
}

}  // namespace
}  // namespace ernie::profiler
