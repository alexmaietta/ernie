// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/profiler.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <nvtx3/nvToolsExt.h>

#include "support/support.h"

namespace ernie::profiler {
namespace {

class ProfilerTest : public testing::Test {
  protected:
    void SetUp() override {
        SetHistory(300);
        Clear();
        BeginFrame();
        SetEnabled(true);
    }

    void TearDown() override {
        SetEnabled(false);
        Clear();
    }
};

Zone MakeZone(std::string_view name, uint32_t track, uint32_t depth, uint64_t startNs,
              uint64_t endNs) {
    return {ZoneId::Named(name), track, depth, startNs, endNs};
}

Frame MakeFrame(uint64_t index, uint64_t startNs, uint64_t endNs, std::vector<Zone> zones) {
    return {index, startNs, endNs, std::move(zones)};
}

std::vector<Zone> FinishFrame() {
    BeginFrame();
    return Frames().back().zones;
}

TEST_F(ProfilerTest, ZoneIdsAreInternedByName) {
    const ZoneId a = ZoneId::Named("Test/A");
    EXPECT_EQ(a, ZoneId::Named("Test/A"));
    EXPECT_NE(a, ZoneId::Named("Test/B"));
    EXPECT_EQ(a.Name(), "Test/A");
}

TEST_F(ProfilerTest, ScopedZonesRecordTheirNesting) {
    EXPECT_TRUE(Enabled());
    {
        ERNIE_ZONE("Test/Outer");
        ERNIE_ZONE("Test/Inner");
    }
    const std::vector<Zone> zones = FinishFrame();
    ASSERT_EQ(zones.size(), 2u);
    EXPECT_EQ(zones[0].id.Name(), "Test/Inner");
    EXPECT_EQ(zones[0].depth, 1u);
    EXPECT_EQ(zones[1].id.Name(), "Test/Outer");
    EXPECT_EQ(zones[1].depth, 0u);
    EXPECT_LE(zones[1].startNs, zones[0].startNs);
    EXPECT_GE(zones[1].endNs, zones[0].endNs);
    EXPECT_FALSE(Tracks().at(zones[0].track).gpu);
}

TEST_F(ProfilerTest, DisabledZonesRecordNothing) {
    SetEnabled(false);
    EXPECT_FALSE(Enabled());
    {
        ERNIE_ZONE("Test/Off");
    }
    EXPECT_TRUE(FinishFrame().empty());
}

TEST_F(ProfilerTest, KeepsOnlyTheNewestFrames) {
    SetHistory(2);
    BeginFrame();
    BeginFrame();
    BeginFrame();
    std::vector<Frame> frames = Frames();
    ASSERT_EQ(frames.size(), 2u);
    EXPECT_EQ(frames[0].index + 1, frames[1].index);
    EXPECT_EQ(frames[1].index + 1, CurrentFrame());
    EXPECT_LE(frames[1].startNs, frames[1].endNs);

    SetHistory(0);
    EXPECT_EQ(Frames().size(), 1u);
}

TEST_F(ProfilerTest, ClearDropsFramesAndTheCurrentZones) {
    BeginFrame();
    {
        ERNIE_ZONE("Test/Cleared");
    }
    Clear();
    EXPECT_TRUE(Frames().empty());
    EXPECT_TRUE(FinishFrame().empty());
}

TEST_F(ProfilerTest, EachThreadGetsItsOwnNamedTrack) {
    {
        ERNIE_ZONE("Test/Main");
    }
    std::thread worker([] {
        SetThreadName("Test worker");
        ERNIE_ZONE("Test/Worker");
    });
    worker.join();

    const std::vector<Zone> zones = FinishFrame();
    ASSERT_EQ(zones.size(), 2u);
    EXPECT_NE(zones[0].track, zones[1].track);
    EXPECT_EQ(Tracks().at(zones[1].track).name, "Test worker");
}

TEST_F(ProfilerTest, GpuZonesJoinTheFrameTheyWereSubmittedIn) {
    const uint32_t gpu = AddGpuTrack("Test GPU");
    EXPECT_TRUE(Tracks().at(gpu).gpu);
    EXPECT_EQ(Tracks().at(gpu).name, "Test GPU");

    const uint64_t submitted = CurrentFrame();
    BeginFrame();
    AddZone(submitted, MakeZone("Test/Late", gpu, 0, 10, 20));
    AddZone(CurrentFrame(), MakeZone("Test/Now", gpu, 0, 10, 20));
    AddZone(CurrentFrame() + 100, MakeZone("Test/Lost", gpu, 0, 10, 20));

    const std::vector<Frame> frames = Frames();
    ASSERT_EQ(frames.back().index, submitted);
    ASSERT_EQ(frames.back().zones.size(), 1u);
    EXPECT_EQ(frames.back().zones[0].id.Name(), "Test/Late");
    const std::vector<Zone> now = FinishFrame();
    ASSERT_EQ(now.size(), 1u);
    EXPECT_EQ(now[0].id.Name(), "Test/Now");
}

TEST(ProfilerReportTest, SummarizesNothingToNothing) {
    const Report report = Summarize({});
    EXPECT_EQ(report.frames, 0u);
    EXPECT_EQ(report.frameMs, 0);
    EXPECT_TRUE(report.tracks.empty());
}

TEST(ProfilerReportTest, AveragesZonesByTrackAndParent) {
    const uint32_t gpu = AddGpuTrack("Test GPU");
    const auto cpu = uint32_t(Tracks().size() + 100);
    std::vector<Frame> frames;
    for (uint64_t i = 0; i < 2; ++i) {
        const uint64_t at = i * 10'000'000;
        frames.push_back(MakeFrame(i, at, at + 10'000'000,
                                   {MakeZone("Test/Solve", cpu, 1, at, at + 1'000'000),
                                    MakeZone("Test/Step", cpu, 0, at, at + 4'000'000),
                                    MakeZone("Test/Trace", gpu, 0, at, at + 5'000'000)}));
    }
    frames[1].zones.push_back(MakeZone("Test/Solve", cpu, 1, 12'000'000, 15'000'000));
    frames[1].zones.push_back(MakeZone("Test/Deep", cpu, 3, 12'500'000, 13'000'000));

    const Report report = Summarize(frames);
    EXPECT_EQ(report.frames, 2u);
    EXPECT_DOUBLE_EQ(report.frameMs, 10);
    ASSERT_EQ(report.tracks.size(), 2u);

    const ReportTrack& gpuTrack = report.tracks[0];
    EXPECT_EQ(gpuTrack.name, "Test GPU");
    EXPECT_TRUE(gpuTrack.gpu);
    EXPECT_DOUBLE_EQ(gpuTrack.ms, 5);

    const ReportTrack& cpuTrack = report.tracks[1];
    EXPECT_EQ(cpuTrack.name, "Track " + std::to_string(cpu));
    EXPECT_FALSE(cpuTrack.gpu);
    EXPECT_DOUBLE_EQ(cpuTrack.ms, 4);
    ASSERT_EQ(cpuTrack.zones.size(), 1u);

    const ReportNode& step = cpuTrack.zones[0];
    EXPECT_EQ(step.name, "Test/Step");
    EXPECT_DOUBLE_EQ(step.ms, 4);
    EXPECT_DOUBLE_EQ(step.calls, 1);
    EXPECT_DOUBLE_EQ(step.percent, 40);
    ASSERT_EQ(step.children.size(), 1u);

    const ReportNode& solve = step.children[0];
    EXPECT_DOUBLE_EQ(solve.ms, 2.5);
    EXPECT_DOUBLE_EQ(solve.calls, 1.5);
    EXPECT_DOUBLE_EQ(solve.minMs, 1);
    EXPECT_DOUBLE_EQ(solve.maxMs, 3);
    ASSERT_EQ(solve.children.size(), 1u);
    EXPECT_EQ(solve.children[0].name, "Test/Deep");
}

TEST(ProfilerReportTest, ZeroLengthFramesHaveNoPercentages) {
    const Report report = Summarize({MakeFrame(0, 5, 5, {MakeZone("Test/Instant", 0, 0, 5, 5)})});
    ASSERT_EQ(report.tracks.size(), 1u);
    EXPECT_EQ(report.tracks[0].zones[0].percent, 0);
}

Report SampleReport() {
    ReportNode leaf{.name = "Leaf", .ms = 0.5, .calls = 1, .percent = 5};
    ReportNode slow{.name = "Slow", .ms = 3, .calls = 1, .percent = 30, .children = {leaf}};
    ReportNode fast{.name = "Fast", .ms = 1, .calls = 2, .percent = 10};
    ReportNode tiny{.name = "Tiny", .ms = 0.01, .calls = 1, .percent = 0.1};
    return {.frames = 4,
            .frameMs = 10,
            .tracks = {{.name = "Main", .gpu = false, .ms = 4, .zones = {fast, slow, tiny}},
                       {.name = "GPU", .gpu = true, .ms = 2, .zones = {}}}};
}

TEST(ProfilerReportTest, TextListsZonesInRecordedOrder) {
    const std::string text = ReportText(SampleReport());
    EXPECT_NE(text.find("Duration (ms)"), std::string::npos);
    EXPECT_NE(text.find("(4 frames)"), std::string::npos);
    EXPECT_NE(text.find("Main (CPU)"), std::string::npos);
    EXPECT_NE(text.find("GPU (GPU)"), std::string::npos);
    EXPECT_NE(text.find("    Leaf"), std::string::npos);
    EXPECT_LT(text.find("Fast"), text.find("Slow"));
    EXPECT_NE(text.find("Tiny"), std::string::npos);
}

TEST(ProfilerReportTest, TextFiltersByTimeAndDepth) {
    const std::string text = ReportText(SampleReport(), {.depth = 1, .minMs = 0.1, .byTime = true});
    EXPECT_LT(text.find("Slow"), text.find("Fast"));
    EXPECT_EQ(text.find("Leaf"), std::string::npos);
    EXPECT_EQ(text.find("Tiny"), std::string::npos);
}

TEST(ProfilerReportTest, JsonHoldsEveryField) {
    Report report = SampleReport();
    report.tracks[0].zones = {{.name = "a\"b\\c\n", .ms = 1, .calls = 2, .percent = 3}};
    EXPECT_EQ(
        ReportJson(report),
        R"({"frames":4,"frameMs":10,"tracks":[)"
        R"({"name":"Main","gpu":false,"ms":4,"zones":[)"
        R"({"name":"a\"b\\c ","ms":1,"calls":2,"percent":3,"minMs":0,"maxMs":0,"children":[]}]},)"
        R"({"name":"GPU","gpu":true,"ms":2,"zones":[]}]})"
        "\n");
}

TEST(ProfilerReportTest, TraceOfNoFramesHasOnlyTheProcessNames) {
    EXPECT_EQ(TraceJson({}),
              R"({"traceEvents":[)"
              R"({"name":"process_name","ph":"M","pid":1,"args":{"name":"CPU"}},)"
              R"({"name":"process_name","ph":"M","pid":2,"args":{"name":"GPU"}},)"
              R"({"name":"thread_name","ph":"M","pid":1,"tid":"frames","args":{"name":"Frames"}})"
              R"(],"displayTimeUnit":"ms"})"
              "\n");
}

TEST(ProfilerReportTest, TracePlacesZonesOnCpuAndGpuThreads) {
    const uint32_t gpu = AddGpuTrack("Test GPU");
    const auto cpu = uint32_t(Tracks().size() + 100);
    const std::string trace = TraceJson({MakeFrame(
        7, 1000, 9000,
        {MakeZone("Test/Cpu", cpu, 0, 2000, 3000), MakeZone("Test/Gpu", gpu, 0, 4000, 8000)})});
    const std::string gpuTid = std::to_string(gpu);
    const std::string cpuTid = std::to_string(cpu);
    EXPECT_NE(
        trace.find(R"({"name":"Frame 7","ph":"X","pid":1,"tid":"frames","ts":0.000,"dur":8.000})"),
        std::string::npos);
    EXPECT_NE(trace.find(R"({"name":"Test/Cpu","ph":"X","pid":1,"tid":)" + cpuTid +
                         R"(,"ts":1.000,"dur":1.000})"),
              std::string::npos);
    EXPECT_NE(trace.find(R"({"name":"Test/Gpu","ph":"X","pid":2,"tid":)" + gpuTid +
                         R"(,"ts":3.000,"dur":4.000})"),
              std::string::npos);
    EXPECT_NE(trace.find(R"({"name":"thread_name","ph":"M","pid":2,"tid":)" + gpuTid +
                         R"(,"args":{"name":"Test GPU"}})"),
              std::string::npos);
    EXPECT_NE(trace.find(R"({"name":"thread_name","ph":"M","pid":1,"tid":)" + cpuTid +
                         R"(,"args":{"name":"Track )" + cpuTid + R"("}})"),
              std::string::npos);
}

class ProfilerNvtxTest : public testing::Test {
  protected:
    void SetUp() override {
        if (!NVTX_SUPPORT_STATIC_INJECTION_LIBRARY) {
            GTEST_SKIP() << "the fake NVTX tool needs NVTX static injection";
        }
        test::UseFreshProcessForDeathTests();
    }
};

int g_fakeDomain = 0;

nvtxDomainHandle_t FakeDomainCreate(const char* name) {
    std::fprintf(stderr, "nvtx domain %s\n", name);
    return reinterpret_cast<nvtxDomainHandle_t>(&g_fakeDomain);
}

nvtxStringHandle_t FakeRegisterString(nvtxDomainHandle_t, const char* text) {
    static uintptr_t handles = 0;
    std::fprintf(stderr, "nvtx register %s\n", text);
    return reinterpret_cast<nvtxStringHandle_t>(++handles);
}

int FakeRangePush(nvtxDomainHandle_t, const nvtxEventAttributes_t* attributes) {
    std::fprintf(stderr, "nvtx push %zu\n", size_t(attributes->message.registered));
    return 0;
}

int FakeRangePop(nvtxDomainHandle_t) {
    std::fprintf(stderr, "nvtx pop\n");
    return 0;
}

void FakeNameOsThread(uint32_t, const char* name) {
    std::fprintf(stderr, "nvtx thread %s\n", name);
}

template <class Function>
void Install(NvtxGetExportTableFunc_t exportTable, NvtxCallbackModule module, unsigned id,
             Function function) {
    const auto* callbacks =
        static_cast<const NvtxExportTableCallbacks*>(exportTable(NVTX_ETID_CALLBACKS));
    NvtxFunctionTable table = nullptr;
    unsigned size = 0;
    callbacks->GetModuleFunctionTable(module, &table, &size);
    *table[id] = reinterpret_cast<NvtxFunctionPointer>(function);
}

int InitializeFakeNvtx(NvtxGetExportTableFunc_t exportTable) {
    Install(exportTable, NVTX_CB_MODULE_CORE2, NVTX_CBID_CORE2_DomainCreateA, FakeDomainCreate);
    Install(exportTable, NVTX_CB_MODULE_CORE2, NVTX_CBID_CORE2_DomainRegisterStringA,
            FakeRegisterString);
    Install(exportTable, NVTX_CB_MODULE_CORE2, NVTX_CBID_CORE2_DomainRangePushEx, FakeRangePush);
    Install(exportTable, NVTX_CB_MODULE_CORE2, NVTX_CBID_CORE2_DomainRangePop, FakeRangePop);
    Install(exportTable, NVTX_CB_MODULE_CORE, NVTX_CBID_CORE_NameOsThreadA, FakeNameOsThread);
    return 1;
}

void ProfileUnderFakeNvtx() {
#if NVTX_SUPPORT_STATIC_INJECTION_LIBRARY
    InitializeInjectionNvtx2_fnptr = InitializeFakeNvtx;
#endif
    AddGpuTrack("Test GPU");
    SetThreadName("Test main");
    for (int i = 0; i < 2; ++i) {
        ERNIE_ZONE("Test/Nvtx");
    }
    std::exit(0);
}

TEST_F(ProfilerNvtxTest, SendsThreadNamesAndRangesToTheTool) {
    const test::ScopedEnvironment on("ERNIE_NVTX", std::nullopt);
    EXPECT_EXIT(ProfileUnderFakeNvtx(), testing::ExitedWithCode(0),
                "^nvtx domain ernie\n"
                "nvtx thread Thread 1\n"
                "nvtx thread Test main\n"
                "nvtx register Test/Nvtx\n"
                "nvtx push 1\nnvtx pop\n"
                "nvtx push 1\nnvtx pop\n$");
}

TEST_F(ProfilerNvtxTest, CanBeTurnedOff) {
    const test::ScopedEnvironment off("ERNIE_NVTX", "0");
    EXPECT_EXIT(ProfileUnderFakeNvtx(), testing::ExitedWithCode(0), "^$");
}

}
}
