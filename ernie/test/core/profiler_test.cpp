// SPDX-License-Identifier: BSD-3-Clause
/// @file profiler_test.cpp
/// @brief Tests of the CPU zone profiler.
#include "ernie/core/profile.h"

#include <gtest/gtest.h>

#include <algorithm>
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

}  // namespace
}  // namespace ernie::profiler
