// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/demo.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

using namespace ernie::app;

TEST(DemoOptions, DefaultsToAWindowRunningUntilClosed) {
    const DemoOptions options =
        ParseDemoOptions(std::vector<std::string>{});
    EXPECT_TRUE(options.viewer);
    EXPECT_FALSE(options.headless);
    EXPECT_TRUE(options.Windowed());
    EXPECT_FALSE(options.numFrames);
    EXPECT_FALSE(options.benchmark);
    EXPECT_EQ(options.renderer, "raster");
}

TEST(DemoOptions, ParsesTheSharedOptionsAndKeepsTheRest) {
    const DemoOptions options =
        ParseDemoOptions(std::vector<std::string>{
            "--viewer", "null", "--num-frames=7", "--test",
            "--output", "out.png", "--rate", "3", "--device",
            "NVIDIA", "--validation", "--no-validation",
            "--headless"});
    EXPECT_FALSE(options.viewer);
    EXPECT_TRUE(options.headless);
    EXPECT_EQ(options.numFrames, 7u);
    EXPECT_TRUE(options.test);
    EXPECT_EQ(options.output, "out.png");
    EXPECT_EQ(options.device, "NVIDIA");
    EXPECT_FALSE(options.validation);
    EXPECT_EQ(options.extra,
              (std::vector<std::string>{"--rate", "3"}));
}

TEST(DemoOptions, BenchmarkTakesOptionalSecondsAndDropsTheViewer) {
    DemoOptions options = ParseDemoOptions(
        std::vector<std::string>{"--benchmark", "--test"});
    EXPECT_TRUE(options.benchmark);
    EXPECT_FALSE(options.benchmarkSeconds);
    EXPECT_FALSE(options.viewer);
    EXPECT_TRUE(options.test);
    options = ParseDemoOptions(
        std::vector<std::string>{"--benchmark", "2.5"});
    EXPECT_EQ(options.benchmarkSeconds, 2.5);
}

TEST(DemoOptions, ProfileFilesImplyProfiling) {
    EXPECT_FALSE(ParseDemoOptions({}).profile);
    EXPECT_TRUE(
        ParseDemoOptions(std::vector<std::string>{"--profile"})
            .profile);
    const DemoOptions options =
        ParseDemoOptions(std::vector<std::string>{
            "--profile-json", "p.json", "--profile-trace=t.json"});
    EXPECT_TRUE(options.profile);
    EXPECT_EQ(options.profileJson, "p.json");
    EXPECT_EQ(options.profileTrace, "t.json");
    EXPECT_TRUE(options.extra.empty());
}

TEST(DemoOptions, RejectsBadValues) {
    EXPECT_THROW(
        ParseDemoOptions(std::vector<std::string>{"--viewer", "gl"}),
        std::invalid_argument);
    EXPECT_THROW(ParseDemoOptions(std::vector<std::string>{
                     "--num-frames", "many"}),
                 std::invalid_argument);
    EXPECT_THROW(
        ParseDemoOptions(std::vector<std::string>{"--output"}),
        std::invalid_argument);
}

TEST(DemoInfo, LabelsByCategoryAndName) {
    const DemoInfo info{"lighting", "dome daylight", "", {}};
    EXPECT_EQ(info.Label(), "lighting: dome daylight");
}
