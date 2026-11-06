// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/cli.h"

#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/util/error.h"
#include "support/support.h"

namespace ernie::app {
namespace {

Command Parse(std::vector<std::string> args) {
    return ParseCommand(args);
}

TEST(CliTest, ParsesEveryOption) {
    const Command defaults = Parse({});
    EXPECT_EQ(defaults.kind, Command::Kind::Viewer);
    EXPECT_EQ(defaults.width, 1280u);
    EXPECT_EQ(defaults.aov, "LdrColor");

    const Command command = Parse({"--device", "cuda:0", "--validation", "--traversal", "software",
                                   "--sampling=hardware", "--size", "800x600", "--vsync", "--aov",
                                   "Normal", "--layout", "layout.ini"});
    EXPECT_EQ(command.device.adapter, "cuda:0");
    EXPECT_TRUE(command.device.validation);
    EXPECT_EQ(command.device.traversal, device::Traversal::Software);
    EXPECT_EQ(command.device.sampling, device::Sampling::Hardware);
    EXPECT_EQ(command.width, 800u);
    EXPECT_EQ(command.height, 600u);
    EXPECT_TRUE(command.vsync);
    EXPECT_EQ(command.aov, "Normal");
    EXPECT_EQ(command.layout, "layout.ini");

    EXPECT_EQ(Parse({"-h"}).kind, Command::Kind::Help);
    EXPECT_EQ(Parse({"--version"}).kind, Command::Kind::Version);
    EXPECT_EQ(Parse({"--devices"}).kind, Command::Kind::Devices);
}

TEST(CliTest, ParsesVersionAndLogLevelFlags) {
    EXPECT_EQ(Parse({"-V"}).kind, Command::Kind::Version);
    EXPECT_FALSE(Parse({}).logLevel);
    EXPECT_EQ(Parse({"-v"}).logLevel, LogLevel::Debug);
    EXPECT_EQ(Parse({"--verbose", "-v"}).logLevel, LogLevel::Debug);
    EXPECT_EQ(Parse({"-q"}).logLevel, LogLevel::Warning);
    EXPECT_EQ(Parse({"--quiet"}).logLevel, LogLevel::Warning);
    EXPECT_THROW(Parse({"-v", "-q"}), Error);
}

TEST(CliTest, RunsTheHeadlessCommandsWithoutTheViewer) {
    for (const char* flag : {"--help", "--version", "--devices"}) {
        SCOPED_TRACE(flag);
        EXPECT_EQ(RunCli({flag}), 0);
    }
    EXPECT_EQ(RunCli({"--size", "nope"}), 2);
}

TEST(CliTest, ParsesTheSceneAndWindowlessOptions) {
    const Command scene =
        Parse({"--worlds", "3", "--camera", "/World/Camera", "--product", "/Render/Mine",
               "--preset", "minimal", "--aovs", "DistanceToCamera,,Normal"});
    EXPECT_EQ(scene.kind, Command::Kind::Viewer);
    EXPECT_EQ(scene.worlds, 3u);
    EXPECT_EQ(scene.camera, "/World/Camera");
    EXPECT_EQ(scene.product, "/Render/Mine");
    EXPECT_EQ(scene.preset, "minimal");
    EXPECT_EQ(scene.aovs, (std::vector<std::string>{"DistanceToCamera", "Normal"}));

    EXPECT_EQ(Parse({"--headless"}).kind, Command::Kind::Render);
    EXPECT_EQ(Parse({"--frames", "4"}).kind, Command::Kind::Render);
    const Command output = Parse({"--output", "out"});
    EXPECT_EQ(output.kind, Command::Kind::Render);
    EXPECT_EQ(output.output, "out");

    const Command benchmark =
        Parse({"--benchmark", "--warmup", "5", "--frames", "50", "--json", "--profile"});
    EXPECT_EQ(benchmark.kind, Command::Kind::Benchmark);
    EXPECT_EQ(benchmark.warmup, 5u);
    EXPECT_EQ(benchmark.frames, 50u);
    EXPECT_TRUE(benchmark.json);
    EXPECT_TRUE(benchmark.profile);
    EXPECT_EQ(Parse({"--help", "--benchmark"}).kind, Command::Kind::Help);

    EXPECT_THROW(Parse({"--json"}), Error);
    EXPECT_THROW(Parse({"--warmup", "3"}), Error);
    EXPECT_THROW(Parse({"--benchmark", "--output", "out"}), Error);
    EXPECT_THROW(Parse({"--preset", "fast"}), Error);
    EXPECT_THROW(Parse({"--worlds", "0"}), Error);
}

TEST(CliTest, RendersEachWorldsAovsWithoutAWindow) {
    const test::TemporaryDirectory directory;
    const std::filesystem::path out = directory.Path() / "frames";
    EXPECT_EQ(RunCommand(Parse({"--output", out.string(), "--size", "32x24", "--preset", "minimal",
                                "--worlds", "2", "--aovs", "DistanceToCamera"})),
              0);
    for (const char* file :
         {"LdrColor_0.png", "LdrColor_1.png", "DistanceToCamera_0.png", "DistanceToCamera_1.png"}) {
        SCOPED_TRACE(file);
        EXPECT_TRUE(std::filesystem::exists(out / file));
    }
    EXPECT_THROW(RunCommand(Parse({"--headless", "--camera", "/Nowhere"})), Error);
    EXPECT_THROW(RunCommand(Parse({"--headless", "--product", "/Render/Nothing"})), Error);
    EXPECT_THROW(RunCommand(Parse({"--headless", "--aovs", "Sparkle"})), Error);
}

TEST(CliTest, BenchmarksPrintOneJsonLine) {
    testing::internal::CaptureStdout();
    const int status = RunCommand(Parse({
        "--benchmark",
        "--warmup",
        "1",
        "--frames",
        "2",
        "--size",
        "16x16",
        "--preset",
        "minimal",
        "--json",
    }));
    const std::string printed = testing::internal::GetCapturedStdout();
    EXPECT_EQ(status, 0);
    EXPECT_TRUE(printed.starts_with("{\"product\": \"/Render/Product\""));
    EXPECT_NE(printed.find("\"pixels\": 256"), std::string::npos);
    EXPECT_NE(printed.find("\"steps\": 2"), std::string::npos);
    EXPECT_TRUE(printed.ends_with("}\n"));
}

TEST(CliTest, RejectsMalformedOptions) {
    EXPECT_THROW(Parse({"--size", "800"}), Error);
    EXPECT_THROW(Parse({"--size", "0x600"}), Error);
    EXPECT_THROW(Parse({"--traversal", "sideways"}), Error);
    EXPECT_THROW(Parse({"--device"}), Error);
    EXPECT_THROW(Parse({"--vsync=yes"}), Error);
    EXPECT_THROW(Parse({"--fullscreen"}), Error);
}

}
}
