// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/viewer/cli/cli.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/util/error.h"

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
