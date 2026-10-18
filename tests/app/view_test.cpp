// SPDX-License-Identifier: BSD-3-Clause
/// @file view_test.cpp
/// @brief Tests of the viewer's command line.
#include "ernie/app/viewer.h"

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

namespace ernie {
namespace {

using app::ParseViewCommand;
using app::ViewCommand;

TEST(ViewCommand, OpensAnEmptySceneInAWindowByDefault) {
    const ViewCommand command = ParseViewCommand({});
    EXPECT_FALSE(command.scene);
    EXPECT_FALSE(command.viewer.offscreen);
    EXPECT_FALSE(command.viewer.vsync) << "frames show what they cost";
    EXPECT_EQ(command.viewer.frames, 0u);
}

TEST(ViewCommand, ReadsASceneAndOptions) {
    const std::array<std::string_view, 9> arguments{"rover.json", "--offscreen", "--frames",
                                                    "3",          "--size",      "320x200",
                                                    "--device",   "llvmpipe",    "--vsync"};
    const ViewCommand command = ParseViewCommand(arguments);
    EXPECT_EQ(command.scene, std::filesystem::path("rover.json"));
    EXPECT_EQ(command.viewer.title, "Ernie - rover.json");
    EXPECT_TRUE(command.viewer.offscreen);
    EXPECT_EQ(command.viewer.frames, 3u);
    EXPECT_EQ(command.viewer.size, glm::uvec2(320, 200));
    EXPECT_EQ(command.viewer.device, "llvmpipe");
    EXPECT_TRUE(command.viewer.vsync);
}

TEST(ViewCommand, RejectsWhatItCannotRead) {
    for (const std::vector<std::string_view>& arguments : {std::vector<std::string_view>{"--bogus"},
                                                           {"--frames"},
                                                           {"--frames", "three"},
                                                           {"--size", "320"},
                                                           {"--size", "0x10"},
                                                           {"a.json", "b.json"}}) {
        EXPECT_THROW(ParseViewCommand(arguments), std::invalid_argument) << arguments.front();
    }
}

}  // namespace
}  // namespace ernie
