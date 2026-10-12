// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/capture.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace ernie::app;

TEST(CaptureName, FillsItsFieldsAndKeepsOthers) {
    CaptureNameFields fields;
    fields.scene = "lunar";
    fields.camera = "Rover";
    fields.frame = 7;
    EXPECT_EQ(ExpandCaptureName("{scene}_{camera}_{frame}", fields),
              "lunar_Rover_0007");
    EXPECT_EQ(ExpandCaptureName("{nope}", fields), "{nope}");
    // Local date and time, digits and dashes.
    const std::string stamp =
        ExpandCaptureName("{date}_{time}", fields);
    EXPECT_EQ(stamp.size(), 19u);
    EXPECT_EQ(stamp[10], '_');
}

TEST(CaptureName, MakesFileNamesSafe) {
    CaptureNameFields fields;
    fields.camera = "a/b:c";
    EXPECT_EQ(ExpandCaptureName("{camera}?", fields), "a_b_c_");
    EXPECT_EQ(ExpandCaptureName("", fields), "capture");
}

TEST(UniquePath, NeverOverwrites) {
    // Its own: ctest runs tests side by side, with seeds alike.
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() /
        ("ernie_capture_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    const std::filesystem::path shot = dir / "shot.png";
    std::filesystem::remove(shot);
    EXPECT_EQ(UniquePath(shot), shot);
    std::ofstream(shot) << "x";
    EXPECT_EQ(UniquePath(shot), dir / "shot_1.png");
    std::ofstream(dir / "shot_1.png") << "x";
    EXPECT_EQ(UniquePath(shot), dir / "shot_2.png");
    std::filesystem::remove_all(dir);
}

TEST(EncodeCommand, AsksFfmpegForTheFormat) {
    const auto has = [](const std::vector<std::string>& command,
                        const std::string& word) {
        return std::find(command.begin(), command.end(), word) !=
               command.end();
    };
    const auto mp4 = EncodeCommand("ffmpeg", "f/s_%04d.png", 30.0,
                                   "f/s.mp4", VideoFormat::Mp4);
    EXPECT_EQ(mp4.front(), "ffmpeg");
    EXPECT_EQ(mp4.back(), "f/s.mp4");
    EXPECT_TRUE(has(mp4, "libx264"));
    EXPECT_TRUE(has(mp4, "yuv420p"));
    EXPECT_TRUE(has(mp4, "30"));
    const auto prores = EncodeCommand("ffmpeg", "f/s_%04d.png", 24.0,
                                      "f/s.mov", VideoFormat::ProRes);
    EXPECT_TRUE(has(prores, "prores_ks"));
    EXPECT_STREQ(VideoExtension(VideoFormat::ProRes), ".mov");
}

TEST(RunProcess, ReportsExitStatusAndMissingPrograms) {
#ifndef _WIN32
    EXPECT_EQ(RunProcess({"/bin/sh", "-c", "exit 3"}), 3);
#endif
    EXPECT_EQ(RunProcess({"/no/such/program"}), -1);
}
