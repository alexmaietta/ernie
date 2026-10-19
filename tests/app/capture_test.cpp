// SPDX-License-Identifier: BSD-3-Clause
/// @file capture_test.cpp
/// @brief Tests of capture naming, unique paths and ffmpeg's encoding command.
#include "ernie/app/capture.h"

#include <gtest/gtest.h>

#include <fstream>

namespace ernie {
namespace {

TEST(Capture, NamesFillInAndStayFileNames) {
    app::CaptureNameFields fields{.scene = "rover", .camera = "Nav/Left", .frame = 7};
    EXPECT_EQ(app::ExpandCaptureName("{scene}_{camera}_{frame}", fields), "rover_Nav_Left_0007");
    EXPECT_EQ(app::ExpandCaptureName("{unknown}", fields), "{unknown}") << "left as written";
    EXPECT_EQ(app::ExpandCaptureName("", fields), "capture");
    const std::string dated = app::ExpandCaptureName("{date}", fields);
    EXPECT_EQ(dated.size(), 10u) << dated;
}

TEST(Capture, NeverOverwritesAFile) {
    const std::filesystem::path folder =
        std::filesystem::temp_directory_path() / "ernie_capture_test";
    std::filesystem::create_directories(folder);
    const std::filesystem::path shot = folder / "shot.png";
    std::ofstream(shot) << "taken";
    EXPECT_EQ(app::UniquePath(shot), folder / "shot_1.png");
    EXPECT_EQ(app::UniquePath(folder / "free.png"), folder / "free.png");
    std::filesystem::remove_all(folder);
}

TEST(Capture, EncodesFramesForSharingOrEditing) {
    const std::vector<std::string> mp4 =
        app::EncodeCommand("ffmpeg", "shot_%04d.png", 24.0, "out.mp4", app::VideoFormat::Mp4);
    EXPECT_EQ(mp4.front(), "ffmpeg");
    EXPECT_EQ(mp4.back(), "out.mp4");
    EXPECT_NE(std::ranges::find(mp4, "libx264"), mp4.end());
    const std::vector<std::string> prores =
        app::EncodeCommand("ffmpeg", "shot_%04d.png", 30.0, "out.mov", app::VideoFormat::ProRes);
    EXPECT_NE(std::ranges::find(prores, "prores_ks"), prores.end());
    EXPECT_STREQ(app::VideoExtension(app::VideoFormat::ProRes), ".mov");
}

}  // namespace
}  // namespace ernie
