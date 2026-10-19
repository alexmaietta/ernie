// SPDX-License-Identifier: BSD-3-Clause
/// @file viewer_test.cpp
/// @brief Tests of the viewer on a device: frames drawn offscreen, and in a window when there is a
/// display.
#include "ernie/app/viewer.h"
#include "ernie/device/buffer.h"
#include "ernie/scene/scene.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>

namespace ernie {
namespace {

using app::Viewer;
using app::ViewerOptions;

/// @return The viewer, or none when there is no Vulkan and it is not required. Its toolbar and
/// heads-up display are off: the tests read and click the small viewport's pixels under them.
std::unique_ptr<Viewer> MakeViewer(Scene& scene, ViewerOptions options) {
    options.shell.display.toolbar = false;
    options.shell.display.hud = false;
    try {
        return std::make_unique<Viewer>(scene, options);
    } catch (const DeviceError&) {
        if (test::VulkanRequired()) {
            throw;
        }
        return nullptr;
    }
}

/// @return The RGBA8 texel at `pixel` of the offscreen image once its last frame is drawn.
std::array<std::uint8_t, 4> ReadTexel(Viewer& viewer, glm::uvec2 pixel) {
    const app::OffscreenPresenter& offscreen = *viewer.Offscreen();
    DeviceContext& context = viewer.Context();
    const Buffer texel(context, 4, MemoryUse::Readback, "Viewer texel");
    const std::array waits{offscreen.Drawn()};
    context.Main()
        .Submit(
            [&](VkCommandBuffer commands) {
                const VkBufferImageCopy region{
                    .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                    .imageOffset = {static_cast<std::int32_t>(pixel.x),
                                    static_cast<std::int32_t>(pixel.y), 0},
                    .imageExtent = {1, 1, 1}};
                vkCmdCopyImageToBuffer(commands, offscreen.Image().Handle(),
                                       VK_IMAGE_LAYOUT_GENERAL, texel.Handle(), 1, &region);
            },
            waits)
        .Wait();
    texel.Invalidate(0, 4);
    std::array<std::uint8_t, 4> rgba{};
    std::memcpy(rgba.data(), texel.Mapped().data(), rgba.size());
    return rgba;
}

TEST(Viewer, ShowsTheFramesViewportImageOffscreen) {
    Scene scene;
    ViewerOptions options{.size = {320, 240}, .offscreen = true, .frames = 3, .validation = true};
    const std::unique_ptr<Viewer> viewer = MakeViewer(scene, options);
    if (!viewer) {
        GTEST_SKIP() << "No Vulkan device";
    }
    viewer->Run();
    EXPECT_EQ(viewer->Frames(), 3u);
    const auto [origin, size] = viewer->GetShell().Viewport();
    // The viewport's bottom right, clear of the heads-up display at its top left.
    const glm::uvec2 corner(static_cast<std::uint32_t>(origin.x + size.x) - 4,
                            static_cast<std::uint32_t>(origin.y + size.y) - 4);
    // The UI clears to black under it, so the colour comes from the frame's viewport image.
    const std::array<std::uint8_t, 4> texel = ReadTexel(*viewer, corner);
    const glm::vec4 clear = viewer->GetShell().GetTheme().viewport;
    for (std::size_t c = 0; c < 4; ++c) {
        EXPECT_NEAR(texel[c], std::round(clear[static_cast<glm::length_t>(c)] * 255.0f), 1.0)
            << "channel " << c;
    }
    test::ExpectClean(viewer->Context());
}

TEST(Viewer, TracesTheSceneIntoTheViewport) {
    Scene scene;
    const PrimId ball = scene.Define("/Ball", "Sphere");
    scene.Set(ball, "radius", 2.0);
    ViewerOptions options{.size = {320, 240}, .offscreen = true, .frames = 3, .validation = true};
    const std::unique_ptr<Viewer> viewer = MakeViewer(scene, options);
    if (!viewer) {
        GTEST_SKIP() << "No Vulkan device";
    }
    if (!viewer->Context().Capabilities().rayQuery) {
        GTEST_SKIP() << viewer->Context().Info().name << " has no ray queries";
    }
    viewer->Run();
    const auto [origin, size] = viewer->GetShell().Viewport();
    // The orbit camera looks at the origin, where the ball is, from 10 m.
    const glm::uvec2 middle(static_cast<std::uint32_t>(origin.x + 0.5f * size.x),
                            static_cast<std::uint32_t>(origin.y + 0.5f * size.y));
    const std::array<std::uint8_t, 4> texel = ReadTexel(*viewer, middle);
    const glm::vec4 clear = viewer->GetShell().GetTheme().viewport;
    EXPECT_GT(texel[0], std::round(clear.r * 255.0f) + 40) << "the lit ball, not the background";
    test::ExpectClean(viewer->Context());
}

TEST(Viewer, ReplaysItsTraceAsTheCameraMoves) {
    Scene scene;
    const PrimId ball = scene.Define("/Ball", "Sphere");
    scene.Set(ball, "radius", 2.0);
    // Each frame shows its own trace, so the image follows the camera at once.
    ViewerOptions options{.size = {320, 240},
                          .offscreen = true,
                          .frames = 6,
                          .validation = true,
                          .asyncRender = false};
    options.shell.display.eco = false;
    const std::unique_ptr<Viewer> viewer = MakeViewer(scene, options);
    if (!viewer) {
        GTEST_SKIP() << "No Vulkan device";
    }
    if (!viewer->Context().Capabilities().rayQuery) {
        GTEST_SKIP() << viewer->Context().Info().name << " has no ray queries";
    }
    viewer->Run();
    EXPECT_TRUE(viewer->Replayed()) << "a still scene's frames replay, a slot each";
    // Aimed 8 m to the side, the ball leaves the middle; the recorded commands read the new pose.
    viewer->GetShell().Camera().target += glm::dvec3(8.0, 0.0, 0.0);
    viewer->RunFrames(2);
    EXPECT_TRUE(viewer->Replayed()) << "a camera move records nothing new";
    const auto [origin, size] = viewer->GetShell().Viewport();
    const glm::uvec2 middle(static_cast<std::uint32_t>(origin.x + 0.5f * size.x),
                            static_cast<std::uint32_t>(origin.y + 0.5f * size.y));
    const std::array<std::uint8_t, 4> texel = ReadTexel(*viewer, middle);
    const glm::vec4 clear = viewer->GetShell().GetTheme().viewport;
    EXPECT_NEAR(texel[0], std::round(clear.r * 255.0f), 2.0) << "the background, past the ball";
    test::ExpectClean(viewer->Context());
}

TEST(Viewer, TracesApartFromTheUiOnTheComputeQueue) {
    Scene scene;
    const PrimId ball = scene.Define("/Ball", "Sphere");
    scene.Set(ball, "radius", 2.0);
    ViewerOptions options{.size = {320, 240}, .offscreen = true, .frames = 6, .validation = true};
    options.shell.display.eco = false;
    const std::unique_ptr<Viewer> viewer = MakeViewer(scene, options);
    if (!viewer) {
        GTEST_SKIP() << "No Vulkan device";
    }
    if (!viewer->Context().Capabilities().rayQuery) {
        GTEST_SKIP() << viewer->Context().Info().name << " has no ray queries";
    }
    if (&viewer->Context().Compute() == &viewer->Context().Main()) {
        GTEST_SKIP() << viewer->Context().Info().name << " has no compute queue of its own";
    }
    EXPECT_TRUE(viewer->AsyncRender());
    viewer->Run();
    const auto [origin, size] = viewer->GetShell().Viewport();
    const glm::uvec2 middle(static_cast<std::uint32_t>(origin.x + 0.5f * size.x),
                            static_cast<std::uint32_t>(origin.y + 0.5f * size.y));
    const float clear = std::round(viewer->GetShell().GetTheme().viewport.r * 255.0f);
    EXPECT_GT(ReadTexel(*viewer, middle)[0], clear + 40) << "the ball";

    // The UI shows the newest finished trace: the camera's move reaches the image within a few
    // frames, however the two queues interleave.
    viewer->GetShell().Camera().target += glm::dvec3(8.0, 0.0, 0.0);
    bool moved = false;
    for (int frame = 0; frame < 20 && !moved; ++frame) {
        viewer->RunFrames(1);
        moved = std::abs(ReadTexel(*viewer, middle)[0] - clear) <= 2.0f;
    }
    EXPECT_TRUE(moved) << "the background, past the ball";
    test::ExpectClean(viewer->Context());
}

TEST(Viewer, ClickingTheViewportSelectsWhatItShows) {
    Scene scene;
    const PrimId ball = scene.Define("/Ball", "Sphere");
    scene.Set(ball, "radius", 2.0);
    ViewerOptions options{.size = {320, 240}, .offscreen = true, .frames = 2, .validation = true};
    const std::unique_ptr<Viewer> viewer = MakeViewer(scene, options);
    if (!viewer) {
        GTEST_SKIP() << "No Vulkan device";
    }
    if (!viewer->Context().Capabilities().rayQuery) {
        GTEST_SKIP() << viewer->Context().Info().name << " has no ray queries";
    }
    viewer->Run();
    const auto [origin, size] = viewer->GetShell().Viewport();
    ImGuiIO& io = ImGui::GetIO();
    // Each run past the frame count draws one more frame.
    const auto clickAt = [&](float x, float y) {
        io.AddMousePosEvent(origin.x + x * size.x, origin.y + y * size.y);
        viewer->Run();
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
        viewer->Run();
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
        viewer->Run();
    };
    clickAt(0.5f, 0.5f);
    EXPECT_EQ(viewer->GetShell().Selected().Paths(), std::vector<Path>{Path("/Ball")});
    clickAt(0.9f, 0.15f);
    EXPECT_TRUE(viewer->GetShell().Selected().Empty()) << "the background clears it";
    test::ExpectClean(viewer->Context());
}

TEST(Viewer, EcoModePausesRenderingUntilSomethingChanges) {
    Scene scene;
    scene.Define("/Ball", "Sphere");
    ViewerOptions options{.size = {320, 240}, .offscreen = true, .frames = 4};
    options.shell.display.ecoFrames = 3;
    const std::unique_ptr<Viewer> viewer = MakeViewer(scene, options);
    if (!viewer) {
        GTEST_SKIP() << "No Vulkan device";
    }
    viewer->Run();
    EXPECT_FALSE(viewer->EcoPaused()) << "still for too few frames";
    viewer->Run();
    viewer->Run();
    EXPECT_TRUE(viewer->EcoPaused());
    scene.Set(*scene.Find("/Ball"), "radius", 2.0);
    viewer->Run();
    EXPECT_FALSE(viewer->EcoPaused()) << "an edit renders again";
    test::ExpectClean(viewer->Context());
}

TEST(Viewer, CapturesScreenshotsAndSequences) {
    Scene scene;
    scene.Define("/Ball", "Sphere");
    ViewerOptions options{.size = {320, 240}, .offscreen = true, .frames = 2};
    const std::unique_ptr<Viewer> viewer = MakeViewer(scene, options);
    if (!viewer) {
        GTEST_SKIP() << "No Vulkan device";
    }
    const std::filesystem::path folder =
        std::filesystem::temp_directory_path() / "ernie_viewer_capture_test";
    std::filesystem::remove_all(folder);
    app::Shell& shell = viewer->GetShell();
    shell.Capture().folder = folder;
    shell.Capture().name = "shot";
    viewer->Run();
    shell.RequestScreenshot();
    viewer->Run();
    ASSERT_EQ(shell.RecentCaptures().size(), 1u);
    EXPECT_EQ(shell.RecentCaptures().back(), folder / "shot.png");
    EXPECT_TRUE(std::filesystem::exists(folder / "shot.png"));

    shell.Capture().name = "sequence";
    shell.RequestSequence({.start = 10.0, .end = 12.0});
    for (int frame = 0; frame < 4; ++frame) {
        viewer->Run();
    }
    EXPECT_FALSE(shell.Recording()) << "done after three frames";
    for (const char* name : {"frame_0000.png", "frame_0001.png", "frame_0002.png"}) {
        EXPECT_TRUE(std::filesystem::exists(folder / "sequence" / name)) << name;
    }
    EXPECT_EQ(scene.Time(), 12.0) << "the sequence stepped the scene's clock";
    std::filesystem::remove_all(folder);
    test::ExpectClean(viewer->Context());
}

#if ERNIE_VIEWER_WINDOW
TEST(Viewer, OpensOnTheLogTab) {
    Scene scene;
    ViewerOptions options{.size = {640, 480}, .offscreen = true, .frames = 5};
    const std::unique_ptr<Viewer> viewer = MakeViewer(scene, options);
    if (!viewer) {
        GTEST_SKIP() << "No Vulkan device";
    }
    viewer->Run();
    const ImGuiWindow* log = ImGui::FindWindowByName("Log");
    ASSERT_NE(log, nullptr);
    EXPECT_TRUE(log->DockTabIsVisible) << "the bottom dock shows the Log";
    for (const char* other : {"Info", "Devices", "Profiler"}) {
        EXPECT_FALSE(ImGui::FindWindowByName(other)->DockTabIsVisible) << other;
    }
}

TEST(Viewer, PresentsToAWindow) {
    if (!app::Window::Available()) {
        GTEST_SKIP() << "No display";
    }
    Scene scene;
    ViewerOptions options{.size = {320, 240}, .frames = 3, .validation = true, .vsync = false};
    std::unique_ptr<Viewer> viewer;
    try {
        viewer = std::make_unique<Viewer>(scene, options);
    } catch (const DeviceError& error) {
        if (error.Kind() != DeviceErrorKind::Unsupported && test::VulkanRequired()) {
            throw;
        }
        GTEST_SKIP() << error.what();
    }
    viewer->Run();
    EXPECT_EQ(viewer->Frames(), 3u);
    test::ExpectClean(viewer->Context());
}
#endif

}  // namespace
}  // namespace ernie
