// SPDX-License-Identifier: BSD-3-Clause
/// @file viewer_test.cpp
/// @brief Tests of the viewer on a device: frames drawn offscreen, and in a window when there is a
/// display.
#include "ernie/app/viewer.h"
#include "ernie/device/buffer.h"
#include "ernie/scene/scene.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <memory>

namespace ernie {
namespace {

using app::Viewer;
using app::ViewerOptions;

/// @return The viewer, or none when there is no Vulkan and it is not required.
std::unique_ptr<Viewer> MakeViewer(const Scene& scene, const ViewerOptions& options) {
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

TEST(Viewer, ClearsTheViewportOffscreen) {
    const Scene scene;
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
    const std::array<std::uint8_t, 4> texel = ReadTexel(*viewer, corner);
    const glm::vec4 clear = viewer->GetShell().GetTheme().viewport;
    for (std::size_t c = 0; c < 4; ++c) {
        EXPECT_NEAR(texel[c], std::round(clear[static_cast<glm::length_t>(c)] * 255.0f), 1.0)
            << "channel " << c;
    }
    test::ExpectClean(viewer->Context());
}

#if ERNIE_VIEWER_WINDOW
TEST(Viewer, PresentsToAWindow) {
    if (!app::Window::Available()) {
        GTEST_SKIP() << "No display";
    }
    const Scene scene;
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
