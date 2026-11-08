// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/viewer/viewer.h"

#include <algorithm>
#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/scene.h"

namespace ernie::app {
namespace {

constexpr uint32_t kWidth = 160;
constexpr uint32_t kHeight = 90;
constexpr auto kPatience = std::chrono::seconds(60);

class OffscreenViewerTest : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        s_device = device::Device::Create();
    }

    static void TearDownTestSuite() {
        s_device = nullptr;
    }

    static Ref<device::Device> s_device;
};

Ref<device::Device> OffscreenViewerTest::s_device;

Ref<Scene> Row(device::Device& device) {
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<Cube>("/Left", {.translate = {-6, 0, 1}});
    scene->Define<Cube>("/Middle", {.translate = {0, 0, 1}});
    scene->Define<Cube>("/Right", {.translate = {6, 0, 1}});
    return scene;
}

bool Inside(const std::optional<Float2>& point, const Viewer& viewer) {
    return point && point->x >= 0 && point->y >= 0 && point->x <= float(viewer.Width()) &&
           point->y <= float(viewer.Height());
}

template <class Done> bool FrameUntil(Viewer& viewer, Done done) {
    const auto deadline = std::chrono::steady_clock::now() + kPatience;
    while (std::chrono::steady_clock::now() < deadline) {
        viewer.Frame();
        if (done()) {
            return true;
        }
    }
    return false;
}

TEST_F(OffscreenViewerTest, FramesTheWholeSceneWhenItOpens) {
    Viewer viewer(Row(*s_device), {.width = kWidth, .height = kHeight, .offscreen = true});
    ASSERT_TRUE(FrameUntil(viewer, [&] { return viewer.Shown() != nullptr; }));
    for (int frame = 0; frame < 4; ++frame) {
        viewer.Frame();
    }
    EXPECT_TRUE(Inside(viewer.WorldToScreen({-6, 0, 1}), viewer));
    EXPECT_TRUE(Inside(viewer.WorldToScreen({0, 0, 1}), viewer));
    EXPECT_TRUE(Inside(viewer.WorldToScreen({6, 0, 1}), viewer));
}

bool Drawing(const Viewer& viewer, const std::string& overlay) {
    const std::vector<std::string> names = viewer.GetRenderer()->GetOverlays()->Names();
    return std::ranges::find(names, overlay) != names.end();
}

TEST_F(OffscreenViewerTest, OutlinesTheSelectionUntilItClears) {
    Viewer viewer(Row(*s_device), {.width = kWidth, .height = kHeight, .offscreen = true});
    viewer.Select({"/Middle"});
    ASSERT_TRUE(FrameUntil(viewer, [&] { return viewer.Shown() != nullptr; }));
    EXPECT_EQ(viewer.Selected(), std::vector<std::string>{"/Middle"});
    EXPECT_TRUE(Drawing(viewer, "viewer/selection"));
    viewer.Select({});
    viewer.Frame();
    EXPECT_TRUE(viewer.Selected().empty());
    EXPECT_FALSE(Drawing(viewer, "viewer/selection"));
}

}
}
