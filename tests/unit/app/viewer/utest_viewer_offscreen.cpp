// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/viewer/viewer.h"
#include "ernie/app/viewer/viewport/viewport.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/app/viewer/panels/panels.h"
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

TEST_F(OffscreenViewerTest, RunsFrameHandlersAroundEachFrameUntilRemoved) {
    Viewer viewer(Row(*s_device), {.width = kWidth, .height = kHeight, .offscreen = true});
    std::vector<std::string> calls;
    uint64_t begin = 0;
    begin = viewer.AddBeginFrameHandler([&] {
        calls.push_back("begin");
        if (calls.size() > 2) {
            viewer.RemoveHandler(begin);
        }
    });
    viewer.AddEndFrameHandler([&] { calls.push_back("end"); });
    for (int frame = 0; frame < 3; ++frame) {
        viewer.Frame();
    }
    EXPECT_EQ(calls, (std::vector<std::string>{"begin", "end", "begin", "end", "end"}));
}

#ifdef ERNIE_VIEWER_XR
TEST_F(OffscreenViewerTest, OffersAnXrPanelThatDrawsUntilStarted) {
    Viewer viewer(Row(*s_device), {.width = kWidth, .height = kHeight, .offscreen = true});
    viewer.ShowPanel("XR", true);
    for (int frame = 0; frame < 4; ++frame) {
        viewer.Frame();
    }
    EXPECT_EQ(viewer.Viewports(), (std::vector<std::string>{"Viewport"}));
}

TEST_F(OffscreenViewerTest, RunsXrInTheFrameLoopAndMirrorsTheLeftEye) {
    const Ref<Scene> scene = Row(*s_device);
    Viewer viewer(scene,
                  {.width = kWidth, .height = kHeight, .offscreen = true, .builtinPanels = false});
    const std::shared_ptr<XrPanel> xr = AddXrPanel(viewer, scene);
    viewer.ShowPanel("XR", true);
    xr->Start();
    ASSERT_TRUE(xr->Running());
    ASSERT_TRUE(FrameUntil(viewer, [&] { return viewer.Shown("XR Mirror") != nullptr; }));
    for (int frame = 0; frame < 10; ++frame) {
        scene->Find("/Middle")->Set(Xformable::kTranslate, Double3{0, 0, double(frame)});
        viewer.Frame();
    }
    xr->Stop();
    EXPECT_FALSE(xr->Running());
    viewer.Frame();
    EXPECT_EQ(viewer.Viewports(), (std::vector<std::string>{"Viewport"}));
    EXPECT_FALSE(scene->Find("/__ernie_xr").has_value());
}
#endif

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

std::vector<std::string> CloudSensors(Scene& scene) {
    std::vector<std::string> sensors;
    std::vector<Prim> stack;
    if (const std::optional<Prim> root = scene.Find(kSessionRoot)) {
        stack.push_back(*root);
    }
    while (!stack.empty()) {
        const Prim prim = stack.back();
        stack.pop_back();
        if (prim.IsA<RenderProduct>() && prim.Name().starts_with("Cloud")) {
            const std::vector<std::string> cameras = scene.Select(prim).Get("camera").GetTargets(0);
            sensors.insert(sensors.end(), cameras.begin(), cameras.end());
        }
        for (const Prim& child : prim.Children()) {
            stack.push_back(child);
        }
    }
    return sensors;
}

TEST_F(OffscreenViewerTest, DrawsTheCloudsOfSelectedLidarsAndRadars) {
    const Ref<Scene> scene = Row(*s_device);
    scene->Define<Lidar>("/Lidar", {.translate = {0, 0, 10},
                                    .scanType = ScanType::SolidState,
                                    .firingsPerScan = 1,
                                    .emitterElevations = {-80, -70, -60}});
    scene->Define<Radar>("/Radar", {.translate = {0, 0, 10}, .rays = {8, 4}});
    Viewer viewer(scene, {.width = kWidth, .height = kHeight, .offscreen = true});
    ASSERT_TRUE(FrameUntil(viewer, [&] { return viewer.Shown() != nullptr; }));
    EXPECT_TRUE(CloudSensors(*scene).empty());
    EXPECT_FALSE(Drawing(viewer, "viewer/clouds"));

    viewer.Select({"/Lidar"});
    ASSERT_TRUE(FrameUntil(viewer, [&] { return Drawing(viewer, "viewer/clouds"); }));
    EXPECT_EQ(CloudSensors(*scene), std::vector<std::string>{"/Lidar"});

    viewer.Select({"/Lidar", "/Radar"});
    ASSERT_TRUE(FrameUntil(viewer, [&] { return CloudSensors(*scene).size() == 2; }));
    EXPECT_TRUE(Drawing(viewer, "viewer/clouds"));

    viewer.Select({});
    viewer.Frame();
    EXPECT_TRUE(CloudSensors(*scene).empty());
    EXPECT_FALSE(Drawing(viewer, "viewer/clouds"));

    scene->Find("/Lidar")->Set(Sensor::kPipelineBinding, std::vector<std::string>{"/Pipe"});
    const Prim pipe = scene->Define<SensorPipeline>("/Pipe");
    const Prim returns = scene->Define<Shader>("/Pipe/Returns", {.id = "LidarReturns"});
    const Prim noise = scene->Define<Shader>("/Pipe/Noise", {.id = "RangeNoise"});
    noise.Connect("returns", returns, "returns");
    pipe.Connect("noisy", noise, "returns");
    viewer.Select({"/Pipe/Noise"});
    ASSERT_TRUE(FrameUntil(viewer, [&] { return Drawing(viewer, "viewer/clouds"); }));
    EXPECT_EQ(CloudSensors(*scene), std::vector<std::string>{"/Lidar"});
}

}
}
