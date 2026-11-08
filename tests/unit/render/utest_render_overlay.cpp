// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/overlay/overlay.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/rotation.h"
#include "ernie/scene/scene.h"
#include "ernie/util/error.h"
#include "support/devices.h"

namespace ernie {
namespace {

using device::Device;
using test::ForEachDevice;

constexpr uint32_t kWidth = 32;
constexpr uint32_t kHeight = 24;
const std::vector<std::string> kProducts = {"/Render/View"};

void Show(Scene& scene, std::vector<std::string> layers) {
    scene.Find("/Render/View")->Set(RenderProduct::kOverlays, std::move(layers));
}

Ref<Scene> Empty(Device& device) {
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<Camera>("/Cam",
                          {.translate = {0, -5, 0}, .orient = LookAt({0, -5, 0}, {0, 0, 0})});
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Color", {.sourceName = RenderVarSource::LdrColor});
    scene->Define<RenderProduct>(
        "/Render/View",
        {.camera = {"/Cam"}, .resolution = {kWidth, kHeight}, .orderedVars = {"/Render/Color"}});
    scene->Select("/Render/View").Get<RenderMode>("renderMode").Set(RenderMode::Minimal);
    Show(*scene, {"*"});
    return scene;
}

Overlay EveryShape() {
    const OverlayStyle red{.color = {1, 0, 0, 1}, .size = 12};
    const std::vector<Float3> square = {{-1, 0, -1}, {1, 0, -1}, {1, 0, 1}, {-1, 0, 1}};
    const std::vector<uint32_t> counts = {4};
    const std::vector<uint32_t> indices = {0, 1, 2, 3};
    Overlay overlay;
    overlay.Line({-2, 0, 2}, {2, 0, 2})
        .Lines(square)
        .Polyline(square, true)
        .Markers(square)
        .Box({0, 0, 0}, {0.5f, 0.5f, 0.5f})
        .Sphere({1, 0, 1}, 0.3f)
        .Circle({-1, 0, 1}, {0, 1, 0}, 0.3f)
        .Frame({0, 0, 0}, Quatd{}, 0.5f)
        .Arrow({0, 0, 0}, {1, 0, 0})
        .Capsule({0, 0, -1}, Quatd{}, 0.2f, 0.3f)
        .Label({0, 0, 1}, "label")
        .Text({.at = {0.1f, 0.1f}}, "text")
        .Triangle({0, 0, 0}, {1, 0, 0}, {0, 0, 1})
        .Polygon(square)
        .Curve(square)
        .Arc({0, 0, 0}, {0, 1, 0}, 1, 0, 1.5f)
        .Rectangle({0, 0, 0}, {0.5f, 0, 0}, {0, 0, 0.5f})
        .Mesh(square, counts, indices, {}, true)
        .Marker({0, 0, 0}, red);
    return overlay;
}

size_t At(uint32_t x, uint32_t y) {
    return (size_t(y) * kWidth + x) * 4;
}

std::vector<uint8_t> Drawn(Renderer& renderer) {
    RenderOutput output = renderer.Step(kProducts, 1.0 / 30);
    output.Wait();
    const FrameOutput& frame = output["/Render/View"].frames.at(0);
    if (frame.overlay.Empty()) {
        return {};
    }
    return frame.overlay.ToHost<uint8_t>();
}

TEST(OverlayTest, DrawsPicksAndHidesOverlays) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Empty(device);
        Renderer renderer(scene);
        Overlays& overlays = *renderer.GetOverlays();
        overlays.Set("shapes", EveryShape());
        const std::vector<uint8_t> drawn = Drawn(renderer);
        ASSERT_EQ(drawn.size(), size_t(kWidth) * kHeight * 4);
        const size_t center = (size_t(kHeight / 2) * kWidth + kWidth / 2) * 4;
        EXPECT_GT(drawn[center], 200);
        EXPECT_LT(drawn[center + 1], 80);

        OverlayPick pick = overlays.Pick("/Render/View", {kWidth / 2.0f, kHeight / 2.0f});
        Drawn(renderer);
        pick.Wait();
        EXPECT_TRUE(pick.Ready());
        const std::optional<OverlayHit> hit = pick.Hit();
        ASSERT_TRUE(hit);
        EXPECT_EQ(hit->overlay, "shapes");
        ASSERT_TRUE(pick.Ray());
        EXPECT_NEAR(pick.Ray()->origin.y, -5, 1e-3);

        Show(*scene, {});
        EXPECT_TRUE(Drawn(renderer).empty());
        Show(*scene, {"sha*"});
        EXPECT_GT(Drawn(renderer)[center], 200);
        overlays.Remove("shapes");
        const std::vector<uint8_t> removed = Drawn(renderer);
        EXPECT_TRUE(removed.empty() || removed[center + 3] == 0);
    });
}

TEST(OverlayTest, AnchorsShapesOnTheScreen) {
    ForEachDevice([](Device& device) {
        Renderer renderer(Empty(device));
        Overlays& overlays = *renderer.GetOverlays();
        const OverlayStyle corner{.color = {1, 0, 0, 1}, .screen = Float2{1, 1}};
        const OverlayStyle top{.color = {0, 1, 0, 1}, .size = 4, .screen = Float2{0, 0}};
        Overlay overlay;
        overlay.Rectangle({-5, -5, 0}, {3, 0, 0}, {0, 3, 0}, corner, true)
            .Marker({4, 4, 0}, top)
            .Line({0, 12, 0}, {10, 12, 0}, {.color = {0, 0, 1, 1}, .screen = Float2{0, 0}});
        overlays.Set("screen", overlay);
        const std::vector<uint8_t> drawn = Drawn(renderer);
        ASSERT_EQ(drawn.size(), size_t(kWidth) * kHeight * 4);
        const size_t red = At(kWidth - 5, kHeight - 5);
        EXPECT_GT(drawn[red], 200);
        EXPECT_GT(drawn[red + 3], 200);
        const size_t green = At(4, 4);
        EXPECT_GT(drawn[green + 1], 200);
        const size_t blue = At(5, 12);
        EXPECT_GT(drawn[blue + 2], 200);
        EXPECT_EQ(drawn[At(kWidth / 2, 4) + 3], 0);

        OverlayPick pick = overlays.Pick("/Render/View", {kWidth - 5.0f, kHeight - 5.0f});
        Drawn(renderer);
        pick.Wait();
        ASSERT_TRUE(pick.Hit());
        EXPECT_EQ(pick.Hit()->overlay, "screen");
    });
}

TEST(OverlayTest, RejectsScreenAnchorsThatNeedTheScene) {
    const OverlayStyle screen{.screen = Float2{0, 0}};
    const std::vector<uint32_t> prims = {0};
    Overlay overlay;
    EXPECT_THROW(overlay.Prims(prims, OverlayPrimShape::Bounds, screen), Error);
    EXPECT_THROW(overlay.Outline(prims, screen), Error);
    EXPECT_THROW(overlay.Text({.at = {0, 0}}, "text", screen), Error);
    OverlayStyle meters = screen;
    meters.units = OverlayUnits::Meters;
    EXPECT_THROW(overlay.Line({0, 0, 0}, {1, 0, 0}, meters), Error);
    const std::vector<Float4> colors = {{1, 0, 0, 1}, {0, 1, 0, 1}};
    EXPECT_THROW(overlay.Prims(prims, OverlayPrimShape::Bounds, {}, 1, {}, colors), Error);
}

TEST(OverlayTest, SizesPrimFramesInPixels) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Empty(device);
        scene->Define<Xform>("/Thing");
        const std::vector<uint32_t> prims = {scene->Find("/Thing")->Index()};
        Renderer renderer(scene);
        Overlay overlay;
        overlay.Prims(prims, OverlayPrimShape::Frames,
                      {.width = 2, .depth = OverlayDepth::Always, .screenSized = true}, 8);
        renderer.GetOverlays()->Set("frames", overlay);
        const std::vector<uint8_t> drawn = Drawn(renderer);
        ASSERT_EQ(drawn.size(), size_t(kWidth) * kHeight * 4);
        const size_t inside = At(kWidth / 2 + 5, kHeight / 2);
        EXPECT_GT(drawn[inside], 150);
        EXPECT_GT(drawn[inside + 3], 150);
        EXPECT_EQ(drawn[At(kWidth / 2 + 12, kHeight / 2) + 3], 0);
    });
}

TEST(OverlayTest, OutlinesOnlyTheSelectedSurfacesEdges) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Empty(device);
        scene->Define<Cube>("/Wall", {.translate = {0, 10, 0}, .scale = {50, 1, 50}});
        scene->Define<Cube>("/Box", {.size = 1});
        Renderer renderer(scene);
        const auto outlined = [&](std::string_view path) {
            const std::vector<uint32_t> prims = {scene->Find(path)->Index()};
            Overlay overlay;
            overlay.Outline(prims, {.color = {1, 0.5f, 0, 1}, .width = 2});
            renderer.GetOverlays()->Set("outline", overlay);
            const std::vector<uint8_t> drawn = Drawn(renderer);
            size_t count = 0;
            for (size_t at = 3; at < drawn.size(); at += 4) {
                count += drawn[at] > 0 ? 1 : 0;
            }
            return count;
        };
        EXPECT_EQ(outlined("/Wall"), 0u);
        EXPECT_GT(outlined("/Box"), 0u);
    });
}

TEST(OverlayTest, OutlinesSelectedPrimsInsideOthers) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Empty(device);
        scene->Define<Cube>("/Outer", {.size = 1});
        scene->Define<Cube>("/Inner", {.size = 0.4f});
        Renderer renderer(scene);
        const auto brightest = [&](std::string_view path) {
            const std::vector<uint32_t> prims = {scene->Find(path)->Index()};
            Overlay overlay;
            overlay.Outline(prims, {.color = {1, 0.5f, 0, 1}, .width = 2});
            renderer.GetOverlays()->Set("outline", overlay);
            const std::vector<uint8_t> drawn = Drawn(renderer);
            uint8_t most = 0;
            for (size_t at = 3; at < drawn.size(); at += 4) {
                most = std::max(most, drawn[at]);
            }
            return most;
        };
        const uint8_t inner = brightest("/Inner");
        EXPECT_GT(inner, 0);
        EXPECT_LT(inner, 160);
        EXPECT_GT(brightest("/Outer"), 200);
    });
}

TEST(OverlayTest, OutlinesThroughCamerasWithoutClippingPlanesOnCpuAdapters) {
    bool tested = false;
    for (const device::AdapterInfo& adapter : device::EnumerateAdapters()) {
        if (adapter.type != device::AdapterType::Cpu || !adapter.unsupported.empty()) {
            continue;
        }
        SCOPED_TRACE(adapter.alias);
        const Ref<Device> device = Device::Create({.adapter = adapter.alias});
        const Ref<Scene> scene = Empty(*device);
        scene->Define<Cube>("/Box", {.size = 1});
        Renderer renderer(scene);
        const std::vector<uint32_t> prims = {scene->Find("/Box")->Index()};
        Overlay overlay;
        overlay.Outline(prims, {.color = {1, 0.5f, 0, 1}, .width = 2});
        renderer.GetOverlays()->Set("outline", overlay);
        const std::vector<uint8_t> drawn = Drawn(renderer);
        size_t count = 0;
        for (size_t at = 3; at < drawn.size(); at += 4) {
            count += drawn[at] > 0 ? 1 : 0;
        }
        EXPECT_GT(count, 0u);
        device->GetQueue().WaitIdle();
        tested = true;
    }
    if (!tested) {
        GTEST_SKIP() << "no CPU adapter";
    }
}

}
}
