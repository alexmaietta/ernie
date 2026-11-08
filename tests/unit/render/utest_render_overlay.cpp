// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/overlay/overlay.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/render/renderer.h"
#include "ernie/scene/rotation.h"
#include "ernie/scene/scene.h"
#include "support/devices.h"

namespace ernie {
namespace {

using device::Device;
using test::ForEachDevice;

constexpr uint32_t kWidth = 32;
constexpr uint32_t kHeight = 24;
const std::vector<std::string> kProducts = {"/Render/View"};

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
        Renderer renderer(Empty(device));
        renderer.SetOverlay("shapes", EveryShape());
        renderer.ShowOverlays("/Render/View");
        const std::vector<uint8_t> drawn = Drawn(renderer);
        ASSERT_EQ(drawn.size(), size_t(kWidth) * kHeight * 4);
        const size_t center = (size_t(kHeight / 2) * kWidth + kWidth / 2) * 4;
        EXPECT_GT(drawn[center], 200);
        EXPECT_LT(drawn[center + 1], 80);

        OverlayPick pick = renderer.PickOverlay("/Render/View", {kWidth / 2.0f, kHeight / 2.0f});
        Drawn(renderer);
        pick.Wait();
        EXPECT_TRUE(pick.Ready());
        const std::optional<OverlayHit> hit = pick.Hit();
        ASSERT_TRUE(hit);
        EXPECT_EQ(hit->overlay, "shapes");
        ASSERT_TRUE(pick.Ray());
        EXPECT_NEAR(pick.Ray()->origin.y, -5, 1e-3);

        renderer.HideOverlays("/Render/View");
        const std::vector<uint8_t> hidden = Drawn(renderer);
        EXPECT_TRUE(hidden.empty() || hidden[center + 3] == 0);
        renderer.ShowOverlays("/Render/View", std::vector<std::string>{"shapes"});
        renderer.RemoveOverlay("shapes");
        const std::vector<uint8_t> removed = Drawn(renderer);
        EXPECT_TRUE(removed.empty() || removed[center + 3] == 0);
    });
}

}
}
