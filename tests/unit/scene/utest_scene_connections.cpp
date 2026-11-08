// SPDX-License-Identifier: BSD-3-Clause
#include <vector>

#include <gtest/gtest.h>

#include "ernie/scene/scene.h"
#include "ernie/util/error.h"
#include "support/devices.h"

namespace ernie {
namespace {

bool Same(const std::vector<Connection>& found, const std::vector<Connection>& expected) {
    if (found.size() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < found.size(); ++i) {
        if (found[i].input != expected[i].input || found[i].source != expected[i].source ||
            found[i].output != expected[i].output) {
            return false;
        }
    }
    return true;
}

TEST(ConnectionsTest, ConnectingAnInputAgainReplacesItAndDisconnectingRemovesIt) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim look = scene->Define<NodeGraph>("/Look");
        const Prim albedo = scene->Define<UsdUVTexture>("/Look/Albedo");
        const Prim uv = scene->Define<UsdPrimvarReader_float2>("/Look/Uv");
        const Prim placement = scene->Define<TexturePlacement>("/Look/Placement");
        const Prim moved = scene->Define<UsdTransform2d>("/Look/Moved", {.in = {1, 2}});
        EXPECT_EQ(moved.Get(UsdTransform2d::kIn), (Float2{1, 2}));

        albedo.Connect("st", uv, "result");
        look.Connect("surface", albedo, "rgb");
        EXPECT_TRUE(Same(albedo.Connections(), {{"st", "/Look/Uv", "result"}}));
        EXPECT_TRUE(Same(look.Connections(), {{"surface", "/Look/Albedo", "rgb"}}));

        albedo.Connect("st", placement, "st");
        EXPECT_TRUE(Same(albedo.Connections(), {{"st", "/Look/Placement", "st"}}));

        albedo.Disconnect("st");
        albedo.Disconnect("st");
        EXPECT_TRUE(albedo.Connections().empty());
        EXPECT_TRUE(Same(look.Connections(), {{"surface", "/Look/Albedo", "rgb"}}));
    });
}

TEST(ConnectionsTest, OnlyConnectablePrimsConnect) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim albedo = scene->Define<UsdUVTexture>("/Albedo");
        const Prim ball = scene->Define<Sphere>("/Ball");
        EXPECT_THROW(ball.Connect("st", albedo, "rgb"), Error);
        EXPECT_THROW(albedo.Connect("st", ball, "rgb"), Error);
        EXPECT_THROW(albedo.Connect("", albedo, "rgb"), Error);
        EXPECT_TRUE(albedo.Connections().empty());
    });
}

}
}
