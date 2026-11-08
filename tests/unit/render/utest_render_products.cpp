// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/render/render_output.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/scene.h"
#include "ernie/util/error.h"
#include "support/devices.h"

namespace ernie {
namespace {

using device::Device;
using test::ForEachDevice;

const std::vector<std::string> kProducts = {"/Render/View"};

Ref<Scene> OneCamera(Device& device) {
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<Camera>("/Cam");
    scene->Define<Sphere>("/Ball", {.translate = {0, 0, -5}});
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Depth", {.sourceName = RenderVarSource::DistanceToCamera});
    scene->Define<RenderProduct>(
        "/Render/View",
        {.camera = {"/Cam"}, .resolution = {4, 3}, .orderedVars = {"/Render/Depth"}});
    return scene;
}

struct Stepped {
    uint64_t address = 0;
    std::vector<int64_t> shape;
};

Stepped Step(Renderer& renderer) {
    RenderOutput output = renderer.Step(kProducts, 0);
    output.Wait();
    const device::Tensor& tensor =
        output["/Render/View"].frames.at(0).Source("DistanceToCamera").tensor;
    return {tensor.Address(), {tensor.Shape().begin(), tensor.Shape().end()}};
}

TEST(ProductsTest, EditsOutsideAProductKeepItsOutputs) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = OneCamera(device);
        Renderer renderer(scene);
        const uint64_t first = Step(renderer).address;
        scene->Define<Sphere>("/Other", {.translate = {0, 2, -5}});
        scene->Find("/Ball")->Set(Xformable::kTranslate, Double3{0, 0, -6});
        EXPECT_EQ(Step(renderer).address, first);
    });
}

TEST(ProductsTest, EditingAProductOrItsSensorPreparesItAgain) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = OneCamera(device);
        Renderer renderer(scene);
        EXPECT_EQ(Step(renderer).shape, (std::vector<int64_t>{1, 3, 4}));
        scene->Find("/Render/View")->Set(RenderProduct::kResolution, Uint2{2, 2});
        EXPECT_EQ(Step(renderer).shape, (std::vector<int64_t>{1, 2, 2}));
        scene->Define<Camera>("/Second");
        scene->Find("/Render/View")
            ->Set(RenderProduct::kCamera, std::vector<std::string>{"/Cam", "/Second"});
        EXPECT_EQ(Step(renderer).shape, (std::vector<int64_t>{2, 2, 2}));
    });
}

TEST(ProductsTest, ABrokenProductFailsOnlyWhenSteppedAndRecoversOnceFixed) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = OneCamera(device);
        scene->Define<RenderProduct>("/Render/Broken", {.orderedVars = {"/Render/Depth"}});
        Renderer renderer(scene);
        Step(renderer);
        EXPECT_THROW(renderer.Step(std::vector<std::string>{"/Render/Broken"}, 0), Error);
        scene->Find("/Render/Broken")
            ->Set(RenderProduct::kCamera, std::vector<std::string>{"/Cam"});
        RenderOutput output = renderer.Step(std::vector<std::string>{"/Render/Broken"}, 0);
        output.Wait();
        EXPECT_EQ(output["/Render/Broken"].frames.size(), 1u);
    });
}

}
}
