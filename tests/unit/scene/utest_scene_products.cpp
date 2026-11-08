// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/util/error.h"
#include "support/devices.h"

namespace ernie {
namespace {

Ref<Scene> TwoProducts(device::Device& device) {
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<Camera>("/Cam");
    scene->Define<Camera>("/Second");
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Depth", {.sourceName = "DistanceToCamera"});
    scene->Define<RenderVar>("/Render/Color", {.sourceName = "LdrColor"});
    scene->Define<RenderVar>("/Render/Ids", {.sourceName = "SemanticSegmentation"});
    scene->Define<RenderProduct>(
        "/Render/Depths",
        {.camera = {"/Cam", "/Second"}, .resolution = {4, 3}, .orderedVars = {"/Render/Depth"}});
    scene->Define<RenderProduct>("/Render/Colors",
                                 {.camera = {"/Cam"},
                                  .resolution = {2, 2},
                                  .orderedVars = {"/Render/Color", "/Render/Ids"}});
    scene->Publish().Wait();
    return scene;
}

TEST(ProductsTest, PlansGroupCapturesByEntryAndStayCachedUntilAProductChanges) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = TwoProducts(device);
        const ProductTable& products = SceneImpl::Get(*scene).GetProducts();
        ProductPlans& plans = SceneImpl::Get(*scene).GetPlans();
        const std::vector<std::string> paths = {"/Render/Depths", "/Render/Colors"};
        const auto plan = plans.Plan(paths);
        const auto& depths = plan->captures[uint32_t(RenderEntry::MinimalData)];
        ASSERT_EQ(depths.size(), 2u);
        EXPECT_EQ(depths[1].layer, 1u);
        EXPECT_EQ(depths[1].firstPixel, 12u);
        EXPECT_EQ(plan->captureCount, 3u);
        EXPECT_EQ(plan->pixelCount, 28u);
        EXPECT_EQ(plan->taxonomies.size(), 1u);
        const std::shared_ptr<const ResolvedProduct> colors =
            products.Find(scene->Find("/Render/Colors")->Index());
        EXPECT_TRUE(colors->semantic);
        EXPECT_EQ(colors->vars[0].shape, (std::vector<int64_t>{1, 2, 2, 4}));
        EXPECT_EQ(plan->captures[uint32_t(colors->entry)].size(), 1u);

        const std::vector<std::string> reordered = {"/Render/Colors", "/Render/Depths"};
        EXPECT_EQ(plans.Plan(reordered)->id, plan->id);
        scene->Find("/Render/Colors")->Set(RealTimeAPI::kMaxBounces, 3);
        scene->Publish().Wait();
        EXPECT_EQ(plans.Plan(paths)->id, plan->id);
        scene->Find("/Render/Colors")->Set(RenderProduct::kResolution, Uint2{3, 3});
        scene->Publish().Wait();
        const auto replanned = plans.Plan(paths);
        EXPECT_NE(replanned->id, plan->id);
        EXPECT_EQ(replanned->pixelCount, 33u);
        EXPECT_EQ(plan->products[0]->width, 2u);
    });
}

TEST(ProductsTest, PlanningAProductItsSensorCannotOutputThrows) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = TwoProducts(device);
        scene->Define<RenderVar>("/Render/Points", {.sourceName = "PointCloud"});
        scene->Find("/Render/Depths")
            ->Set(RenderProduct::kOrderedVars, std::vector<std::string>{"/Render/Points"});
        scene->Publish().Wait();
        const std::vector<std::string> paths = {"/Render/Depths"};
        EXPECT_THROW(SceneImpl::Get(*scene).GetPlans().Plan(paths), Error);
    });
}

TEST(ProductsTest, PlanningAVarNamingNoOutputThrows) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = TwoProducts(device);
        scene->Define<RenderVar>("/Render/Raw", {.sourceName = "raw"});
        scene->Find("/Render/Depths")
            ->Set(RenderProduct::kOrderedVars, std::vector<std::string>{"/Render/Raw"});
        scene->Publish().Wait();
        const std::vector<std::string> paths = {"/Render/Depths"};
        EXPECT_THROW(SceneImpl::Get(*scene).GetPlans().Plan(paths), Error);
    });
}

TEST(ProductsTest, PublishingAgainInAStepKeepsItsCapturesDue) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = TwoProducts(device);
        scene->Find("/Cam")->Set(SensorAPI::kSensorFrameRate, 10.0f);
        scene->Find("/Second")->Set(SensorAPI::kSensorFrameRate, 10.0f);
        scene->Publish({.dt = 0.1}).Wait();
        const CaptureSchedule& captures = SceneImpl::Get(*scene).GetCaptures();
        const uint32_t depths = scene->Find("/Render/Depths")->Index();
        EXPECT_TRUE(captures.Due(depths));
        scene->Find("/Render/Depths")->Set(RealTimeAPI::kMaxBounces, 2);
        scene->Publish().Wait();
        EXPECT_TRUE(captures.Due(depths));
        scene->Publish({.dt = 0.05}).Wait();
        EXPECT_FALSE(captures.Due(depths));
    });
}

TEST(ProductsTest, OnlyCameraProductsDrawOverlays) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = TwoProducts(device);
        scene->Define<Radar>("/Radar");
        const Prim depths = *scene->Find("/Render/Depths");
        depths.Set(RenderProduct::kOverlays, std::vector<std::string>{"*"});
        scene->Publish().Wait();
        const std::vector<std::string> paths = {"/Render/Depths"};
        ProductPlans& plans = SceneImpl::Get(*scene).GetPlans();
        EXPECT_EQ(plans.Plan(paths)->products.size(), 1u);
        EXPECT_EQ(SceneImpl::Get(*scene).GetProducts().Find(depths.Index())->overlays,
                  std::vector<std::string>{"*"});
        depths.Set(RenderProduct::kCamera, std::vector<std::string>{"/Radar"});
        scene->Publish().Wait();
        EXPECT_THROW(plans.Plan(paths), Error);
    });
}

}
}
