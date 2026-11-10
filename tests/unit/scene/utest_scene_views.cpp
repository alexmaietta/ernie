// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/scene/tables/views.h"
#include "support/devices.h"

namespace ernie {
namespace {

std::vector<ViewRecord> Views(Scene& scene, device::Device& device) {
    scene.Publish({.wait = true});
    const ViewTable& views = SceneImpl::Get(scene).GetViews();
    std::vector<ViewRecord> records(views.Count());
    if (!records.empty()) {
        device.GetQueue().Read(views.Views()->Slice(0, records.size() * sizeof(ViewRecord)),
                               records.data());
    }
    return records;
}

uint32_t Start(Scene& scene, device::Device& device, const std::string& product) {
    const ViewTable& views = SceneImpl::Get(scene).GetViews();
    const uint32_t row = SceneImpl::Get(scene).GetStructure().GetPrims().Row(
        scene.Find(product)->Index(), SchemaId::RenderProduct);
    uint32_t start = 0;
    device.GetQueue().Read(views.Starts()->Slice(row * sizeof(uint32_t), sizeof(uint32_t)), &start);
    return start;
}

Prim DefineCamera(Scene& scene, const std::string& path) {
    const Prim camera = scene.Define<Camera>(path);
    camera.Set(Camera::kFocalLength, 25.0f);
    camera.Set(Camera::kHorizontalAperture, 36.0f);
    camera.Set(Camera::kVerticalAperture, 24.0f);
    camera.Set(Camera::kHorizontalApertureOffset, 2.0f);
    camera.Set(Camera::kVerticalApertureOffset, -1.0f);
    camera.Set(Camera::kClippingRange, Float2{0.5f, 200});
    camera.Set(Camera::kShutterOpen, 0.1f);
    camera.Set(Camera::kShutterClose, 0.3f);
    camera.Set(Camera::kShutterReadout, 0.01f);
    return camera;
}

TEST(ViewsTest, ViewsHoldEachCamerasConformedPinholeTerms) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<Scope>("/Render");
        DefineCamera(*scene, "/Cam");
        const Prim product = scene->Define<RenderProduct>(
            "/Render/Wide", {.camera = {"/Cam"}, .resolution = {200, 100}});
        struct Case {
            AspectRatioConformPolicy policy;
            Float2 aperture;
        };
        for (const Case& expected :
             {Case{AspectRatioConformPolicy::ExpandAperture, {48, 24}},
              Case{AspectRatioConformPolicy::CropAperture, {36, 18}},
              Case{AspectRatioConformPolicy::AdjustApertureWidth, {48, 24}},
              Case{AspectRatioConformPolicy::AdjustApertureHeight, {36, 18}},
              Case{AspectRatioConformPolicy::AdjustPixelAspectRatio, {36, 24}}}) {
            SCOPED_TRACE(int(expected.policy));
            product.Set(RenderProduct::kAspectRatioConformPolicy, expected.policy);
            const std::vector<ViewRecord> views = Views(*scene, device);
            ASSERT_EQ(views.size(), 1u);
            const ViewRecord& view = views[0];
            EXPECT_FLOAT_EQ(view.scale.x, 0.5f * expected.aperture.x / 25);
            EXPECT_FLOAT_EQ(view.scale.y, 0.5f * expected.aperture.y / 25);
            EXPECT_FLOAT_EQ(view.bias.x, 2.0f / 25);
            EXPECT_FLOAT_EQ(view.bias.y, -1.0f / 25);
            EXPECT_FLOAT_EQ(view.spread, expected.aperture.y / (100 * 25.0f));
            EXPECT_FLOAT_EQ(view.clipping.x, 0.5f);
            EXPECT_FLOAT_EQ(view.clipping.y, 200);
            EXPECT_FLOAT_EQ(view.shutter, 0.2f);
            EXPECT_FLOAT_EQ(view.readout, 0.01f);
            EXPECT_GT(view.exposure, 0);
            EXPECT_EQ(view.flags, kViewPinhole);
        }
    });
}

TEST(ViewsTest, OnlyUndistortedPerspectiveCamerasArePinholes) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<Scope>("/Render");
        DefineCamera(*scene, "/Plain");
        DefineCamera(*scene, "/Flat").Set(Camera::kProjection, Projection::Orthographic);
        DefineCamera(*scene, "/Bent")
            .Set(LensDistortionAPI::kDistortionModel, DistortionModel::BrownConrady);
        DefineCamera(*scene, "/Cut")
            .Set(Camera::kClippingPlanes, std::vector<Float4>{{0, 0, 1, 0}});
        DefineCamera(*scene, "/Straight")
            .Set(LensDistortionAPI::kDistortionModel, DistortionModel::None);
        scene->Define<RenderProduct>("/Render/A", {.camera = {"/Plain", "/Flat"}});
        scene->Define<RenderProduct>("/Render/B", {.camera = {"/Bent", "/Cut", "/Straight"}});
        const std::vector<ViewRecord> views = Views(*scene, device);
        ASSERT_EQ(views.size(), 5u);
        const uint32_t a = Start(*scene, device, "/Render/A");
        const uint32_t b = Start(*scene, device, "/Render/B");
        EXPECT_EQ(std::min(a, b), 0u);
        EXPECT_EQ(std::max(a, b), a == 0 ? 2u : 3u);
        EXPECT_EQ(views[a].flags, kViewPinhole);
        EXPECT_EQ(views[a + 1].flags, 0u);
        EXPECT_EQ(views[b].flags, 0u);
        EXPECT_EQ(views[b + 1].flags, kViewPinhole);
        EXPECT_EQ(views[b + 1].clippingPlanes.count, 1u);
        EXPECT_EQ(views[b + 2].flags, kViewPinhole);
        EXPECT_EQ(views[b + 2].clippingPlanes.count, 0u);
    });
}

TEST(ViewsTest, ViewsRewriteForCameraAndProductEditsButNotCameraMoves) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        scene->Define<Scope>("/Render");
        const Prim camera = DefineCamera(*scene, "/Cam");
        const Prim product = scene->Define<RenderProduct>(
            "/Render/View", {.camera = {"/Cam"}, .resolution = {64, 64}});
        const ViewTable& table = SceneImpl::Get(*scene).GetViews();
        const float spread = Views(*scene, device)[0].spread;
        const uint32_t runs = table.Runs();

        camera.Set(Xformable::kTranslate, Double3{1, 2, 3});
        Views(*scene, device);
        EXPECT_EQ(table.Runs(), runs);

        camera.Set(Camera::kFocalLength, 50.0f);
        EXPECT_FLOAT_EQ(Views(*scene, device)[0].spread, spread / 2);
        EXPECT_EQ(table.Runs(), runs + 1);

        product.Set(RenderProduct::kResolution, Uint2{64, 32});
        EXPECT_NE(Views(*scene, device)[0].spread, spread / 2);
        EXPECT_EQ(table.Runs(), runs + 2);
    });
}

}
}
