// SPDX-License-Identifier: BSD-3-Clause
/// @file lux_test.cpp
/// @brief Tests of lights' and cameras' schemas and extents.
#include "ernie/scene/schemas/camera.h"
#include "ernie/scene/schemas/lux.h"
#include "ernie/scene/xform.h"

#include "tests/common.h"

#include <gtest/gtest.h>

namespace ernie {
namespace {

using test::BoxNear;

TEST(Lights, ShapedLightsAreBoundedAsUsdLuxComputesThem) {
    Scene scene;
    EXPECT_TRUE(BoxNear(*scene.Extent(SphereLight::Define(scene, "/Sphere").SetRadius(2.0).Id()),
                        glm::dvec3(-2.0), glm::dvec3(2.0)));
    EXPECT_TRUE(BoxNear(*scene.Extent(DiskLight::Define(scene, "/Disk").Id()), {-0.5, -0.5, 0.0},
                        {0.5, 0.5, 0.0}));
    EXPECT_TRUE(BoxNear(*scene.Extent(RectLight::Define(scene, "/Rect").SetWidth(4.0).Id()),
                        {-2.0, -0.5, 0.0}, {2.0, 0.5, 0.0}));
    EXPECT_TRUE(BoxNear(*scene.Extent(CylinderLight::Define(scene, "/Tube").SetLength(6.0).Id()),
                        {-3.0, -0.5, -0.5}, {3.0, 0.5, 0.5}));
    EXPECT_TRUE(BoxNear(*scene.Extent(PortalLight::Define(scene, "/Window").SetHeight(2.0).Id()),
                        {-0.5, -1.0, 0.0}, {0.5, 1.0, 0.0}));
}

TEST(Lights, DistantAndDomeLightsAreLeftOutOfFraming) {
    Scene scene;
    Xform::Define(scene, "/World");
    DistantLight sun = DistantLight::Define(scene, "/World/Sun");
    DomeLight sky = DomeLight::Define(scene, "/World/Sky");
    SphereLight::Define(scene, "/World/Lamp");
    SetXform(scene, "/World/Lamp", {.translate = glm::dvec3(0.0, 0.0, 3.0)});
    EXPECT_FALSE(scene.Extent(sun.Id()));
    EXPECT_FALSE(scene.Extent(sky.Id()));
    EXPECT_EQ(sun.Schema().Extent().kind, ExtentRule::Kind::Unbounded);
    EXPECT_TRUE(BoxNear(scene.WorldBounds("/World"), {-0.5, -0.5, 2.5}, {0.5, 0.5, 3.5}));
}

TEST(Lights, FallbacksAreUsdLuxs) {
    Scene scene;
    EXPECT_EQ(DistantLight::Define(scene, "/Sun").GetIntensity(), 50000.0) << "its own fallback";
    EXPECT_EQ(SphereLight::Define(scene, "/Lamp").GetIntensity(), 1.0);
    EXPECT_EQ(DistantLight(scene, "/Sun").GetAngle(), 0.53);
    EXPECT_EQ(SphereLight(scene, "/Lamp").GetColorTemperature(), 6500.0);
    EXPECT_EQ(DomeLight::Define(scene, "/Sky").GetTextureFormat(), "automatic");
}

TEST(Lights, ShapingAndShadowsAreAppliedApiSchemas) {
    Scene scene;
    SphereLight::Define(scene, "/Spot");
    ShapingAPI cone = ShapingAPI::Apply(scene, "/Spot");
    EXPECT_EQ(cone.GetShapingConeAngle(), 90.0);
    cone.SetShapingConeAngle(30.0);
    EXPECT_EQ(scene.Get("/Spot", "inputs:shaping:cone:angle"), Value(30.0));
    EXPECT_TRUE(ShadowAPI::Apply(scene, "/Spot").GetShadowEnable());
    EXPECT_EQ(SphereLight(scene, "/Spot").GetRadius(), 0.5) << "the light's own attributes stay";
}

TEST(Camera, HasUsdsFallbacksAndNoExtent) {
    Scene scene;
    const Camera camera = Camera::Define(scene, "/Camera");
    EXPECT_EQ(camera.GetFocalLength(), 50.0);
    EXPECT_EQ(camera.GetHorizontalAperture(), 20.955);
    EXPECT_EQ(camera.GetClippingRange(), glm::dvec2(1.0, 1000000.0));
    EXPECT_EQ(camera.GetProjection(), "perspective");
    EXPECT_FALSE(scene.Extent(camera.Id()));
}

}  // namespace
}  // namespace ernie
