// SPDX-License-Identifier: BSD-3-Clause
/// @file lux_test.cpp
/// @brief Tests of lights' and cameras' schemas and extents.
#include "ernie/scene/schemas/camera.h"
#include "ernie/scene/schemas/collection.h"
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

TEST(Lights, EveryLightHasLightApiAndItsLinksBuiltIn) {
    Scene scene;
    const SphereLight lamp = SphereLight::Define(scene, "/Lamp");
    EXPECT_TRUE(scene.HasApi(lamp.Id(), "LightAPI"));
    EXPECT_TRUE(scene.HasApi(lamp.Id(), "CollectionAPI:lightLink"));
    EXPECT_TRUE(scene.AppliedApis(lamp.Id()).empty()) << "built in, not applied";
    EXPECT_EQ(LightAPI(scene, "/Lamp").GetIntensity(), 1.0);
    EXPECT_TRUE(CollectionAPI(scene, "/Lamp", "lightLink").GetIncludeRoot())
        << "a light lights everything unless linked otherwise";
    EXPECT_FALSE(CollectionAPI::Apply(scene, "/Lamp", "props").GetIncludeRoot())
        << "other collections keep CollectionAPI's fallback";
    EXPECT_EQ(LightAPI(scene, DistantLight::Define(scene, "/Sun").GetPath()).GetIntensity(),
              50000.0);
}

TEST(Lights, ErnieAttributesSitBesideUsdLuxs) {
    Scene scene;
    SphereLight strobe = SphereLight::Define(scene, "/Strobe");
    strobe.SetPulseRate(30.0).SetPulseTrigger(PulseTrigger::Exposure).SetPower(100.0);
    strobe.SetSpectrum({400.0f, 1.0f, 700.0f, 0.5f});
    EXPECT_EQ(scene.Get(strobe.Id(), "ernie:pulse:trigger"), Value("exposure"));
    EXPECT_EQ(scene.Get(strobe.Id(), "ernie:power"), Value(100.0));
    EXPECT_EQ(strobe.GetSpectrum().size(), 4u);
    EXPECT_EQ(DistantLight::Define(scene, "/Sun").GetIrradiance(), 1361.0);
    EXPECT_THROW(strobe.SetPolarizationDegree(2.0), std::invalid_argument);
}

TEST(Camera, ErnieApiSchemasAddLensPolarizerAndSensor) {
    Scene scene;
    Camera camera = Camera::Define(scene, "/Camera");
    EXPECT_THROW(ErnieSensorAPI(scene, "/Camera"), std::invalid_argument) << "not applied";
    ErnieSensorAPI sensor = ErnieSensorAPI::Apply(scene, "/Camera");
    EXPECT_EQ(sensor.GetSpectralBands(), 16);
    sensor.SetSensorType("mono");
    EXPECT_THROW(sensor.SetSensorType("film"), std::invalid_argument);
    ErnieLensAPI::Apply(scene, "/Camera").SetLensProjection("fisheye");
    EXPECT_EQ(ErniePolarizerAPI::Apply(scene, "/Camera").GetPolarizerExtinction(), 1000.0);
    EXPECT_EQ(camera.GetFocalLength(), 0.05) << "USD's attributes stay";
    camera.SetShutterRolling(0.5).SetWorld(3);
    EXPECT_EQ(camera.GetWorld(), 3) << "a batch's views are cameras, each in its world";
}

TEST(Camera, HasUsdsFallbacksInMetresAndNoExtent) {
    Scene scene;
    const Camera camera = Camera::Define(scene, "/Camera");
    EXPECT_EQ(camera.GetFocalLength(), 0.05) << "50 mm, whatever the stage's units";
    EXPECT_EQ(camera.GetHorizontalAperture(), 0.020955);
    EXPECT_EQ(camera.GetClippingRange(), glm::dvec2(1.0, 1000000.0));
    EXPECT_EQ(camera.GetProjection(), "perspective");
    EXPECT_FALSE(scene.Extent(camera.Id()));
}

}  // namespace
}  // namespace ernie
