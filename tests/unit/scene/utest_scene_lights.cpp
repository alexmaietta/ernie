// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/io/image.h"
#include "ernie/scene/kernels/lights.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/scene_impl.h"
#include "support/devices.h"

namespace ernie {
namespace {

template <class Test> void ForEachScene(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        test(*scene, device);
    });
}

std::vector<LightRecord> Records(Scene& scene, device::Device& device) {
    scene.Publish({.wait = true});
    const Lights& lights = SceneImpl::Get(scene).GetLights();
    uint32_t total = 0;
    for (const uint32_t count : lights.Counts()) {
        total += count;
    }
    std::vector<LightRecord> records(total);
    if (total > 0) {
        device.GetQueue().Read(lights.Records()->Slice(0, total * sizeof(LightRecord)),
                               records.data());
    }
    return records;
}

LightRecord Find(const std::vector<LightRecord>& records, const Prim& prim) {
    for (const LightRecord& record : records) {
        if (record.prim == prim.Index()) {
            return record;
        }
    }
    ADD_FAILURE() << "no record for " << prim.Path();
    return {};
}

float Luminance(const Float3& rgb) {
    return 0.2126729f * rgb.x + 0.7151522f * rgb.y + 0.0721750f * rgb.z;
}

constexpr float kPi = 3.14159265358979f;

TEST(LightsTest, RecordsResolveEmissionShapeAndFrame) {
    ForEachScene([](Scene& scene, device::Device& device) {
        const Prim sun = scene.Define<DistantLight>(
            "/Sun", {.intensity = 2, .exposure = 1, .color = {1, 0.5f, 0.25f}, .angle = 1});
        const Prim lamp = scene.Define<SphereLight>(
            "/Lamp", {.translate = {1, 2, 3}, .scale = {2, 2, 2}, .radius = 0.25f});
        const std::vector<LightRecord> records = Records(scene, device);
        ASSERT_EQ(records.size(), 2u);
        const LightRecord& distant = records[0];
        EXPECT_EQ(distant.prim, sun.Index());
        EXPECT_EQ(LightType(distant.type), LightType::Distant);
        EXPECT_FLOAT_EQ(distant.radiance.x, 4);
        EXPECT_FLOAT_EQ(distant.radiance.y, 2);
        EXPECT_FLOAT_EQ(distant.radiance.z, 1);
        EXPECT_NEAR(distant.cosAngle, std::cos(0.5 * 3.14159265358979 / 180), 1e-4);
        EXPECT_FLOAT_EQ(distant.axisZ.z, 1);

        const LightRecord& sphere = records[1];
        EXPECT_EQ(sphere.prim, lamp.Index());
        EXPECT_EQ(LightType(sphere.type), LightType::Sphere);
        EXPECT_FLOAT_EQ(sphere.radius, 0.5f);
        EXPECT_FLOAT_EQ(sphere.position.x, 1);
        EXPECT_FLOAT_EQ(sphere.position.z, 3);
        EXPECT_FLOAT_EQ(sphere.radiance.x, 1);
        EXPECT_GT(sphere.power, 0);
    });
}

TEST(LightsTest, WorldsListTheirOwnLightsThenGlobalOnes) {
    ForEachScene([](Scene& scene, device::Device& device) {
        scene.Define<DistantLight>("/Sun");
        const Prim world = scene.Define<World>("/Env");
        scene.Define<SphereLight>("/Env/Lamp", {.intensity = 3});
        scene.Clone(world, std::vector<std::string>{"/Env1"});
        const std::vector<LightRecord> records = Records(scene, device);
        const Lights& lights = SceneImpl::Get(scene).GetLights();
        ASSERT_EQ(lights.Counts().size(), 3u);
        EXPECT_EQ(lights.Counts()[0], 2u);
        EXPECT_EQ(lights.Counts()[1], 2u);
        EXPECT_EQ(lights.Counts()[2], 1u);
        EXPECT_EQ(LightType(records[0].type), LightType::Sphere);
        EXPECT_EQ(LightType(records[1].type), LightType::Distant);
        EXPECT_EQ(LightType(records[4].type), LightType::Distant);

        const uint32_t runs = lights.Runs();
        scene.Publish({.wait = true});
        EXPECT_EQ(lights.Runs(), runs);

        scene.Find("/Env1/Lamp")->Set(LightAPI::kIntensity, 5.0f);
        const std::vector<LightRecord> edited = Records(scene, device);
        EXPECT_EQ(lights.Written(), 2u);
        EXPECT_FLOAT_EQ(edited[2].radiance.x, 5);
        EXPECT_FLOAT_EQ(edited[0].radiance.x, 3);
    });
}

TEST(LightsTest, AppliedLightApiAddsALightAndRemovingDropsIt) {
    ForEachScene([](Scene& scene, device::Device& device) {
        const Prim ball = scene.Define<Sphere>("/Ball");
        EXPECT_TRUE(Records(scene, device).empty());
        scene.Apply<LightAPI>(ball, {.intensity = 7});
        const std::vector<LightRecord> records = Records(scene, device);
        ASSERT_EQ(records.size(), 1u);
        EXPECT_EQ(LightType(records[0].type), LightType::Unsupported);
        EXPECT_FLOAT_EQ(records[0].radiance.x, 7);
        scene.Remove(ball);
        EXPECT_TRUE(Records(scene, device).empty());
    });
}

TEST(LightSchemasTest, AreaLightsTakeShapingShadowAndPhotometricApis) {
    ForEachScene([](Scene& scene, device::Device&) {
        const RectLightPrim panel = scene.Define<RectLight>(
            "/Panel",
            {.width = 2,
             .height = 0.5f,
             .shaping = ShapingAPI::Desc{.shapingFocus = 4, .shapingConeAngle = 30},
             .photometricAreaLight = PhotometricAreaLightAPI::Desc{
                 .physicalIlluminant = PhysicalIlluminant::Blackbody, .photometricPower = 800}});
        EXPECT_EQ(panel.Width(), 2.0f);
        EXPECT_TRUE(panel.HasApi<ShapingAPI>());
        EXPECT_TRUE(panel.HasApi<OmniShapingAPI>());
        EXPECT_TRUE(panel.HasApi<PhotometricAreaLightAPI>());
        EXPECT_TRUE(panel.HasApi<PhysicalLightIlluminantAPI>());
        EXPECT_FALSE(panel.HasApi<ShadowAPI>());
        EXPECT_EQ(panel.Get(ShapingAPI::kShapingConeAngle), 30.0f);
        EXPECT_EQ(panel.Get(ShapingAPI::kShapingConeSoftness), 0.0f);
        EXPECT_EQ(panel.Get(OmniShapingAPI::kRtxShapingForwardAxis), ShapingAxis::MinusZ);
        EXPECT_EQ(panel.Get(PhotometricAreaLightAPI::kPhotometricPower), 800.0f);
        EXPECT_EQ(panel.Get(PhysicalLightIlluminantAPI::kPhysicalIlluminant),
                  PhysicalIlluminant::Blackbody);
        EXPECT_THROW(scene.Unapply<PhysicalLightIlluminantAPI>(panel), Error);
        EXPECT_THROW(scene.Unapply<OmniShapingAPI>(panel), Error);

        const Prim sun = scene.Define<DistantLight>("/Sun");
        EXPECT_FALSE(sun.HasApi<OmniShapingAPI>());
        EXPECT_THROW(scene.Apply<ShapingAPI>(sun), Error);
        EXPECT_THROW(scene.Apply<PhotometricAreaLightAPI>(sun), Error);
        scene.Apply<ShadowAPI>(sun, {.shadowColor = {0.1f, 0.2f, 0.3f}});
        EXPECT_EQ(sun.Get(ShadowAPI::kShadowColor), (Float3{0.1f, 0.2f, 0.3f}));
        scene.Apply<PhotometricDistantLightAPI>(sun);
        EXPECT_EQ(sun.Get(PhotometricDistantLightAPI::kPhotometricIlluminance), 10000.0f);
        EXPECT_EQ(sun.Get(PhysicalLightIlluminantAPI::kPhysicalIlluminant),
                  PhysicalIlluminant::White);
    });
}

TEST(LightSchemasTest, PhotometricApisKeepAnIlluminantAppliedBefore) {
    ForEachScene([](Scene& scene, device::Device&) {
        const Prim disk = scene.Define<DiskLight>("/Disk", {.radius = 0.1f});
        scene.Apply<PhysicalLightIlluminantAPI>(
            disk, {.physicalIlluminant = PhysicalIlluminant::IlluminantD,
                   .physicalCustomIlluminant = {{400, 1}, {500, 2}, {600, 0.5f}}});
        scene.Apply<PhotometricAreaLightAPI>(disk, {.photometricPower = 100});
        EXPECT_EQ(disk.Get(PhotometricAreaLightAPI::kPhotometricPower), 100.0f);
        EXPECT_EQ(disk.Get(PhysicalLightIlluminantAPI::kPhysicalIlluminant),
                  PhysicalIlluminant::IlluminantD);
        const std::vector<Float2> custom =
            disk.Get(PhysicalLightIlluminantAPI::kPhysicalCustomIlluminant);
        ASSERT_EQ(custom.size(), 3u);
        EXPECT_EQ(custom[1].x, 500.0f);
        EXPECT_EQ(custom[1].y, 2.0f);

        scene.Unapply<PhotometricAreaLightAPI>(disk);
        EXPECT_TRUE(disk.HasApi<PhysicalLightIlluminantAPI>());
        scene.Unapply<PhysicalLightIlluminantAPI>(disk);
        EXPECT_FALSE(disk.HasApi<PhysicalLightIlluminantAPI>());

        const Prim tube = scene.Define<CylinderLight>("/Tube", {.length = 2, .treatAsLine = true});
        EXPECT_EQ(tube.Get(CylinderLight::kLength), 2.0f);
        EXPECT_TRUE(tube.Get(CylinderLight::kTreatAsLine));
        EXPECT_TRUE(tube.Get(LightAPI::kVisibleInPrimaryRay));
    });
}

TEST(LightEmissionTest, IlluminantsKeepLuminanceAndSetChromaticity) {
    ForEachScene([](Scene& scene, device::Device& device) {
        const Prim plain = scene.Define<SphereLight>("/Plain", {.intensity = 10});
        const Prim warm = scene.Define<SphereLight>(
            "/Warm", {.intensity = 10, .enableColorTemperature = true, .colorTemperature = 3000});
        const Prim neutral = scene.Define<SphereLight>(
            "/Neutral",
            {.intensity = 10, .enableColorTemperature = true, .colorTemperature = 6500});
        const Prim daylight = scene.Define<SphereLight>(
            "/Daylight", {.intensity = 10,
                          .colorTemperature = 6504,
                          .physicalLightIlluminant = PhysicalLightIlluminantAPI::Desc{
                              .physicalIlluminant = PhysicalIlluminant::IlluminantD}});
        const Prim candle = scene.Define<SphereLight>(
            "/Candle", {.intensity = 10,
                        .colorTemperature = 1900,
                        .physicalLightIlluminant = PhysicalLightIlluminantAPI::Desc{
                            .physicalIlluminant = PhysicalIlluminant::Blackbody}});
        const Prim custom = scene.Define<SphereLight>(
            "/Custom", {.intensity = 10,
                        .physicalLightIlluminant = PhysicalLightIlluminantAPI::Desc{
                            .physicalIlluminant = PhysicalIlluminant::Custom,
                            .physicalCustomIlluminant = {{600, 1}, {700, 1}}}});
        const std::vector<LightRecord> records = Records(scene, device);

        const Float3 white = Find(records, plain).radiance;
        EXPECT_FLOAT_EQ(white.x, 10);
        EXPECT_FLOAT_EQ(white.z, 10);
        for (const Prim& prim : {warm, neutral, daylight, candle, custom}) {
            EXPECT_NEAR(Luminance(Find(records, prim).radiance), 10, 1e-3) << prim.Path();
        }
        const Float3 d65 = Find(records, daylight).radiance;
        EXPECT_NEAR(d65.x, 10, 0.05);
        EXPECT_NEAR(d65.z, 10, 0.05);
        const Float3 neutralRgb = Find(records, neutral).radiance;
        EXPECT_NEAR(neutralRgb.x, 10, 1e-3);
        EXPECT_NEAR(neutralRgb.z, 10, 1e-3);
        const Float3 warmRgb = Find(records, warm).radiance;
        EXPECT_GT(warmRgb.x, warmRgb.y);
        EXPECT_GT(warmRgb.y, warmRgb.z);
        const Float3 candleRgb = Find(records, candle).radiance;
        EXPECT_GT(candleRgb.x / candleRgb.y, warmRgb.x / warmRgb.y);
        const Float3 red = Find(records, custom).radiance;
        EXPECT_GT(red.x, 2 * red.y);
        for (const LightRecord& record : records) {
            EXPECT_GE(record.radiance.x, 0);
            EXPECT_GE(record.radiance.y, 0);
            EXPECT_GE(record.radiance.z, 0);
        }
    });
}

TEST(LightEmissionTest, PhotometricApisSetLevelsInLumensAndLux) {
    ForEachScene([](Scene& scene, device::Device& device) {
        const Prim sun = scene.Define<DistantLight>(
            "/Sun", {.photometricDistantLight =
                         PhotometricDistantLightAPI::Desc{.photometricIlluminance = 10000}});
        const Prim panel = scene.Define<RectLight>(
            "/Panel",
            {.color = {1, 0, 0},
             .width = 2,
             .height = 1,
             .photometricAreaLight = PhotometricAreaLightAPI::Desc{.photometricPower = 1000}});
        const Prim bulb = scene.Define<SphereLight>(
            "/Bulb", {.radius = 0.5f, .photometricAreaLight = PhotometricAreaLightAPI::Desc{}});
        const Prim spot =
            scene.Define<DiskLight>("/Spot", {.radius = 0.1f,
                                              .photometricAreaLight = PhotometricAreaLightAPI::Desc{
                                                  .photometricPower = 0,
                                                  .photometricIlluminance = 500,
                                                  .photometricIlluminanceDistance = 2}});
        const Prim cone = scene.Define<DiskLight>(
            "/Cone",
            {.radius = 0.1f,
             .shaping = ShapingAPI::Desc{.shapingConeAngle = 30},
             .photometricAreaLight = PhotometricAreaLightAPI::Desc{.photometricPower = 100}});
        const Prim halved = scene.Define<RectLight>(
            "/Halved", {.intensity = 10, .normalize = true, .width = 2, .height = 1});
        const std::vector<LightRecord> records = Records(scene, device);

        const double quarter = std::sin(0.53 * 3.14159265358979 / 720);
        const float disc = float(4 * 3.14159265358979 * quarter * quarter);
        EXPECT_NEAR(Luminance(Find(records, sun).radiance) * disc / 10000, 1, 1e-3);
        const Float3 red = Find(records, panel).radiance;
        EXPECT_NEAR(Luminance(red), 1000 / (2 * kPi), 1e-2);
        EXPECT_FLOAT_EQ(red.y, 0);
        EXPECT_NEAR(Luminance(Find(records, bulb).radiance), 1600 / (kPi * 0.25f * 4 * kPi), 1e-2);
        EXPECT_NEAR(Luminance(Find(records, spot).radiance), 500 * 4 / (kPi * 0.01f), 1);
        const float hemisphere = 100 / (kPi * 0.01f * kPi);
        const float narrowed = 1 - std::pow(std::cos(kPi / 6), 2.0f);
        EXPECT_NEAR(Luminance(Find(records, cone).radiance) / (hemisphere / narrowed), 1, 0.02);
        EXPECT_NEAR(Find(records, cone).coneAngle, kPi / 6, 1e-5);
        EXPECT_TRUE((Find(records, cone).flags & kLightShaped) != 0);
        EXPECT_FLOAT_EQ(Find(records, halved).radiance.x, 5);
        EXPECT_TRUE((Find(records, halved).flags & kLightVisible) != 0);
    });
}

TEST(LightsTest, RecordsPointAtCollectionsThatLimitALight) {
    ForEachScene([](Scene& scene, device::Device& device) {
        const Prim sun = scene.Define<DistantLight>("/Sun");
        scene.Define<Cube>("/Blocker");
        LightRecord record = Records(scene, device)[0];
        EXPECT_EQ(record.lightLink, kNoRow);
        EXPECT_EQ(record.shadowLink, kNoRow);
        EXPECT_EQ(record.shadowDistance, -1.0f);

        sun.Set(CollectionAPI::kCollectionExcludes, "shadowLink", {"/Blocker"});
        record = Records(scene, device)[0];
        EXPECT_EQ(record.lightLink, kNoRow);
        EXPECT_NE(record.shadowLink, kNoRow);

        scene.Apply<ShadowAPI>(sun, {.shadowEnable = false, .shadowDistance = 3});
        record = Records(scene, device)[0];
        EXPECT_TRUE((record.flags & kLightShadowless) != 0);
        EXPECT_EQ(record.shadowDistance, 3.0f);
    });
}

TEST(LightsTest, DomesTurnTheirPoleUpAndListTheirPortals) {
    ForEachScene([](Scene& scene, device::Device& device) {
        const Prim window = scene.Define<PortalLight>(
            "/Window", {.translate = {0, 0, 2}, .intensity = 5, .width = 2, .height = 3});
        const Prim sky = scene.Define<DomeLight>("/Sky", {.intensity = 2, .portals = {"/Window"}});
        std::vector<LightRecord> records = Records(scene, device);
        ASSERT_EQ(records.size(), 2u);
        const LightRecord dome = Find(records, sky);
        EXPECT_EQ(LightType(dome.type), LightType::Dome);
        EXPECT_EQ(dome.portals, 1u);
        EXPECT_NEAR(dome.power, 2 * kPi, 2 * kPi * 1e-3f);
        EXPECT_FLOAT_EQ(dome.axisY.z, 1);
        EXPECT_FLOAT_EQ(dome.axisZ.y, -1);
        const LightRecord portal = Find(records, window);
        EXPECT_EQ(LightType(portal.type), LightType::Portal);
        EXPECT_FLOAT_EQ(portal.width, 2);
        EXPECT_FLOAT_EQ(portal.height, 3);
        EXPECT_EQ(portal.power, 0);
        EXPECT_EQ(Luminance(portal.radiance), 0);

        sky.Set(DomeLight::kPoleAxis, DomePoleAxis::Y);
        records = Records(scene, device);
        EXPECT_FLOAT_EQ(Find(records, sky).axisY.y, 1);
        EXPECT_FLOAT_EQ(Find(records, sky).axisZ.z, 1);
    });
}

TEST(LightsTest, RecordsSitRelativeToTheRenderOriginOfTheirViewer) {
    ForEachScene([](Scene& scene, device::Device& device) {
        constexpr double kFar = 1e8;
        scene.Define<Camera>("/Cam", {.translate = {kFar, 0, 0}});
        scene.Define<RenderProduct>("/View", {.camera = {"/Cam"}});
        const Prim lamp = scene.Define<SphereLight>("/Lamp", {.translate = {kFar + 3, 0, 0}});
        scene.Define<PointInstancer>("/Lamps", {.prototypes = {"/Lamps/Protos/Lamp"},
                                                .protoIndices = {0},
                                                .positions = {{float(kFar), 0, 0}}});
        scene.Define<Scope>("/Lamps/Protos");
        const Prim copied =
            scene.Define<SphereLight>("/Lamps/Protos/Lamp", {.translate = {0, 5, 0}});
        const std::vector<LightRecord> records = Records(scene, device);
        EXPECT_NEAR(Find(records, lamp).position.x, 3, 1e-3);
        EXPECT_NEAR(Find(records, lamp).position.y, 0, 1e-3);
        const LightRecord copy = Find(records, copied);
        EXPECT_NEAR(copy.position.x, 0, 1e-3);
        EXPECT_NEAR(copy.position.y, 5, 1e-3);

        scene.Define<World>("/Env0");
        scene.Find("/Lamps")->Set(PointInstancer::kPositions,
                                  std::vector<Float3>{{float(kFar + 16), 0, 0}});
        const std::vector<LightRecord> after = Records(scene, device);
        const std::span<const uint32_t> counts = SceneImpl::Get(scene).GetLights().Counts();
        const std::vector<LightRecord> global(after.end() - counts.back(), after.end());
        EXPECT_NEAR(Find(global, lamp).position.x, 3, 1e-3);
        const LightRecord relaid = Find(global, copied);
        EXPECT_NEAR(relaid.position.x, 16, 1e-3);
        EXPECT_NEAR(relaid.position.y, 5, 1e-3);
    });
}

TEST(LightsTest, DomeRecordsPickUpTheirTextureOnceItLoads) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "ernie_scene_lights_sky.png";
    WritePng(path, Image(4, 4, Rgba{0.2f, 0.4f, 0.8f, 1}));
    ForEachScene([&](Scene& scene, device::Device& device) {
        const Prim sky = scene.Define<DomeLight>("/Sky", {.textureFile = path.string()});
        Records(scene, device);
        SceneImpl::Get(scene).GetTextures().Finish();
        const LightRecord loaded = Find(Records(scene, device), sky);
        EXPECT_NE(loaded.textures, 0u);
        EXPECT_EQ(loaded.textureFormat, uint32_t(DomeTextureFormat::Angular));
    });
}

}
}
