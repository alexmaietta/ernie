// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/camera.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

using ernie::AttributeDef;
using ernie::AttributeKind;
using ernie::CameraParams;
using ernie::Path;
using ernie::PrimTypeDef;
using ernie::Scene;
using ernie::Value;
namespace tokens = ernie::tokens;

TEST(CameraSchema, FallbacksAreTheDefaultCameraParams) {
    // Each declared fallback is the default-constructed field it resolves to, so they cannot
    // drift apart.
    const CameraParams params;
    const ernie::SensorModel& sensor = params.sensor;
    const PrimTypeDef& type = ernie::Camera::Definition();
    const auto fallback = [&type](std::string_view name) {
        const AttributeDef* def = type.Find(name);
        EXPECT_NE(def, nullptr) << name;
        return def != nullptr ? def->fallback : Value();
    };
    EXPECT_EQ(fallback(tokens::kProjection), Value(std::string(tokens::kPerspective)));
    EXPECT_FALSE(params.orthographic);
    EXPECT_EQ(fallback(tokens::kFocalLength), Value(params.focalLengthMm));
    EXPECT_EQ(fallback(tokens::kVerticalAperture), Value(params.verticalApertureMm));
    EXPECT_EQ(fallback(tokens::kHorizontalApertureOffset),
              Value(params.horizontalApertureOffsetMm));
    EXPECT_EQ(fallback(tokens::kVerticalApertureOffset), Value(params.verticalApertureOffsetMm));
    EXPECT_EQ(fallback(tokens::kClippingRange),
              Value(glm::dvec2(params.nearPlane, params.farPlane)));
    EXPECT_EQ(fallback(tokens::kCameraExposure), Value(params.exposureStops));
    EXPECT_EQ(fallback(tokens::kExposureTime), Value(params.exposureTime));
    EXPECT_EQ(fallback(tokens::kExposureIso), Value(params.iso));
    EXPECT_EQ(fallback(tokens::kExposureFStop), Value(params.fStop));
    EXPECT_EQ(fallback(tokens::kExposureResponsivity), Value(params.responsivity));
    EXPECT_EQ(fallback(tokens::kFStop), Value(params.lensFStop));
    EXPECT_EQ(fallback(tokens::kFocusDistance), Value(params.focusDistance));
    EXPECT_EQ(fallback(tokens::kShutterOpen), Value(params.shutterOpen));
    EXPECT_EQ(fallback(tokens::kShutterClose), Value(params.shutterClose));
    EXPECT_EQ(fallback(tokens::kShutterSamples), Value(params.shutterSamples));
    EXPECT_EQ(fallback(tokens::kShutterRolling), Value(params.rollingShutter));
    EXPECT_EQ(fallback(tokens::kCameraScan), Value(std::string(tokens::kFrameScan)));
    EXPECT_FALSE(params.pushbroom);
    EXPECT_EQ(fallback(tokens::kPushbroomLineOffset), Value(params.lineOffsetMm));
    EXPECT_EQ(fallback(tokens::kSpectralRange),
              Value(glm::dvec2(params.bands.minNm, params.bands.maxNm)));
    EXPECT_EQ(fallback(tokens::kSpectralBands), Value(params.bands.count));
    EXPECT_EQ(fallback(tokens::kLensProjection), Value(std::string(tokens::kPerspective)));
    EXPECT_EQ(params.lens.projection, ernie::LensProjection::Perspective);
    EXPECT_EQ(fallback(tokens::kLensVignetting), Value(params.lens.vignetting));
    EXPECT_EQ(fallback(tokens::kLensTransmission), Value(params.lens.transmission));
    EXPECT_EQ(fallback(tokens::kLensRaysStep), Value(static_cast<int>(ernie::LensRayTable{}.step)));
    EXPECT_EQ(fallback(tokens::kSensorType), Value(std::string("rgb")));
    EXPECT_EQ(sensor.channels.size(), ernie::SensorModel::Rgb().channels.size());
    EXPECT_EQ(fallback(tokens::kSensorFullWell), Value(sensor.fullWellElectrons));
    EXPECT_EQ(fallback(tokens::kSensorReadNoise), Value(sensor.readNoiseElectrons));
    EXPECT_EQ(fallback(tokens::kSensorDarkCurrent), Value(sensor.darkCurrentElectrons));
    EXPECT_EQ(fallback(tokens::kSensorPrnu), Value(sensor.prnu));
    EXPECT_EQ(fallback(tokens::kSensorDsnu), Value(sensor.dsnuElectrons));
    EXPECT_EQ(fallback(tokens::kSensorGain), Value(sensor.electronsPerDnAtIso100));
    EXPECT_EQ(fallback(tokens::kSensorAdcBits), Value(sensor.adcBits));
    EXPECT_EQ(fallback(tokens::kSensorBlackLevel), Value(sensor.blackLevelDn));
    EXPECT_EQ(fallback(tokens::kSensorPixelPitch), Value(params.orthographicPixelPitchUm));
    EXPECT_EQ(fallback(tokens::kSensorNoise), Value(sensor.noise));
    EXPECT_EQ(fallback(tokens::kSensorSeed), Value(static_cast<int>(sensor.seed)));
    EXPECT_EQ(fallback(tokens::kPosesDevice), Value(std::string(tokens::kGpu)));
    // The panel ranges follow the limits `CameraParams::Check` enforces.
    EXPECT_EQ(type.Find(tokens::kShutterSamples)->meta.max, CameraParams::kMaxShutterSamples);
    EXPECT_EQ(type.Find(tokens::kSpectralBands)->meta.max, ernie::SpectralBands::kMaxBands);
}

TEST(CameraSchema, UnsetCameraResolvesToTheDefaults) {
    Scene scene;
    ernie::Camera::Define(scene, "/Camera");
    const CameraParams resolved = ernie::CameraParamsOf(scene, "/Camera", 64, 48);
    const CameraParams defaults;
    EXPECT_EQ(resolved.focalLengthMm, defaults.focalLengthMm);
    EXPECT_EQ(resolved.nearPlane, defaults.nearPlane);
    EXPECT_EQ(resolved.farPlane, defaults.farPlane);
    EXPECT_EQ(resolved.shutterSamples, defaults.shutterSamples);
    EXPECT_EQ(resolved.bands.count, defaults.bands.count);
    EXPECT_EQ(resolved.sensor.adcBits, defaults.sensor.adcBits);
    EXPECT_EQ(resolved.sensor.channels.size(), 3u);
    EXPECT_TRUE(resolved.lens.IsPinhole());
}

TEST(CameraSchema, ViewsReadDefaultsAndSetByUsdName) {
    Scene scene;
    ernie::Camera camera = ernie::Camera::Define(scene, "/World/Camera");
    EXPECT_EQ(scene.TypeOf("/World/Camera"), tokens::kCamera);
    EXPECT_EQ(camera.GetFocalLength(), 50.0);
    EXPECT_FALSE(camera.HasFocalLength());
    EXPECT_EQ(camera.GetProjection(), "perspective");
    EXPECT_EQ(camera.GetClippingRange(), glm::dvec2(1.0, 1e6));
    EXPECT_EQ(camera.GetShutterSamples(), 8);
    EXPECT_TRUE(camera.GetLensDistortion().empty());
    EXPECT_EQ(camera.GetLensRays(), nullptr);
    EXPECT_EQ(camera.GetPoses(), nullptr);
    EXPECT_FALSE(camera.GetSensorQe("mono").has_value());

    camera.SetFocalLength(35.0)
        .SetClippingRange(glm::dvec2(0.1, 100.0))
        .SetScan(tokens::kPushbroom)
        .SetSpectralBands(8)
        .SetSensorQeMono({400.0f, 0.5f, 700.0f, 0.5f});
    EXPECT_TRUE(camera.HasFocalLength());
    EXPECT_EQ(*scene.Get("/World/Camera", tokens::kFocalLength), Value(35.0));
    EXPECT_EQ(*scene.Get("/World/Camera", tokens::kCameraScan), Value(std::string("pushbroom")));
    EXPECT_EQ(camera.GetClippingRange(), glm::dvec2(0.1, 100.0));
    EXPECT_EQ(camera.GetSensorQe("mono"), (std::vector<float>{400.0f, 0.5f, 700.0f, 0.5f}));
    EXPECT_FALSE(camera.GetSensorQe("r").has_value());
    // Authored as another numeric type, it still reads as the declared one.
    scene.Set("/World/Camera", tokens::kShutterSamples, 4.0);
    EXPECT_EQ(camera.GetShutterSamples(), 4);

    // What the view sets is what `CameraParamsOf` resolves.
    camera.SetScan(tokens::kFrameScan).SetSensorType("mono").SetSensorAdcBits(10);
    const CameraParams resolved = ernie::CameraParamsOf(scene, "/World/Camera", 64, 48);
    EXPECT_EQ(resolved.focalLengthMm, 35.0);
    EXPECT_EQ(resolved.nearPlane, 0.1);
    EXPECT_EQ(resolved.bands.count, 8);
    EXPECT_EQ(resolved.sensor.adcBits, 10);
    ASSERT_EQ(resolved.sensor.channels.size(), 1u);
    EXPECT_DOUBLE_EQ(resolved.sensor.channels[0].quantumEfficiency(550.0), 0.5);
}

TEST(CameraSchema, SetCameraWritesThroughTheView) {
    Scene scene;
    CameraParams params;
    params.width = 200;
    params.height = 100;
    params.focalLengthMm = 24.0;
    params.shutterSamples = 3;
    ernie::SetCamera(scene, "/Camera", params);
    const ernie::Camera camera(scene, "/Camera");
    EXPECT_EQ(camera.GetFocalLength(), 24.0);
    EXPECT_EQ(camera.GetShutterSamples(), 3);
    EXPECT_DOUBLE_EQ(camera.GetHorizontalAperture(), params.verticalApertureMm * 2.0);
    EXPECT_EQ(camera.GetScan(), "frame");
}

TEST(CameraSchema, ViewsCheckTypes) {
    Scene scene;
    ernie::Camera::Define(scene, "/Camera");
    scene.Define("/Rock", tokens::kSphere);
    EXPECT_NO_THROW(ernie::Imageable(scene, "/Camera"));
    EXPECT_THROW(ernie::Camera(scene, "/Rock"), std::invalid_argument);
    EXPECT_THROW(ernie::Camera(scene, "/Missing"), std::invalid_argument);
    const Scene& constScene = scene;
    ernie::Camera readOnly(constScene, "/Camera");
    EXPECT_EQ(readOnly.GetExposureIso(), 100.0);
    EXPECT_THROW(readOnly.SetExposureIso(200.0), std::logic_error);
}

TEST(CameraSchema, RegistryDescribesTheCamera) {
    const PrimTypeDef& camera = ernie::GetPrimType("Camera");
    EXPECT_EQ(&camera, &ernie::Camera::Definition());
    EXPECT_FALSE(camera.abstract);
    ASSERT_NE(camera.base, nullptr);
    EXPECT_EQ(camera.base->name, "Imageable");

    const AttributeDef* focal = camera.Find(tokens::kFocalLength);
    ASSERT_NE(focal, nullptr);
    EXPECT_EQ(focal->member, "FocalLength");
    EXPECT_EQ(focal->pyName, "focal_length");
    EXPECT_EQ(focal->label, "Focal length");
    EXPECT_EQ(focal->meta.group, "Lens");
    EXPECT_EQ(camera.FindByPyName("exposure_f_stop")->name, tokens::kExposureFStop);
    EXPECT_EQ(camera.Find(tokens::kExposureFStop)->label, "f-number");
    EXPECT_EQ(camera.Find(tokens::kShutterSamples)->kind, AttributeKind::Int);
    EXPECT_EQ(camera.Find(tokens::kLensDistortion)->kind, AttributeKind::FloatArray);
    EXPECT_EQ(camera.Find(tokens::kLensRays)->kind, AttributeKind::Tensor);
    EXPECT_EQ(camera.Find(tokens::kPoses)->kind, AttributeKind::Tensor);
    EXPECT_EQ(camera.Find(tokens::kWorlds)->kind, AttributeKind::Tensor);
    EXPECT_EQ(camera.Find(tokens::kSensorQeMono)->name, "ernie:sensor:qe:mono");

    // Every attribute once, in the viewer's sections.
    const std::set<std::string_view> groups{"Prim",    "Lens",   "Exposure",
                                            "Shutter", "Sensor", "Views"};
    std::set<std::string_view> names;
    for (const AttributeDef* def : camera.AllAttributes()) {
        EXPECT_TRUE(names.insert(def->name).second) << def->name;
        EXPECT_TRUE(groups.contains(def->meta.group)) << def->name;
    }
}
