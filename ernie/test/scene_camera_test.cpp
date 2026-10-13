// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

using ernie::CameraParams;
using ernie::Scene;
namespace tokens = ernie::tokens;

TEST(Camera, ExposureScaleFollowsUsd) {
    CameraParams camera;
    EXPECT_DOUBLE_EQ(camera.LinearExposureScale(), 1.0)
        << "USD's defaults";
    camera.exposureTime = 0.01;
    camera.iso = 200.0;
    camera.fStop = 8.0;
    camera.responsivity = 0.5;
    camera.exposureStops = 1.0;
    EXPECT_DOUBLE_EQ(camera.LinearExposureScale(),
                     0.5 * 0.01 * 2.0 * 2.0 / 64.0);
}

TEST(Camera, IntrinsicsFollowTheApertureAndOffsets) {
    CameraParams camera;
    camera.width = 640;
    camera.height = 480;
    camera.focalLengthMm = 24.0;
    camera.verticalApertureMm = 4.8;
    EXPECT_DOUBLE_EQ(camera.PixelPitchMetres(), 1e-5);
    EXPECT_DOUBLE_EQ(camera.FocalLengthPixels(), 2400.0);
    EXPECT_EQ(camera.PrincipalPoint(), glm::dvec2(320.0, 240.0));
    EXPECT_NEAR(camera.VerticalFov(), 2.0 * std::atan(2.4 / 24.0),
                1e-15);
    camera.horizontalApertureOffsetMm = 0.1;
    camera.verticalApertureOffsetMm = 0.05;
    EXPECT_EQ(camera.PrincipalPoint(), glm::dvec2(330.0, 235.0))
        << "offsets up the image";
}

TEST(Camera, DistortionRoundTrips) {
    ernie::LensModel lens;
    EXPECT_FALSE(lens.HasDistortion());
    EXPECT_EQ(lens.Distort({0.3, -0.2}), glm::dvec2(0.3, -0.2));
    lens.k1 = -0.2;
    lens.k2 = 0.05;
    lens.k3 = -0.01;
    lens.p1 = 0.001;
    lens.p2 = -0.002;
    EXPECT_TRUE(lens.HasDistortion());
    const glm::dvec2 p(0.3, -0.2);
    const double r2 = 0.13;
    const double radial =
        1.0 + r2 * (-0.2 + r2 * (0.05 + r2 * -0.01));
    const glm::dvec2 expected(
        0.3 * radial + 2.0 * 0.001 * 0.3 * -0.2 +
            -0.002 * (r2 + 2.0 * 0.09),
        -0.2 * radial + 0.001 * (r2 + 2.0 * 0.04) +
            2.0 * -0.002 * 0.3 * -0.2);
    EXPECT_NEAR(lens.Distort(p).x, expected.x, 1e-15);
    EXPECT_NEAR(lens.Distort(p).y, expected.y, 1e-15);
    const glm::dvec2 back = lens.Undistort(lens.Distort(p));
    EXPECT_NEAR(back.x, p.x, 1e-9);
    EXPECT_NEAR(back.y, p.y, 1e-9);
}

TEST(Camera, FisheyeMapsFieldAnglesToRadii) {
    ernie::LensModel lens;
    lens.projection = ernie::LensProjection::Fisheye;
    EXPECT_FALSE(lens.IsPinhole());
    // Equidistant: a ray at field angle theta lands at radius theta.
    const double theta = 1.2;
    const glm::dvec2 pinhole =
        std::tan(theta) * glm::dvec2(0.6, -0.8);
    const glm::dvec2 image = lens.Distort(pinhole);
    EXPECT_NEAR(image.x, 0.6 * theta, 1e-12);
    EXPECT_NEAR(image.y, -0.8 * theta, 1e-12);
    EXPECT_NEAR(lens.FieldAngle(image), theta, 1e-12);
    lens.fisheye = {-0.05, 0.01, -0.002, 0.0005};
    const glm::dvec2 back = lens.Undistort(lens.Distort(pinhole));
    EXPECT_NEAR(back.x, pinhole.x, 1e-9);
    EXPECT_NEAR(back.y, pinhole.y, 1e-9);
    const double t2 = theta * theta;
    const double radius =
        theta *
        (1.0 +
         t2 * (-0.05 + t2 * (0.01 + t2 * (-0.002 + t2 * 0.0005))));
    EXPECT_NEAR(glm::length(lens.Distort(pinhole)), radius, 1e-12);
}

TEST(Camera, NaturalVignettingFollowsTheProjection) {
    ernie::LensModel lens;
    const glm::dvec2 p(0.3, 0.4);
    const double cosine2 = 1.0 / 1.25;
    EXPECT_NEAR(lens.NaturalVignetting(p), cosine2 * cosine2, 1e-15)
        << "cos^4 for a pinhole";
    lens.projection = ernie::LensProjection::Fisheye;
    const double theta = 0.5;  // the radius of p, equidistant
    EXPECT_NEAR(lens.NaturalVignetting(p),
                std::cos(theta) * std::sin(theta) / theta, 1e-12);
    EXPECT_DOUBLE_EQ(lens.NaturalVignetting({0.0, 0.0}), 1.0);
}

TEST(Camera, SingletChromaticAberrationFollowsItsGlass) {
    // SCHOTT N-BK7 (n_d 1.5168, V_d 64.17) at 50 mm: the C line
    // focuses about f / V = 0.78 mm further than the F line, blue
    // nearer the lens than red.
    ernie::LensModel lens;
    ernie::SetSingletChromatic(lens, 50.0, 1.5168, 64.17);
    const double fc = lens.chromaticFocusMm(656.27);
    const double ff = lens.chromaticFocusMm(486.13);
    EXPECT_NEAR(fc - ff, 50.0 / 64.17, 0.01);
    EXPECT_NEAR(lens.chromaticFocusMm(587.56), 0.0, 1e-3)
        << "the d line is the reference";
    EXPECT_LT(lens.chromaticFocusMm(400.0), 0.0);
    EXPECT_GT(lens.chromaticFocusMm(800.0), 0.0);
    EXPECT_DOUBLE_EQ(lens.chromaticScale(450.0), 1.0)
        << "no lateral colour with the stop at it";
    ernie::SetSingletChromatic(lens, 50.0, 1.5168, 64.17, 0.5);
    EXPECT_GT(lens.chromaticScale(656.27), 1.0);
    EXPECT_LT(lens.chromaticScale(486.13), 1.0);
    EXPECT_NEAR(lens.chromaticScale(656.27) -
                    lens.chromaticScale(486.13),
                0.5 * (fc - ff) / 50.0, 1e-5);
    EXPECT_THROW(ernie::SetSingletChromatic(lens, 50.0, 0.9, 64.17),
                 std::invalid_argument);
    EXPECT_THROW(
        ernie::SetSingletChromatic(lens, 50.0, 1.5, 64.17, 2.0),
        std::invalid_argument);

    // Read from the prim's attributes, and checked; two pairs also
    // read as the vec4 a four-number Python list becomes.
    ernie::Scene scene;
    ernie::Camera::Define(scene, "/Camera")
        .SetLensChromaticScale({400.0f, 0.99f, 700.0f, 1.01f});
    scene.Set("/Camera", ernie::tokens::kLensChromaticFocus,
              glm::vec4(400.0f, -0.2f, 700.0f, 0.3f));
    CameraParams camera =
        ernie::CameraParamsOf(scene, "/Camera", 64, 48);
    EXPECT_NEAR(camera.lens.chromaticScale(550.0), 1.0, 1e-6);
    EXPECT_NEAR(camera.lens.chromaticFocusMm(700.0), 0.3, 1e-6);
    camera.lens.chromaticScale =
        ernie::Spectrum::FromPairs(std::vector<float>{400.0f, -1.0f});
    EXPECT_THROW(camera.Check(), std::invalid_argument);
}

TEST(Camera, FisheyeFieldIsLimited) {
    CameraParams camera;
    camera.width = 100;
    camera.height = 100;
    camera.lens.projection = ernie::LensProjection::Fisheye;
    camera.SetVerticalFov(
        1.0);  // a corner at 0.5 sqrt(2) radians off axis
    EXPECT_NO_THROW(camera.Check());
    camera.focalLengthMm =
        camera.verticalApertureMm / 3.0;  // corners at 2.1 radians
    EXPECT_NO_THROW(camera.Check())
        << "121.5 degrees off axis, on cube faces";
    camera.focalLengthMm =
        camera.verticalApertureMm / 4.0;  // corners at 2.8 radians
    EXPECT_NO_THROW(camera.Check())
        << "162 degrees off axis, on all six faces";
    camera.focalLengthMm =
        camera.verticalApertureMm / 5.0;  // corners at 3.5 radians
    EXPECT_THROW(camera.Check(), std::invalid_argument)
        << "past 180 degrees";
    camera.SetVerticalFov(1.0);
    camera.lens.k1 = 0.1;
    EXPECT_THROW(camera.Check(), std::invalid_argument)
        << "no Brown-Conrady terms on a fisheye";
}

TEST(Camera, OrthographicSpansItsAperturesInTheScene) {
    CameraParams camera;
    camera.width = 200;
    camera.height = 100;
    camera.orthographic = true;
    camera.verticalApertureMm = 40.0;  // 4 scene units
    camera.verticalApertureOffsetMm =
        4.0;  // the view up by 0.4 units
    camera.transform = ernie::LookAtTransform(
        {1.0, 2.0, 10.0}, {1.0, 2.0, 0.0}, {0, 1, 0});
    EXPECT_DOUBLE_EQ(camera.OrthographicPixelSize(), 0.04);
    EXPECT_DOUBLE_EQ(camera.PixelPitchMetres(), 5e-6)
        << "the sensor's, not the scene's";
    EXPECT_EQ(camera.PrincipalPoint(), glm::dvec2(100.0, 40.0));
    const ernie::CameraView view = camera.RasterView();
    EXPECT_TRUE(view.IsOrthographic());
    EXPECT_FALSE(CameraParams().RasterView().IsOrthographic());
    // The axis lands on the principal point at every depth, and a
    // pixel is 0.04 units.
    for (const float depth : {2.0f, 9.0f}) {
        const glm::vec4 clip =
            view.ViewProjection() *
            glm::vec4(1.0f, 2.0f, 10.0f - depth, 1.0f);
        EXPECT_NEAR(clip.w, 1.0f, 1e-6f);
        EXPECT_NEAR((clip.x + 1.0f) * 100.0f, 100.0f, 1e-4f);
        EXPECT_NEAR((1.0f - clip.y) * 50.0f, 40.0f, 1e-4f);
        // Reverse-Z: 1 at the near plane, 0 at the far.
        EXPECT_NEAR(clip.z,
                    (camera.farPlane - static_cast<double>(depth)) /
                        (camera.farPlane - camera.nearPlane),
                    1e-6);
    }
    const glm::vec4 right =
        view.ViewProjection() * glm::vec4(3.0f, 2.0f, 5.0f, 1.0f);
    EXPECT_NEAR((right.x + 1.0f) * 100.0f, 150.0f, 1e-3f)
        << "2 units is 50 pixels";

    camera.lens.vignetting = true;
    EXPECT_DOUBLE_EQ(camera.IrradiancePerRadiance({0.5, 0.5}),
                     camera.IrradiancePerRadiance({0.0, 0.0}))
        << "telecentric optics";
    EXPECT_NO_THROW(camera.Check());
    camera.lens.projection = ernie::LensProjection::Fisheye;
    EXPECT_THROW(camera.Check(), std::invalid_argument);
}

TEST(Camera, ElectronsFollowTheCameraEquation) {
    CameraParams camera;
    camera.width = 100;
    camera.height = 100;
    camera.verticalApertureMm = 1.0;
    camera.fStop = 4.0;
    camera.exposureTime = 0.002;
    camera.bands = {500.0, 600.0, 1};
    camera.sensor.channels = {
        {"flat", ernie::Spectrum::Constant(0.5)}};
    camera.lens.transmission = 0.9;
    camera.lens.vignetting = true;
    const std::vector<float> radiance{2.0f};
    const double irradiance = 2.0 * M_PI * 0.9 / (4.0 * 16.0);
    const double energy = irradiance * 100.0 * 1e-10 * 0.002;
    const double photons =
        energy * 550e-9 / (6.62607015e-34 * 299792458.0);
    EXPECT_NEAR(camera.Electrons(radiance, 0) / (photons * 0.5), 1.0,
                1e-6);

    const double off = camera.Electrons(radiance, 0, {1.0, 0.0});
    EXPECT_NEAR(off / camera.Electrons(radiance, 0), 0.25, 1e-12)
        << "cos^4 of 45 degrees";
    camera.lens.vignetting = false;
    EXPECT_NEAR(camera.Electrons(radiance, 0, {1.0, 0.0}),
                camera.Electrons(radiance, 0), 1e-9);
    EXPECT_THROW(camera.Electrons(std::vector<float>{1.0f, 2.0f}, 0),
                 std::invalid_argument);
}

TEST(Camera, SensorPresetsAndGain) {
    const ernie::SensorModel rgb = ernie::SensorModel::Rgb();
    ASSERT_EQ(rgb.channels.size(), 3u);
    EXPECT_GT(rgb.channels[0].quantumEfficiency(610.0),
              rgb.channels[2].quantumEfficiency(610.0));
    EXPECT_GT(rgb.channels[2].quantumEfficiency(460.0),
              rgb.channels[0].quantumEfficiency(460.0));
    const ernie::SensorModel mono = ernie::SensorModel::Mono();
    ASSERT_EQ(mono.channels.size(), 1u);
    EXPECT_GT(mono.channels[0].quantumEfficiency(600.0), 0.5);
    EXPECT_NEAR(rgb.ElectronsPerDn(100.0), 20000.0 / 4095.0, 1e-9);
    EXPECT_NEAR(rgb.ElectronsPerDn(400.0), 20000.0 / 4095.0 / 4.0,
                1e-9)
        << "ISO raises gain";
}

TEST(Camera, ReadsUsdAndErnieAttributes) {
    Scene scene;
    ernie::Camera authored =
        ernie::Camera::Define(scene, "/Camera")
            .SetFocalLength(35.0)
            .SetVerticalAperture(10.0)
            .SetVerticalApertureOffset(0.5)
            .SetClippingRange({0.5, 500.0})
            .SetExposure(-1.0)
            .SetExposureTime(0.004)
            .SetExposureIso(400.0)
            .SetExposureFStop(5.6)
            .SetSpectralRange({400.0, 1000.0})
            .SetSpectralBands(24)
            .SetLensDistortion({-0.1f, 0.01f})
            .SetLensVignetting(false)
            .SetSensorType("mono")
            .SetSensorQeMono({400, 0.3f, 900, 0.3f})
            .SetSensorAdcBits(14)
            .SetSensorNoise(false)
            .SetSensorSeed(7);
    const CameraParams camera =
        ernie::CameraParamsOf(scene, "/Camera", 64, 48);
    EXPECT_EQ(camera.width, 64u);
    EXPECT_DOUBLE_EQ(camera.focalLengthMm, 35.0);
    EXPECT_DOUBLE_EQ(camera.verticalApertureOffsetMm, 0.5);
    EXPECT_DOUBLE_EQ(camera.nearPlane, 0.5);
    EXPECT_DOUBLE_EQ(camera.exposureStops, -1.0);
    EXPECT_DOUBLE_EQ(camera.iso, 400.0);
    EXPECT_DOUBLE_EQ(camera.fStop, 5.6);
    EXPECT_EQ(camera.bands,
              (ernie::SpectralBands{400.0, 1000.0, 24}));
    EXPECT_DOUBLE_EQ(camera.lens.k1, -0.1f);
    EXPECT_DOUBLE_EQ(camera.lens.k3, 0.0);
    EXPECT_FALSE(camera.lens.vignetting);
    ASSERT_EQ(camera.sensor.channels.size(), 1u);
    EXPECT_NEAR(camera.sensor.channels[0].quantumEfficiency(650.0),
                0.3, 1e-6);
    EXPECT_EQ(camera.sensor.adcBits, 14);
    EXPECT_FALSE(camera.sensor.noise);
    EXPECT_EQ(camera.sensor.seed, 7u);

    EXPECT_FALSE(camera.orthographic);
    EXPECT_EQ(camera.lens.projection,
              ernie::LensProjection::Perspective);

    authored.SetLensDistortion({})
        .SetLensProjection(tokens::kFisheye)
        .SetLensFisheye({-0.01f, 0.002f});
    const CameraParams fisheye =
        ernie::CameraParamsOf(scene, "/Camera", 64, 48);
    EXPECT_EQ(fisheye.lens.projection,
              ernie::LensProjection::Fisheye);
    EXPECT_DOUBLE_EQ(fisheye.lens.fisheye[1], 0.002f);
    EXPECT_DOUBLE_EQ(fisheye.lens.fisheye[3], 0.0);
    authored.SetLensProjection("panoramic");
    EXPECT_THROW(ernie::CameraParamsOf(scene, "/Camera", 64, 48),
                 std::invalid_argument);
    authored.SetLensProjection(tokens::kPerspective);

    authored.SetProjection(tokens::kOrthographic)
        .SetSensorPixelPitch(3.45);
    const CameraParams orthographic =
        ernie::CameraParamsOf(scene, "/Camera", 64, 48);
    EXPECT_TRUE(orthographic.orthographic);
    EXPECT_DOUBLE_EQ(orthographic.orthographicPixelPitchUm, 3.45);
    EXPECT_DOUBLE_EQ(orthographic.OrthographicPixelSize(),
                     0.1 * orthographic.verticalApertureMm / 48.0)
        << "apertures in tenths of a scene unit, as UsdGeomCamera";
    authored.SetProjection(tokens::kPerspective);

    authored.SetScan(tokens::kPushbroom).SetPushbroomLineOffset(1.5);
    const CameraParams pushbroom =
        ernie::CameraParamsOf(scene, "/Camera", 64, 48);
    EXPECT_TRUE(pushbroom.pushbroom);
    EXPECT_DOUBLE_EQ(pushbroom.lineOffsetMm, 1.5);
    authored.SetScan("whiskbroom");
    EXPECT_THROW(ernie::CameraParamsOf(scene, "/Camera", 64, 48),
                 std::invalid_argument);
    authored.SetScan(tokens::kFrameScan);

    authored.SetSensorType("infrared");
    EXPECT_THROW(ernie::CameraParamsOf(scene, "/Camera", 64, 48),
                 std::invalid_argument);
    authored.SetSensorType("rgb").SetExposureTime(0.0);
    EXPECT_THROW(ernie::CameraParamsOf(scene, "/Camera", 64, 48),
                 std::invalid_argument);
    ernie::Xform::Define(scene, "/NotACamera");
    EXPECT_THROW(ernie::CameraParamsOf(scene, "/NotACamera", 64, 48),
                 std::invalid_argument);
}

TEST(Camera, PoseGivesTheEyeForwardAndRasterView) {
    CameraParams camera;
    camera.width = 64;
    camera.height = 32;
    camera.transform = ernie::LookAtTransform(
        {10.0, -20.0, 30.0}, {10.0, 0.0, 30.0}, {0, 0, 1});
    EXPECT_EQ(camera.Eye(), glm::dvec3(10.0, -20.0, 30.0));
    EXPECT_NEAR(camera.Forward().y, 1.0, 1e-12);

    const ernie::CameraView plain = camera.RasterView();
    EXPECT_NEAR(plain.verticalFov, camera.VerticalFov(), 1e-6);
    EXPECT_FLOAT_EQ(plain.aspect, 2.0f);
    EXPECT_FLOAT_EQ(plain.projection[2][0], 0.0f);
    EXPECT_NEAR(glm::length(glm::vec3(plain.view *
                                      glm::vec4(plain.eye, 1.0f))),
                0.0f, 1e-4f);
    const glm::vec4 ahead =
        plain.ViewProjection() * glm::vec4(10.0f, 50.0f, 30.0f, 1.0f);
    EXPECT_NEAR(ahead.x / ahead.w, 0.0f, 1e-5f)
        << "a point straight ahead projects to the centre";
    EXPECT_NEAR(ahead.y / ahead.w, 0.0f, 1e-5f);

    const ernie::CameraView wide = camera.RasterView(1.5);
    EXPECT_NEAR(std::tan(wide.verticalFov / 2.0f),
                1.5f * std::tan(plain.verticalFov / 2.0f), 1e-5f);
    camera.horizontalApertureOffsetMm = 2.0;
    const ernie::CameraView shifted = camera.RasterView();
    EXPECT_NEAR(shifted.projection[2][0],
                1.0 - 2.0 * camera.PrincipalPoint().x / 64.0, 1e-6);
    EXPECT_NE(shifted.projection[2][0], 0.0f);

    camera.SetVerticalFov(1.0);
    EXPECT_NEAR(camera.VerticalFov(), 1.0, 1e-12);
    camera.transform = glm::dmat4(0.0);
    EXPECT_THROW(camera.Check(), std::invalid_argument);
}

TEST(Camera, SetCameraAndCameraParamsOfRoundTrip) {
    CameraParams camera;
    camera.width = 40;
    camera.height = 20;
    camera.transform = ernie::LookAtTransform(
        {1.0, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0, 0, 1});
    camera.focalLengthMm = 24.0;
    camera.verticalApertureMm = 12.0;
    camera.verticalApertureOffsetMm = 0.5;
    camera.nearPlane = 0.25;
    camera.farPlane = 400.0;
    camera.exposureStops = 1.5;
    camera.exposureTime = 0.01;
    camera.iso = 200.0;
    camera.fStop = 8.0;
    camera.bands = {400.0, 700.0, 6};
    Scene scene;
    ernie::SetCamera(scene, "/Camera", camera);
    const CameraParams back =
        ernie::CameraParamsOf(scene, "/Camera", 40, 20);
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            EXPECT_NEAR(back.transform[c][r], camera.transform[c][r],
                        1e-12);
        }
    }
    EXPECT_DOUBLE_EQ(back.focalLengthMm, 24.0);
    EXPECT_DOUBLE_EQ(back.verticalApertureMm, 12.0);
    EXPECT_DOUBLE_EQ(back.verticalApertureOffsetMm, 0.5);
    EXPECT_DOUBLE_EQ(back.farPlane, 400.0);
    EXPECT_DOUBLE_EQ(back.LinearExposureScale(),
                     camera.LinearExposureScale());
    EXPECT_EQ(back.bands, camera.bands);
    EXPECT_DOUBLE_EQ(ernie::GetNumber(scene, "/Camera",
                                      tokens::kHorizontalAperture,
                                      0.0),
                     24.0)
        << "the horizontal aperture follows the aspect";

    ernie::Camera(scene, "/Camera")
        .SetProjection(tokens::kOrthographic);
    EXPECT_TRUE(
        ernie::CameraParamsOf(scene, "/Camera", 40, 20).orthographic);
}

// An infinite far plane: depth near / distance, never reaching 0, and
// finite planes cut where they are.
TEST(Camera, ReverseZReachesInfinity) {
    const glm::mat4 infinite = ernie::ReversedPerspective(
        0.8f, 1.5f, 0.5f, std::numeric_limits<float>::infinity());
    for (const float distance : {0.5f, 2.0f, 384.4e6f}) {
        const glm::vec4 clip =
            infinite * glm::vec4(0.0f, 0.0f, -distance, 1.0f);
        EXPECT_NEAR(clip.z / clip.w, 0.5f / distance, 1e-6f);
    }
    const glm::mat4 finite =
        ernie::ReversedPerspective(0.8f, 1.5f, 0.5f, 100.0f);
    const glm::vec4 far =
        finite * glm::vec4(0.0f, 0.0f, -100.0f, 1.0f);
    EXPECT_NEAR(far.z / far.w, 0.0f, 1e-6f);
    const glm::vec4 near =
        finite * glm::vec4(0.0f, 0.0f, -0.5f, 1.0f);
    EXPECT_NEAR(near.z / near.w, 1.0f, 1e-6f);
}
