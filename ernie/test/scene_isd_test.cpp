#include "ernie/scene/isd.h"

#include "ernie/scene/instruments.h"
#include "ernie/scene/tokens.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string Fixture(const char* name) {
    return (std::filesystem::path(ERNIE_TEST_FIXTURES) / "isd" / name).string();
}

}  // namespace

TEST(Isd, ReadsTheLroNarrowAngleCamera) {
    const ernie::IsdSensor nac = ernie::ReadIsdFile(Fixture("toughLroNacLineScan.json"));
    EXPECT_TRUE(nac.IsLineScanner());
    EXPECT_EQ(nac.sensor, "LUNAR RECONNAISSANCE ORBITER NARROW ANGLE CAMERA LEFT");
    EXPECT_EQ(nac.lines, 8192u);
    EXPECT_EQ(nac.samples, 2532u);
    EXPECT_DOUBLE_EQ(nac.focalLengthMm, 699.62);
    EXPECT_DOUBLE_EQ(nac.focalToSample[2], 142.857) << "7 micrometre pixels";
    EXPECT_DOUBLE_EQ(nac.sampleSumming, 2.0);
    EXPECT_EQ(nac.distortion, "lrolrocnac");
    ASSERT_EQ(nac.distortionCoefficients.size(), 1u);
    EXPECT_DOUBLE_EQ(nac.distortionCoefficients[0], 1.81e-5);
    ASSERT_EQ(nac.lineRates.size(), 1u);
    EXPECT_DOUBLE_EQ(nac.lineRates[0][2], 0.0118595288);
    EXPECT_DOUBLE_EQ(nac.semiMajorAxis, 1737.4e3) << "radii in metres";
    // Positions in km without a unit, as ALE writes them: about 36 km above the Moon.
    ASSERT_EQ(nac.positions.values.size(), 173u);
    EXPECT_DOUBLE_EQ(nac.positions.values[0].x, -117.33247735320907e3);
    EXPECT_NEAR(glm::length(nac.positions.values[0]) - nac.semiMajorAxis, 36e3, 2e3);
    ASSERT_EQ(nac.pointing.quaternions.values.size(), 1773u);
    const glm::dquat q = nac.pointing.quaternions.values[0];
    EXPECT_DOUBLE_EQ(q.w, 0.0934065633770046) << "scalar first in the ISD";
    EXPECT_DOUBLE_EQ(q.z, 0.9792826872138032);
    EXPECT_EQ(nac.pointing.timeDependentFrames, (std::vector<std::int64_t>{-85600, 1}));
    EXPECT_EQ(nac.bodyRotation.constantFrames, (std::vector<std::int64_t>{31001, 31006}));
    EXPECT_NEAR(nac.bodyRotation.constantRotation[1][0], -0.0003292854223755708, 1e-15)
        << "row 0, column 1 of the row-major matrix";
    EXPECT_EQ(nac.sunPositions.values.size(), 2u);
}

TEST(Isd, ReadsExplicitUnitsAndTheContextCamera) {
    const ernie::IsdSensor orbital = ernie::ReadIsdFile(Fixture("orbitalLineScan.json"));
    EXPECT_EQ(orbital.lines, 16u);
    EXPECT_DOUBLE_EQ(orbital.semiMajorAxis, 1e6);
    EXPECT_EQ(orbital.positions.values.size(), orbital.positions.times.size());
    const ernie::IsdSensor ctx = ernie::ReadIsdFile(Fixture("ctxLineScan.json"));
    EXPECT_EQ(ctx.samples, 5000u);
    EXPECT_EQ(ctx.distortion, "radial");
    EXPECT_EQ(ctx.distortionCoefficients.size(), 3u);
    EXPECT_NEAR(ctx.semiMinorAxis, 3376.2e3, 1e-6);
}

TEST(Isd, RejectsWhatItCannotRead) {
    EXPECT_THROW(ernie::ReadIsd("{"), std::invalid_argument);
    EXPECT_THROW(ernie::ReadIsd("{}"), std::invalid_argument);
    EXPECT_THROW(ernie::ReadIsd(R"({"name_model": 3})"), std::invalid_argument);
    EXPECT_THROW(ernie::ReadIsdFile("/no/such/isd.json"), std::runtime_error);
}

// usgscsm's own expectations for these ISDs (its tests/LineScanCameraTests.cpp), which Ernie's
// line scanner reproduces.
TEST(IsdLineScanner, MatchesUsgscsmOnItsTestScanners) {
    const ernie::IsdLineScanner constant(ernie::ReadIsdFile(Fixture("constVelocityLineScan.json")));
    const glm::dvec3 center = constant.ImageToGround(8.0, 8.0);
    EXPECT_NEAR(center.x, 9.99999500000, 1e-10);
    EXPECT_NEAR(center.y, 0.0, 1e-10);
    EXPECT_NEAR(center.z, 0.00999999500, 1e-10);
    const glm::dvec3 offBody = constant.ImageToGround(0.0, 4.0);
    EXPECT_NEAR(offBody.x, 0.04799688020, 1e-10);
    EXPECT_NEAR(offBody.y, -7.99961602495, 1e-10);
    EXPECT_NEAR(offBody.z, 16.00004799688, 1e-10);
    const glm::dvec2 back = constant.GroundToImage(constant.ImageToGround(8.5, 8.0));
    EXPECT_NEAR(back.x, 8.5, 1e-3);
    EXPECT_NEAR(back.y, 8.0, 1e-3);

    const ernie::IsdLineScanner orbital(ernie::ReadIsdFile(Fixture("orbitalLineScan.json")));
    const glm::dvec3 ground = orbital.ImageToGround(8.5, 8.0);
    EXPECT_NEAR(ground.x, 999999.67004351015, 1e-9);
    EXPECT_NEAR(ground.y, 0.0, 1e-9);
    EXPECT_NEAR(ground.z, -812.35021438796923, 1e-9);
    EXPECT_NEAR(glm::length(orbital.ImageToGround(8.5, 8.0, 100.0)), 1000100.0, 1e-6);
    for (double line = 0.5; line < 16.0; line += 1.0) {
        const glm::dvec2 image = orbital.GroundToImage(orbital.ImageToGround(line, 8.0, 4900.0));
        EXPECT_NEAR(image.x, line, 0.002);
        EXPECT_NEAR(image.y, 8.0, 0.002);
    }
}

TEST(IsdLineScanner, LroNacSeesTheMoonBelowIt) {
    const ernie::IsdLineScanner nac(ernie::ReadIsdFile(Fixture("toughLroNacLineScan.json")));
    // The middle of the image, some 36 km below the camera; samples about 1.4 m apart (2 x 2
    // summed 7 micrometre pixels at 700 mm).
    const double line = 4096.0;
    const glm::dvec3 a = nac.ImageToGround(line, 1266.0);
    const glm::dvec3 b = nac.ImageToGround(line, 1267.0);
    const glm::dvec3 eye = nac.Position(nac.ImageTime(line));
    EXPECT_NEAR(glm::length(a), 1737.4e3, 1e-3) << "on the sphere";
    EXPECT_NEAR(glm::length(eye - a), 36e3, 3e3);
    EXPECT_NEAR(glm::length(b - a), glm::length(eye - a) * 0.014 / 699.62, 0.05);
    for (const double sample : {10.5, 1266.0, 2500.5}) {
        const glm::dvec2 back = nac.GroundToImage(nac.ImageToGround(line, sample, 250.0));
        EXPECT_NEAR(back.x, line, 1e-4) << sample;
        EXPECT_NEAR(back.y, sample, 1e-4) << "through the distortion and back";
    }
}

TEST(IsdLineScanner, StripsFitTheLensDistortionAlongTheLine) {
    for (const char* name : {"toughLroNacLineScan.json", "ctxLineScan.json"}) {
        const ernie::IsdLineScanner scanner(ernie::ReadIsdFile(Fixture(name)));
        const ernie::IsdLineScanner::Strip strip = scanner.StripOf(1000, 64);
        EXPECT_NO_THROW(strip.camera.Check()) << name;
        // Real ISDs' boresight is the sensor's +z, away from usgscsm's look: the camera faces
        // the ground all the same.
        const glm::dvec3 ground = scanner.ImageToGround(1032.0, 0.5 * scanner.Isd().samples);
        const glm::dvec3 forward = -glm::dvec3(strip.to[2]);
        EXPECT_GT(glm::dot(glm::normalize(ground - glm::dvec3(strip.to[3])), forward), 0.99)
            << name;
        EXPECT_NE(strip.camera.lens.k1, 0.0) << name;
        EXPECT_LT(strip.distortionResidualPixels, 0.01) << name;
        EXPECT_GT(std::abs(strip.camera.lens.Undistort({0.02, 0.0}).x - 0.02) *
                      strip.camera.FocalLengthPixels(),
                  1.0)
            << name << ": the distortion moves the line's ends by pixels";
    }
}

TEST(Isd, ReadsTheLroWideAngleCamerasFramelets) {
    const ernie::IsdSensor wac = ernie::ReadIsdFile(Fixture("lroWacPushFrame.json"));
    EXPECT_TRUE(wac.IsPushFrame());
    EXPECT_FALSE(wac.IsLineScanner());
    EXPECT_EQ(wac.lines, 5096u);
    EXPECT_EQ(wac.samples, 1024u);
    EXPECT_EQ(wac.frameletHeight, 14u) << "a visible band's 14 lines";
    EXPECT_EQ(wac.frameletOverlap, 0u);
    EXPECT_FALSE(wac.frameletsFlipped);
    EXPECT_TRUE(wac.frameletOrderReversed);
    EXPECT_DOUBLE_EQ(wac.exposureDuration, 0.039);
    EXPECT_DOUBLE_EQ(wac.interframeDelay, 0.515625);
    EXPECT_THROW(ernie::IsdLineScanner{wac}, std::invalid_argument);
    EXPECT_THROW(ernie::IsdPushFrame(ernie::ReadIsdFile(Fixture("orbitalLineScan.json"))),
                 std::invalid_argument);
}

// usgscsm's own expectations for this ISD (its tests/PushFrameCameraTests.cpp).
TEST(IsdPushFrame, MatchesUsgscsmOnItsTestCamera) {
    const ernie::IsdPushFrame camera(ernie::ReadIsdFile(Fixture("orbitalPushFrame.json")));
    EXPECT_EQ(camera.FrameletLines(), 12u);
    EXPECT_EQ(camera.Framelets(), 30u);
    const glm::dvec3 center = camera.ImageToGround(6.0, 8.0);
    EXPECT_NEAR(center.x, 1000000.0, 1e-9);
    EXPECT_NEAR(center.y, 0.0, 1e-9);
    EXPECT_NEAR(center.z, 0.0, 1e-9);
    const glm::dvec2 image = camera.GroundToImage({1000000.0, 0.0, 0.0});
    EXPECT_NEAR(image.x, 6.0, 0.001);
    EXPECT_NEAR(image.y, 8.0, 0.001);
    // Framelets overlap on the ground, so a point may come back in a neighbour's lines.
    for (double line = 0.5; line < 360.0; line += 1.0) {
        for (double sample = 0.5; sample < 16.0; sample += 1.0) {
            const glm::dvec2 back = camera.GroundToImage(camera.ImageToGround(line, sample));
            EXPECT_NEAR(back.x, line, 10.0);
            EXPECT_NEAR(back.y, sample, 10.0);
        }
    }
    EXPECT_NEAR(glm::length(camera.ImageToGround(6.0, 8.0, 100.0) - center), 100.0, 0.01);
}

TEST(IsdPushFrame, LroWacFrameletsRunBackInTime) {
    const ernie::IsdPushFrame wac(ernie::ReadIsdFile(Fixture("lroWacPushFrame.json")));
    EXPECT_EQ(wac.FrameletLines(), 14u);
    EXPECT_EQ(wac.Framelets(), 364u);
    // `framelet_order_reversed`: the image's top framelet is exposed last.
    const ernie::IsdSensor& isd = wac.Isd();
    EXPECT_DOUBLE_EQ(wac.ImageTime(0.5),
                     isd.startingTime + 0.5 * isd.exposureDuration + 363 * isd.interframeDelay);
    EXPECT_DOUBLE_EQ(wac.ImageTime(5095.5), isd.startingTime + 0.5 * isd.exposureDuration);
    EXPECT_DOUBLE_EQ(wac.ImageTime(14.0) - wac.ImageTime(13.9), -isd.interframeDelay);
    // A framelet's middle line comes back to itself, through the lens's distortion.
    for (const double line : {7.0, 2555.0, 5089.0}) {
        for (const double sample : {10.5, 512.0, 1013.5}) {
            const glm::dvec3 ground = wac.ImageToGround(line, sample);
            EXPECT_NEAR(glm::length(ground), isd.semiMajorAxis, 1e-3);
            const glm::dvec2 back = wac.GroundToImage(ground);
            EXPECT_NEAR(back.x, line, 1e-4) << line << ", " << sample;
            EXPECT_NEAR(back.y, sample, 1e-4) << line << ", " << sample;
        }
    }
}

// Each framelet renders as a frame camera: the ISD's ground points land on its pixels.
TEST(IsdPushFrame, FrameletsAreFrameCamerasOnTheIsdsPixels) {
    for (const char* name : {"lroWacPushFrame.json", "orbitalPushFrame.json"}) {
        const ernie::IsdPushFrame camera(ernie::ReadIsdFile(Fixture(name)));
        const double height = camera.FrameletLines();
        const double samples = camera.Isd().samples;
        for (const std::uint32_t index : {0u, camera.Framelets() / 2, camera.Framelets() - 1}) {
            const ernie::IsdPushFrame::Framelet framelet = camera.FrameletOf(index);
            const ernie::Camera& c = framelet.camera;
            EXPECT_NO_THROW(c.Check()) << name;
            EXPECT_EQ(c.width, camera.Isd().samples);
            EXPECT_EQ(c.height, camera.FrameletLines());
            EXPECT_EQ(framelet.firstLine, index * camera.FrameletLines());
            EXPECT_DOUBLE_EQ(framelet.time, camera.ImageTime(framelet.firstLine + 0.5));
            EXPECT_LT(framelet.distortionResidualPixels, 0.02) << name;
            const glm::dmat4 toCamera = glm::inverse(c.transform);
            for (const double row : {0.5, 0.5 * height, height - 0.5}) {
                for (const double sample : {0.5, 0.5 * samples, samples - 0.5}) {
                    const glm::dvec3 ground =
                        camera.ImageToGround(framelet.firstLine + row, sample);
                    const glm::dvec3 local(toCamera * glm::dvec4(ground, 1.0));
                    ASSERT_LT(local.z, 0.0) << name << ": in front of the camera";
                    const glm::dvec2 d = c.lens.Distort(glm::dvec2(local) / -local.z);
                    glm::dvec2 pixel = c.PrincipalPoint() + glm::dvec2(d.x, -d.y) *
                                                                c.FocalLengthPixels();
                    if (framelet.rowsReversed) {
                        pixel.y = height - pixel.y;
                    }
                    EXPECT_NEAR(pixel.x, sample, 0.02) << name << " " << index << ", " << row;
                    EXPECT_NEAR(pixel.y, row, 0.02) << name << " " << index << ", " << sample;
                }
            }
        }
        EXPECT_THROW(camera.FrameletOf(camera.Framelets()), std::invalid_argument);
    }
}

TEST(LrocNac, AuthorsThePublishedSensorAndReadout) {
    ernie::Scene scene;
    scene.Define("/Camera", ernie::tokens::kCamera);
    scene.Define("/Product", ernie::tokens::kRenderProduct);
    ernie::AuthorLrocNac(scene, "/Camera", "/Product", ernie::LrocNac::Right);
    const ernie::Camera camera = ernie::CameraOf(scene, "/Camera", 64, 8);
    // Robinson et al. 2010, Table 2, NAC-R.
    EXPECT_DOUBLE_EQ(camera.fStop, 3.590);
    EXPECT_DOUBLE_EQ(camera.sensor.ElectronsPerDn(camera.iso), 92.5);
    EXPECT_DOUBLE_EQ(camera.sensor.readNoiseElectrons, 97.0);
    EXPECT_DOUBLE_EQ(camera.sensor.fullWellElectrons, 352000.0);
    EXPECT_EQ(camera.sensor.adcBits, 12);
    ASSERT_EQ(camera.sensor.channels.size(), 1u);
    EXPECT_NEAR(camera.sensor.channels[0].quantumEfficiency(550.0), 0.6, 1e-6);
    EXPECT_NEAR(camera.sensor.channels[0].quantumEfficiency(800.0), 0.0, 1e-6);
    const auto* filters = scene.Get("/Product", ernie::tokens::kFilters);
    ASSERT_NE(filters, nullptr);
    std::vector<std::string> types;
    for (const ernie::Path& filter : *filters->Get<std::vector<ernie::Path>>()) {
        types.push_back(*scene.Get(filter, ernie::tokens::kFilterType)->Get<std::string>());
    }
    EXPECT_EQ(types, (std::vector<std::string>{"qe", "noise", "adc", "readoutTaps", "compandTable",
                                               "cie", "exposure"}));
    EXPECT_THROW(ernie::AuthorLrocNac(scene, "/Product", "/Camera"), std::invalid_argument);
}

TEST(LrocWac, AuthorsEachBandToItsPublishedResponsivity) {
    for (const int name : {320, 360, 415, 565, 605, 645, 690}) {
        const ernie::LrocWacBand band = ernie::LrocWacBandNamed(name);
        const ernie::LrocWacBandSpec spec = ernie::LrocWacBandOf(band);
        ernie::Scene scene;
        scene.Define("/Camera", ernie::tokens::kCamera)
            .Set(ernie::tokens::kVerticalAperture, spec.pixelPitchUm * 1e-3 * 14.0)
            .Set(ernie::tokens::kExposureTime, 0.039);
        scene.Define("/Product", ernie::tokens::kRenderProduct);
        ernie::AuthorLrocWac(scene, "/Camera", "/Product", band);
        const ernie::Camera camera = ernie::CameraOf(scene, "/Camera", 1024, 14);
        // Robinson et al. 2010, Table 3.
        EXPECT_DOUBLE_EQ(camera.sensor.ElectronsPerDn(camera.iso), 25.9);
        EXPECT_DOUBLE_EQ(camera.sensor.readNoiseElectrons, 66.0);
        EXPECT_DOUBLE_EQ(camera.sensor.fullWellElectrons, 46100.0);
        EXPECT_EQ(camera.sensor.adcBits, 11);
        EXPECT_DOUBLE_EQ(camera.fStop, name < 400 ? 5.65 : 5.052);
        // Table 4: a flat spectral radiance of 1 W/m^2/sr/nm (1e9 W/m^3/sr) reads 1e9 / RC DN
        // per second.
        double electrons = 0.0;
        for (const float e : camera.ElectronsPerRadiance(0)) {
            electrons += e;
        }
        const double dnPerSecond = electrons / camera.exposureTime / 25.9;
        EXPECT_NEAR(dnPerSecond / (1e9 / spec.responsivity), 1.0, 1e-4) << name;
        EXPECT_GT(spec.EffectiveQe(), 0.005) << name;
        EXPECT_LT(spec.EffectiveQe(), 0.5) << name;
    }
    EXPECT_EQ(ernie::LrocWacBandNamed(604), ernie::LrocWacBand::Vis605);
    EXPECT_THROW(ernie::LrocWacBandNamed(550), std::invalid_argument);
}
