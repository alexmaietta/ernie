#include "ernie/scene/isd.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <string>

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
