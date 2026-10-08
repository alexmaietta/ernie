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
    EXPECT_DOUBLE_EQ(q.w, 0.9792826872138032) << "scalar last in the ISD";
    EXPECT_DOUBLE_EQ(q.x, 0.0934065633770046);
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
