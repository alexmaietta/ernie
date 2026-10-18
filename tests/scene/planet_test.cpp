// SPDX-License-Identifier: BSD-3-Clause
/// @file planet_test.cpp
/// @brief Tests of terrains, atmospheres, star fields and ground planes, and their extents.
#include "ernie/scene/schemas/planet.h"
#include "ernie/scene/xform.h"

#include "tests/common.h"

#include <gtest/gtest.h>

#include <vector>

namespace ernie {
namespace {

using test::BoxNear;

constexpr double kMoonRadius = 1737.4e3;
constexpr double kEarthRadius = 6371.0e3;

Tensor Heights(std::vector<float> values, std::vector<std::int64_t> shape) {
    return Tensor::Copy<float>(values, std::move(shape));
}

TEST(Planet, ABodysTerrainReachesItsHighestHeight) {
    Scene scene;
    Sphere::Define(scene, "/Moon").SetRadius(kMoonRadius);
    Terrain::Define(scene, "/Moon/Terrain")
        .SetBody(Path("/Moon"))
        .SetHeights(Heights({-500, 0, 250, 1000}, {2, 2}))
        .SetExaggeration(2.0);
    EXPECT_TRUE(BoxNear(*scene.Extent(*scene.Find("/Moon/Terrain")),
                        glm::dvec3(-kMoonRadius - 2000.0), glm::dvec3(kMoonRadius + 2000.0), 1e-6));
}

TEST(Planet, AFlatTerrainCoversItsFootprintAndHeightRange) {
    Scene scene;
    const Terrain site = Terrain::Define(scene, "/Site")
                             .SetHeights(Heights({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {3, 4}))
                             .SetOrigin({10.0, 20.0})
                             .SetCellSize(2.0);
    EXPECT_TRUE(BoxNear(*scene.Extent(site.Id()), {10.0, 16.0, 0.0}, {16.0, 20.0, 11.0}))
        << "four columns east, three rows south of the north-west corner";
}

TEST(Planet, EachAtmosphereSurroundsItsOwnBody) {
    Scene scene;
    Sphere::Define(scene, "/Earth").SetRadius(kEarthRadius);
    Sphere::Define(scene, "/Mars").SetRadius(3389.5e3);
    SetXform(scene, "/Mars", {.translate = glm::dvec3(1e11, 0.0, 0.0)});
    const Atmosphere air = Atmosphere::Define(scene, "/EarthAir").SetBody(Path("/Earth"));
    const Atmosphere dust =
        Atmosphere::Define(scene, "/MarsAir").SetBody(Path("/Mars")).SetPreset("mars").SetTop(80e3);
    EXPECT_TRUE(BoxNear(*scene.Extent(air.Id()), glm::dvec3(-kEarthRadius - 100e3),
                        glm::dvec3(kEarthRadius + 100e3), 1e-6));
    EXPECT_NEAR(scene.Extent(dust.Id())->Center().x, 1e11, 1e-3) << "Mars's, far away";
    EXPECT_NEAR(scene.Extent(dust.Id())->Size().y, 2.0 * (3389.5e3 + 80e3), 1e-6);
    EXPECT_TRUE(scene.Extent(Atmosphere::Define(scene, "/Nowhere").Id())->Empty())
        << "no body, nothing to surround";
}

TEST(Planet, StarsAndGroundPlanesAreLeftOutOfFraming) {
    Scene scene;
    Xform::Define(scene, "/World");
    StarField::Define(scene, "/World/Stars");
    GroundPlane::Define(scene, "/World/Ground");
    Cube::Define(scene, "/World/Rock");
    EXPECT_TRUE(BoxNear(scene.WorldBounds("/World"), glm::dvec3(-1.0), glm::dvec3(1.0)));
    EXPECT_TRUE(scene.HasApi(*scene.Find("/World/Stars"), "LightAPI")) << "stars are a light";
    EXPECT_EQ(StarField(scene, "/World/Stars").GetZeroPoint(), kMagnitudeZeroLux);
}

}  // namespace
}  // namespace ernie
