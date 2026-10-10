// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/terrain_schemas.h"

#include "ernie/scene/light_schemas.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tokens.h"
#include "ernie/util/tensor.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

using ernie::AttributeKind;
using ernie::PrimTypeDef;
using ernie::Scene;
using ernie::Tensor;
using ernie::Value;
namespace tokens = ernie::tokens;

TEST(TerrainSchemas, TerrainReadsFallbacksAndSets) {
    Scene scene;
    ernie::Terrain terrain = ernie::Terrain::Define(scene, "/World/Terrain");
    EXPECT_EQ(scene.TypeOf("/World/Terrain"), "Terrain");
    EXPECT_EQ(terrain.GetHeights(), nullptr);
    EXPECT_EQ(terrain.GetOrigin(), glm::dvec2(0.0));
    EXPECT_EQ(terrain.GetCellSize(), 1.0);
    EXPECT_EQ(terrain.GetExaggeration(), 1.0);
    EXPECT_EQ(terrain.GetNoData(), 0.0);
    EXPECT_FALSE(terrain.HasNoData());
    EXPECT_EQ(terrain.GetPlanetRadius(), 0.0);
    EXPECT_FALSE(terrain.HasPlanetRadius());
    EXPECT_EQ(terrain.GetDisplayColor(), glm::dvec3(0.5));

    const std::vector<float> heights{1.0f, 2.0f, 3.0f, 4.0f};
    terrain.SetHeights(Tensor::Copy<float>(heights, {2, 2}))
        .SetOrigin({10.0, 20.0})
        .SetCellSize(30.0)
        .SetNoData(-9999.0)
        .SetPlanetRadius(1737.4e3);
    ASSERT_NE(terrain.GetHeights(), nullptr);
    EXPECT_EQ(terrain.GetHeights()->Shape()[0], 2);
    EXPECT_EQ(scene.Read("/World/Terrain", tokens::kHeights), terrain.GetHeights());
    EXPECT_EQ(terrain.GetOrigin(), glm::dvec2(10.0, 20.0));
    EXPECT_EQ(*scene.Get("/World/Terrain", tokens::kCellSize), Value(30.0));
    EXPECT_TRUE(terrain.HasNoData());
    EXPECT_EQ(terrain.GetNoData(), -9999.0);
    EXPECT_EQ(terrain.GetPlanetRadius(), 1737.4e3);
    // Authored as a float, it still reads as a double.
    scene.Set("/World/Terrain", tokens::kExaggeration, 2.0f);
    EXPECT_EQ(terrain.GetExaggeration(), 2.0);
}

TEST(TerrainSchemas, StarFieldReadsFallbacksAndSets) {
    Scene scene;
    ernie::StarField stars = ernie::StarField::Define(scene, "/World/Stars");
    EXPECT_EQ(stars.GetMagnitudeLimit(), 99.0);
    EXPECT_EQ(stars.GetZeroPoint(), 2.54e-6);
    EXPECT_EQ(stars.GetExposure(), 0.0);
    EXPECT_EQ(stars.GetMagnitude(), nullptr);
    EXPECT_EQ(stars.GetDirection(), nullptr);

    const std::vector<float> values{1.0f, 2.5f};
    stars.SetMagnitude(Tensor::Copy<float>(values, {2}))
        .SetRightAscension(Tensor::Copy<float>(values, {2}))
        .SetMagnitudeLimit(6.0)
        .SetExposure(-2.0);
    ASSERT_NE(stars.GetMagnitude(), nullptr);
    EXPECT_EQ(stars.GetMagnitude()->Shape()[0], 2);
    EXPECT_TRUE(stars.HasRightAscension());
    EXPECT_FALSE(stars.HasDeclination());
    EXPECT_EQ(*scene.Get("/World/Stars", tokens::kStarsMagnitudeLimit), Value(6.0));
    EXPECT_EQ(*scene.Get("/World/Stars", tokens::kExposure), Value(-2.0));
}

TEST(TerrainSchemas, ViewsCheckTypes) {
    Scene scene;
    ernie::Terrain::Define(scene, "/Terrain");
    ernie::StarField::Define(scene, "/Stars");
    ernie::DistantLight::Define(scene, "/Sun");
    EXPECT_NO_THROW(ernie::Gprim(scene, "/Terrain"));
    EXPECT_NO_THROW(ernie::Imageable(scene, "/Stars"));
    EXPECT_THROW(ernie::Gprim(scene, "/Stars"), std::invalid_argument);
    EXPECT_THROW(ernie::Terrain(scene, "/Stars"), std::invalid_argument);
    EXPECT_THROW(ernie::StarField(scene, "/Sun"), std::invalid_argument);
    EXPECT_THROW(ernie::Terrain(scene, "/Missing"), std::invalid_argument);

    const Scene& constScene = scene;
    ernie::Terrain readOnly(constScene, "/Terrain");
    EXPECT_EQ(readOnly.GetCellSize(), 1.0);
    EXPECT_THROW(readOnly.SetCellSize(2.0), std::logic_error);
}

TEST(TerrainSchemas, RegistryDescribesTheTypes) {
    const PrimTypeDef& terrain = ernie::GetPrimType("Terrain");
    EXPECT_FALSE(terrain.abstract);
    ASSERT_NE(terrain.base, nullptr);
    EXPECT_EQ(terrain.base->name, "Gprim");
    EXPECT_TRUE(terrain.IsA(ernie::GetPrimType("Imageable")));
    EXPECT_EQ(terrain.Find(tokens::kHeights)->kind, AttributeKind::Tensor);
    EXPECT_EQ(terrain.Find(tokens::kOrigin)->kind, AttributeKind::Vec2);
    EXPECT_EQ(terrain.Find(tokens::kCellSize)->label, "Cell size");
    EXPECT_EQ(terrain.Find(tokens::kCellSize)->meta.group, "Terrain");
    EXPECT_EQ(terrain.Find(tokens::kPlanetRadius)->pyName, "planet_radius");
    EXPECT_EQ(terrain.Find(tokens::kNoData)->pyName, "no_data");
    EXPECT_EQ(terrain.Find(tokens::kMaterialBinding)->kind, AttributeKind::Relationship);
    EXPECT_EQ(terrain.Find(tokens::kHeightsDirty), nullptr);

    const PrimTypeDef& stars = ernie::GetPrimType("StarField");
    EXPECT_FALSE(stars.abstract);
    EXPECT_EQ(stars.base, &ernie::GetPrimType("Imageable"));
    EXPECT_EQ(stars.Find(tokens::kDisplayColor), nullptr);
    EXPECT_EQ(stars.Find(tokens::kStarsZeroPoint)->fallback, Value(2.54e-6));
    EXPECT_EQ(stars.Find(tokens::kStarsMagnitudeLimit)->meta.min, -30.0);
    EXPECT_EQ(stars.FindByPyName("color_index")->name, tokens::kStarsColorIndex);
    EXPECT_EQ(stars.FindByPyName("right_ascension")->kind, AttributeKind::Tensor);
    EXPECT_EQ(stars.AllAttributes().front()->name, tokens::kVisibility);

    const std::vector<const PrimTypeDef*> types = ernie::TerrainSchemaTypes();
    ASSERT_EQ(types.size(), 2u);
    EXPECT_EQ(types[0], &terrain);
    EXPECT_EQ(types[1], &stars);
}
