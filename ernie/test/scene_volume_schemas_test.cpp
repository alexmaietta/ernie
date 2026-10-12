// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/volume_schemas.h"

#include "ernie/scene/material_schemas.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tokens.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

using ernie::Path;
using ernie::PrimTypeDef;
using ernie::Scene;
using ernie::Value;
namespace tokens = ernie::tokens;

TEST(VolumeSchemas, OpenVDBAssetReadsFallbacksAndSets) {
    Scene scene;
    ernie::OpenVDBAsset asset =
        ernie::OpenVDBAsset::Define(scene, "/World/Plume/Density");
    EXPECT_EQ(scene.TypeOf("/World/Plume/Density"), "OpenVDBAsset");
    EXPECT_EQ(asset.GetFilePath(), "");
    EXPECT_EQ(asset.GetFieldName(), "");
    EXPECT_EQ(asset.GetFieldIndex(), 0);
    EXPECT_EQ(asset.GetVectorDataRoleHint(), "None");

    asset.SetFilePath("plume.nvdb")
        .SetFieldName("density")
        .SetFieldIndex(1);
    EXPECT_EQ(*scene.Get("/World/Plume/Density", tokens::kFilePath),
              Value(std::string("plume.nvdb")));
    EXPECT_EQ(asset.GetFieldName(), "density");
    EXPECT_EQ(asset.GetFieldIndex(), 1);
    EXPECT_THROW(asset.GetPrim().Set("fieldNmae", std::string("x")),
                 std::invalid_argument);
}

TEST(VolumeSchemas, VolumeNamesItsFields) {
    Scene scene;
    ernie::Volume volume =
        ernie::Volume::Define(scene, "/World/Plume");
    ernie::OpenVDBAsset::Define(scene, "/World/Plume/Density");
    ernie::OpenVDBAsset::Define(scene, "/World/Plume/Temperature");
    EXPECT_TRUE(volume.FieldNames().empty());
    EXPECT_FALSE(volume.HasField("density"));
    EXPECT_EQ(volume.GetField("density"), Path());

    volume.SetField("density", "/World/Plume/Density")
        .SetField("temperature", "/World/Plume/Temperature");
    EXPECT_EQ(volume.GetField("density"),
              Path("/World/Plume/Density"));
    EXPECT_EQ(*scene.Get("/World/Plume", "field:temperature"),
              Value(Path("/World/Plume/Temperature")));
    EXPECT_EQ(volume.FieldNames(),
              (std::vector<std::string>{"density", "temperature"}));

    // Any field name is allowed (field is an open namespace), and the
    // material binding comes from Gprim.
    EXPECT_NO_THROW(scene.Set("/World/Plume", "field:flame",
                              Value(Path("/World/Plume/Density"))));
    volume.SetMaterialBinding("/Looks/Smoke");
    EXPECT_EQ(volume.GetMaterialBinding(), Path("/Looks/Smoke"));

    volume.ClearField("flame");
    EXPECT_EQ(volume.FieldNames().size(), 2u);
    EXPECT_EQ(ernie::FieldRelationship("density"), "field:density");
}

TEST(VolumeSchemas, MaterialHasVolumeInputs) {
    Scene scene;
    ernie::Material material =
        ernie::Material::Define(scene, "/Looks/Smoke");
    EXPECT_EQ(material.GetDensity(), 1.0);
    EXPECT_EQ(material.GetDensityField(), "density");
    EXPECT_EQ(material.GetVolumeAlbedo(), glm::dvec3(0.0));
    EXPECT_EQ(material.GetAnisotropy(), 0.0);
    EXPECT_EQ(material.GetEmissionStrength(), 0.0);
    EXPECT_EQ(material.GetEmissionField(), "");
    EXPECT_EQ(material.GetBlackbody(), 1.0);
    EXPECT_EQ(material.GetTemperature(), 0.0);
    EXPECT_EQ(material.GetTemperatureField(), "temperature");
    EXPECT_EQ(material.GetTemperatureScale(), 1.0);

    material.SetDensity(20.0).SetVolumeAlbedo({0.9, 0.8, 0.7});
    EXPECT_EQ(*scene.Get("/Looks/Smoke", tokens::kVolumeDensity),
              Value(20.0));
    EXPECT_EQ(ernie::Material::Definition()
                  .Find(tokens::kVolumeAnisotropy)
                  ->meta.group,
              "Volume");
}

TEST(VolumeSchemas, TypesAreRegistered) {
    const PrimTypeDef& volume = ernie::GetPrimType(tokens::kVolume);
    EXPECT_EQ(volume.base, &ernie::Gprim::Definition());
    EXPECT_TRUE(volume.IsOpen(tokens::kField));
    const PrimTypeDef& asset =
        ernie::GetPrimType(tokens::kOpenVDBAsset);
    EXPECT_TRUE(asset.IsA(ernie::GetPrimType(tokens::kFieldAsset)));
    EXPECT_TRUE(asset.IsA(ernie::GetPrimType(tokens::kFieldBase)));
    EXPECT_TRUE(ernie::GetPrimType(tokens::kFieldBase).abstract);
    EXPECT_TRUE(asset.Find(tokens::kFilePath)->meta.asset);

    const std::vector<const PrimTypeDef*> types =
        ernie::VolumeSchemaTypes();
    ASSERT_EQ(types.size(), 4u);
    EXPECT_EQ(types[2], &asset);
    EXPECT_EQ(types[3], &volume);
}
