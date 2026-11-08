// SPDX-License-Identifier: BSD-3-Clause
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/scene/scene.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/util/error.h"
#include "support/devices.h"

namespace ernie {
namespace {

TEST(MaterialTableTest, ValueEditsPackWithoutResolvingAndConnectionsResolve) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim red = scene->Define<Material>("/Red");
        const Prim blue = scene->Define<Material>("/Blue");
        red.Connect("surface", scene->Define<Shader>("/Red/Surface", {.id = "UsdPreviewSurface"}),
                    "surface");
        blue.Connect("surface", scene->Define<Shader>("/Blue/Surface", {.id = "UsdPreviewSurface"}),
                     "surface");
        const Prim albedo = scene->Define<Shader>("/Blue/Albedo", {.id = "UsdUVTexture"});
        albedo.Set(UsdUVTexture::kFile, "albedo.png");
        scene->Publish().Wait();
        const MaterialTable& materials = SceneImpl::Get(*scene).GetMaterials();
        EXPECT_EQ(materials.Record(red.Index()).definition, kDefinitionPreviewConstant);
        EXPECT_EQ(materials.Record(blue.Index()).definition, kDefinitionPreviewConstant);
        EXPECT_NE(materials.Record(red.Index()).words, materials.Record(blue.Index()).words);
        const uint32_t resolves = materials.Resolves();

        scene->Find("/Blue/Surface")->Set(UsdPreviewSurface::kRoughness, 0.9f);
        scene->Publish().Wait();
        EXPECT_EQ(materials.Resolves(), resolves);

        scene->Find("/Blue/Surface")->Connect("diffuseColor", albedo, "rgb");
        scene->Publish().Wait();
        EXPECT_EQ(materials.Resolves(), resolves + 1);
        EXPECT_EQ(materials.Record(blue.Index()).definition, kDefinitionPreviewTextured);
        EXPECT_EQ(materials.Record(red.Index()).definition, kDefinitionPreviewConstant);

        scene->Find("/Blue/Surface")->Disconnect("diffuseColor");
        scene->Publish().Wait();
        EXPECT_EQ(materials.Record(blue.Index()).definition, kDefinitionPreviewConstant);
    });
}

TEST(MaterialTableTest, ShadersTakeTheSchemaTheirInfoIdNames) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim texture = scene->Define<Shader>("/Texture", {.id = "UsdUVTexture"});
        EXPECT_EQ(texture.Schema(), SchemaId::UsdUVTexture);
        EXPECT_EQ(texture.Get(Shader::kId), "UsdUVTexture");
        EXPECT_EQ(texture.Get(UsdUVTexture::kWrapS), TextureWrapMode::Repeat);
        const std::vector<std::string> paths = {"/A", "/B"};
        const PrimRange placements = scene->Define<Shader>(paths, {.id = "TexturePlacement"});
        EXPECT_EQ(scene->Handles(placements).back().Schema(), SchemaId::TexturePlacement);
        EXPECT_THROW(scene->Define<Shader>("/Missing", {.id = "NoSuchNode"}), Error);
        EXPECT_THROW(scene->Define<Shader>("/Abstract", {.id = "Shader"}), Error);
        EXPECT_FALSE(scene->Find("/Missing"));
    });
}

TEST(MaterialTableTest, MaterialsWithoutAKnownSurfaceShadeAsDiffuse) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim bare = scene->Define<Material>("/Bare");
        const Prim odd = scene->Define<Material>("/Odd");
        odd.Connect("surface", scene->Define<Shader>("/Odd/Texture", {.id = "UsdUVTexture"}),
                    "rgb");
        scene->Publish().Wait();
        const MaterialTable& materials = SceneImpl::Get(*scene).GetMaterials();
        EXPECT_EQ(materials.Record(bare.Index()).definition, kDefinitionDiffuse);
        EXPECT_EQ(materials.Record(odd.Index()).definition, kDefinitionDiffuse);
    });
}

}
}
