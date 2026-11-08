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

TEST(MaterialTableTest, ShadersChangeNodeWhenTheirInfoIdChanges) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim material = scene->Define<Material>("/Look");
        const Prim surface = scene->Define<Shader>("/Look/Surface", {.id = "UsdUVTexture"});
        const Prim albedo = scene->Define<Shader>("/Look/Albedo", {.id = "UsdUVTexture"});
        surface.Set(UsdUVTexture::kFile, "albedo.png");
        surface.Connect("diffuseColor", albedo, "rgb");
        material.Connect("surface", surface, "surface");
        scene->Publish().Wait();
        const MaterialTable& materials = SceneImpl::Get(*scene).GetMaterials();
        EXPECT_EQ(materials.Record(material.Index()).definition, kDefinitionDiffuse);

        surface.Set(Shader::kId, "UsdPreviewSurface");
        EXPECT_EQ(surface.Schema(), SchemaId::UsdPreviewSurface);
        EXPECT_EQ(surface.Get(Shader::kId), "UsdPreviewSurface");
        EXPECT_EQ(surface.Get(UsdPreviewSurface::kRoughness), 0.5f);
        EXPECT_EQ(surface.Get(UsdPreviewSurface::kNormalStrength), 1.0f);
        EXPECT_THROW(surface.Get(UsdUVTexture::kFile), Error);
        EXPECT_EQ(surface.Connections().size(), 1u);
        EXPECT_EQ(material.Connections().front().source, "/Look/Surface");
        surface.Set(UsdPreviewSurface::kRoughness, 0.25f);
        scene->Publish().Wait();
        EXPECT_EQ(materials.Record(material.Index()).definition, kDefinitionPreviewTextured);

        EXPECT_THROW(surface.Set(Shader::kId, "NoSuchNode"), Error);
        EXPECT_EQ(surface.Schema(), SchemaId::UsdPreviewSurface);
        EXPECT_EQ(surface.Get(UsdPreviewSurface::kRoughness), 0.25f);

        surface.Set(Shader::kId, "UsdUVTexture");
        EXPECT_EQ(surface.Get(UsdUVTexture::kFile), "");
        scene->Publish().Wait();
        EXPECT_EQ(materials.Record(material.Index()).definition, kDefinitionDiffuse);
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
