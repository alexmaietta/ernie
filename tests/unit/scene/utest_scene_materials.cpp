// SPDX-License-Identifier: BSD-3-Clause
#include <gtest/gtest.h>

#include "ernie/scene/scene.h"
#include "ernie/scene/scene_impl.h"
#include "support/devices.h"

namespace ernie {
namespace {

TEST(MaterialTableTest, ValueEditsPackWithoutResolvingAndConnectionsResolve) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim red = scene->DefineMaterial("/Red");
        const Prim blue = scene->DefineMaterial("/Blue");
        const Prim albedo = scene->Define<UsdUVTexture>("/Blue/Albedo", {.file = "albedo.png"});
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

TEST(MaterialTableTest, MaterialsWithoutAKnownSurfaceShadeAsDiffuse) {
    test::ForEachDevice([](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim bare = scene->Define<Material>("/Bare");
        const Prim odd = scene->Define<Material>("/Odd");
        odd.Connect("surface", scene->Define<UsdUVTexture>("/Odd/Texture"), "rgb");
        scene->Publish().Wait();
        const MaterialTable& materials = SceneImpl::Get(*scene).GetMaterials();
        EXPECT_EQ(materials.Record(bare.Index()).definition, kDefinitionDiffuse);
        EXPECT_EQ(materials.Record(odd.Index()).definition, kDefinitionDiffuse);
    });
}

}
}
