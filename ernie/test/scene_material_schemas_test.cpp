// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/material_schemas.h"

#include "ernie/scene/brdf.h"
#include "ernie/scene/material.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tokens.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using ernie::AttributeKind;
using ernie::Path;
using ernie::PrimTypeDef;
using ernie::Scene;
using ernie::Value;
namespace tokens = ernie::tokens;

TEST(MaterialSchemas, MaterialFallbacksMatchMaterialSettings) {
    Scene scene;
    const ernie::Material material = ernie::Material::Define(scene, "/Looks/Rock");
    const ernie::MaterialSettings settings;
    const ernie::DetailSettings& detail = settings.detail;
    EXPECT_EQ(material.GetBrdf(), settings.brdf);
    EXPECT_EQ(glm::vec3(material.GetAlbedo()), settings.albedo);
    EXPECT_EQ(material.GetAlbedoMapEnable(), settings.useAlbedoMap);
    EXPECT_EQ(glm::vec3(material.GetEmissiveColor()), settings.emission);
    EXPECT_EQ(material.GetAlbedoMap(), nullptr);
    EXPECT_EQ(material.GetAlbedoMapOrigin(), glm::dvec2(0.0));
    EXPECT_EQ(material.GetAlbedoMapCellSize(), 1.0);
    EXPECT_EQ(material.GetDetailAlbedo(), nullptr);
    EXPECT_EQ(material.GetDetailNormal(), nullptr);
    EXPECT_FLOAT_EQ(material.GetDetailTileSize(), detail.tileSize);
    EXPECT_FLOAT_EQ(material.GetDetailAlbedoStrength(), detail.albedoStrength);
    EXPECT_FLOAT_EQ(material.GetDetailNormalStrength(), detail.normalStrength);
    EXPECT_EQ(material.GetDetailDirectxNormals(), detail.directXNormals);
    EXPECT_EQ(material.GetDetailHexTiling(), detail.hexTiling);
    EXPECT_FLOAT_EQ(material.GetDetailHexRotation(), detail.hexRotation);
    EXPECT_FLOAT_EQ(material.GetDetailHexContrast(), detail.hexContrast);
    EXPECT_FLOAT_EQ(material.GetDetailHexSharpness(), detail.hexSharpness);
    EXPECT_FALSE(material.HasBrdf());
}

TEST(MaterialSchemas, ShaderFallbacksMatchPreviewSurfaceAndTextureReader) {
    Scene scene;
    const ernie::Shader shader = ernie::Shader::Define(scene, "/Looks/Rock/Surface");
    EXPECT_EQ(shader.GetInfoId(), "");
    EXPECT_EQ(shader.GetDiffuseColor(), glm::dvec3(0.18));
    EXPECT_EQ(shader.GetEmissiveColor(), glm::dvec3(0.0));
    EXPECT_EQ(shader.GetOpacity(), 1.0);
    EXPECT_EQ(shader.GetOpacityThreshold(), 0.0);
    // The preview surface model's parameters read from the shader: the declared fallbacks are the
    // model's defaults.
    const ernie::BrdfRegistry brdfs = ernie::BrdfRegistry::WithBuiltins();
    const ernie::BrdfModel* model = brdfs.Find("preview_surface");
    ASSERT_NE(model, nullptr);
    const PrimTypeDef& type = ernie::Shader::Definition();
    for (const ernie::BrdfParam& param : model->params) {
        if (param.name.starts_with("specularColor")) {
            continue;
        }
        const ernie::AttributeDef* def = type.Find("inputs:" + param.name);
        ASSERT_NE(def, nullptr) << param.name;
        const Value value = def->Read(scene, shader.GetPath());
        if (const auto* number = value.Get<double>()) {
            EXPECT_FLOAT_EQ(static_cast<float>(*number), param.defaultValue) << param.name;
        } else {
            EXPECT_EQ(*value.Get<bool>() ? 1.0f : 0.0f, param.defaultValue) << param.name;
        }
    }
    const glm::dvec3 specular = shader.GetSpecularColor();
    for (int c = 0; c < 3; ++c) {
        EXPECT_FLOAT_EQ(static_cast<float>(specular[c]), model->params[4 + c].defaultValue);
    }
    // A texture's.
    const ernie::TextureReader reader;
    EXPECT_EQ(shader.GetFile(), reader.file);
    EXPECT_EQ(shader.GetTexture(), nullptr);
    EXPECT_EQ(shader.GetWrapS(), "useMetadata");
    EXPECT_EQ(shader.GetWrapT(), "useMetadata");
    EXPECT_EQ(shader.GetSourceColorSpace(), "auto");
    EXPECT_EQ(shader.GetVarname(), reader.uvSet);
    EXPECT_FLOAT_EQ(static_cast<float>(shader.GetRotation()), reader.uvRotation);
    EXPECT_EQ(glm::vec2(shader.GetTranslation()), reader.uvTranslation);
}

TEST(MaterialSchemas, ViewsSetByUsdName) {
    Scene scene;
    ernie::Material material = ernie::Material::Define(scene, "/Looks/Rock");
    material.SetBrdf("hapke").SetAlbedo(glm::dvec3(0.3)).SetDetailTileSize(2.0);
    EXPECT_TRUE(material.HasBrdf());
    EXPECT_EQ(*scene.Get("/Looks/Rock", tokens::kBrdf), Value(std::string("hapke")));
    EXPECT_EQ(*scene.Get("/Looks/Rock", tokens::kDetailTileSize), Value(2.0));
    EXPECT_EQ(material.GetAlbedo(), glm::dvec3(0.3));
    // Authored as float vectors and ints, they read as the declared types.
    scene.Set("/Looks/Rock", tokens::kAlbedo, glm::vec3(0.5f));
    scene.Set("/Looks/Rock", tokens::kDetailHexSharpness, 3);
    EXPECT_EQ(material.GetAlbedo(), glm::dvec3(0.5));
    EXPECT_EQ(material.GetDetailHexSharpness(), 3.0);

    ernie::Shader shader = ernie::Shader::Define(scene, "/Looks/Rock/Surface");
    shader.SetInfoId(tokens::kUsdPreviewSurface).SetRoughness(0.25);
    EXPECT_EQ(*scene.Get("/Looks/Rock/Surface", tokens::kInfoId),
              Value(std::string(tokens::kUsdPreviewSurface)));
    EXPECT_EQ(shader.GetRoughness(), 0.25);
    // USD's int useSpecularWorkflow reads as a bool.
    scene.Set("/Looks/Rock/Surface", tokens::kUseSpecularWorkflow, 1);
    EXPECT_TRUE(shader.GetUseSpecularWorkflow());
}

TEST(MaterialSchemas, ViewsCheckTypes) {
    Scene scene;
    ernie::Material::Define(scene, "/Looks/Rock");
    ernie::Shader::Define(scene, "/Looks/Rock/Surface");
    scene.Define("/Ball", tokens::kSphere);
    EXPECT_NO_THROW(ernie::Imageable(scene, "/Looks/Rock"));
    EXPECT_THROW(ernie::Material(scene, "/Looks/Rock/Surface"), std::invalid_argument);
    EXPECT_THROW(ernie::Shader(scene, "/Looks/Rock"), std::invalid_argument);
    EXPECT_THROW(ernie::Material(scene, "/Ball"), std::invalid_argument);
    EXPECT_THROW(ernie::Material(scene, "/Missing"), std::invalid_argument);
    const Scene& constScene = scene;
    ernie::Material readOnly(constScene, "/Looks/Rock");
    EXPECT_THROW(readOnly.SetBrdf("lambert"), std::logic_error);
}

TEST(MaterialSchemas, RegistryDescribesTypes) {
    const PrimTypeDef* material = ernie::FindPrimType(tokens::kMaterial);
    const PrimTypeDef* shader = ernie::FindPrimType(tokens::kShader);
    ASSERT_NE(material, nullptr);
    ASSERT_NE(shader, nullptr);
    EXPECT_FALSE(material->abstract);
    EXPECT_FALSE(shader->abstract);
    EXPECT_EQ(material->base, &ernie::Imageable::Definition());
    EXPECT_EQ(shader->base, &ernie::Imageable::Definition());
    EXPECT_EQ(material, &ernie::Material::Definition());

    const ernie::AttributeDef* brdf = material->Find(tokens::kBrdf);
    ASSERT_NE(brdf, nullptr);
    EXPECT_EQ(brdf->kind, AttributeKind::Token);
    EXPECT_EQ(brdf->label, "BRDF");
    EXPECT_EQ(brdf->meta.group, "Surface");
    EXPECT_EQ(material->Find(tokens::kAlbedoMap)->kind, AttributeKind::Tensor);
    const ernie::AttributeDef* sharpness = material->FindByPyName("detail_hex_sharpness");
    ASSERT_NE(sharpness, nullptr);
    EXPECT_EQ(sharpness->name, tokens::kDetailHexSharpness);
    EXPECT_EQ(sharpness->meta.group, "Detail textures");
    EXPECT_EQ(sharpness->meta.min, 1.0);
    EXPECT_EQ(sharpness->meta.max, 32.0);
    EXPECT_NE(material->FindByPyName("detail_directx_normals"), nullptr);
    EXPECT_NE(material->Find(tokens::kVisibility), nullptr);

    const ernie::AttributeDef* id = shader->Find(tokens::kInfoId);
    ASSERT_NE(id, nullptr);
    EXPECT_EQ(id->meta.options.size(), 4u);
    EXPECT_TRUE(shader->Find(tokens::kFile)->meta.asset);
    EXPECT_EQ(shader->Find(tokens::kShaderTexture)->kind, AttributeKind::Tensor);
    // Open namespaces and connections stay undeclared.
    EXPECT_EQ(material->Find("inputs:b"), nullptr);
    EXPECT_EQ(shader->Find(tokens::kShaderScale), nullptr);
}
