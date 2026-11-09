// SPDX-License-Identifier: BSD-3-Clause
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/scene/scene.h"
#include "ernie/stage/schema/runtime.h"
#include "ernie/util/error.h"
#include "support/devices.h"

namespace ernie {
namespace {

RuntimeSchemaDesc Puddle() {
    return {.name = "TestPuddle",
            .base = SchemaId::Shader,
            .doc = "A test node.",
            .columns = {{.field = "wetness",
                         .type = ColumnType::Float,
                         .fallback = 0.25f,
                         .minimum = 0,
                         .maximum = 1},
                        {.field = "tint",
                         .type = ColumnType::Float3,
                         .role = ColumnRole::Color,
                         .fallback = Float3{0.1f, 0.2f, 0.3f}},
                        {.field = "label", .type = ColumnType::Token, .fallback = "muddy"}}};
}

TEST(RuntimeSchemaTest, RegisteredNodesAreDefinedByIdAndReadByName) {
    const SchemaId puddle = RegisterSchema(Puddle());
    EXPECT_EQ(RegisterSchema(Puddle()), puddle);
    EXPECT_TRUE(IsRuntime(puddle));
    EXPECT_TRUE(IsA(puddle, SchemaId::Shader));
    EXPECT_EQ(ShaderSchema("TestPuddle"), puddle);
    EXPECT_EQ(Columns(puddle)[1].role, ColumnRole::Color);
    EXPECT_EQ(Columns(puddle)[0].maximum, 1.0f);
    test::ForEachDevice([&](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const Prim prim = scene->Define<Shader>("/Puddle", {.id = "TestPuddle"});
        EXPECT_EQ(prim.Schema(), puddle);
        EXPECT_EQ(prim.Get(Shader::kId), "TestPuddle");
        EXPECT_EQ(prim.Get<float>("wetness"), 0.25f);
        EXPECT_EQ(prim.Get<Float3>("tint"), (Float3{0.1f, 0.2f, 0.3f}));
        prim.Set("wetness", 0.75f);
        scene->Publish().Wait();
        EXPECT_EQ(prim.Get<float>("wetness"), 0.75f);

        const Prim texture = scene->Define<Shader>("/Texture", {.id = "UsdUVTexture"});
        prim.Connect("tint", texture, "rgb");
        EXPECT_EQ(prim.Connections().size(), 1u);

        const Prim plain = scene->Define(std::string("/Plain"), puddle);
        EXPECT_EQ(plain.Get(Shader::kId), "TestPuddle");

        texture.Set(Shader::kId, "TestPuddle");
        EXPECT_EQ(texture.Schema(), puddle);
        EXPECT_EQ(texture.Get<float>("wetness"), 0.25f);
        texture.Set(Shader::kId, "UsdUVTexture");
        EXPECT_EQ(texture.Schema(), SchemaId::UsdUVTexture);
        scene->Publish().Wait();
    });
}

TEST(RuntimeSchemaTest, ClashingAndMalformedSchemasAreRejected) {
    RegisterSchema(Puddle());
    RuntimeSchemaDesc changed = Puddle();
    changed.columns[0].fallback = 0.5f;
    EXPECT_THROW(RegisterSchema(changed), Error);
    EXPECT_THROW(RegisterSchema({.name = "UsdUVTexture", .base = SchemaId::Shader}), Error);
    EXPECT_THROW(
        RegisterSchema(
            {.name = "TestHides",
             .base = SchemaId::Shader,
             .columns = {{.field = "id", .type = ColumnType::Token, .fallback = std::string()}}}),
        Error);
    EXPECT_THROW(
        RegisterSchema({.name = "TestWrongFallback",
                        .base = SchemaId::Shader,
                        .columns = {{.field = "x", .type = ColumnType::Float3, .fallback = 1.0f}}}),
        Error);
    EXPECT_THROW(RegisterSchema({.name = "TestOnAnApi", .base = SchemaId::MaterialBindingAPI}),
                 Error);
    EXPECT_EQ(FindSchemaByName("TestHides"), SchemaId::None);
}

}
}
