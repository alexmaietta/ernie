// SPDX-License-Identifier: BSD-3-Clause
/// @file reference_test.cpp
/// @brief Tests that every view of the registry covers every schema and attribute.
#include "ernie/scene/reference.h"

#include <gtest/gtest.h>

#include <string>

namespace ernie {
namespace {

bool Has(const std::string& text, std::string_view part) {
    return text.find(part) != std::string::npos;
}

TEST(Reference, EverySchemaIsInOneFamily) {
    for (const SchemaDef* schema : Schemas()) {
        int families = 0;
        for (const SchemaFamily& family : SchemaFamilies()) {
            families += static_cast<int>(std::ranges::count(family.usd, schema->name) +
                                         std::ranges::count(family.ernie, schema->name));
        }
        EXPECT_EQ(families, 1) << schema->name;
    }
    for (const SchemaFamily& family : SchemaFamilies()) {
        for (const auto* names : {&family.usd, &family.ernie}) {
            for (const std::string_view name : *names) {
                EXPECT_NE(FindSchema(name), nullptr) << name << " is not registered";
            }
        }
    }
}

TEST(Reference, TheJsonSchemaAndPagesShowEverySchemaAndAttribute) {
    const std::string json = SceneJsonSchema();
    for (const SchemaFamily& family : SchemaFamilies()) {
        const std::string page = ReferencePage(family);
        for (const auto* names : {&family.usd, &family.ernie}) {
            for (const std::string_view name : *names) {
                const SchemaDef& schema = *FindSchema(name);
                EXPECT_TRUE(Has(json, "\"" + std::string(name) + "\": {")) << name;
                EXPECT_TRUE(Has(page, "## " + std::string(name) + "\n")) << name;
                for (const AttributeDef& def : schema.attributes) {
                    const std::string quoted = "`" + std::string(def.name) + "`";
                    EXPECT_TRUE(Has(page, quoted)) << name << " " << def.name;
                    const bool instanced = Has(std::string(def.name), kInstanceName);
                    EXPECT_TRUE(instanced || Has(json, "\"" + std::string(def.name) + "\""))
                        << name << " " << def.name;
                }
            }
        }
    }
}

TEST(Reference, ExamplesLoadWithoutProblems) {
    for (const SchemaDef* schema : Schemas()) {
        const std::string example = ExampleScene(*schema);
        EXPECT_EQ(example.empty(), schema->kind == SchemaKind::Abstract) << schema->name;
        for (const SceneProblem& problem :
             ValidateScene(example.empty() ? "{\"prims\": {}}" : example)) {
            ADD_FAILURE() << schema->name << ": " << problem.Message();
        }
    }
}

TEST(Reference, TheUsdPluginDeclaresErniesOwnSchemasOnly) {
    const UsdSchemaPlugin plugin = ErnieUsdSchemaPlugin();
    for (const SchemaDef* schema : Schemas()) {
        const std::string name(schema->name);
        const bool declared = Has(plugin.generatedSchema, "class " + name + " \"" + name + "\"") ||
                              Has(plugin.generatedSchema, "class \"" + name + "\"");
        EXPECT_EQ(declared, IsErnieSchema(name)) << name;
        EXPECT_EQ(Has(plugin.plugInfo, "\"schemaIdentifier\": \"" + name + "\""), declared) << name;
    }
    EXPECT_TRUE(Has(plugin.plugInfo, "\"bases\": [\"UsdLuxBoundableLightBase\"]")) << "LaserLight";
}

}  // namespace
}  // namespace ernie
