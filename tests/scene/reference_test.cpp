// SPDX-License-Identifier: BSD-3-Clause
/// @file reference_test.cpp
/// @brief Tests that every view of the registry covers every schema and attribute.
#include "ernie/scene/reference.h"

#include "tests/common.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

namespace ernie {
namespace {

using test::Contains;

TEST(Reference, SchemasAreListedOnceBasesFirst) {
    const std::span<const SchemaDef* const> schemas = Schemas();
    for (auto at = schemas.begin(); at != schemas.end(); ++at) {
        EXPECT_EQ(std::count(schemas.begin(), schemas.end(), *at), 1) << (*at)->name;
        if ((*at)->base != nullptr) {
            EXPECT_NE(std::find(schemas.begin(), at, (*at)->base), at) << (*at)->name;
        }
    }
}

TEST(Reference, TheJsonSchemaAndPagesShowEverySchemaAndAttribute) {
    const std::string json = SceneJsonSchema();
    for (const SchemaFamily& family : SchemaFamilies()) {
        const std::string page = ReferencePage(family);
        for (const auto* schemas : {&family.usd, &family.ernie}) {
            for (const SchemaDef* schema : *schemas) {
                const std::string name(schema->name);
                EXPECT_TRUE(Contains(json, "\"" + name + "\": {")) << name;
                EXPECT_TRUE(Contains(page, "## " + name + "\n")) << name;
                for (const AttributeDef& def : schema->attributes) {
                    const std::string attribute(def.name);
                    EXPECT_TRUE(Contains(page, "`" + attribute + "`")) << name << " " << attribute;
                    EXPECT_TRUE(Contains(attribute, kInstanceName) ||
                                Contains(json, "\"" + attribute + "\""))
                        << name << " " << attribute;
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
        const bool declared =
            Contains(plugin.generatedSchema, "class " + name + " \"" + name + "\"") ||
            Contains(plugin.generatedSchema, "class \"" + name + "\"");
        EXPECT_EQ(declared, IsErnieSchema(name)) << name;
        EXPECT_EQ(Contains(plugin.plugInfo, "\"schemaIdentifier\": \"" + name + "\""), declared)
            << name;
    }
    EXPECT_TRUE(Contains(plugin.plugInfo, "\"bases\": [\"UsdLuxBoundableLightBase\"]"))
        << "LaserLight";
}

}  // namespace
}  // namespace ernie
