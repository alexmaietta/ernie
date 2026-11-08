// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/material/codegen.h"

#include <gtest/gtest.h>

namespace ernie {
namespace {

TEST(MaterialCodegenTest, DispatchLinksOnlyTheBuiltinsWithoutGeneratedDefinitions) {
    const DefinitionSet builtins;
    const auto sources = DispatchSources(builtins);
    ASSERT_EQ(sources.size(), 1u);
    const std::string& dispatch = sources.at(kDispatchFile);
    EXPECT_NE(dispatch.find("return definition < 4;"), std::string::npos);
    EXPECT_EQ(dispatch.find("import ernie.material.generated"), std::string::npos);
    EXPECT_FALSE(builtins.Contains(4));
    EXPECT_TRUE(builtins.Contains(3));
}

TEST(MaterialCodegenTest, GeneratedDefinitionsAddAnImportACaseAndTheirModule) {
    const DefinitionSet set({{.id = 5,
                              .type = "Wet5",
                              .module = "ernie.material.generated.d5",
                              .source = "struct Wet5 {}"}});
    const auto sources = DispatchSources(set);
    ASSERT_EQ(sources.size(), 2u);
    EXPECT_EQ(sources.at("ernie/material/generated/d5.slang"), "struct Wet5 {}");
    const std::string& dispatch = sources.at(kDispatchFile);
    EXPECT_NE(dispatch.find("import ernie.material.generated.d5;"), std::string::npos);
    EXPECT_NE(dispatch.find("|| definition == 5"), std::string::npos);
    EXPECT_NE(dispatch.find("StandardDefinition<Wet5, Spectrum>.Shade(context, closure);"),
              std::string::npos);
    EXPECT_TRUE(set.Contains(5));
    EXPECT_FALSE(set.Contains(6));
    EXPECT_NE(set.Hash(), DefinitionSet().Hash());
}

}
}
