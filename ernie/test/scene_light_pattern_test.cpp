// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/light_pattern.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

namespace {

ernie::LightPatternModel Stripes() {
    ernie::LightPatternModel model;
    model.id = "stripes";
    model.typeName = "Stripes";
    model.slang = "import light_pattern;\nstruct Stripes : "
                  "ILightPattern { ... }";
    model.params = {{"period", 0.1f, ""}, {"duty", 0.5f, ""}};
    model.extent = 0.5;
    return model;
}

}  // namespace

TEST(LightPatternRegistry, RegistersPatternsByIdAndChecksThem) {
    ernie::LightPatternRegistry registry;
    registry.Add(Stripes());
    ASSERT_NE(registry.Find("stripes"), nullptr);
    EXPECT_EQ(registry.IndexOf("stripes"), 0);
    EXPECT_EQ(registry.IndexOf("rings"), -1);
    EXPECT_EQ(registry.Find("stripes")->Defaults(),
              (std::vector<float>{0.1f, 0.5f}));

    EXPECT_THROW(registry.Add(Stripes()), std::invalid_argument)
        << "taken";
    ernie::LightPatternModel model = Stripes();
    model.id = "line";
    EXPECT_THROW(registry.Add(model), std::invalid_argument)
        << "a built-in's";
    model = Stripes();
    model.id = "2stripes";
    EXPECT_THROW(registry.Add(model), std::invalid_argument);
    model = Stripes();
    model.id = "rings";
    model.extent = 0.0;
    EXPECT_THROW(registry.Add(model), std::invalid_argument);
    model.extent = 1.0;
    model.typeName = "Rings";
    EXPECT_THROW(registry.Add(model), std::invalid_argument)
        << "its source does not mention Rings";
    model = Stripes();
    model.id = "many";
    model.params.resize(17);
    EXPECT_THROW(registry.Add(model), std::invalid_argument);
}

TEST(LightPatternModules, DispatchEveryPatternByIndex) {
    ernie::LightPatternRegistry registry;
    registry.Add(Stripes());
    const std::vector<ernie::ShaderModuleSource> modules =
        ernie::LightPatternModules(registry);
    ASSERT_EQ(modules.size(), 3u);
    EXPECT_EQ(modules[0].name, "light_pattern");
    EXPECT_EQ(modules[1].name, "light_pattern_stripes");
    EXPECT_EQ(modules[2].name, "light_patterns");
    EXPECT_NE(modules[2].source.find("case 0"), std::string::npos);
    EXPECT_NE(modules[2].source.find("Stripes pattern"),
              std::string::npos);
}
