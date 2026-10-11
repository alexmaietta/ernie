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
    model.slang = "import light_pattern;\npublic struct Stripes : "
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
    model.id = "image";
    EXPECT_THROW(registry.Add(model), std::invalid_argument)
        << "a built-in's, though this registry has none";
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

TEST(LightPatternRegistry, BuiltinsComeFirst) {
    ernie::LightPatternRegistry registry =
        ernie::LightPatternRegistry::WithBuiltins();
    const std::vector<std::string> ids = {"spot", "line", "grid",
                                          "image", "dots"};
    ASSERT_EQ(registry.Models().size(), ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        EXPECT_EQ(registry.IndexOf(ids[i]), static_cast<int>(i));
        // Only the image needs integrating.
        EXPECT_EQ(registry.Models()[i].normalized, ids[i] != "image")
            << ids[i];
    }
    EXPECT_EQ(registry.Find("grid")->params.size(), 4u);
    EXPECT_THROW(registry.Add(registry.Models()[0]),
                 std::invalid_argument);
    registry.Add(Stripes());
    EXPECT_EQ(registry.IndexOf("stripes"), 5);
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
