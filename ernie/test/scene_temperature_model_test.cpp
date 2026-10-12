// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/temperature_model.h"

#include <gtest/gtest.h>

#include <stdexcept>

TEST(TemperatureModel, BuiltinsRegisterAndDispatch) {
    const auto registry = ernie::TemperatureModelRegistry::WithBuiltins();
    ASSERT_EQ(registry.IndexOf("balance"), 0);
    ASSERT_EQ(registry.IndexOf("diurnal"), 1);
    EXPECT_EQ(registry.Find("balance")->Defaults(),
              (std::vector<float>{-1.0f, -1.0f, 3.0f, 0.0f, 1.0f}));
    // The diurnal model's lunar table: its sizes, then 19 x 96 kelvin.
    const std::vector<float> data = registry.Find("diurnal")->defaultData();
    ASSERT_EQ(data.size(), 2u + 19u * 96u);
    EXPECT_EQ(data[0], 19.0f);
    EXPECT_EQ(data[1], 96.0f);
    const auto modules = ernie::TemperatureModelModules(registry);
    ASSERT_EQ(modules.size(), 4u);
    EXPECT_EQ(modules.back().name, "temperature_models");
    EXPECT_NE(modules.back().source.find("case 1"), std::string::npos);
}

TEST(TemperatureModel, AddChecksTheModel) {
    auto registry = ernie::TemperatureModelRegistry::WithBuiltins();
    ernie::TemperatureModel model;
    model.id = "warm";
    model.typeName = "Warm";
    model.slang = "import temperature_model;\npublic struct Warm : "
                  "ITemperatureModel {}";
    registry.Add(model);
    EXPECT_EQ(registry.IndexOf("warm"), 2);
    EXPECT_THROW(registry.Add(model), std::invalid_argument);
    model.id = "balance";
    EXPECT_THROW(registry.Add(model), std::invalid_argument);
    model.id = "cold";
    model.typeName = "Cold";
    EXPECT_THROW(registry.Add(model), std::invalid_argument);
}

TEST(TemperatureModel, AbsorptivityAndEmissivityWeighTheSpectra) {
    EXPECT_NEAR(ernie::SolarAbsorptivity(ernie::Spectrum::FromPairs(
                    std::vector<float>{300.0f, 0.2f, 2000.0f, 0.2f})),
                0.8, 1e-6);
    EXPECT_NEAR(ernie::ThermalEmissivity([](double) { return 0.9; }, 300.0),
                0.9, 1e-9);
    // Black beyond 8 um only: a black body at 300 K emits about 86%
    // of its power there (the fraction below lambda T = 2400 um K is
    // 0.140).
    EXPECT_NEAR(ernie::ThermalEmissivity(
                    [](double nm) { return nm > 8000.0 ? 1.0 : 0.0; }, 300.0),
                0.86, 0.01);
}
