// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/prim_schemas.h"
#include "ernie/scene/tokens.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace ernie::app {
namespace {

namespace tokens = ernie::tokens;

std::vector<std::string> Titles(const std::vector<PropertySection>& sections) {
    std::vector<std::string> titles;
    for (const PropertySection& section : sections) {
        titles.push_back(section.title);
    }
    return titles;
}

const PropertyField* FieldOf(const std::vector<PropertySection>& sections,
                             std::string_view attribute) {
    for (const PropertySection& section : sections) {
        for (const PropertyField& field : section.fields) {
            if (field.attribute == attribute) {
                return &field;
            }
        }
    }
    return nullptr;
}

TEST(PrimSchemasTest, LightPanelsComeFromTheRegistry) {
    Scene scene;
    scene.Define("/Lamp", tokens::kSphereLight);
    scene.Define("/Sun", tokens::kDistantLight);
    scene.Define("/Sky", tokens::kDomeLight);
    const BrdfRegistry brdfs = BrdfRegistry::WithBuiltins();
    const SensorFilterRegistry filters = SensorFilterRegistry::WithStandard();

    const std::vector<PropertySection> lamp = SchemaOf(scene, "/Lamp", brdfs, filters);
    EXPECT_EQ(Titles(lamp),
              (std::vector<std::string>{"Light", "Shape", "Spectrum", "Photometric", "Prim"}));
    // The type's own attributes before its bases' in a section; arrays are not edited.
    EXPECT_EQ(lamp[1].fields.front().attribute, tokens::kRadius);
    EXPECT_EQ(lamp[1].fields.back().attribute, tokens::kLightPower);
    EXPECT_EQ(FieldOf(lamp, tokens::kLightSpectrum), nullptr);
    const PropertyField* color = FieldOf(lamp, tokens::kColor);
    ASSERT_NE(color, nullptr);
    EXPECT_EQ(color->kind, PropertyField::Kind::Color);
    EXPECT_EQ(color->label, "Colour");
    const PropertyField* illuminant = FieldOf(lamp, tokens::kPhysicalIlluminant);
    ASSERT_NE(illuminant, nullptr);
    EXPECT_EQ(illuminant->kind, PropertyField::Kind::Token);
    EXPECT_EQ(illuminant->options.size(), 4u);
    EXPECT_EQ(FieldOf(lamp, tokens::kPhotometricPower)->fallback, Value(1600.0));

    const std::vector<PropertySection> sun = SchemaOf(scene, "/Sun", brdfs, filters);
    EXPECT_EQ(FieldOf(sun, tokens::kAngle)->fallback, Value(0.53));
    EXPECT_EQ(FieldOf(sun, tokens::kPhotometricIlluminance)->fallback, Value(10000.0));
    EXPECT_EQ(FieldOf(sun, tokens::kPhotometricPower), nullptr);

    const std::vector<PropertySection> sky = SchemaOf(scene, "/Sky", brdfs, filters);
    EXPECT_EQ(FieldOf(sky, tokens::kTextureFile)->kind, PropertyField::Kind::Text);
    EXPECT_EQ(FieldOf(sky, tokens::kLightTexture), nullptr);
}

}  // namespace
}  // namespace ernie::app
