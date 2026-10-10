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

TEST(PrimSchemasTest, MaterialPanelsComeFromTheRegistryAndTheBrdf) {
    Scene scene;
    scene.Define("/Looks/Rock", tokens::kMaterial).Set(tokens::kBrdf, std::string("hapke"));
    scene.Define("/Looks/Rock/Tex", tokens::kShader)
        .Set(tokens::kInfoId, std::string(tokens::kUsdUVTexture));
    scene.Define("/Looks/Rock/Node", tokens::kShader);
    const BrdfRegistry brdfs = BrdfRegistry::WithBuiltins();
    const SensorFilterRegistry filters = SensorFilterRegistry::WithStandard();

    const std::vector<PropertySection> rock = SchemaOf(scene, "/Looks/Rock", brdfs, filters);
    EXPECT_EQ(Titles(rock), (std::vector<std::string>{"Surface", "Reflectance model",
                                                      "Detail textures", "Prim"}));
    const PropertyField* brdf = FieldOf(rock, tokens::kBrdf);
    ASSERT_NE(brdf, nullptr);
    EXPECT_EQ(brdf->kind, PropertyField::Kind::Token);
    EXPECT_EQ(brdf->options.size(), brdfs.Models().size());
    EXPECT_EQ(rock[1].fields.size(), brdfs.Find("hapke")->params.size());
    EXPECT_EQ(rock[1].fields.front().attribute, "inputs:" + brdfs.Find("hapke")->params[0].name);
    EXPECT_EQ(FieldOf(rock, tokens::kAlbedoMapEnable)->label, "Use albedo map");
    EXPECT_EQ(FieldOf(rock, tokens::kDetailHexSharpness)->max, 32.0);
    EXPECT_EQ(FieldOf(rock, tokens::kAlbedoMap), nullptr);

    // A shader shows the inputs of its info:id; all of them without one.
    EXPECT_EQ(Titles(SchemaOf(scene, "/Looks/Rock/Tex", brdfs, filters)),
              (std::vector<std::string>{"Shader", "Texture", "Prim"}));
    EXPECT_EQ(Titles(SchemaOf(scene, "/Looks/Rock/Node", brdfs, filters)),
              (std::vector<std::string>{"Shader", "Preview surface", "Texture", "Transform 2D",
                                        "Primvar reader", "Prim"}));
}

TEST(PrimSchemasTest, GeometryPanelsComeFromTheRegistry) {
    Scene scene;
    scene.Define("/Capsule", tokens::kCapsule);
    scene.Define("/Mesh", tokens::kMesh);
    scene.Define("/Ground", tokens::kGroundPlane);
    const BrdfRegistry brdfs = BrdfRegistry::WithBuiltins();
    const SensorFilterRegistry filters = SensorFilterRegistry::WithStandard();

    // Radius, height, then the axis its base declares.
    const std::vector<PropertySection> capsule = SchemaOf(scene, "/Capsule", brdfs, filters);
    EXPECT_EQ(Titles(capsule), (std::vector<std::string>{"Shape", "Geometry", "Prim"}));
    ASSERT_EQ(capsule[0].fields.size(), 3u);
    EXPECT_EQ(capsule[0].fields[0].fallback, Value(0.5));
    EXPECT_EQ(capsule[0].fields[1].label, "Height");
    EXPECT_EQ(capsule[0].fields[1].max, 1000.0);
    EXPECT_EQ(capsule[0].fields[2].options, (std::vector<std::string>{"X", "Y", "Z"}));
    EXPECT_EQ(FieldOf(capsule, tokens::kDisplayColor)->kind, PropertyField::Kind::Color);

    // A mesh's arrays are not edited.
    EXPECT_EQ(Titles(SchemaOf(scene, "/Mesh", brdfs, filters)),
              (std::vector<std::string>{"Geometry", "Prim"}));

    const std::vector<PropertySection> ground = SchemaOf(scene, "/Ground", brdfs, filters);
    EXPECT_EQ(Titles(ground), (std::vector<std::string>{"Ground", "Prim"}));
    EXPECT_EQ(FieldOf(ground, tokens::kChecker)->label, "Checkerboard");
    EXPECT_EQ(FieldOf(ground, tokens::kCheckerSize)->min, 0.001);
    EXPECT_EQ(FieldOf(ground, tokens::kCheckerColor1)->label, "Checker colour 1");
    EXPECT_EQ(FieldOf(ground, tokens::kMaterialBinding)->kind, PropertyField::Kind::Target);
    EXPECT_EQ(FieldOf(ground, tokens::kDisplayColor), nullptr);
}

TEST(PrimSchemasTest, RenderPanelsEditTheCameraAsOneTarget) {
    Scene scene;
    scene.Define("/Settings", tokens::kRenderSettings);
    scene.Define("/Product", tokens::kRenderProduct);
    scene.Define("/Tex", tokens::kShader);
    const BrdfRegistry brdfs = BrdfRegistry::WithBuiltins();
    const SensorFilterRegistry filters = SensorFilterRegistry::WithStandard();

    // The camera, a list of targets in the registry, leads the image section as one target;
    // the other lists of targets are not edited.
    for (const std::vector<PropertySection>& sections :
         {RenderSettingsSchema(), SchemaOf(scene, "/Settings", brdfs, filters),
          SchemaOf(scene, "/Product", brdfs, filters)}) {
        const PropertyField* camera = FieldOf(sections, tokens::kCameraRel);
        ASSERT_NE(camera, nullptr);
        EXPECT_EQ(camera->kind, PropertyField::Kind::Target);
        EXPECT_EQ(camera->options, std::vector<std::string>{std::string(tokens::kCamera)});
        EXPECT_EQ(sections[0].fields[0].attribute, tokens::kCameraRel);
        EXPECT_EQ(sections[0].fields[1].attribute, tokens::kResolution);
        int cameras = 0;
        for (const PropertySection& section : sections) {
            for (const PropertyField& field : section.fields) {
                cameras += field.attribute == tokens::kCameraRel ? 1 : 0;
            }
        }
        EXPECT_EQ(cameras, 1);
        EXPECT_EQ(FieldOf(sections, tokens::kProducts), nullptr);
        EXPECT_EQ(FieldOf(sections, tokens::kOrderedVars), nullptr);
    }
    // A 4-vector that is not a colour has no field.
    EXPECT_EQ(FieldOf(SchemaOf(scene, "/Tex", brdfs, filters), tokens::kShaderBias), nullptr);
}

}  // namespace
}  // namespace ernie::app
