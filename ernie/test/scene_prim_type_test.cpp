// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tokens.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <set>
#include <string>

using ernie::AttributeDef;
using ernie::AttributeKind;
using ernie::Path;
using ernie::PrimTypeDef;
using ernie::Scene;
using ernie::Value;
namespace tokens = ernie::tokens;

TEST(PrimType, ViewsReadFallbacksAndSetByUsdName) {
    Scene scene;
    ernie::SphereLight lamp = ernie::SphereLight::Define(scene, "/World/Lamp");
    EXPECT_EQ(scene.TypeOf("/World/Lamp"), "SphereLight");
    EXPECT_EQ(lamp.GetRadius(), 0.5);
    EXPECT_FALSE(lamp.HasRadius());
    EXPECT_EQ(lamp.GetColor(), glm::dvec3(1.0));
    EXPECT_EQ(lamp.GetPhotometricPower(), 1600.0);
    EXPECT_TRUE(lamp.GetSpectrum().empty());

    lamp.SetRadius(0.2).SetIntensity(3.0).SetPhysicalIlluminant(tokens::kBlackbody);
    EXPECT_TRUE(lamp.HasRadius());
    EXPECT_EQ(lamp.GetRadius(), 0.2);
    EXPECT_EQ(*scene.Get("/World/Lamp", tokens::kRadius), Value(0.2));
    EXPECT_EQ(*scene.Get("/World/Lamp", tokens::kIntensity), Value(3.0));
    EXPECT_EQ(lamp.GetPhysicalIlluminant(), "blackbody");
    // Authored as another numeric type, it still reads as the declared one.
    scene.Set("/World/Lamp", tokens::kShapingConeAngle, 45);
    EXPECT_EQ(lamp.GetShapingConeAngle(), 45.0);
    EXPECT_FALSE(lamp.IsPhotometric());
    lamp.SetPhotometricPower(800.0);
    EXPECT_TRUE(lamp.IsPhotometric());
}

TEST(PrimType, ViewsCheckTypesAndConstness) {
    Scene scene;
    ernie::DistantLight::Define(scene, "/Sun");
    scene.Define("/Rock", tokens::kSphere);
    EXPECT_NO_THROW(ernie::Light(scene, "/Sun"));
    EXPECT_NO_THROW(ernie::NonboundableLight(scene, "/Sun"));
    EXPECT_THROW(ernie::BoundableLight(scene, "/Sun"), std::invalid_argument);
    EXPECT_THROW(ernie::SphereLight(scene, "/Rock"), std::invalid_argument);
    EXPECT_THROW(ernie::Light(scene, "/Missing"), std::invalid_argument);

    // A generic light view reads what every light has, with the derived type's registry entry.
    const ernie::Light light(scene, "/Sun");
    EXPECT_EQ(light.PrimType().name, "DistantLight");
    EXPECT_EQ(light.GetColorTemperature(), 6500.0);

    const Scene& constScene = scene;
    ernie::DistantLight readOnly(constScene, "/Sun");
    EXPECT_EQ(readOnly.GetAngle(), 0.53);
    EXPECT_THROW(readOnly.SetAngle(1.0), std::logic_error);
}

TEST(PrimType, RegistryListsTypesWithBasesAndAttributes) {
    const PrimTypeDef& sphere = ernie::GetPrimType("SphereLight");
    EXPECT_FALSE(sphere.abstract);
    ASSERT_NE(sphere.base, nullptr);
    EXPECT_EQ(sphere.base->name, "BoundableLightBase");
    EXPECT_TRUE(sphere.base->abstract);
    EXPECT_TRUE(sphere.IsA(ernie::GetPrimType("Light")));
    EXPECT_FALSE(sphere.IsA(ernie::GetPrimType("NonboundableLightBase")));
    EXPECT_EQ(ernie::FindPrimType("Nope"), nullptr);
    EXPECT_THROW(ernie::GetPrimType("Nope"), std::invalid_argument);

    const AttributeDef* radius = sphere.Find(tokens::kRadius);
    ASSERT_NE(radius, nullptr);
    EXPECT_EQ(radius->member, "Radius");
    EXPECT_EQ(radius->pyName, "radius");
    EXPECT_EQ(radius->kind, AttributeKind::Double);
    EXPECT_EQ(radius->fallback, Value(0.5));
    EXPECT_EQ(sphere.FindByPyName("color_temperature")->name, tokens::kColorTemperature);
    EXPECT_EQ(sphere.Find(tokens::kColorTemperature)->label, "Temperature");
    EXPECT_EQ(sphere.Find(tokens::kEnableColorTemperature)->label, "Use temperature");
    EXPECT_EQ(sphere.Find(tokens::kShapingConeSoftness)->label, "Cone softness");
    EXPECT_EQ(sphere.Find(tokens::kAngle), nullptr);

    // Bases' attributes first, each once.
    const std::vector<const AttributeDef*> all = sphere.AllAttributes();
    EXPECT_EQ(all.front()->name, tokens::kVisibility);
    EXPECT_EQ(all[2]->name, tokens::kIntensity);
    EXPECT_EQ(all.back()->name, tokens::kShapingConeSoftness);
    std::set<std::string_view> names;
    for (const AttributeDef* def : all) {
        EXPECT_TRUE(names.insert(def->name).second) << def->name;
    }

    // Each family's photometric illuminance has its own default.
    EXPECT_EQ(ernie::GetPrimType("DistantLight").Find(tokens::kPhotometricIlluminance)->fallback,
              Value(10000.0));
    EXPECT_EQ(ernie::GetPrimType("RectLight").Find(tokens::kPhotometricIlluminance)->fallback,
              Value(0.0));
    EXPECT_EQ(ernie::GetPrimType("DomeLight").Find(tokens::kLightTexture)->kind,
              AttributeKind::Tensor);
}

TEST(PrimType, GenericReadsMatchTheViewsFallbacks) {
    Scene scene;
    for (const PrimTypeDef* type : ernie::PrimTypes()) {
        if (type->abstract) {
            continue;
        }
        const Path path = Path::Root().Child(std::string(type->name));
        scene.Define(path, type->name);
        EXPECT_EQ(ernie::PrimTypeOf(scene, path), type);
        for (const AttributeDef* def : type->AllAttributes()) {
            if (def->kind == AttributeKind::Tensor || def->kind == AttributeKind::Relationship) {
                EXPECT_TRUE(def->Read(scene, path).Empty()) << def->name;
                continue;
            }
            EXPECT_EQ(def->Read(scene, path), def->fallback) << type->name << " " << def->name;
        }
    }
}

TEST(PrimType, CoerceConvertsToTheDeclaredType) {
    const PrimTypeDef& sphere = ernie::GetPrimType("SphereLight");
    EXPECT_EQ(sphere.Find(tokens::kRadius)->Coerce(Value(2)), Value(2.0));
    EXPECT_EQ(sphere.Find(tokens::kTreatAsPoint)->Coerce(Value(1)), Value(true));
    EXPECT_EQ(sphere.Find(tokens::kColor)->Coerce(Value(glm::vec3(0.5f))), Value(glm::dvec3(0.5)));
    EXPECT_EQ(sphere.Find(tokens::kLightSpectrum)->Coerce(Value(glm::dvec2(500.0, 1.0))),
              Value(std::vector<float>{500.0f, 1.0f}));
    EXPECT_THROW(sphere.Find(tokens::kRadius)->Coerce(Value("big")), std::invalid_argument);
    EXPECT_THROW(sphere.Find(tokens::kTreatAsPoint)->Coerce(Value(2)), std::invalid_argument);
    EXPECT_THROW(ernie::GetPrimType("DomeLight").Find(tokens::kLightTexture)->Coerce(Value(1.0)),
                 std::invalid_argument);
    EXPECT_EQ(ernie::SnakeCase("PhotometricIlluminanceDistance"),
              "photometric_illuminance_distance");
}

TEST(PrimType, ViewsSetTransformsInTheirChains) {
    Scene scene;
    const glm::dmat4 lifted = glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 2.0));
    const ernie::Sphere rock =
        ernie::Sphere::Define(scene, "/Rock").SetTransform(lifted).SetRadius(0.5);
    EXPECT_EQ(rock.GetLocalTransform(), lifted);
    EXPECT_EQ(rock.GetRadius(), 0.5);
    ernie::XformOps ops;
    ops.translate = glm::dvec3(1.0, 2.0, 3.0);
    const ernie::DistantLight sun = ernie::DistantLight::Define(scene, "/Sun").SetXform(ops);
    EXPECT_EQ(glm::dvec3(sun.GetLocalTransform()[3]), glm::dvec3(1.0, 2.0, 3.0));
}
