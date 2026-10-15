// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tokens.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using ernie::AttributeDef;
using ernie::AttributeKind;
using ernie::Path;
using ernie::PrimTypeDef;
using ernie::Scene;
using ernie::Value;
namespace tokens = ernie::tokens;

TEST(PrimType, ViewsReadFallbacksAndSetByUsdName) {
    Scene scene;
    ernie::SphereLight lamp =
        ernie::SphereLight::Define(scene, "/World/Lamp");
    EXPECT_EQ(scene.TypeOf("/World/Lamp"), "SphereLight");
    EXPECT_EQ(lamp.GetRadius(), 0.5);
    EXPECT_FALSE(lamp.HasRadius());
    EXPECT_EQ(lamp.GetColor(), glm::dvec3(1.0));
    EXPECT_EQ(lamp.GetPhotometricPower(), 1600.0);
    EXPECT_TRUE(lamp.GetSpectrum().empty());

    lamp.SetRadius(0.2).SetIntensity(3.0).SetPhysicalIlluminant(
        tokens::kBlackbody);
    EXPECT_TRUE(lamp.HasRadius());
    EXPECT_EQ(lamp.GetRadius(), 0.2);
    EXPECT_EQ(*scene.Get("/World/Lamp", tokens::kRadius), Value(0.2));
    EXPECT_EQ(*scene.Get("/World/Lamp", tokens::kIntensity),
              Value(3.0));
    EXPECT_EQ(lamp.GetPhysicalIlluminant(), "blackbody");
    // Authored as another numeric type, it still reads as the
    // declared one.
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
    EXPECT_THROW(ernie::BoundableLight(scene, "/Sun"),
                 std::invalid_argument);
    EXPECT_THROW(ernie::SphereLight(scene, "/Rock"),
                 std::invalid_argument);
    EXPECT_THROW(ernie::Light(scene, "/Missing"),
                 std::invalid_argument);

    // A generic light view reads what every light has, with the
    // derived type's registry entry.
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
    EXPECT_FALSE(
        sphere.IsA(ernie::GetPrimType("NonboundableLightBase")));
    EXPECT_EQ(ernie::FindPrimType("Nope"), nullptr);
    EXPECT_THROW(ernie::GetPrimType("Nope"), std::invalid_argument);

    const AttributeDef* radius = sphere.Find(tokens::kRadius);
    ASSERT_NE(radius, nullptr);
    EXPECT_EQ(radius->member, "Radius");
    EXPECT_EQ(radius->pyName, "radius");
    EXPECT_EQ(radius->kind, AttributeKind::Double);
    EXPECT_EQ(radius->fallback, Value(0.5));
    EXPECT_EQ(sphere.FindByPyName("color_temperature")->name,
              tokens::kColorTemperature);
    EXPECT_EQ(sphere.Find(tokens::kColorTemperature)->label,
              "Temperature");
    EXPECT_EQ(sphere.Find(tokens::kEnableColorTemperature)->label,
              "Use temperature");
    EXPECT_EQ(sphere.Find(tokens::kShapingConeSoftness)->label,
              "Cone softness");
    EXPECT_EQ(sphere.Find(tokens::kAngle), nullptr);

    // Bases' attributes first, each once.
    const std::vector<const AttributeDef*> all =
        sphere.AllAttributes();
    EXPECT_EQ(all.front()->name, tokens::kVisibility);
    EXPECT_EQ(all[3]->name, tokens::kIntensity);
    EXPECT_EQ(all.back()->name, tokens::kShapingConeSoftness);
    std::set<std::string_view> names;
    for (const AttributeDef* def : all) {
        EXPECT_TRUE(names.insert(def->name).second) << def->name;
    }

    // Each family's photometric illuminance has its own default.
    EXPECT_EQ(ernie::GetPrimType("DistantLight")
                  .Find(tokens::kPhotometricIlluminance)
                  ->fallback,
              Value(10000.0));
    EXPECT_EQ(ernie::GetPrimType("RectLight")
                  .Find(tokens::kPhotometricIlluminance)
                  ->fallback,
              Value(0.0));
    EXPECT_EQ(ernie::GetPrimType("DomeLight")
                  .Find(tokens::kLightTexture)
                  ->kind,
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
            if (def->kind == AttributeKind::Tensor ||
                def->kind == AttributeKind::Relationship) {
                EXPECT_TRUE(def->Read(scene, path).Empty())
                    << def->name;
                continue;
            }
            EXPECT_EQ(def->Read(scene, path), def->fallback)
                << type->name << " " << def->name;
        }
    }
}

TEST(PrimType, CoerceConvertsToTheDeclaredType) {
    const PrimTypeDef& sphere = ernie::GetPrimType("SphereLight");
    EXPECT_EQ(sphere.Find(tokens::kRadius)->Coerce(Value(2)),
              Value(2.0));
    EXPECT_EQ(sphere.Find(tokens::kTreatAsPoint)->Coerce(Value(1)),
              Value(true));
    EXPECT_EQ(
        sphere.Find(tokens::kColor)->Coerce(Value(glm::vec3(0.5f))),
        Value(glm::dvec3(0.5)));
    EXPECT_EQ(sphere.Find(tokens::kLightSpectrum)
                  ->Coerce(Value(glm::dvec2(500.0, 1.0))),
              Value(std::vector<float>{500.0f, 1.0f}));
    EXPECT_THROW(sphere.Find(tokens::kRadius)->Coerce(Value("big")),
                 std::invalid_argument);
    EXPECT_THROW(sphere.Find(tokens::kTreatAsPoint)->Coerce(Value(2)),
                 std::invalid_argument);
    EXPECT_THROW(ernie::GetPrimType("DomeLight")
                     .Find(tokens::kLightTexture)
                     ->Coerce(Value(1.0)),
                 std::invalid_argument);
    EXPECT_EQ(ernie::SnakeCase("PhotometricIlluminanceDistance"),
              "photometric_illuminance_distance");
}

TEST(PrimType, ViewsSetTransformsInTheirChains) {
    Scene scene;
    const glm::dmat4 lifted =
        glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 2.0));
    const ernie::Sphere rock = ernie::Sphere::Define(scene, "/Rock")
                                   .SetTransform(lifted)
                                   .SetRadius(0.5);
    EXPECT_EQ(rock.GetLocalTransform(), lifted);
    EXPECT_EQ(rock.GetRadius(), 0.5);
    ernie::XformOps ops;
    ops.translate = glm::dvec3(1.0, 2.0, 3.0);
    const ernie::DistantLight sun =
        ernie::DistantLight::Define(scene, "/Sun").SetXform(ops);
    EXPECT_EQ(glm::dvec3(sun.GetLocalTransform()[3]),
              glm::dvec3(1.0, 2.0, 3.0));
}

TEST(PrimType, ViewsClearAndPassThroughInTheirChains) {
    Scene scene;
    // Clear, Set and Write return the derived view, so typed setters
    // chain after them.
    ernie::SphereLight lamp =
        ernie::SphereLight::Define(scene, "/Lamp")
            .SetRadius(0.2)
            .Set("user:custom", Value(3))
            .Write("user:samples", ernie::Tensor::Copy<float>(
                                       std::vector<float>{1, 2}, {2}))
            .ClearRadius()
            .SetIntensity(2.0);
    EXPECT_FALSE(lamp.HasRadius());
    EXPECT_EQ(lamp.GetRadius(), 0.5) << "the fallback again";
    EXPECT_EQ(scene.Get("/Lamp", tokens::kRadius), nullptr);
    EXPECT_EQ(*scene.Get("/Lamp", "user:custom"), Value(3));
    EXPECT_EQ(scene.Read("/Lamp", "user:samples")->Elements(), 2);
    EXPECT_EQ(lamp.GetIntensity(), 2.0);
    lamp.ClearRadius();  // Clearing what is unset changes nothing.

    // Clearing a tensor removes the written array.
    ernie::Mesh mesh = ernie::Mesh::Define(scene, "/Mesh")
                           .SetPoints(ernie::Tensor::Copy<float>(
                               std::vector<float>(9, 0.0f), {3, 3}));
    EXPECT_TRUE(mesh.ClearPoints().GetPoints() == nullptr);

    // The prim handle, for what views do not cover.
    const ernie::Prim prim = lamp.GetPrim();
    EXPECT_EQ(prim.GetPath(), Path("/Lamp"));
    EXPECT_EQ(prim.Type(), "SphereLight");
    const Scene& constScene = scene;
    EXPECT_THROW(ernie::SphereLight(constScene, "/Lamp").GetPrim(),
                 std::logic_error);
    EXPECT_THROW(
        ernie::SphereLight(constScene, "/Lamp").ClearIntensity(),
        std::logic_error);
}

TEST(PrimType, Vec4AttributesReadFourVectors) {
    Scene scene;
    scene.Define("/Tex", tokens::kShader);
    const ernie::Shader texture(scene, "/Tex");
    const AttributeDef& bias =
        *texture.PrimType().Find(tokens::kShaderBias);
    EXPECT_EQ(bias.kind, AttributeKind::Vec4);
    EXPECT_EQ(bias.fallback, Value(glm::dvec4(0.0)));
    EXPECT_EQ(texture.GetBias(), glm::dvec4(0.0));
    // A vec4 (four numbers from Python) or a dvec4 reads; other types
    // read as the fallback.
    scene.Set("/Tex", tokens::kShaderBias,
              glm::vec4(-1.0f, -1.0f, -1.0f, 0.0f));
    EXPECT_EQ(texture.GetBias(), glm::dvec4(-1.0, -1.0, -1.0, 0.0));
    ernie::Shader(scene, "/Tex").SetBias(glm::dvec4(0.5));
    EXPECT_EQ(*scene.Get("/Tex", tokens::kShaderBias),
              Value(glm::dvec4(0.5)));
    EXPECT_EQ(bias.Read(scene, "/Tex"), Value(glm::dvec4(0.5)));
    // Another type is refused (`Validate`).
    EXPECT_THROW(
        scene.Set("/Tex", tokens::kShaderBias, glm::dvec3(1.0)),
        std::invalid_argument);

    EXPECT_EQ(bias.Coerce(Value(glm::vec4(1.0f))),
              Value(glm::dvec4(1.0)));
    try {
        bias.Coerce(Value(glm::dvec3(1.0)));
        ADD_FAILURE() << "a 3-vector is not a 4-vector";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(std::string(error.what()),
                  "'inputs:bias' must be a 4-vector");
    }
}

TEST(PrimType, RelationshipListsTakeOneTargetOrSeveral) {
    Scene scene;
    ernie::RenderProduct product =
        ernie::RenderProduct::Define(scene, "/Render/Product");
    EXPECT_TRUE(product.GetCamera().empty());
    EXPECT_FALSE(product.HasCamera());

    // One target (a Path or its text) is stored as a Path, as before;
    // a list as a list.
    product.SetCamera("/Cam");
    EXPECT_EQ(*scene.Get("/Render/Product", tokens::kCameraRel),
              Value(Path("/Cam")));
    EXPECT_EQ(product.GetCamera(), std::vector<Path>{Path("/Cam")});
    product.SetCamera(std::string("/Other")).SetCamera(Path("/Cam"));
    EXPECT_EQ(product.GetCamera(), std::vector<Path>{Path("/Cam")});
    product.SetCamera({"/A", "/B"});
    EXPECT_EQ(*scene.Get("/Render/Product", tokens::kCameraRel),
              Value(std::vector<Path>{Path("/A"), Path("/B")}));
    EXPECT_EQ(product.GetCamera(),
              (std::vector<Path>{Path("/A"), Path("/B")}));
    product.SetCamera(std::vector<Path>{Path("/A")});
    EXPECT_EQ(*scene.Get("/Render/Product", tokens::kCameraRel),
              Value(std::vector<Path>{Path("/A")}));

    // An empty list is authored, apart from unset.
    product.SetOrderedVars({});
    EXPECT_TRUE(product.HasOrderedVars());
    EXPECT_TRUE(product.GetOrderedVars().empty());
    EXPECT_FALSE(product.ClearOrderedVars().HasOrderedVars());

    const AttributeDef& camera =
        *product.PrimType().Find(tokens::kCameraRel);
    EXPECT_EQ(camera.kind, AttributeKind::Relationships);
    EXPECT_EQ(camera.fallback, Value(std::vector<Path>{}));
    EXPECT_EQ(camera.Read(scene, "/Render/Product"),
              Value(std::vector<Path>{Path("/A")}));
    // Coerce keeps one target a Path, as stored, and makes lists of
    // strings paths.
    EXPECT_EQ(camera.Coerce(Value("/Cam")), Value(Path("/Cam")));
    EXPECT_EQ(camera.Coerce(Value(Path("/Cam"))),
              Value(Path("/Cam")));
    EXPECT_EQ(
        camera.Coerce(Value(std::vector<std::string>{"/A", "/B"})),
        Value(std::vector<Path>{Path("/A"), Path("/B")}));
    EXPECT_EQ(camera.Coerce(Value(std::vector<float>{})),
              Value(std::vector<Path>{}));
    EXPECT_THROW(camera.Coerce(Value(2.0)), std::invalid_argument);
    EXPECT_THROW(
        camera.Coerce(Value(std::vector<std::string>{"not a path"})),
        std::invalid_argument);
    // Another type is refused (`Validate`).
    EXPECT_THROW(scene.Set("/Render/Product", tokens::kCameraRel, 3),
                 std::invalid_argument);
}

TEST(PrimType, FloatArraysTakeOneNumber) {
    const AttributeDef& widths =
        *ernie::GetPrimType(tokens::kGeomPoints)
             .Find(tokens::kWidths);
    EXPECT_EQ(widths.kind, AttributeKind::FloatArray);
    EXPECT_EQ(widths.Coerce(Value(0.5)),
              Value(std::vector<float>{0.5f}));
    EXPECT_EQ(widths.Coerce(Value(2)),
              Value(std::vector<float>{2.0f}));
    EXPECT_EQ(widths.Coerce(Value(glm::dvec4(1.0))),
              Value(std::vector<float>(4, 1.0f)));
}

TEST(PrimType, SettingChecksNamesAndTypesAgainstTheDeclaration) {
    Scene scene;
    scene.Define("/Lamp", tokens::kSphereLight);
    const auto message = [&](std::string_view attribute,
                             Value value) {
        try {
            scene.Set("/Lamp", attribute, std::move(value));
        } catch (const std::invalid_argument& error) {
            return std::string(error.what());
        }
        return std::string();
    };
    // A misspelt name, plain or in one of the type's own namespaces,
    // and a value of another type.
    EXPECT_NE(
        message("radus", 1.0).find("did you mean 'inputs:radius'"),
        std::string::npos);
    EXPECT_NE(message("inputs:radus", 1.0)
                  .find("did you mean 'inputs:radius'"),
              std::string::npos);
    EXPECT_NE(
        message(tokens::kRadius, "big").find("must be a number"),
        std::string::npos);
    EXPECT_NE(message(tokens::kTreatAsPoint, glm::dvec3(1.0))
                  .find("'/Lamp'"),
              std::string::npos);
    // Numbers of any type for numbers, ints for bools, as the readers
    // take them.
    EXPECT_EQ(message(tokens::kRadius, 2), "");
    EXPECT_EQ(message(tokens::kShadowEnable, 0), "");
    // Custom data in a namespace of its own, primvars, transform ops
    // and connections are free.
    EXPECT_EQ(message("user:speed", 3.0), "");
    EXPECT_EQ(message("primvars:st", std::vector<float>{0.0f, 1.0f}),
              "");
    EXPECT_EQ(message("xformOp:translate:pivot", glm::dvec3(1.0)),
              "");
    EXPECT_EQ(
        message("inputs:color.connect", "/Looks/Tex.outputs:rgb"),
        "");
    // Writing arrays checks names only.
    EXPECT_THROW(scene.Write("/Lamp", "radus",
                             ernie::Tensor::Copy<float>(
                                 std::vector<float>{1}, {1})),
                 std::invalid_argument);

    // A shader's inputs depend on its info:id, and grouping prims
    // hold any custom data.
    scene.Define("/Looks/Tex", tokens::kShader)
        .Set("inputs:anything", 1.0);
    scene.Define("/World", tokens::kXform).Set("speed", 3.0);
    // Untyped and unregistered prims are not checked.
    scene.Set("/Looks", "anything", 1.0);
    scene.Define("/Custom", "MyType").Set("anything", 1.0);
}

TEST(PrimType, DeclarationsSayWhatTheirChangesStamp) {
    // One attribute declared on several types stamps the same bit on
    // each, as `DirtyOf` reads it.
    std::map<std::string_view, std::optional<ernie::Dirty>> bits;
    for (const PrimTypeDef* type : ernie::PrimTypes()) {
        for (const AttributeDef& def : type->attributes) {
            const auto [it, added] =
                bits.emplace(def.name, def.meta.dirty);
            EXPECT_EQ(it->second, def.meta.dirty)
                << def.name << " on " << type->name;
            EXPECT_EQ(ernie::DirtyOf(def.name),
                      def.meta.dirty.value_or(
                          def.name.starts_with("primvars:")
                              ? ernie::Dirty::Primvars
                              : ernie::Dirty::Params))
                << def.name;
        }
    }
    EXPECT_EQ(ernie::DirtyOf(tokens::kPoints), ernie::Dirty::Points);
    EXPECT_EQ(ernie::DirtyOf(tokens::kPrototypes),
              ernie::Dirty::Topology);
    EXPECT_EQ(ernie::DirtyOf("user:speed"), ernie::Dirty::Params);
}

TEST(PrimType, SettersTakeRangesOfNumbers) {
    Scene scene;
    ernie::SphereLight lamp =
        ernie::SphereLight::Define(scene, "/World/Lamp");
    // Vectors from any contiguous range, exactly as many numbers.
    lamp.SetColor(std::array<float, 3>{0.5f, 0.25f, 1.0f});
    EXPECT_EQ(lamp.GetColor(), glm::dvec3(0.5, 0.25, 1.0));
    const std::vector<int> ones = {1, 1, 1};
    lamp.SetColor(std::span<const int>(ones));
    EXPECT_EQ(lamp.GetColor(), glm::dvec3(1.0));
    EXPECT_THROW(lamp.SetColor(std::vector<float>{1.0f, 2.0f}),
                 std::invalid_argument);
    // Float arrays from numbers of any type.
    lamp.SetSpectrum(std::vector<double>{400.0, 1.0, 700.0, 0.5});
    EXPECT_EQ(lamp.GetSpectrum(),
              (std::vector<float>{400.0f, 1.0f, 700.0f, 0.5f}));

    // Tensors of a shape, one dimension by default.
    ernie::Mesh mesh = ernie::Mesh::Define(scene, "/World/Mesh");
    const std::vector<float> points = {0, 0, 0, 1, 0, 0, 0, 1, 0};
    mesh.SetPoints(points, {3, 3});
    ASSERT_NE(mesh.GetPoints(), nullptr);
    EXPECT_EQ(mesh.GetPoints()->Shape(),
              (std::vector<std::int64_t>{3, 3}));
    EXPECT_EQ(mesh.GetPoints()->Type(), ernie::DType::Float32);
    mesh.SetFaceVertexIndices(std::vector<std::int32_t>{0, 1, 2});
    EXPECT_EQ(mesh.GetFaceVertexIndices()->Shape(),
              (std::vector<std::int64_t>{3}));
    EXPECT_THROW(mesh.SetPoints(points, {2, 3}),
                 std::invalid_argument);

    // OrArray arrays as float32 rows of a vector's numbers.
    mesh.WriteDisplayColor(
        std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1});
    ASSERT_NE(mesh.ReadDisplayColor(), nullptr);
    EXPECT_EQ(mesh.ReadDisplayColor()->Shape(),
              (std::vector<std::int64_t>{3, 3}));
    EXPECT_EQ(mesh.ReadDisplayColor()->Type(), ernie::DType::Float32);
    EXPECT_THROW(mesh.WriteDisplayColor(std::vector<float>{1, 0}),
                 std::invalid_argument);
}
