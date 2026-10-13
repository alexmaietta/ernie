// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/session/materialx.h"

#include "ernie/device/shader_compiler.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tokens.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie {
namespace {

/// A gold standard_surface whose base colour a constant node gives,
/// as USD import brings a MaterialX network in.
Scene GoldScene() {
    Scene scene;
    Material::Define(scene, "/Looks/Gold")
        .GetPrim()
        .Set(std::string(tokens::kMtlxSurface) + ".connect",
             std::string("/Looks/Gold/Surface.outputs:out"));
    Shader::Define(scene, "/Looks/Gold/Surface")
        .SetInfoId("ND_standard_surface_surfaceshader")
        .GetPrim()
        .Set("inputs:metalness", 1.0)
        .Set("inputs:base_color.connect",
             std::string("/Looks/Gold/Tint.outputs:out"));
    Shader::Define(scene, "/Looks/Gold/Tint")
        .SetInfoId("ND_constant_color3")
        .GetPrim()
        .Set("inputs:value", glm::dvec3(0.94, 0.78, 0.37));
    return scene;
}

TEST(MaterialX, NetworksBecomeDocuments) {
    const Scene scene = GoldScene();
    const std::optional<MaterialXSource> source =
        MaterialXOf(scene, Path("/Looks/Gold"));
    ASSERT_TRUE(source.has_value());
    EXPECT_EQ(source->element, "Gold");
    EXPECT_NE(source->document.find("standard_surface"),
              std::string::npos);
    EXPECT_NE(source->document.find("0.94, 0.78, 0.37"),
              std::string::npos)
        << source->document;
    EXPECT_NE(source->document.find("surfacematerial"),
              std::string::npos);
    EXPECT_FALSE(MaterialXOf(scene, Path("/Looks/Gold/Surface")))
        << "only Materials have MaterialX surfaces";
}

TEST(MaterialX, GeneratedSlangCompiles) {
    const MaterialXSlang slang = GenerateMaterialXSlang(
        *MaterialXOf(GoldScene(), Path("/Looks/Gold")));
    EXPECT_NE(slang.pixel.find("NG_standard_surface"),
              std::string::npos);
    const ShaderCompiler compiler;
    const std::vector<ShaderModuleSource> modules{
        {"gold", slang.pixel}};
    EXPECT_FALSE(
        compiler.CompileSpirv(modules, "fragmentMain").empty());
}

TEST(MaterialX, FilesNameTheirMaterials) {
    const std::filesystem::path dir =
        std::filesystem::path(testing::TempDir()) / "ernie_materialx";
    std::filesystem::create_directories(dir);
    const std::filesystem::path file = dir / "plastic.mtlx";
    std::ofstream(file) << R"(<?xml version="1.0"?>
<materialx version="1.39">
  <open_pbr_surface name="Red" type="surfaceshader">
    <input name="base_color" type="color3" value="0.8, 0.1, 0.1" />
  </open_pbr_surface>
  <surfacematerial name="Plastic" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="Red" />
  </surfacematerial>
  <surfacematerial name="Other" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="Red" />
  </surfacematerial>
</materialx>
)";
    Scene scene;
    Material material = Material::Define(scene, "/Looks/Plastic");
    material.SetMaterialXFile(file.string());
    const std::optional<MaterialXSource> first =
        MaterialXOf(scene, Path("/Looks/Plastic"));
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->element, "Plastic")
        << "the first renderable one";
    EXPECT_EQ(first->base, dir);
    material.SetMaterialXElement("Other");
    EXPECT_EQ(MaterialXOf(scene, Path("/Looks/Plastic"))->element,
              "Other");
    EXPECT_NE(GenerateMaterialXSlang(*first).pixel.find("open_pbr"),
              std::string::npos);
    material.SetMaterialXElement("Missing");
    EXPECT_THROW(MaterialXOf(scene, Path("/Looks/Plastic")),
                 std::invalid_argument);
}

TEST(MaterialX, UnknownNodesThrow) {
    Scene scene = GoldScene();
    Shader(scene, "/Looks/Gold/Tint").SetInfoId("ND_no_such_node");
    EXPECT_THROW(MaterialXOf(scene, Path("/Looks/Gold")),
                 std::invalid_argument);
}

}  // namespace
}  // namespace ernie
