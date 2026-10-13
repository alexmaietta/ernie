// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/session/materialx.h"

#include "ernie/device/shader_compiler.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tokens.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
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

std::string Fixture(const char* name) {
    std::ifstream in(std::filesystem::path(ERNIE_TEST_FIXTURES) /
                     "materialx" / name);
    return {std::istreambuf_iterator<char>(in),
            std::istreambuf_iterator<char>()};
}

/// Compiles `shader`'s code under the stub bridge
/// (fixtures/materialx), as the shading pass will its own.
/// @throws std::invalid_argument with Slang's diagnostics.
void Compile(const ShaderCompiler& compiler,
             const MaterialXShader& shader, const std::string& name) {
    const std::vector<ShaderModuleSource> modules{
        {name, Fixture("stub_bridge.slang") + shader.code +
                   Fixture("stub_main.slang")}};
    compiler.CompileSpirv(modules, "stubMain");
}

TEST(MaterialX, ShadersReadTheirValuesFromUniforms) {
    const MaterialXShader shader = GenerateMaterialXShader(
        *MaterialXOf(GoldScene(), Path("/Looks/Gold")));
    EXPECT_TRUE(shader.surface);
    EXPECT_FALSE(shader.transparent);
    EXPECT_NE(shader.code.find("NG_standard_surface"),
              std::string::npos);
    EXPECT_NE(shader.code.find("ernieMxLightResponse"),
              std::string::npos)
        << "lit by Ernie's lights";
    const auto uniform = [&](std::string_view path) {
        for (const MaterialXUniform& u : shader.uniforms) {
            if (u.path == path) {
                return u;
            }
        }
        return MaterialXUniform{};
    };
    EXPECT_EQ(uniform("Surface/metalness").type, "float");
    EXPECT_DOUBLE_EQ(std::stod(uniform("Surface/metalness").value),
                     1.0);
    EXPECT_EQ(uniform("Tint/value").type, "color3");
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
    ASSERT_EQ(std::sscanf(uniform("Tint/value").value.c_str(),
                          "%lf, %lf, %lf", &r, &g, &b),
              3);
    EXPECT_DOUBLE_EQ(g, 0.78);
    // Laid out one after another, scalar.
    std::uint32_t end = 0;
    for (const MaterialXUniform& u : shader.uniforms) {
        EXPECT_EQ(u.offset, end) << u.variable;
        end = u.offset +
              (u.type == "float" || u.type == "integer" ||
                       u.type == "boolean"
                   ? 4u
               : u.type == "vector2"                       ? 8u
               : u.type == "vector3" || u.type == "color3" ? 12u
               : u.type == "vector4" || u.type == "color4" ? 16u
               : u.type == "matrix33"                      ? 36u
                                                           : 64u);
    }
    EXPECT_EQ(shader.uniformBytes, end);
    const ShaderCompiler compiler;
    Compile(compiler, shader, "gold");
}

TEST(MaterialX, FilesNameTheirMaterials) {
    const std::filesystem::path dir =
        std::filesystem::path(testing::TempDir()) / "ernie_materialx";
    std::filesystem::create_directories(dir);
    const std::filesystem::path file = dir / "plastic.mtlx";
    std::ofstream(file) << R"(<?xml version="1.0"?>
<materialx version="1.39">
  <image name="Grain" type="color3">
    <input name="file" type="filename" value="grain.png" colorspace="srgb_texture" />
    <input name="uaddressmode" type="string" value="mirror" />
  </image>
  <open_pbr_surface name="Red" type="surfaceshader">
    <input name="base_color" type="color3" nodename="Grain" />
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
    const MaterialXShader shader = GenerateMaterialXShader(*first);
    EXPECT_NE(shader.code.find("open_pbr"), std::string::npos);
    ASSERT_EQ(shader.textures.size(), 1u);
    EXPECT_EQ(shader.textures[0].slot, 0u);
    EXPECT_EQ(shader.textures[0].file, (dir / "grain.png").string())
        << "relative to the document";
    EXPECT_EQ(shader.textures[0].uAddress, 3) << "mirror";
    EXPECT_EQ(shader.textures[0].vAddress, 2) << "periodic";
    const ShaderCompiler compiler;
    Compile(compiler, shader, "plastic");
    material.SetMaterialXElement("Missing");
    EXPECT_THROW(MaterialXOf(scene, Path("/Looks/Plastic")),
                 std::invalid_argument);
}

// Every renderable element of MaterialX's test suite and examples
// generated for Ernie and compiled under the stub bridge: a
// compliance check of code generation, minutes long, so run on demand
// (--gtest_also_run_disabled_tests). It lists what fails.
TEST(MaterialX, DISABLED_TestSuiteCompiles) {
    const std::filesystem::path resources =
        MaterialXLibraries().parent_path() / "resources" /
        "Materials";
    ASSERT_TRUE(std::filesystem::is_directory(resources))
        << resources;
    const ShaderCompiler compiler;
    std::size_t passed = 0;
    std::vector<std::string> failures;
    for (const char* folder : {"TestSuite", "Examples"}) {
        for (const auto& entry :
             std::filesystem::recursive_directory_iterator(resources /
                                                           folder)) {
            const std::filesystem::path& file = entry.path();
            const std::string text = file.string();
            if (file.extension() != ".mtlx" ||
                text.find("_options.mtlx") != std::string::npos ||
                text.find("/libraries/") != std::string::npos ||
                text.find("/locale/") != std::string::npos) {
                continue;
            }
            std::vector<std::string> elements;
            try {
                elements = MaterialXElements(file);
            } catch (const std::exception& e) {
                failures.push_back(text + ": " + e.what());
                continue;
            }
            for (const std::string& element : elements) {
                Scene scene;
                Material::Define(scene, "/Looks/M")
                    .SetMaterialXFile(text)
                    .SetMaterialXElement(element);
                try {
                    Compile(compiler,
                            GenerateMaterialXShader(*MaterialXOf(
                                scene, Path("/Looks/M"))),
                            file.stem().string());
                    ++passed;
                } catch (const std::exception& e) {
                    std::string what = e.what();
                    failures.push_back(
                        text + " " + element + ": " +
                        what.substr(0, what.find('\n', 200)));
                }
            }
        }
    }
    std::string list;
    for (const std::string& failure : failures) {
        list += "\n" + failure;
    }
    std::printf("%zu compiled, %zu failed%s\n", passed,
                failures.size(), list.c_str());
    EXPECT_TRUE(failures.empty());
}

TEST(MaterialX, UnknownNodesThrow) {
    Scene scene = GoldScene();
    Shader(scene, "/Looks/Gold/Tint").SetInfoId("ND_no_such_node");
    EXPECT_THROW(MaterialXOf(scene, Path("/Looks/Gold")),
                 std::invalid_argument);
}

}  // namespace
}  // namespace ernie
