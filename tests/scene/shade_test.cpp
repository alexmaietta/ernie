// SPDX-License-Identifier: BSD-3-Clause
/// @file shade_test.cpp
/// @brief Tests of UsdShade materials, shaders and their connections.
#include "ernie/scene/schemas/geom.h"
#include "ernie/scene/schemas/shade.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace ernie {
namespace {

/// A red UsdPreviewSurface material whose colour comes from a texture.
void RedMaterial(Scene& scene) {
    Material::Define(scene, "/Looks/Red");
    Shader::Define(scene, "/Looks/Red/Surface").SetId("UsdPreviewSurface");
    Shader::Define(scene, "/Looks/Red/Albedo").SetId("UsdUVTexture");
    scene.Set("/Looks/Red/Surface", "inputs:roughness", 0.4);
    scene.Set("/Looks/Red/Albedo", "inputs:file", std::string("red.png"));
    Connect(scene, "/Looks/Red", "outputs:surface",
            {Path("/Looks/Red/Surface"), "outputs:surface"});
    Connect(scene, "/Looks/Red/Surface", "inputs:diffuseColor",
            {Path("/Looks/Red/Albedo"), "outputs:rgb"});
}

TEST(Shade, ConnectionsRoundTripAsUsdPropertyPaths) {
    Scene scene;
    RedMaterial(scene);
    EXPECT_EQ(FindConnection(scene, "/Looks/Red", "outputs:surface"),
              (Connection{Path("/Looks/Red/Surface"), "outputs:surface"}));
    EXPECT_EQ(scene.Get("/Looks/Red/Surface", "inputs:diffuseColor.connect"),
              Value("/Looks/Red/Albedo.outputs:rgb"))
        << "as USD writes the connection";
    EXPECT_FALSE(FindConnection(scene, "/Looks/Red/Surface", "inputs:roughness")) << "a value";
    EXPECT_EQ(scene.Get("/Looks/Red/Surface", "inputs:roughness"), Value(0.4));

    scene.Set("/Looks/Red/Surface", "inputs:metallic.connect", std::string("/Looks/Red/Albedo"));
    EXPECT_THROW(FindConnection(scene, "/Looks/Red/Surface", "inputs:metallic"),
                 std::invalid_argument)
        << "a prim with no property";
}

TEST(Shade, MaterialsAreNodeGraphsOutsideTheImageableTree) {
    Scene scene;
    RedMaterial(scene);
    const PrimId red = *scene.Find("/Looks/Red");
    EXPECT_TRUE(scene.Schema(red)->IsA(NodeGraph::Definition()));
    EXPECT_FALSE(scene.Schema(red)->IsA(Imageable::Definition())) << "no visibility or world";
    EXPECT_EQ(Shader(scene, "/Looks/Red/Surface").GetId(), "UsdPreviewSurface");
    EXPECT_THROW(scene.Set(red, "displayColor", glm::dvec3(1.0)), std::invalid_argument)
        << "only inputs and outputs are open";
    EXPECT_TRUE(scene.WorldBounds("/Looks").Empty());
}

TEST(Shade, PrimsBindMaterialsThroughTheBindingApi) {
    Scene scene;
    RedMaterial(scene);
    Sphere::Define(scene, "/Ball");
    MaterialBindingAPI::Apply(scene, "/Ball").SetMaterialBinding(Path("/Looks/Red"));
    EXPECT_EQ(MaterialBindingAPI(scene, "/Ball").GetMaterialBinding(), Path("/Looks/Red"));
}

}  // namespace
}  // namespace ernie
