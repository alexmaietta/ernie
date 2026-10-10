// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/delegate.h"
#include "ernie/scene/camera.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "support/delegate_scenes.h"
#include "support/vulkan_fixture.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using ernie::Path;
using ernie::RasterDelegate;
using ernie::Scene;
using ernie::Tensor;
using namespace ernie::testing;
namespace tokens = ernie::tokens;

namespace {

/// A 16 m quad at the origin under MeshScene's overhead camera, with texture coordinates (0, 0)
/// to (1, 1) from its corner at (-8, -8) to (8, 8), bound to /Looks/M: a UsdPreviewSurface whose
/// UsdUVTexture /Looks/M/Tex reads primvar st. Its two products render LdrColor and HdrColor.
Scene TexturedScene() {
    Scene scene;
    ernie::Camera::Define(scene, "/World/Camera").SetFocalLength(18.0).SetSensorNoise(false);
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 60.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, ernie::DistantLightTransform({0.0, 0.0, 1.0}));
    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetResolution(glm::dvec2(kSize, kSize))
        .SetAmbient(0.0);
    DefineQuad(scene, "/World/Quad", 8.0f);
    const std::array<float, 8> st{0, 0, 1, 0, 1, 1, 0, 1};
    scene.Write("/World/Quad", "primvars:st", Tensor::Copy<float>(st, {4, 2}));
    ernie::Mesh(scene, "/World/Quad").SetMaterialBinding("/Looks/M");

    ernie::Material::Define(scene, "/Looks/M");
    ernie::Shader::Define(scene, "/Looks/M/Surface")
        .SetInfoId(tokens::kUsdPreviewSurface)
        .SetDiffuseColor(glm::dvec3(0.5));
    ernie::Shader::Define(scene, "/Looks/M/Tex")
        .SetInfoId(tokens::kUsdUVTexture)
        .SetSourceColorSpace("raw");
    ernie::Shader::Define(scene, "/Looks/M/St")
        .SetInfoId(tokens::kUsdPrimvarReaderFloat2)
        .SetVarname("st");
    ernie::Connect(scene, "/Looks/M", tokens::kSurfaceOutput, "/Looks/M/Surface",
                   tokens::kSurfaceOutput);
    ernie::Connect(scene, "/Looks/M/Tex", tokens::kSt, "/Looks/M/St", "outputs:result");

    std::vector<Path> vars;
    for (const std::string_view name : {"HdrColor", "LdrColor"}) {
        const Path var("/Render/Vars/" + std::string(name));
        ernie::RenderVar::Define(scene, var).SetSourceName(name);
        vars.push_back(var);
    }
    ernie::RenderProduct::Define(scene, "/Render/Out")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(kSize, kSize));
    scene.Set("/Render/Out", tokens::kOrderedVars, vars);
    return scene;
}

/// Render var `var` of one step of /Render/Out, on the CPU.
Tensor Render(RasterDelegate& delegate, const Scene& scene, const std::string& var) {
    const Path product[] = {"/Render/Out"};
    return delegate.Step(scene, product)
        .at("/Render/Out")
        .vars.at(var)
        .Map(ernie::MapDevice::Cpu)
        .tensor.Contiguous();
}

/// HdrColor's green at the image centre (the quad's middle).
float CentreGreen(RasterDelegate& delegate, const Scene& scene) {
    const Tensor hdr = Render(delegate, scene, "HdrColor");
    const std::size_t centre = (kSize / 2) * kSize + kSize / 2;
    return hdr.As<float>()[4 * centre + 1];
}

/// HdrColor's green at the pixel showing quad point (x, y).
float GreenAt(const Scene& scene, const Tensor& hdr, float x, float y) {
    const glm::vec4 clip =
        ernie::CameraParamsOf(scene, "/World/Camera", kSize, kSize).RasterView().ViewProjection() *
        glm::vec4(x, y, 0.0f, 1.0f);
    const glm::vec2 ndc = glm::vec2(clip) / clip.w;
    const auto col = static_cast<std::size_t>((ndc.x * 0.5f + 0.5f) * kSize);
    const auto row = static_cast<std::size_t>((0.5f - ndc.y * 0.5f) * kSize);
    return hdr.As<float>()[4 * (row * kSize + col) + 1];
}

/// /Looks/M as a plain grey preview surface lit and seen from straight above, where its specular
/// lobe shows.
Scene PreviewScene() {
    Scene scene = TexturedScene();
    ernie::Shader(scene, "/Looks/M/Surface").SetDiffuseColor(glm::dvec3(0.6));
    scene.Set("/World/Sun", tokens::kTransform, ernie::DistantLightTransform({0.3, 0.0, 1.0}));
    return scene;
}

}  // namespace

TEST_F(RasterDelegateTest, AlbedoTexturesFollowTextureCoordinatesWithTheImageUpright) {
    // Red, green in the top row; blue, white in the bottom: t = 0 is the image's bottom row, so
    // the quad shows the image as it is, red top left. A 2 x 2 image filters bilinearly across
    // the whole quad (pure only at texel centres, which the sampled pixels miss slightly) and the
    // preview surface's white highlight greys it, so each quadrant's own channel leads.
    Scene scene = TexturedScene();
    const std::array<std::uint8_t, 16> texels{255, 0, 0,   255, 0,   255, 0,   255,
                                              0,   0, 255, 255, 255, 255, 255, 255};
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetTexture(Tensor::Copy<std::uint8_t>(texels, {2, 2, 4}))
        .SetWrapS("clamp")
        .SetWrapT("clamp");
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kDiffuseColor, "/Looks/M/Tex",
                   "outputs:rgb");
    RasterDelegate delegate(*context_);
    const Tensor ldr = Render(delegate, scene, "LdrColor");
    const auto colour = [&](float x, float y) {
        const std::uint8_t* t = TexelAt(scene, ldr, {x, y, 0.0f});
        return glm::ivec3(t[0], t[1], t[2]);
    };
    const glm::ivec3 topLeft = colour(-4.0f, 4.0f);
    const glm::ivec3 topRight = colour(4.0f, 4.0f);
    const glm::ivec3 bottomLeft = colour(-4.0f, -4.0f);
    const glm::ivec3 bottomRight = colour(4.0f, -4.0f);
    const auto leads = [](glm::ivec3 rgb, int channel) {
        return rgb[channel] > rgb[(channel + 1) % 3] + 80 && rgb[channel] > rgb[(channel + 2) % 3] + 80;
    };
    EXPECT_TRUE(leads(topLeft, 0)) << "red " << topLeft.r << " " << topLeft.g << " " << topLeft.b;
    EXPECT_TRUE(leads(topRight, 1)) << "green";
    EXPECT_TRUE(leads(bottomLeft, 2)) << "blue";
    EXPECT_GT(std::min({bottomRight.r, bottomRight.g, bottomRight.b}), 150) << "white";
    EXPECT_LT(std::max({bottomRight.r, bottomRight.g, bottomRight.b}) -
                  std::min({bottomRight.r, bottomRight.g, bottomRight.b}),
              30);

    // The texture's scale multiplies it: a quarter of red stays red, dimmer.
    scene.Set("/Looks/M/Tex", tokens::kShaderScale, glm::vec4(0.25f, 0.25f, 0.25f, 1.0f));
    const Tensor dim = Render(delegate, scene, "LdrColor");
    EXPECT_LT(TexelAt(scene, dim, {-4.0f, 4.0f, 0.0f})[0] + 40, topLeft.r);
}

TEST_F(RasterDelegateTest, NormalMapsTiltTheSurfaceAlongItsTextureCoordinates) {
    // A normal map tilted 30 degrees towards +s (+x on the quad), stored as n / 2 + 1/2 with the
    // usual scale 2 and bias -1, under a sun 45 degrees towards +x: Lambert's cos(i) rises from
    // cos 45 to cos 15; with the coordinates turned 180 degrees the tilt faces away, cos 75. The
    // Material names Lambert, which a preview surface network then shades with.
    Scene scene = TexturedScene();
    ernie::Material(scene, "/Looks/M").SetBrdf("lambert");
    scene.Set("/World/Sun", tokens::kTransform, ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    const float tilt = glm::radians(30.0f);
    const std::array<float, 4> normal{0.5f * std::sin(tilt) + 0.5f, 0.5f,
                                      0.5f * std::cos(tilt) + 0.5f, 1.0f};
    ernie::Shader(scene, "/Looks/M/Tex").SetTexture(Tensor::Copy<float>(normal, {1, 1, 4}));
    scene.Set("/Looks/M/Tex", tokens::kShaderScale, glm::vec4(2.0f, 2.0f, 2.0f, 1.0f));
    scene.Set("/Looks/M/Tex", tokens::kShaderBias, glm::vec4(-1.0f, -1.0f, -1.0f, 0.0f));
    RasterDelegate delegate(*context_);
    const float flat = CentreGreen(delegate, scene);
    ASSERT_GT(flat, 0.0f);

    ernie::Connect(scene, "/Looks/M/Surface", tokens::kNormalInput, "/Looks/M/Tex",
                   "outputs:rgb");
    const float towards = CentreGreen(delegate, scene);
    const auto cosine = [](double degrees) {
        return static_cast<float>(std::cos(glm::radians(degrees)));
    };
    EXPECT_NEAR(towards / flat, cosine(15.0) / cosine(45.0), 0.03f);

    ernie::Shader::Define(scene, "/Looks/M/Turn")
        .SetInfoId(tokens::kUsdTransform2d)
        .SetRotation(180.0);
    ernie::Connect(scene, "/Looks/M/Turn", tokens::kIn, "/Looks/M/St", "outputs:result");
    ernie::Connect(scene, "/Looks/M/Tex", tokens::kSt, "/Looks/M/Turn", "outputs:result");
    const float away = CentreGreen(delegate, scene);
    EXPECT_NEAR(away / flat, cosine(75.0) / cosine(45.0), 0.03f);
}

TEST_F(RasterDelegateTest, PreviewSurfacesShadeWithGgxAndTheirTexturesPerPixel) {
    // A metallic texture with 0 on the image's left and 1 on its right (in b, as glTF's
    // metallicRoughness), and a roughness texture of 0.3 (in g): each half shades as the plain
    // material of the same constant metallic and roughness.
    Scene scene = PreviewScene();
    RasterDelegate delegate(*context_);
    const auto render = [&](float metallic, float roughness) {
        ernie::Shader(scene, "/Looks/M/Surface").SetMetallic(metallic).SetRoughness(roughness);
        return Render(delegate, scene, "HdrColor");
    };
    const Tensor dielectric = render(0.0f, 0.3f);
    const Tensor metal = render(1.0f, 0.3f);
    const Tensor rougher = render(0.0f, 0.8f);
    const float left = GreenAt(scene, dielectric, -4.0f, 0.0f);
    const float right = GreenAt(scene, metal, 4.0f, 0.0f);
    EXPECT_GT(std::abs(GreenAt(scene, metal, -4.0f, 0.0f) - left), 0.05f * left)
        << "metal and dielectric differ";
    EXPECT_GT(std::abs(GreenAt(scene, rougher, -4.0f, 0.0f) - left), 0.02f * left)
        << "and so does roughness";

    const std::array<std::uint8_t, 8> metallicRoughness{0, 77, 0, 255, 0, 77, 255, 255};
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetTexture(Tensor::Copy<std::uint8_t>(metallicRoughness, {1, 2, 4}))
        .SetWrapS("clamp");
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kMetallic, "/Looks/M/Tex", "outputs:b");
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kRoughness, "/Looks/M/Tex", "outputs:g");
    const Tensor textured = render(0.5f, 0.9f);  // the textures replace these
    EXPECT_NEAR(GreenAt(scene, textured, -4.0f, 0.0f), left, 0.03f * left);
    EXPECT_NEAR(GreenAt(scene, textured, 4.0f, 0.0f), right, 0.03f * right);
}

TEST_F(RasterDelegateTest, OpacityThresholdsCutSurfacesOutOfImagesAndShadows) {
    // The quad's left half has opacity 0 (in a texture's a) under a threshold of 0.5, over a
    // brighter ground 5 m below, with the sun overhead casting shadows: through the cut half the
    // camera sees the ground lit as it is clear of the quad (no shadow from what is cut out),
    // and the right half still shows the quad.
    Scene scene = TexturedScene();
    ernie::DistantLight(scene, "/World/Sun").SetShadowEnable(true);
    // A dark, rough quad, so it shows darker than the ground.
    ernie::Shader(scene, "/Looks/M/Surface").SetDiffuseColor(glm::dvec3(0.05)).SetRoughness(1.0);
    ernie::Material::Define(scene, "/Looks/Ground").SetAlbedo(glm::dvec3(0.9));
    ernie::GroundPlane::Define(scene, "/World/Ground")
        .SetHeight(-5.0)
        .SetMaterialBinding("/Looks/Ground");
    const std::array<std::uint8_t, 8> alpha{255, 255, 255, 0, 255, 255, 255, 255};
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetTexture(Tensor::Copy<std::uint8_t>(alpha, {1, 2, 4}))
        .SetWrapS("clamp");
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kOpacity, "/Looks/M/Tex", "outputs:a");
    RasterDelegate delegate(*context_);
    const Tensor opaque = Render(delegate, scene, "HdrColor");
    const float clear = GreenAt(scene, opaque, -14.0f, 0.0f);
    EXPECT_LT(GreenAt(scene, opaque, -4.0f, 0.0f), 0.5f * clear)
        << "without a threshold opacity draws opaque: the quad, darker than the ground";

    ernie::Shader(scene, "/Looks/M/Surface").SetOpacityThreshold(0.5);
    const Tensor cut = Render(delegate, scene, "HdrColor");
    EXPECT_NEAR(GreenAt(scene, cut, -4.0f, 0.0f), clear, 0.05f * clear)
        << "the ground through the cut half, unshadowed";
    EXPECT_NEAR(GreenAt(scene, cut, 4.0f, 0.0f), GreenAt(scene, opaque, 4.0f, 0.0f), 1e-4f)
        << "the opaque half unchanged";
}

TEST_F(RasterDelegateTest, TexturesFollowTheSceneTheyComeFrom) {
    // Two scenes whose texture prims share a path and an attribute version but hold different
    // images: the second's must not show the first's (a cache keyed without the scene did).
    RasterDelegate delegate(*context_);
    const auto render = [&](std::uint8_t red) {
        Scene scene = TexturedScene();
        const std::array<std::uint8_t, 4> texel{red, 0, 0, 255};
        ernie::Shader(scene, "/Looks/M/Tex")
            .SetTexture(Tensor::Copy<std::uint8_t>(texel, {1, 1, 4}));
        ernie::Connect(scene, "/Looks/M/Surface", tokens::kDiffuseColor, "/Looks/M/Tex",
                       "outputs:rgb");
        const Tensor hdr = Render(delegate, scene, "HdrColor");
        const std::size_t centre = (kSize / 2) * kSize + kSize / 2;
        return hdr.As<float>()[4 * centre];
    };
    const float dark = render(0);
    const float bright = render(255);
    EXPECT_GT(bright, dark + 0.1f);
}
