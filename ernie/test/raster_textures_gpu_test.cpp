// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/compute_interop.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/camera.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/util/texture_bake.h"
#include "support/delegate_scenes.h"
#include "support/vulkan_fixture.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using ernie::Path;
using ernie::RasterDelegate;
using ernie::Scene;
using ernie::Tensor;
using namespace ernie::testing;
namespace tokens = ernie::tokens;

namespace {

/// Sets or unsets an environment variable: setenv and unsetenv are
/// POSIX's, _putenv_s Windows's (an empty value unsets).
void SetEnv(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void UnsetEnv(const char* name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

/// A 16 m quad at the origin under MeshScene's overhead camera, with
/// texture coordinates (0, 0) to (1, 1) from its corner at (-8, -8)
/// to (8, 8), bound to /Looks/M: a UsdPreviewSurface whose
/// UsdUVTexture /Looks/M/Tex reads primvar st. Its two products
/// render LdrColor and HdrColor.
Scene TexturedScene() {
    Scene scene;
    ernie::Camera::Define(scene, "/World/Camera")
        .SetFocalLength(18.0)
        .SetSensorNoise(false);
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 60.0},
                                     {0.0, 0.0, 0.0},
                                     {0.0, 1.0, 0.0}));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform,
             ernie::DistantLightTransform({0.0, 0.0, 1.0}));
    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetResolution(glm::dvec2(kSize, kSize))
        .SetAmbient(0.0);
    DefineQuad(scene, "/World/Quad", 8.0f);
    const std::array<float, 8> st{0, 0, 1, 0, 1, 1, 0, 1};
    scene.Write("/World/Quad", "primvars:st",
                Tensor::Copy<float>(st, {4, 2}));
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
    ernie::Connect(scene, "/Looks/M", tokens::kSurfaceOutput,
                   "/Looks/M/Surface", tokens::kSurfaceOutput);
    ernie::Connect(scene, "/Looks/M/Tex", tokens::kSt, "/Looks/M/St",
                   "outputs:result");

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
Tensor Render(RasterDelegate& delegate, const Scene& scene,
              const std::string& var) {
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
float GreenAt(const Scene& scene, const Tensor& hdr, float x,
              float y) {
    const glm::vec4 clip =
        ernie::CameraParamsOf(scene, "/World/Camera", kSize, kSize)
            .RasterView()
            .ViewProjection() *
        glm::vec4(x, y, 0.0f, 1.0f);
    const glm::vec2 ndc = glm::vec2(clip) / clip.w;
    const auto col =
        static_cast<std::size_t>((ndc.x * 0.5f + 0.5f) * kSize);
    const auto row =
        static_cast<std::size_t>((0.5f - ndc.y * 0.5f) * kSize);
    return hdr.As<float>()[4 * (row * kSize + col) + 1];
}

/// A DDS file of `width` x `height` with `mips` mips: `fourCc`'s
/// blocks, or with no FourCC 32-bit RGBA texels, then `payload` (the
/// mips one after another); written to the test's temporary directory
/// as `name`.
std::string WriteDds(const std::string& name, std::uint32_t width,
                     std::uint32_t height, std::string_view fourCc,
                     std::span<const std::uint8_t> payload,
                     std::uint32_t mips = 1) {
    std::array<std::uint32_t, 32> header{};
    std::memcpy(&header[0], "DDS ", 4);
    header[1] = 124;
    header[2] = 0x1007 | (mips > 1 ? 0x20000u : 0u);
    header[3] = height;
    header[4] = width;
    header[7] = mips;
    header[19] = 32;
    if (!fourCc.empty()) {
        header[20] = 0x4;
        std::memcpy(&header[21], fourCc.data(), 4);
    } else {
        header[20] = 0x41;
        header[22] = 32;
        header[23] = 0xffu;
        header[24] = 0xff00u;
        header[25] = 0xff0000u;
        header[26] = 0xff000000u;
    }
    const std::string path = ::testing::TempDir() + name;
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(header.data()), 128);
    file.write(reinterpret_cast<const char*>(payload.data()),
               static_cast<std::streamsize>(payload.size()));
    return path;
}

/// A KTX2 file of VkFormat `vkFormat`, `width` x `height`, one level
/// holding `payload`, written as `name`.
std::string WriteKtx2(const std::string& name, std::uint32_t vkFormat,
                      std::uint32_t width, std::uint32_t height,
                      std::span<const std::uint8_t> payload) {
    std::array<std::uint32_t, 26> header{};
    constexpr std::array<std::uint8_t, 12> kIdentifier{
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32,
        0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
    std::memcpy(header.data(), kIdentifier.data(), 12);
    header[3] = vkFormat;
    header[4] = 1;
    header[5] = width;
    header[6] = height;
    header[9] = 1;     // faces
    header[10] = 1;    // levels
    header[20] = 104;  // level 0's offset, then its lengths
    header[22] = static_cast<std::uint32_t>(payload.size());
    header[24] = header[22];
    const std::string path = ::testing::TempDir() + name;
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(header.data()), 104);
    file.write(reinterpret_cast<const char*>(payload.data()),
               static_cast<std::streamsize>(payload.size()));
    return path;
}

/// HdrColor's rgb at the pixels showing quad points (+-4, +-4).
std::vector<float> Quadrants(RasterDelegate& delegate,
                             const Scene& scene) {
    const Tensor hdr = Render(delegate, scene, "HdrColor");
    std::vector<float> rgb;
    for (const glm::vec2 point :
         {glm::vec2(-4, 4), glm::vec2(4, 4), glm::vec2(-4, -4),
          glm::vec2(4, -4)}) {
        const glm::vec4 clip =
            ernie::CameraParamsOf(scene, "/World/Camera", kSize,
                                  kSize)
                .RasterView()
                .ViewProjection() *
            glm::vec4(point, 0.0f, 1.0f);
        const glm::vec2 ndc = glm::vec2(clip) / clip.w;
        const auto col =
            static_cast<std::size_t>((ndc.x * 0.5f + 0.5f) * kSize);
        const auto row =
            static_cast<std::size_t>((0.5f - ndc.y * 0.5f) * kSize);
        for (std::size_t c = 0; c < 3; ++c) {
            rgb.push_back(
                hdr.As<float>()[4 * (row * kSize + col) + c]);
        }
    }
    return rgb;
}

/// /Looks/M as a plain grey preview surface lit and seen from
/// straight above, where its specular lobe shows.
Scene PreviewScene() {
    Scene scene = TexturedScene();
    ernie::Shader(scene, "/Looks/M/Surface")
        .SetDiffuseColor(glm::dvec3(0.6));
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({0.3, 0.0, 1.0}));
    return scene;
}

}  // namespace

TEST_F(RasterDelegateTest,
       AlbedoTexturesFollowTextureCoordinatesWithTheImageUpright) {
    // Red, green in the top row; blue, white in the bottom: t = 0 is
    // the image's bottom row, so the quad shows the image as it is,
    // red top left. A 2 x 2 image filters bilinearly across the whole
    // quad (pure only at texel centres, which the sampled pixels miss
    // slightly) and the preview surface's white highlight greys it,
    // so each quadrant's own channel leads.
    Scene scene = TexturedScene();
    const std::array<std::uint8_t, 16> texels{
        255, 0, 0,   255, 0,   255, 0,   255,
        0,   0, 255, 255, 255, 255, 255, 255};
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetTexture(Tensor::Copy<std::uint8_t>(texels, {2, 2, 4}))
        .SetWrapS("clamp")
        .SetWrapT("clamp");
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kDiffuseColor,
                   "/Looks/M/Tex", "outputs:rgb");
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
        return rgb[channel] > rgb[(channel + 1) % 3] + 80 &&
               rgb[channel] > rgb[(channel + 2) % 3] + 80;
    };
    EXPECT_TRUE(leads(topLeft, 0)) << "red " << topLeft.r << " "
                                   << topLeft.g << " " << topLeft.b;
    EXPECT_TRUE(leads(topRight, 1)) << "green";
    EXPECT_TRUE(leads(bottomLeft, 2)) << "blue";
    EXPECT_GT(std::min({bottomRight.r, bottomRight.g, bottomRight.b}),
              150)
        << "white";
    EXPECT_LT(
        std::max({bottomRight.r, bottomRight.g, bottomRight.b}) -
            std::min({bottomRight.r, bottomRight.g, bottomRight.b}),
        30);

    // The texture's scale multiplies it: a quarter of red stays red,
    // dimmer.
    scene.Set("/Looks/M/Tex", tokens::kShaderScale,
              glm::vec4(0.25f, 0.25f, 0.25f, 1.0f));
    const Tensor dim = Render(delegate, scene, "LdrColor");
    EXPECT_LT(TexelAt(scene, dim, {-4.0f, 4.0f, 0.0f})[0] + 40,
              topLeft.r);
}

TEST_F(RasterDelegateTest,
       NormalMapsTiltTheSurfaceAlongItsTextureCoordinates) {
    // A normal map tilted 30 degrees towards +s (+x on the quad),
    // stored as n / 2 + 1/2 with the usual scale 2 and bias -1, under
    // a sun 45 degrees towards +x: Lambert's cos(i) rises from cos 45
    // to cos 15; with the coordinates turned 180 degrees the tilt
    // faces away, cos 75. The Material names Lambert, which a preview
    // surface network then shades with.
    Scene scene = TexturedScene();
    ernie::Material(scene, "/Looks/M").SetBrdf("lambert");
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    const float tilt = glm::radians(30.0f);
    const std::array<float, 4> normal{
        0.5f * std::sin(tilt) + 0.5f, 0.5f,
        0.5f * std::cos(tilt) + 0.5f, 1.0f};
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetTexture(Tensor::Copy<float>(normal, {1, 1, 4}));
    scene.Set("/Looks/M/Tex", tokens::kShaderScale,
              glm::vec4(2.0f, 2.0f, 2.0f, 1.0f));
    scene.Set("/Looks/M/Tex", tokens::kShaderBias,
              glm::vec4(-1.0f, -1.0f, -1.0f, 0.0f));
    RasterDelegate delegate(*context_);
    const float flat = CentreGreen(delegate, scene);
    ASSERT_GT(flat, 0.0f);

    ernie::Connect(scene, "/Looks/M/Surface", tokens::kNormalInput,
                   "/Looks/M/Tex", "outputs:rgb");
    const float towards = CentreGreen(delegate, scene);
    const auto cosine = [](double degrees) {
        return static_cast<float>(std::cos(glm::radians(degrees)));
    };
    EXPECT_NEAR(towards / flat, cosine(15.0) / cosine(45.0), 0.03f);

    ernie::Shader::Define(scene, "/Looks/M/Turn")
        .SetInfoId(tokens::kUsdTransform2d)
        .SetRotation(180.0);
    ernie::Connect(scene, "/Looks/M/Turn", tokens::kIn, "/Looks/M/St",
                   "outputs:result");
    ernie::Connect(scene, "/Looks/M/Tex", tokens::kSt,
                   "/Looks/M/Turn", "outputs:result");
    const float away = CentreGreen(delegate, scene);
    EXPECT_NEAR(away / flat, cosine(75.0) / cosine(45.0), 0.03f);
}

TEST_F(RasterDelegateTest,
       PreviewSurfacesShadeWithGgxAndTheirTexturesPerPixel) {
    // A metallic texture with 0 on the image's left and 1 on its
    // right (in b, as glTF's metallicRoughness), and a roughness
    // texture of 0.3 (in g): each half shades as the plain material
    // of the same constant metallic and roughness.
    Scene scene = PreviewScene();
    RasterDelegate delegate(*context_);
    const auto render = [&](float metallic, float roughness) {
        ernie::Shader(scene, "/Looks/M/Surface")
            .SetMetallic(metallic)
            .SetRoughness(roughness);
        return Render(delegate, scene, "HdrColor");
    };
    const Tensor dielectric = render(0.0f, 0.3f);
    const Tensor metal = render(1.0f, 0.3f);
    const Tensor rougher = render(0.0f, 0.8f);
    const float left = GreenAt(scene, dielectric, -4.0f, 0.0f);
    const float right = GreenAt(scene, metal, 4.0f, 0.0f);
    EXPECT_GT(std::abs(GreenAt(scene, metal, -4.0f, 0.0f) - left),
              0.05f * left)
        << "metal and dielectric differ";
    EXPECT_GT(std::abs(GreenAt(scene, rougher, -4.0f, 0.0f) - left),
              0.02f * left)
        << "and so does roughness";

    const std::array<std::uint8_t, 8> metallicRoughness{
        0, 77, 0, 255, 0, 77, 255, 255};
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetTexture(
            Tensor::Copy<std::uint8_t>(metallicRoughness, {1, 2, 4}))
        .SetWrapS("clamp");
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kMetallic,
                   "/Looks/M/Tex", "outputs:b");
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kRoughness,
                   "/Looks/M/Tex", "outputs:g");
    const Tensor textured =
        render(0.5f, 0.9f);  // the textures replace these
    EXPECT_NEAR(GreenAt(scene, textured, -4.0f, 0.0f), left,
                0.03f * left);
    EXPECT_NEAR(GreenAt(scene, textured, 4.0f, 0.0f), right,
                0.03f * right);
}

TEST_F(RasterDelegateTest,
       OpacityThresholdsCutSurfacesOutOfImagesAndShadows) {
    // The quad's left half has opacity 0 (in a texture's a) under a
    // threshold of 0.5, over a brighter ground 5 m below, with the
    // sun overhead casting shadows: through the cut half the camera
    // sees the ground lit as it is clear of the quad (no shadow from
    // what is cut out), and the right half still shows the quad.
    Scene scene = TexturedScene();
    ernie::DistantLight(scene, "/World/Sun").SetShadowEnable(true);
    // A dark, rough quad, so it shows darker than the ground.
    ernie::Shader(scene, "/Looks/M/Surface")
        .SetDiffuseColor(glm::dvec3(0.05))
        .SetRoughness(1.0);
    ernie::Material::Define(scene, "/Looks/Ground")
        .SetAlbedo(glm::dvec3(0.9));
    ernie::GroundPlane::Define(scene, "/World/Ground")
        .SetHeight(-5.0)
        .SetMaterialBinding("/Looks/Ground");
    const std::array<std::uint8_t, 8> alpha{255, 255, 255, 0,
                                            255, 255, 255, 255};
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetTexture(Tensor::Copy<std::uint8_t>(alpha, {1, 2, 4}))
        .SetWrapS("clamp");
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kOpacity,
                   "/Looks/M/Tex", "outputs:a");
    RasterDelegate delegate(*context_);
    const Tensor opaque = Render(delegate, scene, "HdrColor");
    const float clear = GreenAt(scene, opaque, -14.0f, 0.0f);
    EXPECT_LT(GreenAt(scene, opaque, -4.0f, 0.0f), 0.5f * clear)
        << "without a threshold opacity draws opaque: the quad, "
           "darker than the ground";

    ernie::Shader(scene, "/Looks/M/Surface").SetOpacityThreshold(0.5);
    const Tensor cut = Render(delegate, scene, "HdrColor");
    EXPECT_NEAR(GreenAt(scene, cut, -4.0f, 0.0f), clear,
                0.05f * clear)
        << "the ground through the cut half, unshadowed";
    EXPECT_NEAR(GreenAt(scene, cut, 4.0f, 0.0f),
                GreenAt(scene, opaque, 4.0f, 0.0f), 1e-4f)
        << "the opaque half unchanged";
}

TEST_F(RasterDelegateTest, TexturesFollowTheSceneTheyComeFrom) {
    // Two scenes whose texture prims share a path and an attribute
    // version but hold different images: the second's must not show
    // the first's (a cache keyed without the scene did).
    RasterDelegate delegate(*context_);
    const auto render = [&](std::uint8_t red) {
        Scene scene = TexturedScene();
        const std::array<std::uint8_t, 4> texel{red, 0, 0, 255};
        ernie::Shader(scene, "/Looks/M/Tex")
            .SetTexture(Tensor::Copy<std::uint8_t>(texel, {1, 1, 4}));
        ernie::Connect(scene, "/Looks/M/Surface",
                       tokens::kDiffuseColor, "/Looks/M/Tex",
                       "outputs:rgb");
        const Tensor hdr = Render(delegate, scene, "HdrColor");
        const std::size_t centre = (kSize / 2) * kSize + kSize / 2;
        return hdr.As<float>()[4 * centre];
    };
    const float dark = render(0);
    const float bright = render(255);
    EXPECT_GT(bright, dark + 0.1f);
}

TEST_F(RasterDelegateTest, StoredTexturesRenderAsTheirDecodedTwins) {
    // The same 2 x 2 RGBA8 image as a tensor (decoded, mipped and
    // uploaded) and as a DDS file (uploaded as stored); then a BC1
    // block of pure red in a KTX2 file against a red tensor.
    Scene scene = TexturedScene();
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kDiffuseColor,
                   "/Looks/M/Tex", "outputs:rgb");
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetWrapS("clamp")
        .SetWrapT("clamp");
    const std::array<std::uint8_t, 16> texels{
        255, 0, 0,   255, 0,   255, 0,   255,
        0,   0, 255, 255, 255, 255, 255, 255};
    RasterDelegate delegate(*context_);
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetTexture(Tensor::Copy<std::uint8_t>(texels, {2, 2, 4}));
    const std::vector<float> decoded = Quadrants(delegate, scene);
    scene.Clear("/Looks/M/Tex", tokens::kShaderTexture);
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetFile(WriteDds("stored_rgba8.dds", 2, 2, "", texels));
    const std::vector<float> stored = Quadrants(delegate, scene);
    ASSERT_EQ(stored.size(), decoded.size());
    for (std::size_t i = 0; i < stored.size(); ++i) {
        EXPECT_NEAR(stored[i], decoded[i], 0.01f * decoded[i] + 1e-4f)
            << "value " << i;
    }

    // BC1 (VkFormat 133): both endpoints red 565, every index 0.
    const std::array<std::uint8_t, 8> red{0x00, 0xF8, 0x00, 0xF8,
                                          0,    0,    0,    0};
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetFile(WriteKtx2("stored_bc1.ktx2", 133, 4, 4, red));
    const std::vector<float> bc1 = Quadrants(delegate, scene);
    scene.Clear("/Looks/M/Tex", tokens::kFile);
    const std::array<std::uint8_t, 4> texel{255, 0, 0, 255};
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetTexture(Tensor::Copy<std::uint8_t>(texel, {1, 1, 4}));
    const std::vector<float> tensor = Quadrants(delegate, scene);
    ASSERT_GT(tensor[0], tensor[1] + 0.05f);
    for (std::size_t i = 0; i < bc1.size(); ++i) {
        EXPECT_NEAR(bc1[i], tensor[i], 0.01f * tensor[i] + 1e-4f)
            << "value " << i;
    }
}

TEST_F(RasterDelegateTest, StoredBc5NormalMapsRebuildZ) {
    // As NormalMapsTiltTheSurfaceAlongItsTextureCoordinates, from a
    // BC5 (ATI2) DDS file: x = sin 30 and y = 0 stored as n / 2 + 1/2
    // in two channels, z rebuilt in the shader.
    Scene scene = TexturedScene();
    ernie::Material(scene, "/Looks/M").SetBrdf("lambert");
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    // Each channel's block: both endpoints the value, every index 0.
    const std::array<std::uint8_t, 16> block{
        191, 191, 0, 0, 0, 0, 0, 0, 128, 128, 0, 0, 0, 0, 0, 0};
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetFile(WriteDds("stored_bc5.dds", 4, 4, "ATI2", block));
    scene.Set("/Looks/M/Tex", tokens::kShaderScale,
              glm::vec4(2.0f, 2.0f, 2.0f, 1.0f));
    scene.Set("/Looks/M/Tex", tokens::kShaderBias,
              glm::vec4(-1.0f, -1.0f, -1.0f, 0.0f));
    RasterDelegate delegate(*context_);
    const float flat = CentreGreen(delegate, scene);
    ASSERT_GT(flat, 0.0f);
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kNormalInput,
                   "/Looks/M/Tex", "outputs:rgb");
    const float towards = CentreGreen(delegate, scene);
    EXPECT_NEAR(towards / flat,
                std::cos(glm::radians(15.0f)) /
                    std::cos(glm::radians(45.0f)),
                0.03f);

    // Without the usual scale and bias the file cannot be read as
    // normals.
    scene.Set("/Looks/M/Tex", tokens::kShaderBias, glm::vec4(0.0f));
    EXPECT_THROW(Render(delegate, scene, "HdrColor"),
                 std::invalid_argument);
}

TEST_F(RasterDelegateTest, MaterialsReadingOneTextureShareItsUpload) {
    // Two materials reading one 256 x 256 RGBA8 file hold one upload
    // of it; reading two files, two. The second file's upload goes
    // with its last reader.
    const std::vector<std::uint8_t> texels(256 * 256 * 4, 200);
    constexpr std::uint64_t kBytes = 256 * 256 * 4;
    const std::string first =
        WriteDds("shared_a.dds", 256, 256, "", texels);
    const std::string second =
        WriteDds("shared_b.dds", 256, 256, "", texels);
    Scene scene = TexturedScene();
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kDiffuseColor,
                   "/Looks/M/Tex", "outputs:rgb");
    ernie::Shader(scene, "/Looks/M/Tex").SetFile(first);
    // A second material with its own texture reader, sharing
    // /Looks/M's coordinates.
    ernie::Material::Define(scene, "/Looks/N");
    ernie::Shader::Define(scene, "/Looks/N/Surface")
        .SetInfoId(tokens::kUsdPreviewSurface);
    ernie::Shader::Define(scene, "/Looks/N/Tex")
        .SetInfoId(tokens::kUsdUVTexture)
        .SetSourceColorSpace("raw")
        .SetFile(first);
    ernie::Connect(scene, "/Looks/N", tokens::kSurfaceOutput,
                   "/Looks/N/Surface", tokens::kSurfaceOutput);
    ernie::Connect(scene, "/Looks/N/Surface", tokens::kDiffuseColor,
                   "/Looks/N/Tex", "outputs:rgb");
    ernie::Connect(scene, "/Looks/N/Tex", tokens::kSt, "/Looks/M/St",
                   "outputs:result");
    RasterDelegate delegate(*context_);
    // Each state renders twice, so what a first frame allocates is in
    // every measurement.
    const auto bytes = [&] {
        Render(delegate, scene, "HdrColor");
        Render(delegate, scene, "HdrColor");
        return delegate.Stats().gpuBytes;
    };
    const std::uint64_t shared = bytes();

    ernie::Shader(scene, "/Looks/N/Tex").SetFile(second);
    const std::uint64_t separate = bytes();
    EXPECT_EQ(separate, shared + kBytes);

    ernie::Shader(scene, "/Looks/N/Tex").SetFile(first);
    EXPECT_EQ(bytes(), shared);
}

TEST_F(RasterDelegateTest,
       BasisUniversalTexturesRenderAsTheirSources) {
    // The fixtures' 16 x 16 quadrants (fixtures/ktx2) as a tensor,
    // and as ETC1S and UASTC KTX2 files transcoded to BC7: solid
    // colours survive both codecs closely.
    const auto fixture = [](const char* name) {
        return (std::filesystem::path(ERNIE_TEST_FIXTURES) / "ktx2" /
                name)
            .string();
    };
    Scene scene = TexturedScene();
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kDiffuseColor,
                   "/Looks/M/Tex", "outputs:rgb");
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetWrapS("clamp")
        .SetWrapT("clamp");
    std::vector<std::uint8_t> texels(16 * 16 * 4);
    for (std::size_t y = 0; y < 16; ++y) {
        for (std::size_t x = 0; x < 16; ++x) {
            const bool right = x >= 8;
            const bool bottom = y >= 8;
            const std::array<std::uint8_t, 4> colour =
                !bottom
                    ? (right ? std::array<std::uint8_t, 4>{0, 255, 0,
                                                           255}
                             : std::array<std::uint8_t, 4>{255, 0, 0,
                                                           255})
                    : (right ? std::array<std::uint8_t, 4>{255, 255,
                                                           255, 255}
                             : std::array<std::uint8_t, 4>{0, 0, 255,
                                                           255});
            std::copy(colour.begin(), colour.end(),
                      texels.begin() + static_cast<std::ptrdiff_t>(
                                           4 * (y * 16 + x)));
        }
    }
    RasterDelegate delegate(*context_);
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetTexture(Tensor::Copy<std::uint8_t>(texels, {16, 16, 4}));
    const std::vector<float> decoded = Quadrants(delegate, scene);
    scene.Clear("/Looks/M/Tex", tokens::kShaderTexture);
    // ETC1S quantises colour coarsely (saturated colours gain a
    // little of the others); UASTC keeps them within a few percent.
    const std::array<std::pair<const char*, bool>, 2> files{
        {{"quadrants_etc1s.ktx2", true},
         {"quadrants_uastc_zstd.ktx2", false}}};
    for (const auto& [name, etc1s] : files) {
        ernie::Shader(scene, "/Looks/M/Tex").SetFile(fixture(name));
        const std::vector<float> transcoded =
            Quadrants(delegate, scene);
        for (std::size_t i = 0; i < transcoded.size(); ++i) {
            EXPECT_NEAR(transcoded[i], decoded[i],
                        etc1s ? 0.05f : 0.03f * decoded[i] + 1e-3f)
                << name << " value " << i;
        }
    }

    // The normal maps, transcoded to BC5, tilt the surface 30 degrees
    // as NormalMapsTiltTheSurfaceAlongItsTextureCoordinates's does.
    ernie::Material(scene, "/Looks/M").SetBrdf("lambert");
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    scene.Set("/Looks/M/Tex", tokens::kShaderScale,
              glm::vec4(2.0f, 2.0f, 2.0f, 1.0f));
    scene.Set("/Looks/M/Tex", tokens::kShaderBias,
              glm::vec4(-1.0f, -1.0f, -1.0f, 0.0f));
    scene.Clear("/Looks/M/Surface",
                std::string(tokens::kDiffuseColor) +
                    std::string(tokens::kConnectSuffix));
    const float flat = CentreGreen(delegate, scene);
    ASSERT_GT(flat, 0.0f);
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kNormalInput,
                   "/Looks/M/Tex", "outputs:rgb");
    for (const char* name :
         {"tilted_normal_etc1s.ktx2", "tilted_normal_uastc.ktx2"}) {
        ernie::Shader(scene, "/Looks/M/Tex").SetFile(fixture(name));
        EXPECT_NEAR(CentreGreen(delegate, scene) / flat,
                    std::cos(glm::radians(15.0f)) /
                        std::cos(glm::radians(45.0f)),
                    0.03f)
            << name;
    }
}

namespace {

/// A 1024 x 1024 texture with every mip, forced virtual: the first
/// frames sample the mip tail, the tiles the view asks for load over
/// the next ones, and the picture is the one the texture uploaded
/// whole draws, far and close.
void ExpectVirtualAlbedoLoadsTheTilesInView(
    const ernie::VulkanContext& context) {
    // The quadrants as 32-texel checks of their colour and black, so
    // fine and coarse mips differ; each mip the 2 x 2 average of the
    // one before.
    constexpr std::uint32_t kSide = 1024;
    std::vector<ernie::Image> levels{ernie::Image(kSide, kSide)};
    const glm::vec4 kQuadrants[2][2] = {
        {glm::vec4(1, 0, 0, 1), glm::vec4(0, 1, 0, 1)},
        {glm::vec4(0, 0, 1, 1), glm::vec4(1, 1, 1, 1)}};
    for (std::uint32_t y = 0; y < kSide; ++y) {
        for (std::uint32_t x = 0; x < kSide; ++x) {
            const bool lit = (x / 32 + y / 32) % 2 == 0;
            levels[0].At(x, y) =
                lit ? kQuadrants[y / (kSide / 2)][x / (kSide / 2)]
                    : glm::vec4(0, 0, 0, 1);
        }
    }
    levels = ernie::MipChain(levels[0], ernie::MipFilter::Average);
    std::vector<std::uint8_t> texels;
    for (const ernie::Image& level : levels) {
        const std::vector<std::uint8_t> bytes =
            ernie::ToRgba8(level, ernie::ImageEncoding::Linear);
        texels.insert(texels.end(), bytes.begin(), bytes.end());
    }
    const auto mips = static_cast<std::uint32_t>(levels.size());
    const std::string file = WriteDds("virtual_quadrants.dds", kSide,
                                      kSide, "", texels, mips);
    Scene scene = TexturedScene();
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kDiffuseColor,
                   "/Looks/M/Tex", "outputs:rgb");
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetWrapS("clamp")
        .SetWrapT("clamp")
        .SetFile(file);
    RasterDelegate whole(context);
    scene.Set("/Looks/M/Tex", tokens::kVirtualTexture,
              std::string("off"));
    const std::vector<float> reference = Quadrants(whole, scene);

    scene.Set("/Looks/M/Tex", tokens::kVirtualTexture,
              std::string("on"));
    RasterDelegate delegate(context);
    std::vector<float> last;
    for (int frame = 0; frame < 3; ++frame) {
        last = Quadrants(delegate, scene);
    }
    for (std::size_t i = 0; i < last.size(); ++i) {
        EXPECT_NEAR(last[i], reference[i],
                    0.01f * reference[i] + 1e-4f)
            << "value " << i;
    }
    // At 48 pixels the quad asks for no finer mip than the tail's.
    // Closer and at 256 pixels it asks for finer ones, whose tiles
    // load over the next frames.
    ernie::RenderSettings(scene, "/Render/Settings")
        .SetResolution(glm::dvec2(256.0));
    ernie::RenderProduct(scene, "/Render/Out")
        .SetResolution(glm::dvec2(256.0));
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 20.0},
                                     {0.0, 0.0, 0.0},
                                     {0.0, 1.0, 0.0}));
    Render(delegate, scene, "HdrColor");
    const std::uint64_t before = delegate.Stats().gpuBytes;
    for (int frame = 0; frame < 4; ++frame) {
        Render(delegate, scene, "HdrColor");
    }
    // Sparse images grow as tiles bind; pages are allocated up front.
    if (context.SupportsSparseResidency()) {
        EXPECT_GT(delegate.Stats().gpuBytes, before)
            << "tiles were loaded";
    }

    // Close up the picture is the whole texture's, but for filtering
    // differences at the quadrants' edges (the paged path filters
    // isotropically).
    const Tensor close = Render(delegate, scene, "HdrColor");
    scene.Set("/Looks/M/Tex", tokens::kVirtualTexture,
              std::string("off"));
    const Tensor full = Render(whole, scene, "HdrColor");
    double difference = 0.0;
    double total = 0.0;
    for (std::int64_t i = 0; i < close.Elements(); ++i) {
        difference +=
            std::abs(close.As<float>()[i] - full.As<float>()[i]);
        total += std::abs(full.As<float>()[i]);
    }
    EXPECT_LT(difference, 0.02 * total);
}

}  // namespace

TEST_F(RasterDelegateTest, VirtualAlbedoTexturesLoadTheTilesInView) {
    // On a sparse image.
    if (!context_->SupportsSparseResidency()) {
        GTEST_SKIP() << "no sparse image residency";
    }
    ExpectVirtualAlbedoLoadsTheTilesInView(*context_);
}

namespace {

/// As `RasterDelegateTest`, with sparse residency off: virtual
/// textures page in software.
class PagedRasterDelegateTest : public RasterDelegateTest {
  protected:
    ernie::VulkanContextOptions Options() const override {
        ernie::VulkanContextOptions options =
            RasterDelegateTest::Options();
        options.disableSparseResidency = true;
        return options;
    }
};

}  // namespace

TEST_F(PagedRasterDelegateTest,
       VirtualAlbedoTexturesLoadTheTilesInView) {
    // In pages, filtered by the shader.
    ExpectVirtualAlbedoLoadsTheTilesInView(*context_);
}

TEST_F(RasterDelegateTest, BakedTexturesRenderAsTheirSources) {
    // The quadrants (64 x 64) mipped and baked to BC7 render as the
    // image uploaded whole; a tilted normal map baked to BC5 tilts
    // the surface as the plain one does.
    Scene scene = TexturedScene();
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kDiffuseColor,
                   "/Looks/M/Tex", "outputs:rgb");
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetWrapS("clamp")
        .SetWrapT("clamp");
    const glm::vec4 kQuadrants[2][2] = {
        {glm::vec4(1, 0, 0, 1), glm::vec4(0, 1, 0, 1)},
        {glm::vec4(0, 0, 1, 1), glm::vec4(1, 1, 1, 1)}};
    ernie::Image quadrants(64, 64);
    std::vector<std::uint8_t> texels;
    for (std::uint32_t y = 0; y < 64; ++y) {
        for (std::uint32_t x = 0; x < 64; ++x) {
            const glm::vec4 c = kQuadrants[y / 32][x / 32];
            quadrants.At(x, y) = c;
            for (int i = 0; i < 4; ++i) {
                texels.push_back(
                    static_cast<std::uint8_t>(c[i] * 255.0f));
            }
        }
    }
    RasterDelegate delegate(*context_);
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetTexture(Tensor::Copy<std::uint8_t>(texels, {64, 64, 4}));
    const std::vector<float> decoded = Quadrants(delegate, scene);
    scene.Clear("/Looks/M/Tex", tokens::kShaderTexture);
    const std::string colour =
        ::testing::TempDir() + "baked_quadrants.dds";
    ernie::WriteDdsFile(
        colour,
        ernie::EncodeTexture(
            ernie::MipChain(quadrants, ernie::MipFilter::Average),
            ernie::TextureFormat::Bc7, ernie::ImageEncoding::Linear));
    ernie::Shader(scene, "/Looks/M/Tex").SetFile(colour);
    const std::vector<float> baked = Quadrants(delegate, scene);
    for (std::size_t i = 0; i < baked.size(); ++i) {
        EXPECT_NEAR(baked[i], decoded[i], 0.03f * decoded[i] + 1e-3f)
            << "value " << i;
    }

    ernie::Material(scene, "/Looks/M").SetBrdf("lambert");
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    scene.Set("/Looks/M/Tex", tokens::kShaderScale,
              glm::vec4(2.0f, 2.0f, 2.0f, 1.0f));
    scene.Set("/Looks/M/Tex", tokens::kShaderBias,
              glm::vec4(-1.0f, -1.0f, -1.0f, 0.0f));
    scene.Clear("/Looks/M/Surface",
                std::string(tokens::kDiffuseColor) +
                    std::string(tokens::kConnectSuffix));
    const float flat = CentreGreen(delegate, scene);
    ASSERT_GT(flat, 0.0f);
    const float tilt = glm::radians(30.0f);
    const ernie::Image normals(
        16, 16,
        glm::vec4(0.5f * std::sin(tilt) + 0.5f, 0.5f,
                  0.5f * std::cos(tilt) + 0.5f, 1.0f));
    const std::string normal =
        ::testing::TempDir() + "baked_normal.dds";
    ernie::WriteDdsFile(
        normal,
        ernie::EncodeTexture(
            ernie::MipChain(normals, ernie::MipFilter::NormalSlopes),
            ernie::TextureFormat::Bc5, ernie::ImageEncoding::Linear));
    ernie::Shader(scene, "/Looks/M/Tex").SetFile(normal);
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kNormalInput,
                   "/Looks/M/Tex", "outputs:rgb");
    EXPECT_NEAR(CentreGreen(delegate, scene) / flat,
                std::cos(glm::radians(15.0f)) /
                    std::cos(glm::radians(45.0f)),
                0.03f);
}

TEST_F(RasterDelegateTest, ImageFilesUploadAsBc7KeptBetweenRuns) {
    // A 256 x 256 PNG albedo of the quadrants uploads as BC7, under
    // half the bytes of its tensor twin's RGBA8, and renders as the
    // twin does. Its BC7 is kept in the texture cache, from which a
    // second delegate uploads it: replaced there by black, it renders
    // black.
    if (!context_->SupportsTextureCompressionBc()) {
        GTEST_SKIP() << "The device cannot sample BC7";
    }
    const std::filesystem::path cache =
        std::filesystem::path(::testing::TempDir()) / "texture_cache";
    std::filesystem::remove_all(cache);
    SetEnv("ERNIE_TEXTURE_CACHE", cache.string().c_str());

    constexpr int kSide = 256;
    std::vector<std::uint8_t> texels;
    for (int y = 0; y < kSide; ++y) {
        for (int x = 0; x < kSide; ++x) {
            const bool right = x >= kSide / 2;
            const bool bottom = y >= kSide / 2;
            // Red, green on top; blue, white below.
            texels.push_back(right == bottom ? 255 : 0);
            texels.push_back(right ? 255 : 0);
            texels.push_back(bottom ? 255 : 0);
            texels.push_back(255);
        }
    }
    const std::string png =
        ::testing::TempDir() + "cached_quadrants.png";
    ASSERT_NE(stbi_write_png(png.c_str(), kSide, kSide, 4,
                             texels.data(), kSide * 4),
              0);
    Scene scene = TexturedScene();
    ernie::Connect(scene, "/Looks/M/Surface", tokens::kDiffuseColor,
                   "/Looks/M/Tex", "outputs:rgb");
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetWrapS("clamp")
        .SetWrapT("clamp");
    const auto quadrants = [&](RasterDelegate& delegate) {
        Quadrants(delegate, scene);
        const std::vector<float> rgb = Quadrants(delegate, scene);
        return std::make_pair(rgb, delegate.Stats().gpuBytes);
    };

    RasterDelegate twin(*context_);
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetTexture(
            Tensor::Copy<std::uint8_t>(texels, {kSide, kSide, 4}));
    const auto [decoded, rgba8Bytes] = quadrants(twin);
    scene.Clear("/Looks/M/Tex", tokens::kShaderTexture);
    ernie::Shader(scene, "/Looks/M/Tex").SetFile(png);
    RasterDelegate first(*context_);
    const auto [compressed, bc7Bytes] = quadrants(first);
    // RGBA8's 256 x 256 and mips are 349 524 bytes, BC7's 87 408.
    EXPECT_LT(bc7Bytes, rgba8Bytes - 200000);
    for (std::size_t i = 0; i < compressed.size(); ++i) {
        EXPECT_NEAR(compressed[i], decoded[i],
                    0.03f * decoded[i] + 1e-3f)
            << "value " << i;
    }

    std::vector<std::filesystem::path> entries;
    for (const auto& entry :
         std::filesystem::directory_iterator(cache)) {
        entries.push_back(entry.path());
    }
    ASSERT_EQ(entries.size(), 1u);
    ernie::WriteDdsFile(
        entries[0],
        ernie::EncodeTexture(
            ernie::MipChain(
                ernie::Image(kSide, kSide, glm::vec4(0, 0, 0, 1)),
                ernie::MipFilter::Average),
            ernie::TextureFormat::Bc7Srgb,
            ernie::ImageEncoding::Srgb));
    RasterDelegate second(*context_);
    const std::vector<float> cached = quadrants(second).first;
    // Black albedo, as a black tensor renders: the surface's specular
    // reflection alone.
    scene.Clear("/Looks/M/Tex", tokens::kFile);
    ernie::Shader(scene, "/Looks/M/Tex")
        .SetTexture(Tensor::Copy<std::uint8_t>(
            std::vector<std::uint8_t>(kSide * kSide * 4, 0),
            {kSide, kSide, 4}));
    const std::vector<float> black = quadrants(twin).first;
    for (std::size_t i = 0; i < cached.size(); ++i) {
        EXPECT_NEAR(cached[i], black[i], 1e-3f) << "value " << i;
    }
    UnsetEnv("ERNIE_TEXTURE_CACHE");
    std::filesystem::remove_all(cache);
}

TEST_F(RasterDelegateTest,
       TemperatureTexturesSetTheTemperatureAcrossAMesh) {
    // The quad's material at 250 K with a 2 x 1 temperature texture
    // of 0 (the material's) and 400 K: the Temperature var reads the
    // texels' centres, bilinear between them; the prim's own
    // temperature replaces both, and thermal emission off reads none.
    Scene scene = TexturedScene();
    const Path var("/Render/Vars/Temperature");
    ernie::RenderVar::Define(scene, var).SetSourceName("Temperature");
    scene.Set("/Render/Out", tokens::kOrderedVars,
              std::vector<Path>{var});
    ernie::RenderSettings(scene, "/Render/Settings")
        .SetThermalEnable(true);
    const std::array<float, 2> texels{0.0f, 400.0f};
    ernie::Material(scene, "/Looks/M")
        .SetThermalTemperature(250.0)
        .SetThermalTemperatureTexture(
            Tensor::Copy<float>(texels, {1, 2}));
    RasterDelegate delegate(*context_);
    const auto at = [&](float x) {
        const Tensor kelvin = Render(delegate, scene, "Temperature");
        const glm::vec4 clip =
            ernie::CameraParamsOf(scene, "/World/Camera", kSize,
                                  kSize)
                .RasterView()
                .ViewProjection() *
            glm::vec4(x, 0.0f, 0.0f, 1.0f);
        const auto col = static_cast<std::size_t>(
            (clip.x / clip.w * 0.5f + 0.5f) * kSize);
        return kelvin.As<float>()[(kSize / 2) * kSize + col];
    };
    // Outside the texels' centres (u 0.25 and 0.75) the edges clamp.
    EXPECT_NEAR(at(-6.0f), 250.0f, 0.01f) << "0 is the material's";
    EXPECT_NEAR(at(6.0f), 400.0f, 0.01f);
    // Within a pixel of u 0.5, where it climbs 50 K a metre.
    EXPECT_NEAR(at(0.0f), 200.0f, 40.0f)
        << "bilinear between 0 and 400";
    ernie::Mesh(scene, "/World/Quad").SetThermalTemperature(500.0);
    EXPECT_NEAR(at(6.0f), 500.0f, 0.01f);
    ernie::RenderSettings(scene, "/Render/Settings")
        .SetThermalEnable(false);
    EXPECT_EQ(at(6.0f), 0.0f);
}

TEST_F(RasterDelegateTest,
       SolverTemperaturesPerPointWriteBackFromHostAndCuda) {
    // The quad's points at 300 K on its left edge and 500 K on its
    // right, written as a solver steps: interpolated across it, in
    // place of the prim's own, updated in place from the host and,
    // with CUDA, from device arrays.
    Scene scene = TexturedScene();
    const Path var("/Render/Vars/Temperature");
    ernie::RenderVar::Define(scene, var).SetSourceName("Temperature");
    scene.Set("/Render/Out", tokens::kOrderedVars,
              std::vector<Path>{var});
    ernie::RenderSettings(scene, "/Render/Settings")
        .SetThermalEnable(true);
    ernie::Mesh(scene, "/World/Quad").SetThermalTemperature(250.0);
    RasterDelegate delegate(*context_);
    const auto at = [&](float x) {
        const Tensor kelvin = Render(delegate, scene, "Temperature");
        const glm::vec4 clip =
            ernie::CameraParamsOf(scene, "/World/Camera", kSize,
                                  kSize)
                .RasterView()
                .ViewProjection() *
            glm::vec4(x, 0.0f, 0.0f, 1.0f);
        const auto col = static_cast<std::size_t>(
            (clip.x / clip.w * 0.5f + 0.5f) * kSize);
        return kelvin.As<float>()[(kSize / 2) * kSize + col];
    };
    EXPECT_EQ(at(0.0f), 250.0f)
        << "the prim's own without a solver's";
    const auto write = [&](std::array<float, 4> kelvin) {
        scene.Write("/World/Quad", tokens::kThermalTemperatures,
                    Tensor::Copy<float>(kelvin, {4}));
    };
    write({300.0f, 500.0f, 500.0f, 300.0f});
    // Within a pixel of x = 0, where it climbs 12.5 K a metre.
    EXPECT_NEAR(at(0.0f), 400.0f, 10.0f);
    EXPECT_LT(at(-6.0f), at(6.0f));
    write({600.0f, 600.0f, 600.0f, 600.0f});
    EXPECT_NEAR(at(0.0f), 600.0f, 0.01f) << "the next step, in place";

    std::string why;
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    const std::array<float, 4> device{700.0f, 700.0f, 700.0f, 700.0f};
    scene.Write("/World/Quad", tokens::kThermalTemperatures,
                cuda->Upload(Tensor::Copy<float>(device, {4})));
    EXPECT_NEAR(at(0.0f), 700.0f, 0.01f) << "from the CUDA device";
    const std::array<float, 4> next{720.0f, 720.0f, 720.0f, 720.0f};
    scene.Write("/World/Quad", tokens::kThermalTemperatures,
                cuda->Upload(Tensor::Copy<float>(next, {4})));
    EXPECT_NEAR(at(0.0f), 720.0f, 0.01f);
    ernie::RenderSettings(scene, "/Render/Settings")
        .SetThermalEnable(false);
    EXPECT_EQ(at(0.0f), 0.0f);
}
