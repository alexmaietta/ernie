#include "ernie/raster/delegate.h"
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
    scene.Define("/World/Camera", tokens::kCamera)
        .Set(tokens::kTransform,
             ernie::LookAtTransform({0.0, 0.0, 60.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}))
        .Set(tokens::kFocalLength, 18.0f)
        .Set(tokens::kSensorNoise, false);
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, ernie::DistantLightTransform({0.0, 0.0, 1.0}));
    scene.Define("/Render/Settings", tokens::kRenderSettings)
        .Set(tokens::kResolution, glm::ivec2(kSize, kSize))
        .Set(tokens::kAmbient, 0.0f);
    DefineQuad(scene, "/World/Quad", 8.0f);
    const std::array<float, 8> st{0, 0, 1, 0, 1, 1, 0, 1};
    scene.Write("/World/Quad", "primvars:st", Tensor::Copy<float>(st, {4, 2}));
    scene.Set("/World/Quad", tokens::kMaterialBinding, Path("/Looks/M"));

    scene.Define("/Looks/M", tokens::kMaterial);
    scene.Define("/Looks/M/Surface", tokens::kShader)
        .Set(tokens::kInfoId, std::string(tokens::kUsdPreviewSurface))
        .Set(tokens::kDiffuseColor, glm::vec3(0.5f));
    scene.Define("/Looks/M/Tex", tokens::kShader)
        .Set(tokens::kInfoId, std::string(tokens::kUsdUVTexture))
        .Set(tokens::kSourceColorSpace, std::string("raw"));
    scene.Define("/Looks/M/St", tokens::kShader)
        .Set(tokens::kInfoId, std::string(tokens::kUsdPrimvarReaderFloat2))
        .Set(tokens::kVarname, std::string("st"));
    ernie::Connect(scene, "/Looks/M", tokens::kSurfaceOutput, "/Looks/M/Surface",
                   tokens::kSurfaceOutput);
    ernie::Connect(scene, "/Looks/M/Tex", tokens::kSt, "/Looks/M/St", "outputs:result");

    std::vector<Path> vars;
    for (const std::string_view name : {"HdrColor", "LdrColor"}) {
        const Path var("/Render/Vars/" + std::string(name));
        scene.Define(var, tokens::kRenderVar).Set(tokens::kSourceName, std::string(name));
        vars.push_back(var);
    }
    scene.Define("/Render/Out", tokens::kRenderProduct)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(kSize, kSize))
        .Set(tokens::kOrderedVars, vars);
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

}  // namespace

TEST_F(RasterDelegateTest, AlbedoTexturesFollowTextureCoordinatesWithTheImageUpright) {
    // Red, green in the top row; blue, white in the bottom: t = 0 is the image's bottom row, so
    // the quad shows the image as it is, red top left. Primaries come back through spectral
    // upsampling and the camera's response less than pure, so each quadrant's own channel leads.
    Scene scene = TexturedScene();
    const std::array<std::uint8_t, 16> texels{255, 0, 0,   255, 0,   255, 0,   255,
                                              0,   0, 255, 255, 255, 255, 255, 255};
    scene.Write("/Looks/M/Tex", tokens::kShaderTexture,
                Tensor::Copy<std::uint8_t>(texels, {2, 2, 4}));
    scene.Set("/Looks/M/Tex", tokens::kWrapS, std::string("clamp"));
    scene.Set("/Looks/M/Tex", tokens::kWrapT, std::string("clamp"));
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
    // cos 45 to cos 15; with the coordinates turned 180 degrees the tilt faces away, cos 75.
    Scene scene = TexturedScene();
    scene.Set("/World/Sun", tokens::kTransform, ernie::DistantLightTransform({1.0, 0.0, 1.0}));
    const float tilt = glm::radians(30.0f);
    const std::array<float, 4> normal{0.5f * std::sin(tilt) + 0.5f, 0.5f,
                                      0.5f * std::cos(tilt) + 0.5f, 1.0f};
    scene.Write("/Looks/M/Tex", tokens::kShaderTexture, Tensor::Copy<float>(normal, {1, 1, 4}));
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

    scene.Define("/Looks/M/Turn", tokens::kShader)
        .Set(tokens::kInfoId, std::string(tokens::kUsdTransform2d))
        .Set(tokens::kRotation, 180.0);
    ernie::Connect(scene, "/Looks/M/Turn", tokens::kIn, "/Looks/M/St", "outputs:result");
    ernie::Connect(scene, "/Looks/M/Tex", tokens::kSt, "/Looks/M/Turn", "outputs:result");
    const float away = CentreGreen(delegate, scene);
    EXPECT_NEAR(away / flat, cosine(75.0) / cosine(45.0), 0.03f);
}
