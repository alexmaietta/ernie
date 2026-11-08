// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/render/render_output.h"
#include "ernie/render/renderer.h"
#include "ernie/render/shaders.h"
#include "ernie/scene/scene.h"
#include "support/devices.h"
#include "support/support.h"

namespace ernie {
namespace {

using device::Device;
using test::ForEachDevice;

const std::filesystem::path& Scratch() {
    static const test::TemporaryDirectory directory;
    return directory.Path();
}

constexpr uint32_t kSize = 8;
const std::vector<std::string> kProducts = {"/Render/View"};

struct Wall {
    Ref<Scene> scene;
    Prim surface;
    Prim spectrum;
    Prim camera;
    Prim product;
};

Wall SpectralWall(Device& device) {
    Wall wall{.scene = Scene::Create(device)};
    Scene& scene = *wall.scene;
    scene.Define<Scope>("/Looks");
    const Prim material = scene.Define<Material>("/Looks/Paint");
    wall.surface = scene.Define<Shader>("/Looks/Paint/Surface", {.id = "UsdPreviewSurface"});
    material.Connect("surface", wall.surface, "surface");
    wall.spectrum = scene.Define<Shader>("/Looks/Paint/Spectrum", {.id = "SampledSpectrum"});
    scene.Define<Plane>("/Wall", {.materialBinding = {"/Looks/Paint"},
                                  .translate = {0, 0, -10},
                                  .width = 100,
                                  .length = 100});
    wall.camera = scene.Define<Camera>("/Cam", {.focalLength = 50});
    scene.Define<Scope>("/Render");
    scene.Define<RenderVar>("/Render/Color", {.sourceName = "HdrColor"});
    wall.product = scene.Define<RenderProduct>("/Render/View", {.renderMode = RenderMode::RealTime,
                                                                .camera = {"/Cam"},
                                                                .resolution = {kSize, kSize},
                                                                .orderedVars = {"/Render/Color"}});
    return wall;
}

Float3 Center(Renderer& renderer) {
    RenderOutput output = renderer.Step(kProducts, 0);
    output.Wait();
    const std::vector<float> pixels =
        output["/Render/View"].frames.at(0).Source("HdrColor").tensor.ToHost<float>();
    const uint32_t at = (kSize / 2 * kSize + kSize / 2) * 4;
    return {pixels[at], pixels[at + 1], pixels[at + 2]};
}

void ExpectNear(Float3 a, Float3 b) {
    EXPECT_NEAR(a.x, b.x, 1e-3f);
    EXPECT_NEAR(a.y, b.y, 1e-3f);
    EXPECT_NEAR(a.z, b.z, 1e-3f);
}

TEST(SpectraTest, FlatSpectraRenderLikeTheirGreyInRgbAndBands) {
    ForEachDevice([](Device& device) {
        Wall wall = SpectralWall(device);
        wall.surface.Set(UsdPreviewSurface::kDiffuseColor, Float3{0.5f, 0.5f, 0.5f});
        wall.spectrum.Set(SampledSpectrum::kWavelengths, std::vector<float>{300, 1000});
        wall.spectrum.Set(SampledSpectrum::kValues, std::vector<float>{0.5f, 0.5f});
        Renderer renderer(wall.scene);
        for (const SpectralMode mode : {SpectralMode::Rgb, SpectralMode::Bands}) {
            wall.product.Set(RenderProduct::kSpectralMode, mode);
            wall.surface.Disconnect("diffuseColor");
            const Float3 grey = Center(renderer);
            ASSERT_GT(grey.y, 0.05f);
            wall.surface.Connect("diffuseColor", wall.spectrum, "spectrum");
            ExpectNear(Center(renderer), grey);
        }
    });
}

TEST(SpectraTest, SpectraColourTheImageAndFollowEachCamerasBands) {
    ForEachDevice([](Device& device) {
        Wall wall = SpectralWall(device);
        wall.surface.Connect("diffuseColor", wall.spectrum, "spectrum");
        wall.spectrum.Set(SampledSpectrum::kWavelengths,
                          std::vector<float>{380, 619, 620, 680, 681, 780});
        wall.spectrum.Set(SampledSpectrum::kValues, std::vector<float>{0, 0, 1, 1, 0, 0});
        Renderer renderer(wall.scene);
        const Float3 red = Center(renderer);
        EXPECT_GT(red.x, 3 * red.z);
        EXPECT_GT(red.x, red.y);

        wall.product.Set(RenderProduct::kSpectralMode, SpectralMode::Bands);
        wall.camera.Set(Camera::kDetectorSpectralRange, Float2{700, 1000});
        wall.spectrum.Set(SampledSpectrum::kWavelengths, std::vector<float>{300, 1000});
        wall.spectrum.Set(SampledSpectrum::kValues, std::vector<float>{1, 1});
        const Float3 white = Center(renderer);
        ASSERT_GT(white.x, 0.05f);
        wall.spectrum.Set(SampledSpectrum::kValues, std::vector<float>{0, 0});
        const Float3 specular = Center(renderer);
        wall.spectrum.Set(SampledSpectrum::kWavelengths, std::vector<float>{300, 699, 700, 1000});
        wall.spectrum.Set(SampledSpectrum::kValues, std::vector<float>{1, 1, 0, 0});
        EXPECT_NEAR(Center(renderer).x, specular.x, 1e-3f);
        wall.spectrum.Set(SampledSpectrum::kWavelengths,
                          std::vector<float>{700, 812.4f, 812.6f, 1000});
        wall.spectrum.Set(SampledSpectrum::kValues, std::vector<float>{0, 0, 1, 1});
        const Float3 high = Center(renderer);
        wall.spectrum.Set(SampledSpectrum::kValues, std::vector<float>{1, 1, 0, 0});
        const Float3 low = Center(renderer);
        EXPECT_GT(high.x - specular.x, 0.2f * (white.x - specular.x));
        EXPECT_GT(low.x - specular.x, 0.2f * (white.x - specular.x));
        EXPECT_NEAR(high.x + low.x - specular.x, white.x, 0.01f * white.x);
    });
}

TEST(SpectraTest, NodeSpectralInputsTakeValuesAndSpectra) {
    const std::filesystem::path file = Scratch() / "ernie_render_spectra_node.slang";
    std::ofstream(file) << R"(
import ernie.material;

struct TestSpectralBase {
    SpectralValue reflectance = { 1, 1, float3(0.5, 0.5, 0.5) };

    void Material(inout SurfaceInputs m, ShadeContext c) {
        m.baseColor = reflectance;
    }
}
)";
    ForEachDevice([&](Device& device) {
        Wall wall = SpectralWall(device);
        EXPECT_EQ(LoadShaders(*wall.scene, file), std::vector<std::string>{"TestSpectralBase"});
        wall.surface.Set(Shader::kId, "TestSpectralBase");
        EXPECT_EQ(wall.surface.Get<Float3>("reflectance"), (Float3{0.5f, 0.5f, 0.5f}));
        wall.spectrum.Set(SampledSpectrum::kWavelengths, std::vector<float>{300, 1000});
        wall.spectrum.Set(SampledSpectrum::kValues, std::vector<float>{0.5f, 0.5f});
        Renderer renderer(wall.scene);
        const Float3 grey = Center(renderer);
        ASSERT_GT(grey.y, 0.05f);
        wall.surface.Connect("reflectance", wall.spectrum, "spectrum");
        ExpectNear(Center(renderer), grey);
        wall.spectrum.Set(SampledSpectrum::kValues, std::vector<float>{0.25f, 0.25f});
        EXPECT_LT(Center(renderer).y, grey.y);
    });
}

}
}
