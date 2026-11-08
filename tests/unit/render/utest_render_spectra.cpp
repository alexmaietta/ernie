// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
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

TEST(SpectraTest, RgbColoursGetASmoothTailBeyondTheVisibleAndWarnOnce) {
    ForEachDevice([](Device& device) {
        Wall wall = SpectralWall(device);
        wall.product.Set(RenderProduct::kSpectralMode, SpectralMode::Bands);
        wall.camera.Set(Camera::kDetectorSpectralRange, Float2{700, 1000});
        Renderer renderer(wall.scene);
        const test::CapturedLog log(LogLevel::Warning);
        wall.surface.Set(UsdPreviewSurface::kDiffuseColor, Float3{0.5f, 0.5f, 0.5f});
        const Float3 grey = Center(renderer);
        wall.spectrum.Set(SampledSpectrum::kWavelengths, std::vector<float>{300, 1000});
        wall.spectrum.Set(SampledSpectrum::kValues, std::vector<float>{0.5f, 0.5f});
        wall.surface.Connect("diffuseColor", wall.spectrum, "spectrum");
        EXPECT_NEAR(Center(renderer).x, grey.x, 0.02f * grey.x);
        wall.surface.Disconnect("diffuseColor");
        Center(renderer);
        size_t warnings = 0;
        for (const auto& [level, message] : log.Messages()) {
            warnings += message.find("has only RGB colours") != std::string::npos ? 1 : 0;
        }
        EXPECT_EQ(warnings, 1u);
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

struct Returns {
    Ref<Scene> scene;
    Prim surface;
    Prim spectrum;
    Prim lidar;
};

Returns Wall(Device& device, float elevation) {
    Returns wall{.scene = Scene::Create(device)};
    Scene& scene = *wall.scene;
    scene.Define<Scope>("/Looks");
    const Prim material = scene.Define<Material>("/Looks/Paint");
    wall.surface = scene.Define<Shader>("/Looks/Paint/Surface", {.id = "UsdPreviewSurface"});
    material.Connect("surface", wall.surface, "surface");
    wall.spectrum = scene.Define<Shader>("/Looks/Paint/Spectrum", {.id = "SampledSpectrum"});
    scene.Define<Plane>(
        "/Wall",
        {.materialBinding = {"/Looks/Paint"}, .translate = {0, 0, -5}, .width = 0, .length = 0});
    wall.lidar = scene.Define<Lidar>(
        "/Lidar",
        {.scanType = ScanType::SolidState, .firingsPerScan = 1, .emitterElevations = {elevation}});
    scene.Define<Radar>("/Radar", {.fieldOfView = {2, 2}, .rays = {kSize, kSize}});
    scene.Define<Scope>("/Render");
    scene.Define<RenderVar>("/Render/Points",
                            {.sourceName = "PointCloud", .channels = {"Intensity"}});
    scene.Define<RenderProduct>("/Render/Scan",
                                {.camera = {"/Lidar"}, .orderedVars = {"/Render/Points"}});
    scene.Define<RenderVar>("/Render/Echo", {.sourceName = "Intensity"});
    scene.Define<RenderProduct>("/Render/Radar",
                                {.camera = {"/Radar"}, .orderedVars = {"/Render/Echo"}});
    return wall;
}

float LidarIntensity(Renderer& renderer) {
    RenderOutput output = renderer.Step(std::vector<std::string>{"/Render/Scan"}, 0);
    output.Wait();
    const std::vector<float> values = output["/Render/Scan"]
                                          .frames.at(0)
                                          .Source("PointCloud")
                                          .channels.at("Intensity")
                                          .ToHost<float>();
    return values.empty() ? 0.0f : values.front();
}

float RadarIntensity(Renderer& renderer) {
    RenderOutput output = renderer.Step(std::vector<std::string>{"/Render/Radar"}, 0);
    output.Wait();
    const std::vector<float> values =
        output["/Render/Radar"].frames.at(0).Source("Intensity").tensor.ToHost<float>();
    return values.at((kSize / 2) * kSize + kSize / 2);
}

TEST(SpectraTest, LidarReadsMaterialsAtItsLaserWavelength) {
    ForEachDevice([](Device& device) {
        Returns wall = Wall(device, 0);
        wall.surface.Connect("diffuseColor", wall.spectrum, "spectrum");
        wall.spectrum.Set(SampledSpectrum::kWavelengths, std::vector<float>{380, 780, 800, 1000});
        Renderer renderer(wall.scene);
        wall.spectrum.Set(SampledSpectrum::kValues, std::vector<float>{0.9f, 0.9f, 0.1f, 0.1f});
        const float visible = LidarIntensity(renderer);
        wall.spectrum.Set(SampledSpectrum::kValues, std::vector<float>{0.1f, 0.1f, 0.9f, 0.9f});
        const float infrared = LidarIntensity(renderer);
        ASSERT_GT(visible, 0);
        EXPECT_NEAR(infrared / visible, 9, 0.1f);

        wall.lidar.Set(Lidar::kBeamWavelength, 550.0f);
        EXPECT_NEAR(LidarIntensity(renderer) / infrared, 1.0f / 9, 0.01f);

        wall.lidar.Set(Lidar::kBeamWavelength, 905.0f);
        wall.spectrum.Set(SampledSpectrum::kValues, std::vector<float>{0.5f, 0.5f, 0.5f, 0.5f});
        const float flat = LidarIntensity(renderer);
        wall.surface.Disconnect("diffuseColor");
        wall.surface.Set(UsdPreviewSurface::kDiffuseColor, Float3{0.5f, 0.5f, 0.5f});
        EXPECT_NEAR(LidarIntensity(renderer), flat, 0.03f * flat);
    });
}

TEST(SpectraTest, RetroreflectionAndRadarBackscatterComeFromTheMaterial) {
    const std::filesystem::path file = Scratch() / "ernie_render_spectra_returns.slang";
    std::ofstream(file) << R"(
import ernie.material;

struct TestReturns {
    float retro = 0;
    float backscatter = 1;

    void Material(inout SurfaceInputs m, ShadeContext c) {
        m.baseColor = SpectralValue.Constant(0.5);
        m.retroreflection = retro;
        m.radarBackscatter = backscatter;
    }
}
)";
    ForEachDevice([&](Device& device) {
        Returns wall = Wall(device, 45);
        LoadShaders(*wall.scene, file);
        wall.surface.Set(Shader::kId, "TestReturns");
        Renderer renderer(wall.scene);
        const float diffuse = LidarIntensity(renderer);
        ASSERT_GT(diffuse, 0);
        wall.surface.Set("retro", 1.0f);
        EXPECT_NEAR(LidarIntensity(renderer) / diffuse, std::sqrt(2.0f), 0.01f);

        const float echo = RadarIntensity(renderer);
        ASSERT_GT(echo, 0);
        wall.surface.Set("backscatter", 0.25f);
        EXPECT_NEAR(RadarIntensity(renderer) / echo, 0.25f, 1e-3f);
    });
}

}
}
