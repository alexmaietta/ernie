// SPDX-License-Identifier: BSD-3-Clause
#include "support/delegate_scenes.h"

#include "ernie/raster/render_targets.h"
#include "ernie/scene/camera.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string_view>

namespace ernie::testing {

Tensor HeightsOf(const TestRaster& raster) {
    auto owner = std::make_shared<std::vector<float>>(raster.values);
    return Tensor(
        owner->data(), DType::Float32,
        {static_cast<std::int64_t>(raster.height), static_cast<std::int64_t>(raster.width)}, {}, {},
        owner);
}

Scene TerrainScene(const TestRaster& raster) {
    Scene scene;
    scene.Define("/World/Terrain", tokens::kTerrain)
        .Write(tokens::kHeights, HeightsOf(raster))
        .Set(tokens::kOrigin, glm::dvec2(raster.originX, raster.originY))
        .Set(tokens::kCellSize, raster.cellSize)
        .Set(tokens::kMaterialBinding, Path("/Looks/Rock"));
    scene.Define("/Looks/Rock", tokens::kMaterial).Set(tokens::kAlbedo, glm::vec3(0.6f));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, DistantLightTransform({1.0, 1.0, 0.6}));
    scene.Define("/Render/Settings", tokens::kRenderSettings)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(kSize, kSize))
        .Set(tokens::kClearColor, glm::vec3(0.0f, 0.0f, 1.0f));
    return scene;
}

void DefineQuad(Scene& scene, const Path& path, float half) {
    const std::array<float, 12> points{-half, -half, 0, half,  -half, 0,
                                       half,  half,  0, -half, half,  0};
    const std::array<std::int64_t, 1> counts{4};
    const std::array<std::int64_t, 4> indices{0, 1, 2, 3};
    scene.Define(path, tokens::kMesh)
        .Write(tokens::kPoints, Tensor::Copy<float>(points, {4, 3}))
        .Write(tokens::kFaceVertexCounts, Tensor::Copy<std::int64_t>(counts, {1}))
        .Write(tokens::kFaceVertexIndices, Tensor::Copy<std::int64_t>(indices, {4}));
}

Scene MeshScene() {
    Scene scene;
    scene.Define("/World/Camera", tokens::kCamera)
        .Set(tokens::kTransform,
             LookAtTransform({0.0, 0.0, 60.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}))
        .Set(tokens::kFocalLength, 18.0f);
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, DistantLightTransform({0.0, 0.0, 1.0}));
    scene.Define("/Render/Settings", tokens::kRenderSettings)
        .Set(tokens::kResolution, glm::ivec2(kSize, kSize))
        .Set(tokens::kAmbient, 0.0f);
    scene.Define("/Looks/Red", tokens::kMaterial).Set(tokens::kAlbedo, glm::vec3(1.0f, 0.0f, 0.0f));
    DefineQuad(scene, "/World/Quad");
    scene.Set("/World/Quad", tokens::kTransform,
              glm::translate(glm::dmat4(1.0), glm::dvec3(-12.0, 0.0, 0.0)));
    scene.Set("/World/Quad", tokens::kMaterialBinding, Path("/Looks/Red"));

    const std::array<float, 9> positions{12, -10, 0, 12, 0, 0, 12, 10, 0};
    scene.Define("/World/Rocks", tokens::kPointInstancer)
        .Write(tokens::kPositions, Tensor::Copy<float>(positions, {3, 3}))
        .Set(tokens::kPrototypes, std::vector<Path>{"/World/Rocks/Rock"});
    DefineQuad(scene, "/World/Rocks/Rock", 1.5f);
    scene.Set("/World/Rocks/Rock", tokens::kDisplayColor, glm::vec3(0.0f, 1.0f, 0.0f));
    return scene;
}

Scene CullingScene() {
    Scene scene = MeshScene();
    std::vector<float> positions;
    for (int i = -20; i <= 20; ++i) {
        positions.insert(positions.end(), {5.0f * static_cast<float>(i), -20.0f, 0.0f});
    }
    scene.Define("/World/Row", tokens::kPointInstancer)
        .Write(tokens::kPositions, Tensor::Copy<float>(positions, {41, 3}))
        .Set(tokens::kPrototypes, std::vector<Path>{"/World/Rocks/Rock"});
    return scene;
}

const std::uint8_t* TexelAt(const Scene& scene, const Tensor& image, glm::vec3 p) {
    const glm::vec4 clip =
        CameraParamsOf(scene, "/World/Camera", kSize, kSize).RasterView().ViewProjection() *
        glm::vec4(p, 1.0f);
    const glm::vec2 ndc = glm::vec2(clip) / clip.w;
    const auto col = static_cast<std::size_t>((ndc.x * 0.5f + 0.5f) * kSize);
    const auto row = static_cast<std::size_t>((0.5f - ndc.y * 0.5f) * kSize);
    return image.As<std::uint8_t>() + 4 * (row * kSize + col);
}

Scene FlatScene() {
    Scene scene;
    scene.Define("/World/Terrain", tokens::kTerrain)
        .Write(tokens::kHeights, HeightsOf(TestRaster::Centred(65, 65, 2.0)))
        .Set(tokens::kOrigin, glm::dvec2(-65.0, 65.0))
        .Set(tokens::kCellSize, 2.0)
        .Set(tokens::kMaterialBinding, Path("/Looks/Grey"));
    scene.Define("/Looks/Grey", tokens::kMaterial).Set(tokens::kAlbedo, glm::vec3(0.5f));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, DistantLightTransform({0.0, 0.0, 1.0}))
        .Set(tokens::kShadowEnable, false);
    scene.Define("/World/Camera", tokens::kCamera)
        .Set(tokens::kTransform,
             LookAtTransform({0.0, 0.0, 100.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}))
        .Set(tokens::kSensorNoise, false);
    std::vector<Path> vars;
    for (const std::string_view name : RenderTargets::VarNames()) {
        const Path var = Path("/Render/Vars").Child(std::string(name));
        scene.Define(var, tokens::kRenderVar).Set(tokens::kSourceName, std::string(name));
        vars.push_back(var);
    }
    scene.Define("/Render/Settings", tokens::kRenderSettings)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(kVarSize, kVarSize))
        .Set(tokens::kColorMode, tokens::kHillshade)
        .Set(tokens::kAmbient, 0.0f);
    scene.Define("/Render/Product", tokens::kRenderProduct).Set(tokens::kOrderedVars, vars);
    scene.Set("/Render/Settings", tokens::kProducts, std::vector<Path>{"/Render/Product"});
    return scene;
}

std::map<std::string, Tensor, std::less<>> MapVars(RenderProductSetOutputs& outputs) {
    std::map<std::string, Tensor, std::less<>> tensors;
    for (auto& [name, var] : outputs.at("/Render/Product").vars) {
        tensors[name] = var.Map(MapDevice::Cpu).tensor;
    }
    return tensors;
}

std::pair<std::size_t, std::size_t> PixelOf(const Scene& scene, float x, float y) {
    const glm::vec4 clip =
        CameraParamsOf(scene, "/World/Camera", kSize, kSize).RasterView().ViewProjection() *
        glm::vec4(x, y, 0.0f, 1.0f);
    const glm::vec2 ndc = glm::vec2(clip) / clip.w;
    return {static_cast<std::size_t>((0.5f - ndc.y * 0.5f) * kVarSize),
            static_cast<std::size_t>((ndc.x * 0.5f + 0.5f) * kVarSize)};
}

std::vector<std::uint8_t> Bytes(const Tensor& image) {
    const auto bytes =
        static_cast<std::size_t>(image.Elements()) * (image.Type() == DType::UInt8 ? 1 : 4);
    const auto* data = static_cast<const std::uint8_t*>(image.Data());
    return {data, data + bytes};
}

std::size_t BytesDiffering(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
    std::size_t differing = 0;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        differing += std::abs(int(a[i]) - int(b[i])) > 8 ? 1u : 0u;
    }
    return differing + (a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}

}  // namespace ernie::testing
